import hashlib
import numpy as np
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import Resume, JobDescription, AnalysisResult, Session
from .serializers import ResumeSerializer, JobDescriptionSerializer, AnalysisResultSerializer, SessionSerializer
from nlp.parser import parse_resume
from nlp.matcher import (extract_skills, calculate_match, identify_gaps,
                          get_strengths_and_weaknesses, extract_years_of_experience,
                          get_chunked_embedding)
from nlp.analysis import generate_suggestions, generate_interview_questions

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

try:
    from pgvector.django import L2Distance
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    L2Distance = None


ALLOWED_MIME_TYPES = {
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
}


def get_or_create_session(session_key):
    """FIX 7: Get or create a session by key."""
    if not session_key:
        return None
    session, _ = Session.objects.get_or_create(session_key=session_key)
    return session


class SessionViewSet(viewsets.ReadOnlyModelViewSet):
    """FIX 7: Endpoint to retrieve session history."""
    serializer_class = SessionSerializer

    def get_queryset(self):
        session_key = self.request.query_params.get('session_key')
        if session_key:
            return Session.objects.filter(session_key=session_key)
        return Session.objects.none()

    @action(detail=False, methods=['get'])
    def history(self, request):
        """Returns all analyses for a given session_key."""
        session_key = request.query_params.get('session_key')
        if not session_key:
            return Response({"error": "session_key is required"}, status=status.HTTP_400_BAD_REQUEST)
        analyses = AnalysisResult.objects.filter(
            session__session_key=session_key
        ).select_related('resume', 'job_description').order_by('-created_at')
        return Response(AnalysisResultSerializer(analyses, many=True).data)


class ResumeViewSet(viewsets.ModelViewSet):
    queryset = Resume.objects.all()
    serializer_class = ResumeSerializer

    def perform_create(self, serializer):
        resume = serializer.save()
        try:
            text = parse_resume(resume.file.path)
            skills = extract_skills(text, contextual=True)
            resume.parsed_text = text
            resume.extracted_skills = skills
            resume.save()
        except Exception as e:
            print(f"Parsing error: {e}")

    def create(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Read file content once for MIME check + hash
        file_bytes = file_obj.read()
        file_obj.seek(0)

        # FIX 8: MIME type validation (only when python-magic is available)
        if MAGIC_AVAILABLE:
            try:
                mime = magic.from_buffer(file_bytes, mime=True)
            except Exception:
                mime = file_obj.content_type
            if mime not in ALLOWED_MIME_TYPES:
                return Response(
                    {"error": f"Unsupported file type '{mime}'. Only PDF and DOCX are allowed."},
                    status=status.HTTP_400_BAD_REQUEST
                )

        # FIX 3: Deduplication by SHA-256 hash

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = Resume.objects.filter(file_hash=file_hash).first()
        if existing:
            return Response(ResumeSerializer(existing).data, status=status.HTTP_200_OK)

        session_key = request.data.get('session_key')
        session = get_or_create_session(session_key)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        resume = serializer.save(file_hash=file_hash, session=session)

        # Parse, extract skills, and compute embedding
        try:
            text = parse_resume(resume.file.path)
            skills = extract_skills(text, contextual=True)
            # FIX 10: Compute and store sentence embedding for pgvector search
            emb = get_chunked_embedding(text)
            if emb.dim() > 1:
                emb = emb.squeeze(0)
            resume.parsed_text = text
            resume.extracted_skills = skills
            resume.embedding = emb.cpu().numpy().tolist()
            resume.save()
        except Exception as e:
            print(f"Parsing error: {e}")

        return Response(ResumeSerializer(resume).data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'])
    def semantic_search(self, request):
        """
        FIX 10: Find the top-N stored resumes most semantically similar to a job description.
        POST body: { "job_description": "...", "top_n": 5 }
        """
        if not PGVECTOR_AVAILABLE:
            return Response({"error": "pgvector not available on this deployment."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        jd_text = request.data.get('job_description', '')
        top_n = int(request.data.get('top_n', 5))
        if not jd_text:
            return Response({"error": "job_description is required"}, status=status.HTTP_400_BAD_REQUEST)

        jd_emb = get_chunked_embedding(jd_text)
        if jd_emb.dim() > 1:
            jd_emb = jd_emb.squeeze(0)
        jd_vector = jd_emb.cpu().numpy().tolist()

        results = (
            Resume.objects
            .exclude(embedding__isnull=True)
            .order_by(L2Distance('embedding', jd_vector))[:top_n]
        )
        return Response(ResumeSerializer(results, many=True).data)




class JobDescriptionViewSet(viewsets.ModelViewSet):
    queryset = JobDescription.objects.all()
    serializer_class = JobDescriptionSerializer

    def create(self, request, *args, **kwargs):
        session_key = request.data.get('session_key')
        session = get_or_create_session(session_key)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(session=session)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class AnalysisViewSet(viewsets.ModelViewSet):
    queryset = AnalysisResult.objects.all()
    serializer_class = AnalysisResultSerializer

    @action(detail=False, methods=['post'])
    def analyze(self, request):
        resume_id = request.data.get('resume_id')
        job_id = request.data.get('job_id')
        session_key = request.data.get('session_key')

        if not resume_id or not job_id:
            return Response({"error": "resume_id and job_id are required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resume = Resume.objects.get(id=resume_id)
            job = JobDescription.objects.get(id=job_id)
        except (Resume.DoesNotExist, JobDescription.DoesNotExist):
            return Response({"error": "Resume or Job Description not found"}, status=status.HTTP_404_NOT_FOUND)

        # FIX 7: Link to session
        session = get_or_create_session(session_key)

        # Combine job title + description so seniority keywords in the title are detected
        full_jd_text = f"{job.title}\n{job.description_text}"

        # FIX 1: Re-extract skills live at analysis time (contextual for resume, simple for JD)
        live_resume_skills = extract_skills(resume.parsed_text or "", contextual=True)

        # FIX 1: Save the freshly extracted skills back to the resume record
        if set(live_resume_skills) != set(resume.extracted_skills or []):
            resume.extracted_skills = live_resume_skills
            resume.save(update_fields=['extracted_skills'])

        # Perform analysis — now returns (score, breakdown)
        match_percentage, score_breakdown = calculate_match(resume.parsed_text, full_jd_text, live_resume_skills)
        missing_skills = identify_gaps(live_resume_skills, full_jd_text)

        # FIX 4: Compute strengths and weaknesses
        resume_exp, num_internships = extract_years_of_experience(resume.parsed_text or "")
        strengths, weaknesses = get_strengths_and_weaknesses(
            live_resume_skills, full_jd_text, match_percentage, resume_exp, num_internships
        )

        suggestions = generate_suggestions(missing_skills, match_percentage)
        questions = generate_interview_questions(live_resume_skills, missing_skills)

        analysis = AnalysisResult.objects.create(
            resume=resume,
            job_description=job,
            match_percentage=match_percentage,
            missing_skills=missing_skills,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            interview_questions=questions,
            score_breakdown=score_breakdown,  # FIX 2
            session=session,                  # FIX 7
        )

        return Response(AnalysisResultSerializer(analysis).data, status=status.HTTP_201_CREATED)
