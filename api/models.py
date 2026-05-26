from django.db import models
from django.contrib.postgres.fields import ArrayField


class Session(models.Model):
    """
    FIX 7: Lightweight anonymous session to track a user's analysis history.
    A session_key is generated on the frontend and sent with every request.
    No login required — stored in browser localStorage.
    """
    session_key = models.CharField(max_length=64, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_active = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Session {self.session_key[:8]}... ({self.created_at.strftime('%Y-%m-%d')})"


class Resume(models.Model):
    file = models.FileField(upload_to='resumes/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    parsed_text = models.TextField(blank=True, null=True)
    extracted_skills = ArrayField(models.CharField(max_length=255), default=list, blank=True)
    # FIX 3: SHA-256 hash of file content for deduplication
    file_hash = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    # FIX 7: Optional link to a session
    session = models.ForeignKey(Session, on_delete=models.SET_NULL, null=True, blank=True, related_name='resumes')
    # FIX 10: Embedding stored as JSON (migrates to pgvector VectorField post docker rebuild)
    embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Resume {self.id} - {self.uploaded_at.strftime('%Y-%m-%d')}"


class JobDescription(models.Model):
    title = models.CharField(max_length=255, blank=True)
    description_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    # FIX 7: Optional link to a session
    session = models.ForeignKey(Session, on_delete=models.SET_NULL, null=True, blank=True, related_name='jobs')

    def __str__(self):
        return self.title or f"Job {self.id}"


class AnalysisResult(models.Model):
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name='analyses')
    job_description = models.ForeignKey(JobDescription, on_delete=models.CASCADE, related_name='analyses')
    match_percentage = models.FloatField()
    missing_skills = ArrayField(models.CharField(max_length=255), default=list)
    strengths = ArrayField(models.CharField(max_length=255), default=list)
    weaknesses = ArrayField(models.CharField(max_length=255), default=list)
    improvement_suggestions = models.TextField(blank=True, null=True)
    interview_questions = ArrayField(models.TextField(), default=list)
    # FIX 2: Per-component score breakdown stored as JSON
    score_breakdown = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # FIX 7: Optional link to a session
    session = models.ForeignKey(Session, on_delete=models.SET_NULL, null=True, blank=True, related_name='analyses')

    def __str__(self):
        return f"Analysis {self.id}: {self.match_percentage}% Match"
