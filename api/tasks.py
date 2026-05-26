from celery import shared_task
from nlp.matcher import extract_skills, calculate_match, identify_gaps, get_strengths_and_weaknesses, extract_years_of_experience
from nlp.analysis import generate_suggestions, generate_interview_questions


@shared_task(bind=True, max_retries=3)
def run_analysis_async(self, resume_id, job_id, session_id=None):
    """
    FIX 9: Async Celery task for NLP analysis pipeline.
    Called by the analyze endpoint — offloads heavy ML work to a background worker.
    """
    from api.models import Resume, JobDescription, AnalysisResult, Session

    try:
        resume = Resume.objects.get(id=resume_id)
        job = JobDescription.objects.get(id=job_id)
        session = Session.objects.get(id=session_id) if session_id else None

        full_jd_text = f"{job.title}\n{job.description_text}"
        live_resume_skills = extract_skills(resume.parsed_text or "", contextual=True)

        # Save freshly extracted skills
        if set(live_resume_skills) != set(resume.extracted_skills or []):
            resume.extracted_skills = live_resume_skills
            resume.save(update_fields=['extracted_skills'])

        match_percentage, score_breakdown = calculate_match(resume.parsed_text, full_jd_text, live_resume_skills)
        missing_skills = identify_gaps(live_resume_skills, full_jd_text)

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
            score_breakdown=score_breakdown,
            session=session,
        )
        return analysis.id

    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
