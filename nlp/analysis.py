def generate_suggestions(gaps, match_percentage):
    """
    Generates resume improvement suggestions based on match results.
    """
    suggestions = []
    
    if match_percentage < 40:
        suggestions.append("Critical: Your resume lacks many core requirements for this role. Consider a significant rewrite.")
    elif match_percentage < 70:
        suggestions.append("Improvement: You have a good foundation, but need to better align your experience with the job keywords.")
    else:
        suggestions.append("Strong Match: Your resume is well-aligned. Focus on quantifying your achievements.")

    if gaps:
        suggestions.append(f"Skill Gap: The following key skills were found in the job description but not in your resume: {', '.join(gaps[:5])}. If you have these skills, make sure they are explicitly mentioned.")
    
    suggestions.append("ATS Tip: Use standard headings like 'Experience', 'Education', and 'Skills' to ensure parser compatibility.")
    suggestions.append("ATS Tip: Avoid using images, charts, or complex multi-column layouts that can confuse older ATS systems.")
    
    return "\n".join(suggestions)

def generate_interview_questions(found_skills, missing_skills):
    """
    Generates potential interview questions based on the candidate's profile and gaps.
    """
    questions = []
    
    # Questions for skills they have
    for skill in found_skills[:2]:
        questions.append(f"Can you walk me through a complex problem you solved using {skill}?")
        questions.append(f"In your experience, what are the most common pitfalls when working with {skill}, and how do you avoid them?")
        
    # Questions for skills they might be missing (testing adaptability or equivalent experience)
    for skill in missing_skills[:2]:
        questions.append(f"The role requires experience with {skill}. While not explicitly on your resume, do you have equivalent experience or a plan to get up to speed quickly?")
        
    if not questions:
        questions.append("Tell me about your most significant professional achievement.")
        questions.append("How do you stay updated with the latest trends in your field?")
        
    return questions
