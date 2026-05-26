import spacy
from sentence_transformers import SentenceTransformer, util
import re
from datetime import datetime

# Load models once
try:
    nlp = spacy.load("en_core_web_md")
except:
    nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Expanded tech, business, and soft skills for keyword matching
COMMON_SKILLS = [
    # Tech / Engineering
    "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue", "Node.js",
    "Django", "Flask", "FastAPI", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Machine Learning", "Deep Learning",
    "NLP", "PyTorch", "TensorFlow", "Pandas", "NumPy", "Scikit-learn", "Git", "CI/CD",
    "Agile", "Scrum", "REST API", "GraphQL", "TypeScript", "SQL", "NoSQL",

    # Human Resources & Management
    "Human Resources", "HR", "Talent Acquisition", "Employee Engagement",
    "Performance Management", "HR Operations", "Organizational Development",
    "Policy Implementation", "Onboarding", "Recruitment", "Payroll", "Benefits Administration",

    # Business & Strategy
    "Leadership", "Project Management", "Operations", "Strategic Planning",
    "Budgeting", "Sales", "Marketing", "Business Strategy", "Business Development",
    "Data Analysis", "Financial Modeling", "Product Management", "Risk Management",

    # Soft Skills & General
    "Communication", "Problem Solving", "Team Building", "Negotiation",
    "Public Speaking", "Customer Service", "Writing", "Research", "Time Management",
    "Critical Thinking", "Adaptability", "Collaboration", "Mentoring"
]

# Action verbs that indicate a skill is being used in context (Fix 1)
ACTION_VERBS = {
    "developed", "built", "designed", "implemented", "led", "managed", "used",
    "utilized", "applied", "worked", "experience", "experienced", "proficient",
    "skilled", "expertise", "knowledge", "deployed", "maintained", "created",
    "architected", "established", "coordinated", "handled", "conducted", "oversaw",
    "improved", "analyzed", "delivered", "executed", "spearheaded", "drove"
}

def extract_skills(text, contextual=True):
    """
    Extracts skills from text using keyword matching.
    - Removed noisy spaCy NER (ORG/PRODUCT) which mis-labels company names as skills.
    - contextual=True (for resumes): each skill must appear in a Skills section OR
      within 10 tokens of an action verb. Prevents keyword-stuffing exploits.
    - contextual=False (for job descriptions): skips verb-proximity check since JDs
      list required skills as bullet points without action verbs.
    """
    found_skills = set()
    text_lower = text.lower()

    # FIX 5: Fuzzy section header detection — handles ALL CAPS, colons, extra spaces,
    # and synonyms like 'Technical Skills', 'Core Competencies', 'Tech Stack' etc.
    skills_section_pattern = (
        r'(?:technical\s+)?(?:skills?|technologies|tools?|expertise|competencies|'
        r'tech\s+stack|core\s+skills?|key\s+skills?|proficiencies|qualifications)'
        r'[:\s\n]+(.*?)(?:\n\s*\n|\Z)'
    )
    skills_sections = re.findall(skills_section_pattern, text_lower, re.IGNORECASE | re.DOTALL)
    skills_section_text = " ".join(skills_sections)

    if not contextual:
        # FIX 6: For job descriptions — simple keyword scan, no verb proximity needed
        for skill in COMMON_SKILLS:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)
        return list(found_skills)

    # Contextual mode for resumes
    doc = nlp(text_lower)
    verb_positions = {token.i for token in doc if token.lemma_ in ACTION_VERBS or token.pos_ == "VERB"}

    for skill in COMMON_SKILLS:
        skill_lower = skill.lower()
        pattern = r'\b' + re.escape(skill_lower) + r'\b'

        # Skills section match — always valid regardless of context
        if re.search(pattern, skills_section_text):
            found_skills.add(skill)
            continue

        # Otherwise validate contextually: must be near an action verb
        for match in re.finditer(pattern, text_lower):
            match_start = match.start()
            char_to_tok = {tok.idx: tok.i for tok in doc}
            closest_tok_i = min(
                (abs(idx - match_start), tok_i) for idx, tok_i in char_to_tok.items()
            )[1] if char_to_tok else 0

            context_valid = any(
                abs(closest_tok_i - v_i) <= 10
                for v_i in verb_positions
            )
            if context_valid:
                found_skills.add(skill)
                break

    return list(found_skills)


def parse_date(date_str):
    date_str = date_str.strip()
    if re.match(r'^\d{4}$', date_str):
        return datetime.strptime(date_str, "%Y")
    for fmt in ["%b %Y", "%B %Y", "%m/%Y", "%m-%Y", "%b. %Y", "%B, %Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def merge_overlapping_intervals(intervals):
    """
    FIX 3: Merge overlapping date intervals to avoid double-counting.
    E.g., [(2020-01, 2022-01), (2020-06, 2021-06)] → [(2020-01, 2022-01)]
    Input: list of (start_month_index, end_month_index)
    Output: merged list
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping — extend if needed
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def calculate_duration_in_years(text, exclude_internships=False):
    """
    Section-aware, overlap-safe date duration calculation.
    
    When exclude_internships=True:
      - Splits the resume into sections by heading keywords.
      - Only extracts date ranges from sections identified as full-time
        employment (e.g. 'Work Experience', 'Professional Experience').
      - Skips sections that are Education or Internship sections entirely.
      - Merges overlapping intervals to avoid double-counting side projects.
    
    When exclude_internships=False:
      - Extracts all date ranges regardless of section (used for counting
        total experience including internships).
    """
    if not text:
        return 0.0

    date_pattern = re.compile(
        r'((?:[A-Z][a-z]{2,8}\.?\s+)?\d{4})\s*(?:-|to|–)\s*((?:[A-Z][a-z]{2,8}\.?\s+)?\d{4}|Present|Current)',
        re.IGNORECASE
    )

    if not exclude_internships:
        # Simple mode: collect all date ranges
        intervals = []
        for m in date_pattern.finditer(text):
            start_date = parse_date(m.group(1))
            end_date = datetime.now() if m.group(2).lower() in ['present', 'current'] else parse_date(m.group(2))
            if start_date and end_date and end_date >= start_date:
                intervals.append((start_date.year * 12 + start_date.month,
                                   end_date.year * 12 + end_date.month))
        merged = merge_overlapping_intervals(intervals)
        return round(sum(e - s for s, e in merged) / 12.0, 1)

    # ── SECTION-AWARE mode ────────────────────────────────────────────────────
    # Keywords that mark the START of each section type
    FULLTIME_HEADERS = re.compile(
        r'^\s*(?:work\s+experience|professional\s+experience|employment|career\s+history|experience)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    SKIP_HEADERS = re.compile(
        r'^\s*(?:education|internship|internships|training|certification|certifications|'
        r'projects|academic|publications|awards|extracurricular|activities|'
        r'volunteer|languages|references|declaration)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    ANY_HEADER = re.compile(
        r'^\s*(?:work\s+experience|professional\s+experience|employment|career\s+history|experience|'
        r'education|internship|internships|training|certification|certifications|'
        r'projects|academic|publications|awards|extracurricular|activities|'
        r'volunteer|languages|references|declaration|skills?|summary|objective|profile)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )

    # Split text into (section_header, section_body) pairs
    lines = text.split('\n')
    sections = []          # list of (is_fulltime: bool, section_text: str)
    current_is_fulltime = False
    current_lines = []

    for line in lines:
        if ANY_HEADER.match(line):
            # Save previous section
            if current_lines:
                sections.append((current_is_fulltime, '\n'.join(current_lines)))
            # Determine type of new section
            current_is_fulltime = bool(FULLTIME_HEADERS.match(line))
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_is_fulltime, '\n'.join(current_lines)))

    # If no sections were found (resume has no recognised headers),
    # fall back to the window-based internship exclusion
    fulltime_sections = [body for (is_ft, body) in sections if is_ft]
    if not fulltime_sections:
        # Fallback: exclude date ranges that appear near internship keywords
        internship_marker = re.compile(r'\b(intern|internship|trainee)\b', re.IGNORECASE)
        education_marker  = re.compile(
            r'\b(bachelor|master|b\.?tech|m\.?tech|b\.?e|mba|degree|diploma|university|college|school|cgpa|gpa)\b',
            re.IGNORECASE
        )
        intervals = []
        for m in date_pattern.finditer(text):
            window = text[max(0, m.start()-400): min(len(text), m.end()+400)]
            if internship_marker.search(window) or education_marker.search(window):
                continue
            start_date = parse_date(m.group(1))
            end_date = datetime.now() if m.group(2).lower() in ['present', 'current'] else parse_date(m.group(2))
            if start_date and end_date and end_date >= start_date:
                intervals.append((start_date.year * 12 + start_date.month,
                                   end_date.year * 12 + end_date.month))
        merged = merge_overlapping_intervals(intervals)
        return round(sum(e - s for s, e in merged) / 12.0, 1)

    # Collect date ranges only from full-time sections
    intervals = []
    for body in fulltime_sections:
        for m in date_pattern.finditer(body):
            start_date = parse_date(m.group(1))
            end_date = datetime.now() if m.group(2).lower() in ['present', 'current'] else parse_date(m.group(2))
            if start_date and end_date and end_date >= start_date:
                intervals.append((start_date.year * 12 + start_date.month,
                                   end_date.year * 12 + end_date.month))

    merged = merge_overlapping_intervals(intervals)
    return round(sum(e - s for s, e in merged) / 12.0, 1)



def extract_years_of_experience(text, is_job_description=False):
    """
    Extracts the years of experience.
    Uses explicit regex for Job Descriptions, Date Math (overlap-safe) for Resumes,
    and Internship counting for freshers.
    Returns a tuple: (full_time_years, num_internships)
    """
    if not text:
        return (0.0, 0)

    extracted_years = 0.0
    num_internships = 0

    # 1. Try explicit regex (Crucial for Job Descriptions)
    pattern = r'(\d+)(?:\+|-)?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        try:
            years = [int(m) for m in matches if int(m) < 40]
            if years:
                extracted_years = float(max(years))
        except ValueError:
            pass

    # 2. Try Date Math — explicitly exclude internship date ranges
    if extracted_years == 0.0:
        extracted_years = calculate_duration_in_years(text, exclude_internships=True)

    # 3. Fresher Fallback: Count Internships
    internship_pattern = r'\b(intern|internship)\b'
    intern_matches = re.findall(internship_pattern, text, re.IGNORECASE)
    num_internships = len(intern_matches)

    if is_job_description:
        is_senior = bool(re.search(r'\b(senior|lead|principal|staff)\b', text, re.IGNORECASE))
        is_junior = bool(re.search(r'\b(junior|entry\s*level|associate|fresher)\b', text, re.IGNORECASE))

        if is_senior:
            extracted_years = max(4.0, extracted_years)
        elif is_junior and extracted_years == 0.0:
            extracted_years = 1.0

        return (extracted_years, 0)

    if extracted_years < 1.0:
        return (0.0, num_internships)

    return (extracted_years, num_internships)


def get_chunked_embedding(text, chunk_size=400):
    """
    FIX 2: Token truncation fix.
    Splits long text into overlapping chunks of ~400 tokens, encodes each chunk
    independently, and returns the averaged embedding to ensure no section of
    a long resume is silently dropped.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return model.encode(text, convert_to_tensor=True)

    chunks = []
    stride = chunk_size // 2  # 50% overlap between chunks
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    embeddings = model.encode(chunks, convert_to_tensor=True)
    # Average all chunk embeddings
    avg_embedding = embeddings.mean(dim=0)
    return avg_embedding


def calculate_match(resume_text, job_description, resume_skills=None):
    """
    Calculates a final match score (0-100%) using a WEIGHTED COMPOSITE model.
    Each component has a fixed maximum contribution to prevent score inflation:
      - Semantic Similarity : 40 pts max
      - Experience Score    : 30 pts max  (with role-mismatch penalty)
      - Skill Score         : 20 pts max  (overlap 15 + volume 5)
      - Summary Relevance   : 10 pts max
    """
    if not resume_text or not job_description:
        return 0.0

    # ── 1. SEMANTIC SIMILARITY (max 40 pts) ─────────────────────────────────
    embeddings1 = get_chunked_embedding(resume_text)
    embeddings2 = get_chunked_embedding(job_description)
    if embeddings1.dim() == 1:
        embeddings1 = embeddings1.unsqueeze(0)
    if embeddings2.dim() == 1:
        embeddings2 = embeddings2.unsqueeze(0)

    cosine_sim = float(util.pytorch_cos_sim(embeddings1, embeddings2)[0][0])
    semantic_score = round(cosine_sim * 40.0, 2)   # scale 0-1 → 0-40 pts
    print(f"[SCORE] Semantic: {cosine_sim:.2f} → {semantic_score:.1f}/40 pts")

    # ── 2. EXPERIENCE SCORE (max 30 pts) ────────────────────────────────────
    job_exp, _ = extract_years_of_experience(job_description, is_job_description=True)
    resume_exp, num_internships = extract_years_of_experience(resume_text, is_job_description=False)

    is_job_senior = bool(re.search(r'\b(senior|lead|principal|staff)\b', job_description, re.IGNORECASE))
    is_job_junior = bool(re.search(r'\b(junior|entry\s*level|associate|fresher)\b', job_description, re.IGNORECASE))

    # Effective experience: internships only count for non-senior roles
    if is_job_senior:
        effective_exp = resume_exp          # Internships DO NOT count for senior roles
    else:
        effective_exp = resume_exp if resume_exp >= 1.0 else (num_internships * 0.5)

    print(f"[SCORE] job_exp={job_exp}, resume_exp={resume_exp}, internships={num_internships}, effective={effective_exp}")
    print(f"[SCORE] is_job_senior={is_job_senior}, is_job_junior={is_job_junior}")

    # Base experience score (0–20 pts): full-time >> internships
    if resume_exp >= 1.0:
        # Professionals: scale based on years, cap at 20
        exp_base = min(20.0, resume_exp * 3.0)
    elif num_internships > 0 and not is_job_senior:
        # Freshers: each internship worth 2 pts, cap at 10
        exp_base = min(10.0, num_internships * 2.0)
    else:
        exp_base = 0.0

    # Experience gap modifier (–10 pts max) vs role requirement
    if job_exp > 0:
        if effective_exp >= job_exp:
            exp_gap_score = 10.0    # Full marks for meeting requirement
        else:
            gap = job_exp - effective_exp
            exp_gap_score = max(0.0, 10.0 - gap * 3.0)   # lose 3 pts per year short
    else:
        exp_gap_score = 5.0         # No explicit requirement stated: neutral

    experience_score = round(exp_base + exp_gap_score, 2)   # max = 30 pts
    experience_score = min(30.0, experience_score)

    # Role mismatch hard penalties (applied AFTER cap)
    is_candidate_senior = effective_exp >= 4.0
    is_candidate_junior = effective_exp <= 1.5

    if is_job_senior and is_candidate_junior:
        experience_score = max(0.0, experience_score - 20.0)   # severe penalty
        print(f"[SCORE] ROLE MISMATCH: junior→senior, -20 exp pts")
    elif is_job_junior and is_candidate_senior:
        experience_score = min(30.0, experience_score + 5.0)   # slight boost
        print(f"[SCORE] ROLE MATCH: senior→junior, +5 exp pts")

    print(f"[SCORE] Experience score: {experience_score:.1f}/30 pts")

    # ── 3. SKILL SCORE (max 20 pts) ─────────────────────────────────────────
    skill_score = 0.0
    if resume_skills is not None:
        job_skills = extract_skills(job_description, contextual=False)  # FIX 6: no verb check for JDs
        if job_skills:
            matched = [s for s in job_skills if s.lower() in [r.lower() for r in resume_skills]]
            overlap_ratio = len(matched) / len(job_skills)
            skill_score += overlap_ratio * 15.0   # up to 15 pts for perfect overlap
            print(f"[SCORE] Skill overlap: {len(matched)}/{len(job_skills)} → +{overlap_ratio*15:.1f} pts")

        # Volume bonus: rewarded for breadth of skills (up to 5 pts)
        vol = min(5.0, len(resume_skills) * 0.3)
        skill_score += vol
        print(f"[SCORE] Skill volume: {len(resume_skills)} skills → +{vol:.1f} pts")

    skill_score = min(20.0, round(skill_score, 2))
    print(f"[SCORE] Skill score: {skill_score:.1f}/20 pts")

    # ── 4. NEATNESS SCORE (max 10 pts) ───────────────────────────────────
    neatness_score = 5.0  # Base neatness

    # Check for bullet points (indicates good formatting)
    bullet_points = len(re.findall(r'^[ \t]*[-*•]\s+', resume_text, re.MULTILINE))
    if bullet_points > 10:
        neatness_score += 2.5
    elif bullet_points > 3:
        neatness_score += 1.5

    # Check for sections (ALL CAPS lines or Camel Case followed by newline/colon)
    sections = len(re.findall(r'^[A-Z][a-zA-Z\s]+[:]?\s*$', resume_text, re.MULTILINE))
    if sections >= 3:
        neatness_score += 1.5

    # Reward reasonable density (not too dense, not too sparse)
    if len(resume_text) > 0:
        density = len(resume_text.splitlines()) / len(resume_text)
        if density < 0.01: # extremely dense, very long lines
            neatness_score -= 2.0
        elif density > 0.02: # good amount of spacing
            neatness_score += 1.0

    neatness_score = round(max(0.0, min(10.0, neatness_score)), 2)
    print(f"[SCORE] Neatness score: {neatness_score:.1f}/10 pts")

    # ── FINAL COMPOSITE ─────────────────────────────────────────────────────
    total = semantic_score + experience_score + skill_score + neatness_score
    final = round(max(0.0, min(100.0, total)), 2)
    print(f"[SCORE] FINAL = {semantic_score:.1f} + {experience_score:.1f} + {skill_score:.1f} + {neatness_score:.1f} = {final}%")
    # Return breakdown alongside final score
    breakdown = {
        "semantic": round(semantic_score, 2),
        "experience": round(experience_score, 2),
        "skills": round(skill_score, 2),
        "neatness": round(neatness_score, 2),
        "total": final
    }
    return final, breakdown


# (duplicate old calculate_match removed)



def identify_gaps(resume_skills, job_text):
    """
    Identifies skills mentioned in job description but missing from resume.
    Uses contextual=False since JDs list skills without action verbs.
    """
    job_skills = extract_skills(job_text, contextual=False)  # FIX 6
    resume_skills_lower = [s.lower() for s in resume_skills]
    gaps = [skill for skill in job_skills if skill.lower() not in resume_skills_lower]
    return gaps


def get_strengths_and_weaknesses(resume_skills, job_text, match_percentage, resume_exp, num_internships):
    """
    FIX 4: Derives strengths and weaknesses from the analysis.
    Returns (strengths: list[str], weaknesses: list[str])
    """
    job_skills = extract_skills(job_text, contextual=False)
    resume_skills_lower = [s.lower() for s in resume_skills]

    # Strengths: skills from JD that the candidate HAS
    matched = [s for s in job_skills if s.lower() in resume_skills_lower]
    strengths = matched[:5]
    if resume_exp >= 4.0:
        strengths.append(f"{resume_exp} years of full-time professional experience")
    elif resume_exp >= 1.0:
        strengths.append(f"{resume_exp} years of professional experience")
    elif num_internships > 0:
        strengths.append(f"{num_internships} internship(s) demonstrating hands-on exposure")

    # Weaknesses: missing required skills + experience gap
    gaps = [s for s in job_skills if s.lower() not in resume_skills_lower]
    weaknesses = gaps[:5]
    if not resume_exp and not num_internships:
        weaknesses.append("No professional experience detected on resume")

    return strengths[:5], weaknesses[:5]
