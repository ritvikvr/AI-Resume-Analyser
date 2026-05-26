import streamlit as st
import requests
import os
import uuid
import json

# Backend API URL - use environment variable for Docker
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# FIX 7: Generate or retrieve persistent session key from browser session state
if "session_key" not in st.session_state:
    st.session_state.session_key = str(uuid.uuid4())

SESSION_KEY = st.session_state.session_key

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info("""
This AI Resume Analyzer uses **Sentence Transformers** for semantic matching and **spaCy** for skill extraction.
It helps you bypass ATS filters by identifying missing keywords and aligning your resume with job requirements.
""")

st.sidebar.divider()
st.sidebar.caption(f"🔑 Session: `{SESSION_KEY[:8]}...`")

# FIX 7: Session History in sidebar
if st.sidebar.button("📜 Load My History"):
    try:
        hist_resp = requests.get(f"{API_URL}/sessions/history/", params={"session_key": SESSION_KEY})
        if hist_resp.ok:
            history = hist_resp.json()
            if history:
                st.sidebar.markdown("### Past Analyses")
                for item in history[:5]:
                    jd = item.get("job_description", {})
                    title = jd.get("title", "Untitled") if isinstance(jd, dict) else "Untitled"
                    score = item.get("match_percentage", 0)
                    date = item.get("created_at", "")[:10]
                    st.sidebar.markdown(f"- **{title}** — {score}% _{date}_")
            else:
                st.sidebar.info("No history yet for this session.")
        else:
            st.sidebar.warning("Could not load history.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("📄 AI Resume Analyzer & Job Matcher")
st.markdown("Analyze your resume against job descriptions to find skill gaps and improve your ATS score.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

with col2:
    st.header("2. Job Description")
    job_title = st.text_input("Job Title (Optional)", placeholder="e.g. Senior Software Engineer")
    job_description = st.text_area("Paste the job description here", height=300)

if st.button("🚀 Analyze Match", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a resume first.")
    elif not job_description:
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing... this may take a moment."):
            try:
                # 1. Upload Resume (with session key)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resume_resp = requests.post(
                    f"{API_URL}/resumes/",
                    files=files,
                    data={"session_key": SESSION_KEY}
                )
                if resume_resp.status_code not in (200, 201):
                    st.error(f"Error uploading resume: {resume_resp.text}")
                    st.stop()
                resume_id = resume_resp.json().get("id")

                # 2. Upload Job Description (with session key)
                job_data = {
                    "title": job_title,
                    "description_text": job_description,
                    "session_key": SESSION_KEY,
                }
                job_resp = requests.post(f"{API_URL}/jobs/", json=job_data)
                if job_resp.status_code != 201:
                    st.error(f"Error saving job description: {job_resp.text}")
                    st.stop()
                job_id = job_resp.json().get("id")

                # 3. Perform Analysis
                analysis_data = {
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "session_key": SESSION_KEY,
                }
                analysis_resp = requests.post(f"{API_URL}/analysis/analyze/", json=analysis_data)
                if analysis_resp.status_code != 201:
                    st.error(f"Error performing analysis: {analysis_resp.text}")
                    st.stop()

                result = analysis_resp.json()

                # ── Display Results ───────────────────────────────────────
                st.balloons()
                st.divider()
                st.header("📊 Analysis Results")

                match_pct = result.get("match_percentage", 0)
                st.subheader(f"Match Score: {match_pct}%")
                st.progress(match_pct / 100)

                if match_pct > 75:
                    st.success("Great match! Your resume is highly compatible with this role.")
                elif match_pct > 50:
                    st.warning("Good match, but there's room for improvement.")
                else:
                    st.error("Low match. Significant adjustments recommended.")

                # FIX 2: Score Breakdown Bar Chart
                breakdown = result.get("score_breakdown", {})
                if breakdown:
                    st.divider()
                    st.subheader("🎯 Score Breakdown")
                    bd_col1, bd_col2, bd_col3, bd_col4 = st.columns(4)
                    with bd_col1:
                        sem = breakdown.get("semantic", 0)
                        st.metric("Semantic Match", f"{sem:.1f}/40", help="How closely your resume's meaning matches the JD")
                        st.progress(sem / 40)
                    with bd_col2:
                        exp = breakdown.get("experience", 0)
                        st.metric("Experience", f"{exp:.1f}/30", help="Years of relevant experience vs role requirements")
                        st.progress(exp / 30)
                    with bd_col3:
                        skills_s = breakdown.get("skills", 0)
                        st.metric("Skills", f"{skills_s:.1f}/20", help="Skill overlap + volume of technical skills")
                        st.progress(skills_s / 20)
                    with bd_col4:
                        neatness = breakdown.get("neatness", 0)
                        st.metric("Neatness", f"{neatness:.1f}/10", help="Resume layout neatness, bullet points, and structure")
                        st.progress(neatness / 10)

                st.divider()
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    # FIX 4: Show Strengths
                    strengths = result.get("strengths", [])
                    if strengths:
                        st.subheader("✅ Your Strengths")
                        for s in strengths:
                            st.write(f"- {s}")

                    st.subheader("🛠 Missing Skills")
                    missing = result.get("missing_skills", [])
                    if missing:
                        for skill in missing:
                            st.write(f"- {skill}")
                    else:
                        st.write("No major skill gaps identified!")

                    # FIX 4: Show Weaknesses
                    weaknesses = result.get("weaknesses", [])
                    if weaknesses:
                        st.subheader("⚠️ Areas to Improve")
                        for w in weaknesses:
                            st.write(f"- {w}")

                    st.subheader("💡 Suggestions")
                    st.info(result.get("improvement_suggestions", "No suggestions available."))

                with res_col2:
                    st.subheader("❓ Interview Prep")
                    st.write("Potential questions based on your profile:")
                    questions = result.get("interview_questions", [])
                    for q in questions:
                        st.write(f"- {q}")

                # Skill Gap Roadmap
                if missing:
                    st.divider()
                    st.subheader("🛤 Skill Gap Roadmap")
                    st.write("Focus your learning on these key areas to become a better fit:")
                    for i, skill in enumerate(missing[:3]):
                        st.markdown(f"**{i+1}. {skill}**: Research common interview questions and build a small project using this technology.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
