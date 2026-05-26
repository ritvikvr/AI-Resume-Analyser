# AI Resume Analyzer & Job Matcher

A powerful tool to help job seekers optimize their resumes for ATS systems using AI.

## Features
- **Resume Parsing**: Support for PDF and DOCX files.
- **Skill Extraction**: Uses spaCy to identify key technical skills.
- **Semantic Matching**: Uses Sentence Transformers (all-MiniLM-L6-v2) for deep semantic comparison between resumes and job descriptions.
- **Skill Gap Analysis**: Identifies missing keywords and provides a learning roadmap.
- **Interview Preparation**: Generates tailored interview questions based on your profile.
- **Interactive Dashboard**: Modern UI built with Streamlit.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Django & Django REST Framework
- **Database**: SQLite (default for prototype)
- **AI/NLP**: spaCy, Sentence Transformers

## Setup Instructions

### 1. Prerequisite
Ensure you have Python 3.10+ installed.

### 2. Clone & Setup Environment
```bash
# Navigate to project
cd ai-resume-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_md
```

### 3. Initialize Database
```bash
python manage.py migrate
```

### 4. Run with Docker (Recommended)
The easiest way to run the entire stack (Database, Backend, Frontend) is using Docker Compose:

```bash
cd ai-resume-analyzer
docker-compose up --build
```
This will start:
- **PostgreSQL** on port 5432
- **Django Backend** on port 8000
- **Streamlit Frontend** on port 8501
- **pgAdmin** on port 5050 (Login: admin@admin.com / admin)

### 5. Run Manually (Alternative)
You need to run two separate terminals:

**Terminal 1: Start Django Backend**
```bash
source venv/bin/activate
python manage.py runserver
```

**Terminal 2: Start Streamlit Frontend**
```bash
source venv/bin/activate
streamlit run app.py
```

## How to Use
1. Upload your resume (PDF/DOCX).
2. Paste the job description you are targeting.
3. Click "Analyze Match".
4. Review your match score, missing skills, and interview tips.
