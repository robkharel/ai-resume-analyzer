import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer | Rabilal Kharel",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0A0D13; color: #E8EAF0; }
    .main .block-container { padding: 1.5rem 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0f1420 100%);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero h1 { color: #6C63FF; font-size: 2.8rem; margin-bottom: 0.3rem; font-weight: 800; }
    .hero p { color: #8B90A8; font-size: 1.05rem; margin: 0; }
    .hero .badge {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        color: #00D4AA;
        border: 1px solid rgba(0,212,170,0.3);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1f35, #111520);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        border: 1px solid rgba(108,99,255,0.2);
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    .metric-card:hover { border-color: rgba(108,99,255,0.5); transform: translateY(-2px); }
    .metric-num { font-size: 3rem; font-weight: 800; line-height: 1; }
    .metric-label { color: #8B90A8; font-size: 0.85rem; margin-top: 0.4rem; }
    .metric-sub { font-size: 0.8rem; font-weight: 600; margin-top: 0.3rem; }

    .color-green { color: #00D4AA; }
    .color-yellow { color: #FFB142; }
    .color-red { color: #FF6B6B; }
    .color-purple { color: #6C63FF; }

    .section-card {
        background: rgba(24,29,46,0.9);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .section-card h3 { color: #E8EAF0; margin-bottom: 1rem; font-size: 1.1rem; }

    .tag-green {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        color: #00D4AA;
        border: 1px solid rgba(0,212,170,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .tag-red {
        display: inline-block;
        background: rgba(255,107,107,0.1);
        color: #FF6B6B;
        border: 1px solid rgba(255,107,107,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .tag-purple {
        display: inline-block;
        background: rgba(108,99,255,0.1);
        color: #9D97FF;
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .tag-yellow {
        display: inline-block;
        background: rgba(255,177,66,0.1);
        color: #FFB142;
        border: 1px solid rgba(255,177,66,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .progress-bar-bg {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        height: 10px;
        margin: 0.4rem 0 1rem 0;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    .tip-card {
        background: rgba(108,99,255,0.08);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        color: #C5C8D8;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .tip-card strong { color: #9D97FF; }

    .ats-item {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.9rem;
        color: #C5C8D8;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px); }

    .stFileUploader > div {
        background: rgba(24,29,46,0.9) !important;
        border: 2px dashed rgba(108,99,255,0.3) !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea {
        background: rgba(24,29,46,0.9) !important;
        border: 1px solid rgba(108,99,255,0.2) !important;
        border-radius: 12px !important;
        color: #E8EAF0 !important;
    }
    div[data-testid="stMarkdownContainer"] h3 { color: #E8EAF0; }
    .stSpinner { color: #6C63FF; }
</style>
""", unsafe_allow_html=True)

# ── SKILL CATEGORIES ─────────────────────────────────────────
AI_ML_SKILLS = [
    'python','tensorflow','pytorch','keras','scikit','sklearn','numpy','pandas',
    'matplotlib','seaborn','nlp','spacy','nltk','transformers','huggingface',
    'langchain','openai','gpt','bert','lstm','cnn','rnn','neural','deep learning',
    'machine learning','computer vision','reinforcement','regression','classification',
    'clustering','xgboost','lightgbm','random forest','neural network','embedding',
    'vector','rag','llm','fine','tuning','prompt','generative'
]
CLOUD_TOOLS = [
    'azure','aws','gcp','google cloud','docker','kubernetes','mlflow','airflow',
    'spark','hadoop','sql','mongodb','postgresql','mysql','redis','kafka',
    'streamlit','flask','fastapi','django','git','github','jupyter','colab','vscode'
]
SOFT_SKILLS = [
    'communication','teamwork','leadership','problem','analytical','critical',
    'collaborative','initiative','detail','organized','agile','scrum','research',
    'presentation','documentation','cross','functional'
]

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except:
        return ""

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_tfidf_score(resume, job):
    try:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=1)
        mat = vec.fit_transform([resume, job])
        return round(cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100, 1)
    except:
        return 0.0

def get_keyword_score(resume, job_keywords):
    matched = [k for k in job_keywords if k in resume]
    return len(matched), matched

def extract_top_keywords(text, n=40):
    STOP = set(['and','the','to','of','in','a','is','for','on','with','are','be',
                'as','an','at','by','we','you','your','our','this','that','will',
                'have','has','from','or','it','not','they','their','which','also',
                'can','more','about','all','but','than','when','what','who','how',
                'work','working','experience','years','year','team','ability',
                'strong','knowledge','understanding','using','use','used'])
    words = re.findall(r'\b[a-z][a-z0-9\+#]{2,}\b', text.lower())
    freq = Counter(w for w in words if w not in STOP)
    return [w for w, _ in freq.most_common(n)]

def check_ats(resume_text):
    checks = []
    rt = resume_text.lower()
    checks.append(("Contact Info (email/phone)", bool(re.search(r'[\w\.-]+@[\w\.-]+', resume_text) or re.search(r'\d{3}[\s\-]\d{3,4}', resume_text))))
    checks.append(("Skills Section", bool(re.search(r'skill', rt))))
    checks.append(("Experience Section", bool(re.search(r'experience|work history|employment', rt))))
    checks.append(("Education Section", bool(re.search(r'education|degree|university|college|bachelor|master', rt))))
    checks.append(("Dates/Timeline", bool(re.search(r'20\d\d|19\d\d', resume_text))))
    checks.append(("Action Verbs", bool(re.search(r'built|developed|created|designed|implemented|led|managed|analyzed|improved|deployed', rt))))
    checks.append(("Quantified Results", bool(re.search(r'\d+%|\d+ years|\d+\+', resume_text))))
    checks.append(("No Special Characters Issues", len(re.findall(r'[^\x00-\x7F]', resume_text)) < 20))
    return checks

def score_resume_strength(resume_text, ats_checks):
    score = 0
    rt = resume_text.lower()
    passed = sum(1 for _, v in ats_checks if v)
    score += (passed / len(ats_checks)) * 40
    word_count = len(resume_text.split())
    if 300 <= word_count <= 800: score += 20
    elif word_count > 800: score += 10
    tech_count = sum(1 for s in AI_ML_SKILLS if s in rt)
    score += min(tech_count * 2, 20)
    if re.search(r'github|linkedin|portfolio', rt): score += 10
    if re.search(r'project|built|developed|created', rt): score += 10
    return min(round(score), 100)

def categorize_skills(keywords, resume_text):
    rt = resume_text.lower()
    matched_ai = [k for k in AI_ML_SKILLS if k in rt and k in keywords]
    missing_ai = [k for k in keywords if k in AI_ML_SKILLS and k not in rt]
    matched_cloud = [k for k in CLOUD_TOOLS if k in rt and k in keywords]
    missing_cloud = [k for k in keywords if k in CLOUD_TOOLS and k not in rt]
    matched_soft = [k for k in SOFT_SKILLS if k in rt and k in keywords]
    missing_soft = [k for k in keywords if k in SOFT_SKILLS and k not in rt]
    return {
        'AI/ML': (matched_ai[:10], missing_ai[:8]),
        'Cloud & Tools': (matched_cloud[:10], missing_cloud[:8]),
        'Soft Skills': (matched_soft[:8], missing_soft[:6])
    }

def draw_gauge(score):
    color = "#00D4AA" if score >= 70 else "#FFB142" if score >= 45 else "#FF6B6B"
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor('#111520')
    ax.set_facecolor('#111520')
    theta_bg = [i * 3.14159 / 180 for i in range(0, 181)]
    x_bg = [0.5 + 0.38 * __import__('math').cos(t + 3.14159) for t in theta_bg]
    y_bg = [0.3 + 0.38 * __import__('math').sin(t + 3.14159) for t in theta_bg]
    ax.fill_between([0.5 - 0.38, 0.5 + 0.38], [0.3, 0.3], color='#1a1f35')
    wedge_bg = mpatches.Wedge((0.5, 0.3), 0.38, 0, 180, width=0.12,
                               facecolor='#1a1f35', edgecolor='#2a2f45', linewidth=1,
                               transform=ax.transAxes)
    ax.add_patch(wedge_bg)
    angle = score / 100 * 180
    wedge_fg = mpatches.Wedge((0.5, 0.3), 0.38, 0, angle, width=0.12,
                               facecolor=color, transform=ax.transAxes, zorder=5)
    ax.add_patch(wedge_fg)
    ax.text(0.5, 0.42, f"{score}%", transform=ax.transAxes,
            ha='center', va='center', fontsize=32, fontweight='bold', color=color, zorder=10)
    ax.text(0.5, 0.22, "Match Score", transform=ax.transAxes,
            ha='center', va='center', fontsize=11, color='#8B90A8')
    label = "Strong Match" if score >= 70 else "Moderate Match" if score >= 45 else "Weak Match"
    ax.text(0.5, 0.08, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color=color, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    return fig

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🤖 POWERED BY NLP + MACHINE LEARNING</div>
    <h1>AI Resume Analyzer</h1>
    <p>Upload your resume · Paste a job description · Get your ATS score, keyword analysis & personalized tips</p>
    <p style="font-size:0.78rem; color:#5A6080; margin-top:0.8rem;">
        Built by <strong style="color:#6C63FF;">Rabilal Kharel</strong> · 
        M.S. Artificial Intelligence · University of the Cumberlands
    </p>
</div>
""", unsafe_allow_html=True)

# ── INPUT SECTION ────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### 📄 Upload Resume (PDF)")
    uploaded = st.file_uploader("Upload your resume", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        st.success(f"✅ **{uploaded.name}** uploaded successfully!")

with col2:
    st.markdown("### 💼 Paste Job Description")
    job_desc = st.text_area("Job description", height=220,
                             placeholder="Paste the full job description here — the more detail, the better your analysis!",
                             label_visibility="collapsed")
    if job_desc:
        word_count = len(job_desc.split())
        color = "#00D4AA" if word_count > 100 else "#FFB142"
        st.markdown(f"<p style='color:{color}; font-size:0.8rem;'>📝 {word_count} words — {'Great detail!' if word_count > 100 else 'Add more detail for better results'}</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔍 Analyze My Resume — Get My Score")

# ── ANALYSIS ─────────────────────────────────────────────────
if analyze:
    if not uploaded:
        st.error("⚠️ Please upload your resume PDF!")
    elif not job_desc.strip():
        st.error("⚠️ Please paste a job description!")
    elif len(job_desc.split()) < 20:
        st.warning("⚠️ Job description is too short. Paste the full description for accurate results.")
    else:
        with st.spinner("🤖 Analyzing your resume with AI..."):
            resume_raw = extract_pdf_text(uploaded)
            if not resume_raw:
                st.error("❌ Could not read your PDF. Make sure it contains actual text (not a scanned image).")
                st.stop()

            resume_clean = clean(resume_raw)
            job_clean = clean(job_desc)

            # Scores
            tfidf_score = get_tfidf_score(resume_clean, job_clean)
            job_keywords = extract_top_keywords(job_clean, n=50)
            matched_count, matched_kw = get_keyword_score(resume_clean, job_keywords)
            missing_kw = [k for k in job_keywords if k not in resume_clean]
            keyword_score = round((matched_count / max(len(job_keywords), 1)) * 100, 1)
            ats_checks = check_ats(resume_raw)
            ats_score = round(sum(1 for _, v in ats_checks if v) / len(ats_checks) * 100)
            resume_strength = score_resume_strength(resume_raw, ats_checks)
            overall = round((tfidf_score * 0.4) + (keyword_score * 0.35) + (ats_score * 0.25), 1)
            skill_cats = categorize_skills(job_keywords + matched_kw, resume_raw)

        # ── RESULTS ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Your Analysis Results")

        # Score Cards
        c1, c2, c3, c4 = st.columns(4)
        def score_color(s):
            return "color-green" if s >= 70 else "color-yellow" if s >= 45 else "color-red"

        with c1:
            sc = score_color(overall)
            lbl = "🟢 Strong" if overall >= 70 else "🟡 Moderate" if overall >= 45 else "🔴 Weak"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-num {sc}">{overall}%</div>
                <div class="metric-label">Overall Match</div>
                <div class="metric-sub {sc}">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            sc = score_color(ats_score)
            lbl = "✅ ATS Ready" if ats_score >= 75 else "⚠️ Needs Work" if ats_score >= 50 else "❌ Poor"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-num {sc}">{ats_score}%</div>
                <div class="metric-label">ATS Score</div>
                <div class="metric-sub {sc}">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            sc = score_color(keyword_score)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-num {sc}">{matched_count}/{len(job_keywords)}</div>
                <div class="metric-label">Keywords Matched</div>
                <div class="metric-sub color-purple">{keyword_score}% keyword match</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            sc = score_color(resume_strength)
            lbl = "💪 Strong" if resume_strength >= 70 else "📝 Average" if resume_strength >= 45 else "🔧 Weak"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-num {sc}">{resume_strength}%</div>
                <div class="metric-label">Resume Strength</div>
                <div class="metric-sub {sc}">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge + ATS
        g1, g2 = st.columns([1, 1], gap="large")
        with g1:
            fig = draw_gauge(overall)
            st.pyplot(fig)
            plt.close()

        with g2:
            st.markdown('<div class="section-card"><h3>🤖 ATS Compatibility Check</h3>', unsafe_allow_html=True)
            for label, passed in ats_checks:
                icon = "✅" if passed else "❌"
                color = "#00D4AA" if passed else "#FF6B6B"
                st.markdown(f'<div class="ats-item"><span style="color:{color};">{icon}</span> {label}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Keyword Analysis
        st.markdown("### 🔑 Keyword Analysis by Category")
        for cat, (matched, missing) in skill_cats.items():
            if matched or missing:
                with st.expander(f"📂 {cat} — {len(matched)} matched, {len(missing)} missing", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**✅ Found in your resume:**")
                        if matched:
                            st.markdown("".join([f'<span class="tag-green">{k}</span>' for k in matched]), unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color:#5A6080;">None found</span>', unsafe_allow_html=True)
                    with col_b:
                        st.markdown("**⚠️ Missing — add these:**")
                        if missing:
                            st.markdown("".join([f'<span class="tag-red">{k}</span>' for k in missing]), unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color:#00D4AA;">All covered! ✅</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # All keywords
        ka, kb = st.columns(2)
        with ka:
            st.markdown('<div class="section-card"><h3>✅ All Matched Keywords</h3>', unsafe_allow_html=True)
            if matched_kw:
                st.markdown("".join([f'<span class="tag-green">{k}</span>' for k in matched_kw[:25]]), unsafe_allow_html=True)
            else:
                st.info("No keyword matches found.")
            st.markdown('</div>', unsafe_allow_html=True)

        with kb:
            st.markdown('<div class="section-card"><h3>⚠️ Top Missing Keywords</h3>', unsafe_allow_html=True)
            if missing_kw:
                st.markdown("".join([f'<span class="tag-red">{k}</span>' for k in missing_kw[:25]]), unsafe_allow_html=True)
            else:
                st.success("🎉 No major keywords missing!")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Progress bars
        st.markdown("### 📈 Score Breakdown")
        metrics = [
            ("Overall Match", overall, "#6C63FF"),
            ("ATS Compatibility", ats_score, "#00D4AA"),
            ("Keyword Match", keyword_score, "#FFB142"),
            ("Resume Strength", resume_strength, "#FF6B6B"),
        ]
        for label, val, color in metrics:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:0.2rem;'><span style='color:#C5C8D8; font-size:0.9rem;'>{label}</span><span style='color:{color}; font-weight:700;'>{val}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"""<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{val}%; background:{color};"></div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Personalized Tips
        st.markdown("### 💡 Personalized Improvement Tips")

        if overall >= 70:
            st.markdown('<div class="tip-card">🎉 <strong>Excellent match!</strong> Your resume aligns very well with this job. Apply with confidence — you are a strong candidate!</div>', unsafe_allow_html=True)
        elif overall >= 45:
            st.markdown('<div class="tip-card">📝 <strong>Good foundation!</strong> You have solid alignment but can improve. Adding the missing keywords above could boost your score by 20-30 points.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-card">🔧 <strong>Needs tailoring.</strong> Your resume needs to be customized for this specific role. Focus on adding the missing keywords and restructuring your skills section.</div>', unsafe_allow_html=True)

        if ats_score < 75:
            missing_ats = [label for label, passed in ats_checks if not passed]
            st.markdown(f'<div class="tip-card">🤖 <strong>ATS Warning:</strong> Many companies use ATS software to auto-filter resumes. Fix these sections: <strong>{", ".join(missing_ats)}</strong></div>', unsafe_allow_html=True)

        if missing_kw:
            top5 = ", ".join(f"<strong>{k}</strong>" for k in missing_kw[:5])
            st.markdown(f'<div class="tip-card">🔑 <strong>Priority Keywords:</strong> Add these to your Skills or Experience section immediately: {top5}</div>', unsafe_allow_html=True)

        if not re.search(r'\d+%|\d+ years|\d+\+', resume_raw):
            st.markdown('<div class="tip-card">📊 <strong>Add Numbers:</strong> Quantify your achievements! Example: "Improved model accuracy by 15%" or "Built chatbot serving 200+ users" — numbers make your resume 40% more effective.</div>', unsafe_allow_html=True)

        if not re.search(r'github|linkedin|portfolio', resume_raw.lower()):
            st.markdown('<div class="tip-card">🔗 <strong>Add Links:</strong> Include your GitHub (github.com/robkharel) and Portfolio (robkharel.github.io) — AI/ML recruiters always check these!</div>', unsafe_allow_html=True)

        st.markdown('<div class="tip-card">🎯 <strong>Pro Tip:</strong> Tailor your resume for EACH job application. Use the exact phrases from the job description — ATS systems look for exact matches.</div>', unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#5A6080; font-size:0.82rem; padding:1.5rem 0;">
    <strong style="color:#6C63FF;">AI Resume Analyzer</strong> · Built with Python, NLP & Streamlit ·
    By <a href="https://robkharel.github.io" style="color:#00D4AA; text-decoration:none;">Rabilal Kharel</a> ·
    <a href="https://github.com/robkharel" style="color:#6C63FF; text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)
