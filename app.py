import streamlit as st
import anthropic
import plotly.graph_objects as go
import pandas as pd
import time
import random

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CareerCoach AI | Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0A0D13; color: #E8EAF0; }
    .main .block-container { padding: 1.5rem 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0f1420 100%);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { color: #6C63FF; font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem; }
    .hero p { color: #8B90A8; font-size: 1rem; margin: 0; }
    .badge {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        color: #00D4AA;
        border: 1px solid rgba(0,212,170,0.3);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.78rem;
        margin-bottom: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .card {
        background: rgba(24,29,46,0.9);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .card h3 { color: #E8EAF0; margin-bottom: 1rem; }

    .question-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(24,29,46,0.95));
        border: 1px solid rgba(108,99,255,0.35);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .question-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #E8EAF0;
        line-height: 1.6;
    }
    .question-meta {
        font-size: 0.78rem;
        color: #6C63FF;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .score-card {
        background: linear-gradient(135deg, #1a1f35, #111520);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(108,99,255,0.2);
        margin-bottom: 1rem;
    }
    .score-num { font-size: 3.5rem; font-weight: 800; line-height: 1; }
    .score-label { color: #8B90A8; font-size: 0.85rem; margin-top: 0.3rem; }

    .feedback-card {
        background: rgba(0,212,170,0.05);
        border: 1px solid rgba(0,212,170,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .feedback-card.warning {
        background: rgba(255,177,66,0.05);
        border-color: rgba(255,177,66,0.2);
    }
    .feedback-card.danger {
        background: rgba(255,107,107,0.05);
        border-color: rgba(255,107,107,0.2);
    }

    .model-answer {
        background: rgba(108,99,255,0.06);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        color: #C5C8D8;
        font-size: 0.95rem;
        line-height: 1.75;
    }

    .tip-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #C5C8D8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .stat-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .pill-green { background: rgba(0,212,170,0.12); color: #00D4AA; border: 1px solid rgba(0,212,170,0.25); }
    .pill-purple { background: rgba(108,99,255,0.12); color: #9D97FF; border: 1px solid rgba(108,99,255,0.25); }
    .pill-yellow { background: rgba(255,177,66,0.12); color: #FFB142; border: 1px solid rgba(255,177,66,0.25); }
    .pill-red { background: rgba(255,107,107,0.12); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.25); }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stTextArea textarea {
        background: rgba(24,29,46,0.9) !important;
        border: 1px solid rgba(108,99,255,0.2) !important;
        border-radius: 12px !important;
        color: #E8EAF0 !important;
        font-size: 1rem !important;
    }
    .stSelectbox > div > div {
        background: rgba(24,29,46,0.9) !important;
        border: 1px solid rgba(108,99,255,0.2) !important;
        color: #E8EAF0 !important;
    }
    div[data-testid="stMarkdownContainer"] h3 { color: #E8EAF0; }
</style>
""", unsafe_allow_html=True)

# ── QUESTION BANK ────────────────────────────────────────────
QUESTIONS = {
    "Machine Learning Engineer": {
        "Technical": [
            "Explain the difference between supervised, unsupervised, and reinforcement learning with real examples.",
            "What is the bias-variance tradeoff and how do you handle it in practice?",
            "Explain how gradient descent works and its variants (SGD, Adam, RMSprop).",
            "What is overfitting? How do you detect and prevent it?",
            "Explain how a neural network learns — forward pass, backpropagation, and weight updates.",
            "What is the difference between L1 and L2 regularization?",
            "How do you handle imbalanced datasets in classification problems?",
            "Explain the attention mechanism and why it's important in modern NLP.",
            "What is transfer learning and when would you use it?",
            "How do you evaluate a machine learning model beyond accuracy?",
        ],
        "Behavioral": [
            "Tell me about yourself and why you want to be a Machine Learning Engineer.",
            "Describe a challenging ML project you worked on and how you solved it.",
            "How do you stay up-to-date with the rapidly evolving AI/ML field?",
            "Tell me about a time you had to explain a complex ML concept to a non-technical person.",
            "Where do you see yourself in 5 years in the AI industry?",
        ],
        "Python & Coding": [
            "How would you implement a simple linear regression from scratch in Python?",
            "Explain list comprehensions, generators, and decorators in Python.",
            "How do you use NumPy for matrix operations in ML?",
            "What is the difference between a list, tuple, set, and dictionary in Python?",
            "How would you load, clean, and preprocess a dataset using Pandas?",
        ],
        "System Design": [
            "How would you design an ML pipeline for a production recommendation system?",
            "How do you monitor a deployed ML model for performance degradation?",
            "Explain the MLOps lifecycle — from training to deployment to monitoring.",
            "How would you handle real-time predictions at scale?",
        ]
    },
    "Data Scientist": {
        "Technical": [
            "What is the difference between correlation and causation? Give an example.",
            "Explain A/B testing and how you would design one.",
            "How do you handle missing data in a dataset?",
            "Explain Principal Component Analysis (PCA) and when you'd use it.",
            "What statistical tests do you know and when would you use each?",
            "How do you detect and handle outliers in data?",
            "Explain the difference between parametric and non-parametric tests.",
            "How would you build a customer churn prediction model?",
        ],
        "Behavioral": [
            "Tell me about yourself and why you want to be a Data Scientist.",
            "Describe a data analysis project that drove a real business decision.",
            "How do you communicate complex findings to non-technical stakeholders?",
            "Tell me about a time your analysis was wrong — how did you handle it?",
        ],
        "Python & Coding": [
            "How do you use Pandas for data manipulation and analysis?",
            "How would you create visualizations using Matplotlib and Seaborn?",
            "Explain how you would do exploratory data analysis (EDA) on a new dataset.",
            "How do you use Scikit-learn for building and evaluating ML models?",
        ],
        "SQL": [
            "Write a SQL query to find the top 5 customers by total revenue.",
            "Explain the difference between INNER JOIN, LEFT JOIN, and FULL OUTER JOIN.",
            "How would you find duplicate records in a database table?",
            "What are window functions in SQL? Give an example.",
        ]
    },
    "AI Engineer": {
        "Technical": [
            "Explain how Large Language Models (LLMs) work at a high level.",
            "What is RAG (Retrieval Augmented Generation) and when would you use it?",
            "How do you fine-tune a pre-trained model for a specific task?",
            "Explain prompt engineering — what makes a good prompt?",
            "What are vector embeddings and how are they used in AI applications?",
            "How would you build a chatbot using an LLM API?",
            "What is the difference between GPT, BERT, and T5 architectures?",
            "How do you evaluate the quality of LLM outputs?",
        ],
        "Behavioral": [
            "Tell me about yourself and why you want to be an AI Engineer.",
            "Describe an AI application you built and the impact it had.",
            "How do you approach the ethical considerations of AI systems?",
            "Where do you see AI technology in the next 5 years?",
        ],
        "Python & Coding": [
            "How would you integrate an AI API (like OpenAI or Anthropic) into a web app?",
            "How do you build a simple RAG pipeline in Python?",
            "Explain how you would use LangChain for building AI applications.",
            "How would you deploy an AI model as a REST API using FastAPI?",
        ],
        "System Design": [
            "How would you design an AI-powered customer support system?",
            "How do you handle rate limits and costs when using LLM APIs at scale?",
            "Design a document Q&A system using RAG architecture.",
        ]
    }
}

def get_ai_feedback(client, question, answer, role, category):
    prompt = f"""You are an expert technical interviewer and career coach specializing in {role} roles.

A candidate is practicing for a {role} interview. Here is the interview question and their answer:

**Question ({category}):** {question}

**Candidate's Answer:** {answer}

Please provide a comprehensive evaluation with:

1. **SCORE** (1-10): Give an overall score
2. **STRENGTHS** (2-3 bullet points): What they did well
3. **IMPROVEMENTS** (2-3 bullet points): What needs work  
4. **MODEL ANSWER**: A concise, ideal answer for this question (3-5 sentences)
5. **PRO TIP**: One specific tip to make their answer stand out

Format your response EXACTLY like this:
SCORE: [number]/10
STRENGTHS:
- [strength 1]
- [strength 2]
IMPROVEMENTS:
- [improvement 1]
- [improvement 2]
MODEL ANSWER:
[ideal answer here]
PRO TIP:
[specific tip here]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def parse_feedback(feedback_text):
    result = {
        "score": 7,
        "strengths": [],
        "improvements": [],
        "model_answer": "",
        "pro_tip": ""
    }
    try:
        lines = feedback_text.split('\n')
        current_section = None
        model_lines = []
        tip_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('SCORE:'):
                try:
                    score_str = line.replace('SCORE:', '').strip()
                    result['score'] = int(score_str.split('/')[0].strip())
                except:
                    result['score'] = 7
            elif line.startswith('STRENGTHS:'):
                current_section = 'strengths'
            elif line.startswith('IMPROVEMENTS:'):
                current_section = 'improvements'
            elif line.startswith('MODEL ANSWER:'):
                current_section = 'model'
            elif line.startswith('PRO TIP:'):
                current_section = 'tip'
            elif line.startswith('- ') and current_section in ['strengths', 'improvements']:
                result[current_section].append(line[2:])
            elif current_section == 'model' and line:
                model_lines.append(line)
            elif current_section == 'tip' and line:
                tip_lines.append(line)

        result['model_answer'] = ' '.join(model_lines)
        result['pro_tip'] = ' '.join(tip_lines)
    except:
        pass
    return result

def score_color(score):
    if score >= 8: return "#00D4AA"
    elif score >= 6: return "#FFB142"
    else: return "#FF6B6B"

def score_label(score):
    if score >= 9: return "🏆 Excellent!"
    elif score >= 7: return "✅ Good"
    elif score >= 5: return "⚠️ Average"
    else: return "🔧 Needs Work"

# ── SESSION STATE ────────────────────────────────────────────
if 'questions_asked' not in st.session_state:
    st.session_state.questions_asked = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_category' not in st.session_state:
    st.session_state.current_category = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'session_complete' not in st.session_state:
    st.session_state.session_complete = False

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🎯 POWERED BY CLAUDE AI · ANTHROPIC</div>
    <h1>CareerCoach AI</h1>
    <p>AI-powered interview coaching · Practice real questions · Get instant expert feedback · Land your dream job</p>
    <p style="font-size:0.78rem; color:#5A6080; margin-top:0.8rem;">
        Built by <strong style="color:#6C63FF;">Rabilal Kharel</strong> · 
        M.S. Artificial Intelligence · University of the Cumberlands ·
        <a href="https://rabikharel.me" style="color:#00D4AA; text-decoration:none;">Portfolio</a>
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR CONFIG ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-api...",
                             help="Get your free key at console.anthropic.com")
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if st.session_state.scores:
        avg = round(sum(st.session_state.scores) / len(st.session_state.scores), 1)
        st.markdown(f"**Questions:** {len(st.session_state.scores)}")
        st.markdown(f"**Average Score:** {avg}/10")
        st.markdown(f"**Best Score:** {max(st.session_state.scores)}/10")
    else:
        st.markdown("*No questions answered yet*")
    st.markdown("---")
    if st.button("🔄 Reset Session"):
        for key in ['questions_asked', 'scores', 'current_question',
                    'current_category', 'feedback', 'show_feedback', 'session_complete']:
            if key in ['questions_asked', 'scores']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.session_state.show_feedback = False
        st.session_state.session_complete = False
        st.rerun()

# ── MAIN CONFIG ───────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    role = st.selectbox("🎯 Target Role",
                         ["Machine Learning Engineer", "Data Scientist", "AI Engineer"])
with col2:
    categories = list(QUESTIONS[role].keys())
    category = st.selectbox("📂 Question Category", ["🎲 Random"] + categories)
with col3:
    difficulty = st.selectbox("⚡ Session Length",
                               ["Quick (5 questions)", "Standard (10 questions)", "Deep (15 questions)"])

num_questions = int(difficulty.split("(")[1].split(" ")[0])

# ── PROGRESS ─────────────────────────────────────────────────
answered = len(st.session_state.scores)
progress = answered / num_questions if num_questions > 0 else 0

st.markdown(f"""
<div style="margin: 1.5rem 0 0.5rem;">
    <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
        <span style="color:#8B90A8; font-size:0.85rem;">Session Progress</span>
        <span style="color:#6C63FF; font-weight:700;">{answered}/{num_questions} questions</span>
    </div>
    <div style="background:rgba(255,255,255,0.05); border-radius:10px; height:8px; overflow:hidden;">
        <div style="width:{progress*100}%; height:100%; background:linear-gradient(90deg,#6C63FF,#00D4AA); border-radius:10px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── GET QUESTION ─────────────────────────────────────────────
if not st.session_state.session_complete:
    if answered < num_questions:
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            get_q = st.button("🎯 Get Next Interview Question")
        with col_btn2:
            skip = st.button("⏭️ Skip Question")

        if get_q or (skip and st.session_state.current_question):
            if skip:
                st.session_state.current_question = None
                st.session_state.show_feedback = False
                st.session_state.feedback = None

            if get_q:
                if category == "🎲 Random":
                    cat = random.choice(list(QUESTIONS[role].keys()))
                else:
                    cat = category

                available = [q for q in QUESTIONS[role][cat]
                             if q not in st.session_state.questions_asked]
                if not available:
                    st.warning(f"All {cat} questions used! Resetting this category.")
                    st.session_state.questions_asked = [q for q in st.session_state.questions_asked
                                                         if q not in QUESTIONS[role][cat]]
                    available = QUESTIONS[role][cat]

                q = random.choice(available)
                st.session_state.current_question = q
                st.session_state.current_category = cat
                st.session_state.questions_asked.append(q)
                st.session_state.show_feedback = False
                st.session_state.feedback = None

        # Show current question
        if st.session_state.current_question:
            st.markdown(f"""
            <div class="question-card">
                <div class="question-meta">❓ {st.session_state.current_category} Question · {role}</div>
                <div class="question-text">{st.session_state.current_question}</div>
            </div>
            """, unsafe_allow_html=True)

            if not st.session_state.show_feedback:
                answer = st.text_area(
                    "Your Answer",
                    height=180,
                    placeholder="Type your answer here... Take your time, think it through, and answer as you would in a real interview.",
                    label_visibility="collapsed"
                )
                word_count = len(answer.split()) if answer.strip() else 0
                if word_count > 0:
                    color = "#00D4AA" if word_count >= 50 else "#FFB142"
                    st.markdown(f"<p style='color:{color}; font-size:0.8rem;'>📝 {word_count} words — {'Great detail!' if word_count >= 50 else 'Try to give more detail (aim for 50+ words)'}</p>",
                                unsafe_allow_html=True)

                submit = st.button("🤖 Get AI Feedback on My Answer")

                if submit:
                    if not api_key:
                        st.error("⚠️ Please enter your Anthropic API key in the sidebar!")
                    elif not answer.strip():
                        st.error("⚠️ Please type your answer first!")
                    elif len(answer.split()) < 10:
                        st.warning("⚠️ Your answer is too short. Please give a more detailed response.")
                    else:
                        with st.spinner("🤖 Claude AI is analyzing your answer..."):
                            try:
                                client = anthropic.Anthropic(api_key=api_key)
                                feedback_text = get_ai_feedback(
                                    client,
                                    st.session_state.current_question,
                                    answer,
                                    role,
                                    st.session_state.current_category
                                )
                                st.session_state.feedback = parse_feedback(feedback_text)
                                st.session_state.feedback['raw'] = feedback_text
                                st.session_state.scores.append(st.session_state.feedback['score'])
                                st.session_state.show_feedback = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ API Error: {str(e)}\n\nMake sure your API key is correct!")

            # Show feedback
            if st.session_state.show_feedback and st.session_state.feedback:
                fb = st.session_state.feedback
                score = fb['score']
                color = score_color(score)
                label = score_label(score)

                # Score display
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.markdown(f"""<div class="score-card">
                        <div class="score-num" style="color:{color};">{score}/10</div>
                        <div class="score-label">Your Score</div>
                        <div style="color:{color}; font-weight:700; margin-top:0.3rem;">{label}</div>
                    </div>""", unsafe_allow_html=True)
                with f2:
                    st.markdown(f"""<div class="score-card">
                        <div class="score-num" style="color:#6C63FF;">{answered}</div>
                        <div class="score-label">Questions Done</div>
                        <div style="color:#6C63FF; font-weight:700; margin-top:0.3rem;">{num_questions - answered} remaining</div>
                    </div>""", unsafe_allow_html=True)
                with f3:
                    avg_score = round(sum(st.session_state.scores) / len(st.session_state.scores), 1) if st.session_state.scores else 0
                    avg_color = score_color(int(avg_score))
                    st.markdown(f"""<div class="score-card">
                        <div class="score-num" style="color:{avg_color};">{avg_score}</div>
                        <div class="score-label">Session Average</div>
                        <div style="color:{avg_color}; font-weight:700; margin-top:0.3rem;">Keep going! 💪</div>
                    </div>""", unsafe_allow_html=True)

                # Feedback details
                fb_class = "feedback-card" if score >= 7 else "feedback-card warning" if score >= 5 else "feedback-card danger"

                if fb['strengths']:
                    st.markdown(f'<div class="{fb_class}"><strong style="color:#00D4AA;">✅ What You Did Well:</strong>', unsafe_allow_html=True)
                    for s in fb['strengths']:
                        st.markdown(f'<div class="tip-item"><span style="color:#00D4AA;">✓</span> {s}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if fb['improvements']:
                    st.markdown('<div class="feedback-card warning"><strong style="color:#FFB142;">⚠️ Areas to Improve:</strong>', unsafe_allow_html=True)
                    for i in fb['improvements']:
                        st.markdown(f'<div class="tip-item"><span style="color:#FFB142;">→</span> {i}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if fb['model_answer']:
                    st.markdown(f'<div class="model-answer"><strong style="color:#9D97FF;">💡 Model Answer:</strong><br><br>{fb["model_answer"]}</div>', unsafe_allow_html=True)

                if fb['pro_tip']:
                    st.markdown(f'<div class="feedback-card" style="background:rgba(108,99,255,0.06); border-color:rgba(108,99,255,0.25);"><strong style="color:#9D97FF;">🚀 Pro Tip:</strong><br><span style="color:#C5C8D8;">{fb["pro_tip"]}</span></div>', unsafe_allow_html=True)

                if answered >= num_questions:
                    st.session_state.session_complete = True
                    st.rerun()

    else:
        st.session_state.session_complete = True

# ── SESSION COMPLETE ─────────────────────────────────────────
if st.session_state.session_complete and st.session_state.scores:
    scores = st.session_state.scores
    avg = round(sum(scores) / len(scores), 1)
    best = max(scores)
    worst = min(scores)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:2rem 0 1rem;">
        <div style="font-size:3rem;">🎉</div>
        <h2 style="color:#6C63FF; font-size:2rem; margin:0.5rem 0;">Session Complete!</h2>
        <p style="color:#8B90A8;">Here's your performance report</p>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    metrics = [
        (avg, "Average Score", "#6C63FF"),
        (best, "Best Score 🏆", "#00D4AA"),
        (worst, "Lowest Score", "#FF6B6B"),
        (len(scores), "Questions Done", "#FFB142"),
    ]
    for col, (val, label, color) in zip([r1, r2, r3, r4], metrics):
        with col:
            st.markdown(f"""<div class="score-card">
                <div class="score-num" style="color:{color};">{val}</div>
                <div class="score-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # Score chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores)+1)),
        y=scores,
        mode='lines+markers',
        line=dict(color='#6C63FF', width=3),
        marker=dict(size=10, color=[score_color(s) for s in scores],
                    line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(108,99,255,0.08)'
    ))
    fig.add_hline(y=7, line_dash="dash", line_color="#00D4AA",
                  annotation_text="Target Score (7)", annotation_position="right")
    fig.update_layout(
        paper_bgcolor='#111520',
        plot_bgcolor='#111520',
        font=dict(color='#8B90A8'),
        xaxis=dict(title='Question Number', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Score', range=[0,10], gridcolor='rgba(255,255,255,0.05)'),
        title=dict(text='Your Score Progression', font=dict(color='#E8EAF0', size=16)),
        height=300,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Final verdict
    if avg >= 8:
        verdict = "🏆 Outstanding! You're ready to ace real interviews. Apply with confidence!"
        v_color = "#00D4AA"
    elif avg >= 6:
        verdict = "✅ Good performance! A bit more practice and you'll be interview-ready."
        v_color = "#FFB142"
    else:
        verdict = "📚 Keep practicing! Focus on the improvement areas and try again."
        v_color = "#FF6B6B"

    st.markdown(f"""
    <div style="background:rgba(24,29,46,0.9); border:1px solid {v_color}40; border-radius:16px; padding:2rem; text-align:center; margin-top:1rem;">
        <p style="font-size:1.2rem; color:{v_color}; font-weight:700; margin:0;">{verdict}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Start New Session"):
        for key in ['questions_asked', 'scores', 'current_question',
                    'current_category', 'feedback', 'show_feedback', 'session_complete']:
            if key in ['questions_asked', 'scores']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.session_state.show_feedback = False
        st.session_state.session_complete = False
        st.rerun()

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#5A6080; font-size:0.82rem; padding:1rem 0;">
    <strong style="color:#6C63FF;">CareerCoach AI</strong> · Powered by Claude AI (Anthropic) · Built with Python & Streamlit ·
    By <a href="https://rabikharel.me" style="color:#00D4AA; text-decoration:none;">Rabilal Kharel</a>
</div>
""", unsafe_allow_html=True)