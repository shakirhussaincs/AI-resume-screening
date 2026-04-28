import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
if "processed" not in st.session_state:
    st.session_state.processed = False
if "results" not in st.session_state:
    st.session_state.results = []
if "clean_jd" not in st.session_state:
    st.session_state.clean_jd = ""
st.set_page_config(page_title="AI Resume Screener", page_icon="rocket", layout="wide", initial_sidebar_state="expanded")

# === IMPORTS ===
try:
    from pdf_parser import extract_text
    PDF_READY = True
except ImportError:
    PDF_READY = False
    def extract_text(f):
        if isinstance(f, list):
            return {getattr(x,'name',f'file_{i}.pdf'): f"stub python machine learning sql data science {i}" for i,x in enumerate(f)}
        return f"stub python machine learning sql data science {getattr(f,'name','file.pdf')}"

try:
    from preprocessing import clean_resume_text
    PREP_READY = True
except ImportError:
    PREP_READY = False
    def clean_resume_text(text): return str(text).lower().strip()

try:
    from similarity import calculate_similarity, calculate_batch_similarity
    SIM_READY = True
except ImportError:
    SIM_READY = False
    def calculate_similarity(r, j):
        import random; random.seed(len(str(r))+len(str(j))); return round(random.uniform(0.25,0.97),4)
    def calculate_batch_similarity(d, j):
        return sorted([{"filename":k,"score":calculate_similarity(v,j)} for k,v in d.items()],key=lambda x:x["score"],reverse=True)

@st.cache_resource
def load_ml_stack():
    paths = ["MODEL/tfidf_vectorizer.pkl","MODEL/feature_selector.pkl","MODEL/resume_rf_model.pkl","MODEL/label_encoder.pkl"]
    if all(os.path.exists(p) for p in paths):
        try: return tuple(joblib.load(p) for p in paths)
        except: pass
    return None, None, None, None

tfidf, selector, model, encoder = load_ml_stack()
MODELS_READY = all(x is not None for x in [tfidf, selector, model, encoder])
MODEL_FILES = {
    "tfidf_vectorizer.pkl": os.path.exists("MODEL/tfidf_vectorizer.pkl"),
    "feature_selector.pkl": os.path.exists("MODEL/feature_selector.pkl"),
    "resume_rf_model.pkl":  os.path.exists("MODEL/resume_rf_model.pkl"),
    "label_encoder.pkl":    os.path.exists("MODEL/label_encoder.pkl"),
}

# === K-FOLD RESULTS (real notebook output) ===
KFOLD_DATA = {
    "accuracy": {
        "SVM (Linear)":  [0.89877301, 0.90184049, 0.90030675, 0.90950920, 0.90184049],
        "Random Forest": [0.92177914, 0.91564417, 0.91411043, 0.92331288, 0.90644172],
    },
    "precision": {
        "SVM (Linear)":  [0.93349649, 0.93688016, 0.93441462, 0.93898157, 0.93491561],
        "Random Forest": [0.94761273, 0.94919394, 0.94235889, 0.94747054, 0.93958201],
    },
    "recall": {
        "SVM (Linear)":  [0.93183264, 0.92677076, 0.92583451, 0.93600451, 0.93093022],
        "Random Forest": [0.94553357, 0.93929209, 0.93622527, 0.93860807, 0.93094897],
    },
    "f1": {
        "SVM (Linear)":  [0.93004704, 0.92945918, 0.92667178, 0.93567287, 0.93125548],
        "Random Forest": [0.94507837, 0.94150831, 0.93900000, 0.93904501, 0.93143909],
    },
}

METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision (Macro)",
    "recall":    "Recall (Macro)",
    "f1":        "F1-Score (Macro)",
}

# === PIPELINE ===
def predict_category(clean_text):
    if not MODELS_READY:
        import random; random.seed(len(clean_text))
        return random.choice(["Data Scientist","Python Developer","Software Engineer","Java Developer","Machine Learning","DevOps Engineer"])
    try:
        vec = tfidf.transform([clean_text])
        sel = selector.transform(vec)
        pred = model.predict(sel)
        return encoder.inverse_transform(pred)[0]
    except: return "Unknown"

def get_status(score_pct):
    if score_pct >= 85:   return "Excellent Match"
    elif score_pct >= 70: return "Good Match"
    elif score_pct >= 50: return "Low Match"
    else:                 return "Poor Match"

def skill_match(skill, text):
    text = text.lower()
    skill_lower = skill.lower()
    if skill_lower in text:
        return True
    if skill_lower.endswith('s') and skill_lower[:-1] in text:
        return True
    if re.search(r'[\s\(\,\•\-]' + re.escape(skill_lower) + r'[\s\)\,\•\-]', text):
        return True
    if re.search(r'\b' + re.escape(skill_lower) + r'\b', text):
        return True
    words = skill_lower.split()
    if len(words) > 1:
        return any(re.search(r'\b' + re.escape(w) + r'\b', text) for w in words if len(w) > 1)
    return False

def process_resume(file, clean_jd, skills):
    extracted  = extract_text(file)
    raw        = extracted["raw"]
    clean      = extracted["clean"]
    category   = predict_category(clean)
    score      = calculate_similarity(clean, clean_jd)
    score_pct  = round(score * 100, 2)
    matched    = [s for s in skills if skill_match(s, raw)]
    missing    = [s for s in skills if not skill_match(s, raw)]
    return {
        "File Name":      getattr(file,'name',str(file)),
        "Predicted Role": category,
        "Match Score %":  score_pct,
        "Status":         get_status(score_pct),
        "Matched Skills": matched,
        "Missing Skills": missing,
        "Cleaned Text":   clean[:800]+"..." if len(clean)>800 else clean,
        "Raw Text":       raw,
    }

def generate_candidate_summary(result):
    name_hint = result["File Name"].replace(".pdf","").replace("_"," ").replace("-"," ").title()
    category  = result["Predicted Role"]
    score     = result["Match Score %"]
    matched   = result["Matched Skills"]
    missing   = result["Missing Skills"]
    status    = result["Status"]
    if matched:
        skills_sentence = f"demonstrates strong proficiency in **{', '.join(matched[:5])}**"
    else:
        skills_sentence = "does not match the required skill set"
    if missing:
        missing_sentence = f"However, the candidate is missing **{', '.join(missing[:3])}**."
    else:
        missing_sentence = "The candidate meets **all required skills**. ✅"
    summary = f"""
**{name_hint}** is categorized as a **{category}** candidate with a match score of **{score}%** ({status}).

The candidate {skills_sentence}.

{missing_sentence}

Overall this candidate is **{'✅ Recommended' if score >= 70 else '❌ Not Recommended'}** for this role.
"""
    return summary

# === CHART HELPERS ===
def draw_circular_gauge(score_pct):
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#161b2e')
    ax.set_facecolor('#161b2e')
    color = '#10b981' if score_pct >= 70 else '#f59e0b' if score_pct >= 50 else '#ef4444'
    bg = plt.Circle((0,0), 0.8, color='#1c2333')
    ax.add_patch(bg)
    inner = plt.Circle((0,0), 0.55, color='#161b2e')
    ax.add_patch(inner)
    angle = (score_pct / 100) * 360
    wedge = mpatches.Wedge((0,0), 0.8, 90, 90-angle, width=0.25, color=color)
    ax.add_patch(wedge)
    ax.text(0, 0.05, f"{score_pct}%", ha='center', va='center', fontsize=18, fontweight='bold', color='#ffffff')
    ax.text(0, -0.18, 'Match Score', ha='center', va='center', fontsize=7, color='#8b9dc3')
    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def draw_skill_chart(skills_jd, raw_text):
    if not skills_jd:
        skills_jd = ['Python','Machine Learning','SQL','Data Analysis','Skills']
    jd_scores = []
    resume_scores = []
    for skill in skills_jd:
        jd_scores.append(100)
        found = skill_match(skill, raw_text)
        if found:
            count = raw_text.lower().count(skill.lower().split()[0])
            score = min(100, 60 + (count * 15))
        else:
            score = 0
        resume_scores.append(score)
    x = np.arange(len(skills_jd)); w = 0.35
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#161b2e')
    ax.set_facecolor('#161b2e')
    ax.bar(x - w/2, jd_scores,     w, label='Job Requirements', color='#7c3aed', alpha=0.9)
    ax.bar(x + w/2, resume_scores, w, label='Resume Skills',    color='#10b981', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:10] for s in skills_jd], fontsize=8, rotation=15, color='#8b9dc3')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7, facecolor='#1c2333', labelcolor='#c9d1d9')
    ax.set_ylabel('Score', fontsize=8, color='#8b9dc3')
    ax.tick_params(colors='#8b9dc3')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    plt.tight_layout()
    return fig

def draw_kfold_line_chart(metric_key):
    """Draw fold-wise line chart for a given metric comparing SVM vs RF."""
    folds   = [f"Fold {i+1}" for i in range(5)]
    svm_vals = KFOLD_DATA[metric_key]["SVM (Linear)"]
    rf_vals  = KFOLD_DATA[metric_key]["Random Forest"]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor('#161b2e')
    ax.set_facecolor('#1c2333')

    ax.plot(folds, svm_vals, color='#a78bfa', marker='o', linewidth=2.2,
            markersize=7, label='SVM (Linear)', zorder=3)
    ax.plot(folds, rf_vals,  color='#10b981', marker='s', linewidth=2.2,
            markersize=7, label='Random Forest', linestyle='--', zorder=3)

    # Shade area between curves
    ax.fill_between(folds, svm_vals, rf_vals, alpha=0.08, color='#7c3aed')

    # Styling
    y_min = min(min(svm_vals), min(rf_vals)) - 0.005
    y_max = max(max(svm_vals), max(rf_vals)) + 0.005
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.tick_params(colors='#8b9dc3', labelsize=8)
    ax.set_xticklabels(folds, color='#8b9dc3', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.grid(axis='y', color='#30363d', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, facecolor='#1c2333', labelcolor='#c9d1d9',
              edgecolor='#30363d', loc='lower right')
    plt.tight_layout(pad=0.5)
    return fig

def draw_model_comparison_bar():
    """Draw mean scores comparison bar chart for all metrics."""
    metrics = list(METRIC_LABELS.values())
    svm_means = [np.mean(KFOLD_DATA[k]["SVM (Linear)"])  * 100 for k in KFOLD_DATA]
    rf_means  = [np.mean(KFOLD_DATA[k]["Random Forest"]) * 100 for k in KFOLD_DATA]

    x = np.arange(len(metrics)); w = 0.32
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#161b2e')
    ax.set_facecolor('#1c2333')

    bars1 = ax.bar(x - w/2, svm_means, w, label='SVM (Linear)',  color='#a78bfa', alpha=0.9, zorder=3)
    bars2 = ax.bar(x + w/2, rf_means,  w, label='Random Forest', color='#10b981', alpha=0.9, zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}%", ha='center', va='bottom',
                fontsize=7, color='#a78bfa', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}%", ha='center', va='bottom',
                fontsize=7, color='#10b981', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#8b9dc3', fontsize=8)
    ax.set_ylim(88, 97)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors='#8b9dc3', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.grid(axis='y', color='#30363d', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, facecolor='#1c2333', labelcolor='#c9d1d9', edgecolor='#30363d')
    plt.tight_layout(pad=0.5)
    return fig

def draw_std_comparison_bar():
    """Draw standard deviation comparison bar chart for all metrics."""
    metrics = list(METRIC_LABELS.values())
    svm_stds = [np.std(KFOLD_DATA[k]["SVM (Linear)"])  * 100 for k in KFOLD_DATA]
    rf_stds  = [np.std(KFOLD_DATA[k]["Random Forest"]) * 100 for k in KFOLD_DATA]

    x = np.arange(len(metrics)); w = 0.32
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#161b2e')
    ax.set_facecolor('#1c2333')

    bars1 = ax.bar(x - w/2, svm_stds, w, label='SVM (Linear)',  color='#a78bfa', alpha=0.9, zorder=3)
    bars2 = ax.bar(x + w/2, rf_stds,  w, label='Random Forest', color='#10b981', alpha=0.9, zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{bar.get_height():.4f}%", ha='center', va='bottom',
                fontsize=7, color='#a78bfa', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{bar.get_height():.4f}%", ha='center', va='bottom',
                fontsize=7, color='#10b981', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#8b9dc3', fontsize=8)
    ax.set_ylim(0, max(max(svm_stds), max(rf_stds)) * 1.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}%"))
    ax.tick_params(colors='#8b9dc3', labelsize=8)
    ax.set_ylabel('Std Deviation (%)', fontsize=8, color='#8b9dc3')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.grid(axis='y', color='#30363d', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, facecolor='#1c2333', labelcolor='#c9d1d9', edgecolor='#30363d')
    plt.tight_layout(pad=0.5)
    return fig

# === DARK THEME STYLING ===
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

  .stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
  }
  h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa, #7c3aed, #06b6d4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -1px !important;
  }
  h2 { font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; color: #e2d9f3 !important; }
  h3 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #c4b5fd !important; letter-spacing: 0.3px !important; }
  p, li, label { color: #c9d1d9 !important; }
  .stCaption { color: #8b9dc3 !important; font-size: 13px !important; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #161b2e 100%) !important;
    border-right: 1px solid #30363d !important;
  }
  section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  section[data-testid="stSidebar"] h2 { font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; color: #a78bfa !important; -webkit-text-fill-color: #a78bfa !important; }
  section[data-testid="stSidebar"] h3 { color: #c4b5fd !important; font-size: 13px !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
  section[data-testid="stSidebar"] strong { color: #a78bfa !important; }
  section[data-testid="stSidebar"] .stTextArea textarea,
  section[data-testid="stSidebar"] .stTextInput input {
    background: #1c2333 !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
  }
  section[data-testid="stSidebar"] ::placeholder { color: #a78bfa !important; opacity: 1 !important; }
  section[data-testid="stSidebar"] .stTextInput input::placeholder { color: #a78bfa !important; }
  section[data-testid="stSidebar"] .stTextArea textarea::placeholder { color: #a78bfa !important; }

  [data-baseweb="popover"] { background: #1c2333 !important; }
  [data-baseweb="popover"] * { background: #1c2333 !important; color: #a78bfa !important; }
  [data-baseweb="option"]:hover { background: #7c3aed !important; color: #ffffff !important; }
  ul[role="listbox"] { background: #1c2333 !important; }
  ul[role="listbox"] li { color: #a78bfa !important; background: #1c2333 !important; }
  ul[role="listbox"] li:hover { background: #7c3aed !important; color: #ffffff !important; }
  div[data-baseweb="select"] div { background: #1c2333 !important; color: #a78bfa !important; }

  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1c2333, #1a1f35) !important;
    border: 1px solid #30363d !important; border-radius: 14px !important;
    padding: 20px !important; box-shadow: 0 4px 20px rgba(124,58,237,0.1) !important;
  }
  [data-testid="stMetricLabel"] { font-size: 12px !important; color: #8b9dc3 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; font-weight: 500 !important; }
  [data-testid="stMetricValue"] {
    font-size: 2.4rem !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa, #06b6d4) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important; font-family: 'Poppins', sans-serif !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; color: white !important;
    font-weight: 700 !important; font-size: 14px !important; font-family: 'Inter', sans-serif !important;
    border: none !important; border-radius: 10px !important; padding: 12px 20px !important;
    width: 100% !important; box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important; letter-spacing: 0.3px !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important; transform: translateY(-1px) !important;
  }
  .stDownloadButton > button { background: #1c2333 !important; color: #a78bfa !important; border: 1px solid #7c3aed !important; border-radius: 10px !important; font-weight: 600 !important; }
  .stDownloadButton > button:hover { background: #7c3aed !important; color: white !important; }

  .stTabs [data-baseweb="tab-list"] { background: #161b2e !important; border-bottom: 1px solid #30363d !important; gap: 8px !important; padding: 6px 6px 0 6px !important; }
  .stTabs [data-baseweb="tab"] {
    background: #1c2333 !important; color: #8b9dc3 !important; border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important; font-family: 'Inter', sans-serif !important; font-size: 14px !important;
    padding: 10px 20px !important; letter-spacing: 0.3px !important;
    border: 1px solid #30363d !important; border-bottom: none !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; color: #ffffff !important;
    border-color: #7c3aed !important; box-shadow: 0 -4px 12px rgba(124,58,237,0.3) !important;
  }

  .stTextArea textarea, .stTextInput input {
    background: #1c2333 !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
  }
  .stTextArea textarea:focus, .stTextInput input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 2px rgba(124,58,237,0.3) !important; }
  .stSelectbox > div > div { background: #1c2333 !important; border: 1px solid #30363d !important; color: #c9d1d9 !important; border-radius: 8px !important; }
  .stMultiSelect > div > div { background: #1c2333 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
  .stMultiSelect span[data-baseweb="tag"] { background: #7c3aed !important; color: white !important; }
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: #7c3aed !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #10b981) !important; border-radius: 4px !important; }

  .stInfo    { background: #1c2333 !important; border-left: 4px solid #3b82f6 !important; color: #93c5fd !important; border-radius: 8px !important; }
  .stWarning { background: #1c2333 !important; border-left: 4px solid #f59e0b !important; color: #fcd34d !important; border-radius: 8px !important; }
  .stSuccess { background: #1c2333 !important; border-left: 4px solid #10b981 !important; color: #6ee7b7 !important; border-radius: 8px !important; }
  .stError   { background: #1c2333 !important; border-left: 4px solid #ef4444 !important; color: #fca5a5 !important; border-radius: 8px !important; }

  #MainMenu, footer { visibility: hidden; }
  [data-testid="collapsedControl"] { display: block !important; visibility: visible !important; color: #ffffff !important; background: #7c3aed !important; border-radius: 50% !important; }

  .category-card { background: linear-gradient(135deg, #1c2333, #1a1f35); border: 1px solid #30363d; border-radius: 14px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
  .eval-card { background: linear-gradient(135deg, #1c2333, #1a1f35); border: 1px solid #30363d; border-radius: 14px; padding: 18px; }
  .status-strong { background: linear-gradient(135deg, #10b981, #059669); color: white; border-radius: 20px; padding: 6px 16px; font-weight: 700; font-size: 13px; display: inline-block; box-shadow: 0 4px 12px rgba(16,185,129,0.4); font-family: 'Inter', sans-serif; }
  .status-good   { background: linear-gradient(135deg, #7c3aed, #6d28d9); color: white; border-radius: 20px; padding: 6px 16px; font-weight: 700; font-size: 13px; display: inline-block; box-shadow: 0 4px 12px rgba(124,58,237,0.4); font-family: 'Inter', sans-serif; }
  .status-low    { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; border-radius: 20px; padding: 6px 16px; font-weight: 700; font-size: 13px; display: inline-block; box-shadow: 0 4px 12px rgba(245,158,11,0.4); font-family: 'Inter', sans-serif; }

  hr { border-color: #30363d !important; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #161b2e; }
  ::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #6d28d9; }
  [data-testid="stFileUploader"] { background: #1c2333 !important; border: 1px dashed #7c3aed !important; border-radius: 10px !important; }
  [data-testid="stFileUploader"] * { color: #c9d1d9 !important; }
  [data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #30363d !important; }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## Upload & Filter")
    st.markdown("---")
    st.markdown("**Paste Job Description**")
    jd_input = st.text_area("JD", placeholder="Sample JD: Senior Data Scientist. Requires machine learning, SQL, Python...", height=150, label_visibility="collapsed")
    st.markdown("**Upload Resumes (PDFs)**")
    uploaded_files = st.file_uploader("upload", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        names = ", ".join([f.name for f in uploaded_files[:3]])
        st.caption(f"{len(uploaded_files)} files uploaded: {names}" + ("..." if len(uploaded_files) > 3 else ""))
    st.markdown("---")
    st.markdown("### Refine Results")
    min_score = st.slider("Min Match Score (%)", 0, 100, 60, 5)
    st.markdown("**Target Roles**")
    target_roles = st.multiselect("roles", [
        "AI Engineer", "Backend Developer", "Blockchain", "Blockchain Developer",
        "Business Analyst", "Cloud Engineer", "Cybersecurity Analyst", "Data Science",
        "Database", "Database Administrator", "DevOps", "Digital Media",
        "DotNet Developer", "ETL Developer", "Engineering Manager", "Frontend Developer",
        "Full Stack Developer", "Java Developer", "Machine Learning Engineer",
        "Mobile Developer", "Network Security Engineer", "Principal Engineer",
        "Product Manager", "Python Developer", "QA Engineer", "React Developer",
        "SAP Developer", "SQL Developer", "Site Reliability Engineer",
        "Software Developer", "System Administrator", "Technical Lead",
        "Technical Writer", "Testing", "UI/UX Designer", "Web Designing"
    ], default=[], label_visibility="collapsed")
    st.markdown("**Required Skills**")
    skills_input = st.text_input("Required Skills", placeholder="Python, SQL, Machine Learning...", label_visibility="collapsed")
    st.markdown("---")
    apply_clicked = st.button("Apply Filters & Process", use_container_width=True)
    st.markdown("---")
    st.markdown("### Module Status")
    st.markdown(("✅" if PDF_READY   else "⚠️ STUB") + " pdf_parser.py")
    st.markdown(("✅" if PREP_READY  else "⚠️ STUB") + " preprocessing.py")
    st.markdown(("✅" if SIM_READY   else "⚠️ STUB") + " similarity.py")
    for fname, exists in MODEL_FILES.items():
        st.markdown(("✅" if exists else "❌ MISSING") + f" {fname}")
    if MODELS_READY: st.success("All models loaded!")
    else: st.warning("MODEL/ files not found - using stubs")

# === MAIN ===
st.markdown("# 🚀 AI Candidate Ranking")
st.markdown("<p style='color:#8b9dc3;font-size:15px;font-family:Inter,sans-serif;margin-top:-10px'>Upload resumes + job description — ML categorises & cosine similarity ranks candidates</p>", unsafe_allow_html=True)
st.markdown("---")

tab_bulk, tab_single, tab_eval = st.tabs([
    "Bulk Screening (100+ PDFs)",
    "Single Resume Analysis",
    "📊 Model Evaluation"
])

if apply_clicked:
    st.session_state.processed = True

# =============================================
# === MODEL EVALUATION TAB ===
# =============================================
with tab_eval:
    st.markdown("## Model Evaluation — Stratified 5-Fold Cross Validation")
    st.markdown("<p style='color:#8b9dc3;font-size:14px;margin-top:-10px'>Comparing SVM (Linear) vs Random Forest across 5 folds — Accuracy, Precision, Recall, F1</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Summary metric cards ──
    st.markdown("### Overall Mean Scores")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    cols_svm = [c1, c2, c3, c4]
    cols_rf  = [c5, c6, c7, c8]
    metric_keys = list(KFOLD_DATA.keys())

    for col, key in zip(cols_svm, metric_keys):
        val = np.mean(KFOLD_DATA[key]["SVM (Linear)"]) * 100
        col.metric(f"SVM {METRIC_LABELS[key].split()[0]}", f"{val:.2f}%")

    for col, key in zip(cols_rf, metric_keys):
        val = np.mean(KFOLD_DATA[key]["Random Forest"]) * 100
        col.metric(f"RF {METRIC_LABELS[key].split()[0]}", f"{val:.2f}%")

    st.markdown("---")

    # ── Overall comparison bar chart ──
    st.markdown("### Mean Score Comparison — All Metrics")
    fig_bar = draw_model_comparison_bar()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Std deviation comparison bar chart ──
    st.markdown("### Standard Deviation Comparison — All Metrics")
    st.caption("Lower std = more consistent model across folds")
    fig_std = draw_std_comparison_bar()
    st.pyplot(fig_std, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Fold-wise line charts, 2 per row ──
    st.markdown("### Fold-wise Performance (per Metric)")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    chart_cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for col, key in zip(chart_cols, metric_keys):
        with col:
            svm_vals = KFOLD_DATA[key]["SVM (Linear)"]
            rf_vals  = KFOLD_DATA[key]["Random Forest"]
            svm_mean = np.mean(svm_vals) * 100
            rf_mean  = np.mean(rf_vals)  * 100
            svm_std  = np.std(svm_vals)  * 100
            rf_std   = np.std(rf_vals)   * 100

            st.markdown(f"**{METRIC_LABELS[key]}**")

            # Mini stats row
            sc1, sc2 = st.columns(2)
            sc1.markdown(
                f"<div style='background:#1c2333;border:1px solid #30363d;border-radius:10px;padding:10px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b9dc3'>SVM Mean</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:#a78bfa'>{svm_mean:.2f}%</div>"
                f"<div style='font-size:10px;color:#8b9dc3'>±{svm_std:.2f}</div></div>",
                unsafe_allow_html=True
            )
            sc2.markdown(
                f"<div style='background:#1c2333;border:1px solid #30363d;border-radius:10px;padding:10px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b9dc3'>RF Mean</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:#10b981'>{rf_mean:.2f}%</div>"
                f"<div style='font-size:10px;color:#8b9dc3'>±{rf_std:.2f}</div></div>",
                unsafe_allow_html=True
            )

            fig_line = draw_kfold_line_chart(key)
            st.pyplot(fig_line, use_container_width=True)
            plt.close()

    st.markdown("---")

    # ── Fold-wise data table ──
    st.markdown("### Raw Fold Data")
    folds = [f"Fold {i+1}" for i in range(5)]
    rows = []
    for key, label in METRIC_LABELS.items():
        for model_name in ["SVM (Linear)", "Random Forest"]:
            vals = KFOLD_DATA[key][model_name]
            row = {
                "Model": model_name,
                "Metric": label,
                **{f"Fold {i+1}": f"{v*100:.4f}%" for i, v in enumerate(vals)},
                "Mean": f"{np.mean(vals)*100:.4f}%",
                "Std":  f"{np.std(vals)*100:.4f}%",
            }
            rows.append(row)
    eval_df = pd.DataFrame(rows)

    def highlight_model(val):
        if val == "SVM (Linear)":  return "color:#a78bfa;font-weight:600"
        if val == "Random Forest": return "color:#10b981;font-weight:600"
        return ""

    styled_eval = eval_df.style.map(highlight_model, subset=["Model"])
    st.dataframe(styled_eval, use_container_width=True, hide_index=True)

    # ── Winner badge ──
    st.markdown("---")
    st.markdown("### Verdict")
    vc1, vc2, vc3 = st.columns([1, 2, 1])
    with vc2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1c2333,#1a1f35);border:1px solid #10b981;
                    border-radius:16px;padding:24px;text-align:center'>
            <div style='font-size:12px;color:#8b9dc3;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>
                Best Performing Model
            </div>
            <div style='font-size:1.8rem;font-weight:800;color:#10b981;margin-bottom:4px'>
                🏆 Random Forest
            </div>
            <div style='font-size:13px;color:#8b9dc3'>
                Outperforms SVM on all 4 metrics across all 5 folds<br>
                <span style='color:#a78bfa'>F1: 93.88%</span> vs <span style='color:#8b9dc3'>SVM 93.06%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# === SCREENING TABS ===
# =============================================
if st.session_state.processed:
    if apply_clicked:
        if not jd_input:
            st.error("Please paste a Job Description in the sidebar.")
            st.stop()
        if not uploaded_files:
            st.error("Please upload at least one PDF resume.")
            st.stop()
        if "results_df" not in st.session_state:
            st.session_state.results_df = None
        if "results_df_f" not in st.session_state:
            st.session_state.results_df_f = None

    skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
    clean_jd = clean_resume_text(jd_input)

    # SINGLE MODE
    if len(uploaded_files) == 1:
        with tab_single:
            with st.spinner("Analysing resume..."):
                result = process_resume(uploaded_files[0], clean_jd, skills_list)
            score     = result["Match Score %"]
            category  = result["Predicted Role"]
            matched   = result["Matched Skills"]
            missing   = result["Missing Skills"]
            clean_txt = result["Cleaned Text"]
            raw_txt   = result["Raw Text"]

            if score >= 85:   status_label, status_sub, status_cls = "Strong Match", "Ready for Interview", "status-strong"
            elif score >= 70: status_label, status_sub, status_cls = "Good Match",   "Worth Reviewing",    "status-good"
            else:             status_label, status_sub, status_cls = "Low Match",    "May Not Qualify",    "status-low"

            st.markdown("## Individual Candidate Analysis")
            col1, col2, col3 = st.columns([1, 1.2, 1])
            with col1:
                fig = draw_circular_gauge(score)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with col2:
                st.markdown(f"<div class='category-card'><div style='font-size:12px;color:#8b9dc3;margin-bottom:6px'>ML Predicted Category</div><div style='font-size:1.6rem;font-weight:800;color:#a78bfa'>{category}</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='category-card' style='text-align:center'><div style='font-size:12px;color:#8b9dc3;margin-bottom:10px'>Status</div><span class='{status_cls}'>{status_label}</span><div style='font-size:12px;color:#8b9dc3;margin-top:10px'>{status_sub}</div></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Similarity Breakdown")
            chart_col, kw_col = st.columns([1.3, 1])
            with chart_col:
                st.markdown("**Key Skill Alignment**")
                fig2 = draw_skill_chart(skills_list, raw_txt)
                st.pyplot(fig2, use_container_width=True)
                plt.close()
            with kw_col:
                st.markdown("**Keyword Overlap**")
                kc1, kc2 = st.columns(2)
                with kc1:
                    st.markdown("**✅ In Resume + JD**")
                    for s in matched: st.markdown(f"- {s}")
                    if not matched: st.caption("None found")
                with kc2:
                    st.markdown("**❌ Missing from Resume**")
                    for s in missing: st.markdown(f"- {s}")
                    if not missing: st.caption("None missing")

            st.markdown("---")
            st.markdown("### Candidate Summary")
            summary = generate_candidate_summary(result)
            st.markdown(summary)

        with tab_bulk:
            st.info("Only 1 file uploaded - switch to Single Resume Analysis tab.")

    # BULK MODE
    else:
        with tab_bulk:
            if apply_clicked:
                progress_bar = st.progress(0)
                status_text  = st.empty()
                results = []
                for i, file in enumerate(uploaded_files):
                    status_text.caption(f"Processing {file.name} ({i+1}/{len(uploaded_files)})")
                    results.append(process_resume(file, clean_jd, skills_list))
                    progress_bar.progress((i+1) / len(uploaded_files))
                status_text.empty(); progress_bar.empty()
                st.session_state.results = results

            if not st.session_state.results:
                st.info("Click Apply Filters & Process to start.")
                st.stop()

            df = pd.DataFrame(st.session_state.results)
            df = df.sort_values("Match Score %", ascending=False).reset_index(drop=True)
            df.index += 1

            df_f = df[df["Match Score %"] >= min_score].copy()
            st.session_state.results_df = df
            st.session_state.results_df_f = df_f
            if target_roles: df_f = df_f[df_f["Predicted Role"].isin(target_roles)]

            total     = len(df)
            filtered  = len(df_f)
            top_score = df["Match Score %"].max() if total > 0 else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("🗂️ Total Resumes", total)
            m2.metric("✅ Filtered Candidates", filtered)
            m3.metric("🏆 Top Score", f"{top_score}%")

            st.markdown("---")
            lc1, lc2 = st.columns([3,1])
            with lc1: st.markdown("### Ranking Leaderboard")
            with lc2:
                csv = df_f.drop(columns=["Matched Skills","Missing Skills","Cleaned Text","Raw Text"], errors="ignore").to_csv().encode("utf-8")
                st.download_button("Export Results (CSV)", csv, "screening_results.csv", "text/csv", use_container_width=True)

            display_df = df_f[["File Name","Predicted Role","Match Score %","Status"]].copy()
            display_df.columns = ["File Name","Predicted Job Category","Match Score %","Status"]
            display_df.index.name = "Rank"

            def color_status(val):
                if "Excellent" in str(val): return "color:#10b981;font-weight:700"
                if "Good"      in str(val): return "color:#a78bfa;font-weight:600"
                if "Low"       in str(val): return "color:#f59e0b;font-weight:600"
                if "Poor"      in str(val): return "color:#ef4444;font-weight:600"
                return ""

            styled = (display_df.style
                      .applymap(color_status, subset=["Status"])
                      .background_gradient(subset=["Match Score %"], cmap="Purples", vmin=0, vmax=100)
                      .format({"Match Score %": "{:.1f}%"}))
            st.dataframe(styled, use_container_width=True, height=380)

            st.markdown("---")
            st.markdown("### Quick Resume View")
            st.caption("Select a candidate to read text:")
            selected = st.selectbox("candidate", df["File Name"].tolist(), label_visibility="collapsed")
            if selected:
                row = df[df["File Name"] == selected].iloc[0]
                st.markdown(generate_candidate_summary(row))

        with tab_single:
            st.info(f"{len(uploaded_files)} files uploaded - see Bulk Screening tab for the leaderboard.")

else:
    with tab_bulk:
        st.markdown("<div style='text-align:center;padding:80px 20px'><h3 style='color:#8b9dc3'>Bulk Screening Mode</h3><p style='color:#4a5568;font-size:14px'>Upload 2-100+ PDFs + paste a Job Description in the sidebar, then click Apply Filters & Process.</p></div>", unsafe_allow_html=True)
    with tab_single:
        st.markdown("<div style='text-align:center;padding:80px 20px'><h3 style='color:#8b9dc3'>Single Resume Analysis</h3><p style='color:#4a5568;font-size:14px'>Upload 1 PDF + paste a Job Description in the sidebar, then click Apply Filters & Process.</p></div>", unsafe_allow_html=True)

st.markdown("---")
