# app.py
# ==============================================================================
# MULTIMODAL CONSISTENCY DECISION-SUPPORT DEMO
# TMCL / Align11 Framework
# Final clean version
# ==============================================================================

import streamlit as st
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from transformers import BertTokenizer
import yaml
from io import BytesIO

from model.tmcl import load_tmcl_from_hub
from model.explain import explain_pair


st.set_page_config(
    page_title="Proposed Model Demo – Multimodal Consistency Assessment",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 2.5rem !important; padding-bottom: 3rem !important; max-width: 1500px; }
.stApp { background-color: #f8fafc; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #07162f 0%, #0b1f3a 100%); }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
.stButton > button { border-radius: 8px; min-height: 3rem; font-weight: 650; }
.stButton > button[kind="primary"] { background-color: #2563eb; color: white; border: none; }
.stButton > button[kind="primary"]:hover { background-color: #1d4ed8; color: white; }
.section-header { font-size: 1.05rem; font-weight: 700; color: #172033; margin-bottom: 0.55rem; }
.section-eyebrow { font-size: 0.78rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.7rem; }
.status-online { display: inline-block; background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; padding: 0.32rem 0.70rem; border-radius: 999px; font-size: 0.80rem; font-weight: 700; }
.status-offline { display: inline-block; background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; padding: 0.32rem 0.70rem; border-radius: 999px; font-size: 0.80rem; font-weight: 700; }
.assessment-card { border-radius: 12px; padding: 1.30rem 1.35rem; min-height: 145px; }
.assessment-low { background: #f0fdf4; border: 1px solid #86efac; }
.assessment-medium { background: #fffbeb; border: 1px solid #fcd34d; }
.assessment-high { background: #fff7ed; border: 1px solid #fdba74; }
.assessment-title { font-size: 1.28rem; font-weight: 750; color: #172033; margin-top: 0.50rem; margin-bottom: 0.40rem; }
.assessment-description { font-size: 0.94rem; line-height: 1.5; color: #475569; }
.risk-badge { display: inline-block; padding: 0.28rem 0.65rem; border-radius: 999px; font-size: 0.75rem; font-weight: 750; }
.risk-low { background: #dcfce7; color: #166534; }
.risk-medium { background: #fef3c7; color: #92400e; }
.risk-high { background: #ffedd5; color: #9a3412; }
.human-control-note { background: #eff6ff; border-left: 4px solid #2563eb; padding: 0.85rem 1rem; border-radius: 7px; color: #1e3a8a; font-size: 0.90rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
THRESHOLD = config["inference"]["default_threshold"]
IMAGE_SIZE = config["model"]["image_size"]
LOW_RISK_THRESHOLD = 0.70
HIGH_RISK_THRESHOLD = 0.40


@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = config["model"]["repo_id"]
    filename = config["model"]["filename"]
    hf_token = st.secrets.get("HF_TOKEN", None)
    try:
        model = load_tmcl_from_hub(
            repo_id=repo_id, filename=filename, token=hf_token,
            device=device, freeze=True, attn_implementation="eager"
        )
        model_status = "Model Loaded"
    except Exception as e:
        st.error(f"Could not load model: {e}")
        model = None
        model_status = "Model Not Loaded"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, tokenizer, transform, device, model_status


def create_attention_overlay(image_pil, vit_map):
    if vit_map is None:
        return image_pil
    try:
        vit_map = np.asarray(vit_map)
        vit_min, vit_max = np.min(vit_map), np.max(vit_map)
        if vit_max > vit_min:
            vit_map = (vit_map - vit_min) / (vit_max - vit_min)
        att_img = Image.fromarray((vit_map * 255).astype(np.uint8)).resize(
            image_pil.size, resample=Image.BILINEAR
        )
        att_np = np.array(att_img) / 255.0
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(image_pil)
        ax.imshow(att_np, cmap="jet", alpha=0.46)
        ax.axis("off")
        fig.tight_layout(pad=0)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=130)
        buf.seek(0)
        overlay = Image.open(buf).convert("RGB")
        plt.close(fig)
        return overlay
    except Exception as e:
        st.error(f"Error creating attention overlay: {e}")
        return image_pil


def get_assessment(similarity):
    if similarity >= LOW_RISK_THRESHOLD:
        return {
            "label": "No significant mismatch detected", "risk": "Low", "icon": "✓",
            "css": "assessment-low", "badge": "risk-low",
            "message": "The model found substantial alignment between the image and the submitted description.",
            "action": "Review the evidence and confirm whether the description adequately represents the image.",
        }
    elif similarity >= HIGH_RISK_THRESHOLD:
        return {
            "label": "Review recommended", "risk": "Medium", "icon": "⚠",
            "css": "assessment-medium", "badge": "risk-medium",
            "message": "The model found partial alignment but also identified uncertainty between the image and the submitted description.",
            "action": "Review the highlighted evidence before making a final assessment.",
        }
    else:
        return {
            "label": "Potential mismatch detected", "risk": "High", "icon": "⚠",
            "css": "assessment-high", "badge": "risk-high",
            "message": "The model identified substantial differences between the image and the submitted description.",
            "action": "Carefully review the image and textual evidence before making a final decision.",
        }


def get_model_confidence(similarity):
    distance = min(abs(similarity - LOW_RISK_THRESHOLD), abs(similarity - HIGH_RISK_THRESHOLD))
    if distance >= 0.15: return "High"
    elif distance >= 0.07: return "Moderate"
    return "Low"


def safe_top_terms(result, n=4):
    return (result.get("top_tokens") or [])[:n]


if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "human_reviews" not in st.session_state: st.session_state.human_reviews = []


with st.sidebar:
    st.markdown("### ◈ Proposed Model Demo")
    st.caption("TMCL Decision-Support Framework")
    st.markdown("---")
    page = st.radio("Navigation", ["New Analysis", "Dashboard", "History", "Analytics", "Settings", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Application Domain**")
    application_domain = st.selectbox("Domain", ["General (Natural Images)", "Customs & Trade", "Customs X-ray (Future)", "Document Verification"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**User / Reviewer**")
    reviewer = st.selectbox("Reviewer", ["demo_user", "officer_01"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""**Risk Guide**

🟢 **Low** (0.70 – 1.00)  
No significant mismatch

🟡 **Medium** (0.40 – 0.69)  
Needs review

🟠 **High** (0.00 – 0.39)  
Potential mismatch""")
    st.markdown("---")
    st.caption("© 2026 Mugimba Kakure Jude")


model, tokenizer, transform, device, model_status = load_resources()


if page in ["New Analysis", "Dashboard"]:
    title_col, status_col = st.columns([5, 1.2])
    with title_col:
        st.title("Analyze Image & Description")
        st.caption("The model assesses alignment between the image and submitted description. The final assessment remains with the human reviewer.")
    with status_col:
        if model_status == "Model Loaded":
            st.markdown('<br><span class="status-online">● Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<br><span class="status-offline">● Offline</span>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-header">1. Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, use_container_width=True)
            st.caption(f"Selected: {uploaded_file.name}")
        else:
            st.info("Upload an image to begin the assessment.")
            image_pil = None
    with col2:
        st.markdown('<div class="section-header">2. Description</div>', unsafe_allow_html=True)
        description = st.text_area("Description", height=210, placeholder="Enter the description associated with the image...", max_chars=512, label_visibility="collapsed")
        st.caption(f"{len(description)} / 512 characters")

    st.markdown("")
    analyze_clicked = st.button("🔎  Analyze Image & Description", type="primary", use_container_width=True)

    if analyze_clicked:
        if image_pil is None:
            st.warning("Please upload an image before running the analysis.")
        elif not description.strip():
            st.warning("Please enter the corresponding description.")
        elif model is None:
            st.error("The model is currently unavailable.")
        else:
            with st.spinner("Analyzing multimodal consistency and generating evidence..."):
                result = explain_pair(
                    image_path=image_pil, document=description.strip(), model=model,
                    tokenizer=tokenizer, transform=transform, threshold=THRESHOLD,
                    device=device, show_plot=False, top_k_tokens=12
                )
                similarity = float(result["similarity"])
                result["margin"] = similarity - THRESHOLD
                result["verified_at"] = datetime.now().strftime("%b %d, %Y %H:%M:%S")
                result["description"] = description.strip()
                result["image_pil"] = image_pil
                result["human_assessment"] = get_assessment(similarity)
                result["model_confidence"] = get_model_confidence(similarity)
                st.session_state.last_result = result
                assessment = result["human_assessment"]
                desc_short = description.strip()[:48] + ("..." if len(description.strip()) > 48 else "")
                st.session_state.history.insert(0, {
                    "id": f"#{1001 + len(st.session_state.history)}",
                    "description_short": desc_short,
                    "score": round(similarity, 3),
                    "model_assessment": assessment["label"],
                    "risk": assessment["risk"],
                    "reviewer": reviewer,
                    "time": result["verified_at"],
                })

    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        similarity = float(res["similarity"])
        assessment = res.get("human_assessment", get_assessment(similarity))
        confidence = res.get("model_confidence", get_model_confidence(similarity))

        st.markdown("---")
        st.markdown('<div class="section-eyebrow">Model Assessment</div>', unsafe_allow_html=True)

        result_col, risk_col, meaning_col = st.columns([2.4, 1, 1.4], gap="large")
        with result_col:
            st.markdown(f"""
<div class="assessment-card {assessment['css']}">
<span class="risk-badge {assessment['badge']}">{assessment['risk']} Risk</span>
<div class="assessment-title">{assessment['icon']} {assessment['label']}</div>
<div class="assessment-description">{assessment['message']}</div>
</div>
""", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("**Model Confidence**")
                st.markdown(f"### {confidence}")
                st.caption("Relative confidence based on distance from assessment boundaries.")
            with m2:
                st.markdown("**Similarity Score**")
                st.markdown(f"### {similarity:.2f}")
                st.progress(min(max(similarity, 0.0), 1.0))
                st.caption("Higher scores indicate stronger image–description alignment.")
        with risk_col:
            st.markdown("#### Risk Level")
            if assessment["risk"] == "Low": st.success("LOW\n\nNo significant mismatch")
            elif assessment["risk"] == "Medium": st.warning("MEDIUM\n\nNeeds Review")
            else: st.warning("HIGH\n\nPotential Mismatch")
            st.caption(assessment["action"])
        with meaning_col:
            st.markdown("#### What does this mean?")
            st.write(assessment["message"])
            st.info("Review the evidence below before making your own assessment.")

        st.markdown("")
        st.markdown('<div class="section-eyebrow">Evidence Used by the Model</div>', unsafe_allow_html=True)
        exp1, exp2, exp3 = st.columns(3, gap="large")

        with exp1:
            st.markdown("#### 👁 Image Evidence")
            st.caption("Highlighted regions contributed most strongly to the model assessment.")
            if res.get("vit_attention_map") is not None:
                try:
                    overlay = create_attention_overlay(res["image_pil"], res["vit_attention_map"])
                    st.image(overlay, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render attention map: {e}")
                    st.image(res["image_pil"], use_container_width=True)
            else:
                st.image(res["image_pil"], use_container_width=True)
                st.caption("Visual attention evidence is unavailable.")

        with exp2:
            st.markdown("#### 📊 Description Evidence")
            st.caption("Terms that contributed most strongly to the model comparison.")
            tokens = res.get("top_tokens", []) or []
            scores = res.get("top_scores", []) or []
            if tokens and len(scores) > 0:
                try:
                    tokens_to_plot = tokens[:10]
                    scores_to_plot = np.asarray(scores[:10], dtype=float)
                    fig, ax = plt.subplots(figsize=(5.0, 4.0))
                    colours = sns.color_palette("Blues", n_colors=len(tokens_to_plot))
                    ax.barh(range(len(tokens_to_plot)), scores_to_plot[::-1], color=colours[::-1])
                    ax.set_yticks(range(len(tokens_to_plot)))
                    ax.set_yticklabels(tokens_to_plot[::-1], fontsize=9)
                    ax.set_xlabel("Relative Importance", fontsize=9)
                    max_score = float(np.max(scores_to_plot)) if len(scores_to_plot) else 1.0
                    if max_score > 0: ax.set_xlim(0, max_score * 1.18)
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not render token chart: {e}")
            else:
                st.info("No textual evidence scores available.")

        with exp3:
            st.markdown("#### 💡 Assessment Summary")
            top_terms = safe_top_terms(res, n=4)
            if assessment["risk"] == "Low":
                st.success("The model found substantial alignment between the image and description.")
                if top_terms: st.write("**Influential terms:** " + ", ".join(top_terms))
                st.info("This result does not independently establish that the description is correct. Human review remains appropriate.")
            elif assessment["risk"] == "Medium":
                st.warning("The model found partial agreement, but the result lies in the review range.")
                if top_terms: st.write("**Influential terms:** " + ", ".join(top_terms))
                st.info("Examine the highlighted regions and terms before deciding.")
            else:
                st.warning("The model identified substantial image–description inconsistency.")
                if top_terms: st.write("**Influential terms:** " + ", ".join(top_terms))
                st.info("Treat this as a review signal rather than an automated final decision.")

        st.markdown("---")
        st.markdown('<div class="section-eyebrow">Your Review & Decision</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="human-control-note">
<strong>Human decision authority:</strong>
The model provides decision-support evidence only.
Review the original image, submitted description and model evidence before recording your assessment.
</div>
""", unsafe_allow_html=True)

        st.write("**Based on the image, description and model evidence, what is your assessment?**")
        human_decision = st.radio("Human assessment", ["Confirm Match", "Request Review", "Confirm Mismatch"], horizontal=True, label_visibility="collapsed", key="human_decision")

        rc1, rc2 = st.columns(2, gap="large")
        with rc1:
            st.markdown("##### Do you agree with the model assessment?")
            agreement = st.radio("Agreement", ["Agree", "Disagree", "Unsure"], horizontal=True, label_visibility="collapsed", key="model_agreement")
            disagreement_reason = ""
            if agreement == "Disagree":
                disagreement_reason = st.text_area("Why do you disagree? (Optional)", placeholder="Describe what the model may have missed...", max_chars=300, key="disagreement_reason")
        with rc2:
            st.markdown("##### Additional Comments")
            reviewer_comment = st.text_area("Comments", placeholder="Enter any observations...", max_chars=500, label_visibility="collapsed", key="reviewer_comment")

        _, submit_col = st.columns([3, 1])
        with submit_col:
            if st.button("Submit Assessment", type="primary", use_container_width=True):
                st.session_state.human_reviews.insert(0, {
                    "analysis_time": res["verified_at"],
                    "review_time": datetime.now().strftime("%b %d, %Y %H:%M:%S"),
                    "reviewer": reviewer,
                    "application_domain": application_domain,
                    "similarity": round(similarity, 4),
                    "model_assessment": assessment["label"],
                    "model_risk": assessment["risk"],
                    "human_decision": human_decision,
                    "agreement": agreement,
                    "disagreement_reason": disagreement_reason,
                    "comments": reviewer_comment,
                })
                st.success("Assessment recorded successfully.")

        with st.expander("⚙ Technical Model Details", expanded=False):
            t1, t2, t3, t4 = st.columns(4)
            t1.markdown(f"**Raw Similarity**\n\n### {similarity:.4f}")
            t2.markdown(f"**Threshold (τ)**\n\n### {THRESHOLD:.4f}")
            t3.markdown(f"**Margin**\n\n### {res['margin']:+.4f}")
            t4.markdown(f"**Device**\n\n### {str(device).upper()}")
            st.caption(f"Timestamp: {res['verified_at']} · Model: TMCL / Align11 (Frozen)")
            st.warning("Confidence is a decision-support heuristic, not a calibrated probability.")

        with st.expander("🛡 Risk Interpretation Guide", expanded=False):
            st.markdown("""
| Similarity | Model Assessment | Risk | Recommended Action |
|---|---|---|---|
| **≥ 0.70** | No significant mismatch | Low | Review and confirm |
| **0.40 – 0.69** | Review recommended | Medium | Examine evidence carefully |
| **< 0.40** | Potential mismatch | High | Detailed human review |
""")
            st.caption("Risk categories are decision-support indicators only.")


elif page == "History":
    st.title("Analysis History")
    if not st.session_state.history: st.info("No analyses have been completed yet.")
    else: st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("Human Review History")
    if not st.session_state.human_reviews: st.info("No human assessments submitted yet.")
    else: st.dataframe(pd.DataFrame(st.session_state.human_reviews), use_container_width=True, hide_index=True)

elif page == "Analytics":
    st.title("Analytics")
    if not st.session_state.history: st.info("Complete some analyses first.")
    else:
        df = pd.DataFrame(st.session_state.history)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Total**\n\n### {len(df)}")
        c2.markdown(f"**Low Risk**\n\n### {(df['risk']=='Low').sum()}")
        c3.markdown(f"**Needs Review**\n\n### {(df['risk']=='Medium').sum()}")
        c4.markdown(f"**Potential Mismatch**\n\n### {(df['risk']=='High').sum()}")
        if st.session_state.human_reviews:
            rdf = pd.DataFrame(st.session_state.human_reviews)
            rate = (rdf["agreement"] == "Agree").sum() / len(rdf) * 100
            st.markdown(f"**Human–Model Agreement Rate**\n\n### {rate:.1f}%")

elif page == "Settings":
    st.title("Settings")
    st.write(f"Model threshold (τ): **{THRESHOLD:.4f}**")
    st.write(f"Image size: **{IMAGE_SIZE} × {IMAGE_SIZE}**")
    st.write(f"Device: **{str(device).upper()}**")
    st.markdown("---")
    st.write(f"Low risk ≥ **{LOW_RISK_THRESHOLD:.2f}**")
    st.write(f"Medium risk: **{HIGH_RISK_THRESHOLD:.2f} – {LOW_RISK_THRESHOLD:.2f}**")
    st.write(f"High risk < **{HIGH_RISK_THRESHOLD:.2f}**")
    st.info("Edit `config.yaml` to change the model threshold permanently.")

elif page == "About":
    st.title("About")
    st.markdown("""
### Multimodal Consistency Decision-Support Demo

Human-centered multimodal consistency assessment based on the **TMCL / Align11** framework.

**Workflow**  
Image + Description → Model Assessment → Evidence → Human Review → Human Decision

The model provides decision-support evidence only.  
Final authority remains with the human reviewer.
""")