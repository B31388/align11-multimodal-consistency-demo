# app.py
# ------------------------------------------------------------------------------
# Multimodal Consistency Verification Demo
# TMCL / Align11 Framework – Stage-1 + Stage-2 Explainability
# ------------------------------------------------------------------------------

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from transformers import BertTokenizer
import yaml

# Local modules
from model.tmcl import load_tmcl_from_hub
from model.explain import explain_pair

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Proposed Model Demo – Multimodal Consistency Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .consistent-box {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .inconsistent-box {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load config
# ----------------------------------------------------------------------
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
THRESHOLD = config["inference"]["default_threshold"]
IMAGE_SIZE = config["model"]["image_size"]
MAX_LENGTH = config["model"]["max_length"]

# ----------------------------------------------------------------------
# Load model (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    REPO_ID = config["model"]["repo_id"]
    FILENAME = config["model"]["filename"]

    # Token from Streamlit secrets (never hardcode it)
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)

    try:
        model = load_tmcl_from_hub(
            repo_id=REPO_ID,
            filename=FILENAME,
            token=HF_TOKEN,
            device=device,
            freeze=True,
            attn_implementation="eager"
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
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    return model, tokenizer, transform, device, model_status


def create_attention_overlay(image_pil: Image.Image, vit_map: np.ndarray) -> Image.Image:
    if vit_map is None:
        return image_pil

    att_img = Image.fromarray((vit_map * 255).astype(np.uint8)).resize(
        image_pil.size, resample=Image.BILINEAR
    )
    att_np = np.array(att_img) / 255.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_pil)
    ax.imshow(att_np, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=120)
    buf.seek(0)
    overlay = Image.open(buf).convert("RGB")
    plt.close(fig)
    return overlay


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 Proposed Model Demo")
    st.caption("Multimodal Consistency Verification")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "New Verification", "History", "Analytics", "Settings", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Application Domain**")
    domain = st.selectbox(
        "Domain",
        ["General (Natural Images)", "Customs X-ray (Future)", "Document Verification"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Officer**")
    st.selectbox("User", ["demo_user", "officer_01"], label_visibility="collapsed")

    st.markdown("---")
    st.caption("© 2026 Proposed Model Demo  \nAlign11 Framework")

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
model, tokenizer, transform, device, model_status = load_resources()

col_status1, col_status2 = st.columns([6, 1])
with col_status2:
    if model_status == "Model Loaded":
        st.success("Model Loaded", icon="✅")
    else:
        st.error("Model Not Loaded")

# ----------------------------------------------------------------------
# Main pages
# ----------------------------------------------------------------------
if page in ["Dashboard", "New Verification"]:

    st.title("New Consistency Verification")
    st.markdown("Upload an image and enter the corresponding description to verify consistency.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Image (Natural Image)")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, use_container_width=True)
        else:
            st.info("Please upload an image to begin.")
            image_pil = None

    with col2:
        st.subheader("2. Description / Text")
        description = st.text_area(
            "Enter the textual description of the image.",
            height=180,
            placeholder="A black mountain bike with red details standing on a dirt path near a lake...",
            max_chars=512
        )
        st.caption(f"{len(description)} / 512")

    st.markdown("")
    verify_clicked = st.button("🔍  Verify Consistency", use_container_width=True, type="primary")

    if verify_clicked:
        if image_pil is None:
            st.warning("Please upload an image first.")
        elif not description.strip():
            st.warning("Please enter a description.")
        elif model is None:
            st.error("Model is not loaded. Check your Hugging Face repo_id and token.")
        else:
            with st.spinner("Running multimodal consistency check + explanations..."):
                result = explain_pair(
                    image_path=image_pil,
                    document=description.strip(),
                    model=model,
                    tokenizer=tokenizer,
                    transform=transform,
                    threshold=THRESHOLD,
                    device=device,
                    show_plot=False,
                    top_k_tokens=12
                )

                margin = result["similarity"] - THRESHOLD
                result["margin"] = margin
                result["verified_at"] = datetime.now().strftime("%b %d, %Y %H:%M:%S")
                result["description"] = description.strip()
                result["image_pil"] = image_pil

                st.session_state.last_result = result

                st.session_state.history.insert(0, {
                    "id": f"#{1000 + len(st.session_state.history) + 1}",
                    "description_short": description.strip()[:45] + ("..." if len(description) > 45 else ""),
                    "score": round(result["similarity"], 2),
                    "decision": result["decision"].upper(),
                    "time": result["verified_at"]
                })

    # Results
    if st.session_state.last_result is not None:
        res = st.session_state.last_result

        st.markdown("---")
        st.subheader("Verification Result")

        col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

        with col_res1:
            if res["decision"] == "Consistent":
                st.markdown(f"""
                <div class="consistent-box">
                    <h2 style="color:#059669; margin:0;">✅ CONSISTENT</h2>
                    <p style="margin:0.3rem 0 0 0;">The image and description are consistent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="inconsistent-box">
                    <h2 style="color:#dc2626; margin:0;">❌ INCONSISTENT</h2>
                    <p style="margin:0.3rem 0 0 0;">The image and description do not match.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            st.metric("Decision Threshold (τ)", f"{THRESHOLD:.2f}")
            st.caption(f"Verified At\n{res['verified_at']}")

        with col_res3:
            st.metric("Margin", f"{res['margin']:+.2f}")
            st.caption("Model\nAlign11 (Frozen)")

        # Explanations
        st.markdown("")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            st.markdown("#### 👁 Visual Explanation")
            if res["vit_attention_map"] is not None:
                overlay = create_attention_overlay(res["image_pil"], res["vit_attention_map"])
                st.image(overlay, use_container_width=True)
            else:
                st.image(res["image_pil"], use_container_width=True)

        with exp_col2:
            st.markdown("#### 📊 Textual Explanation")
            if res["top_tokens"]:
                fig, ax = plt.subplots(figsize=(5, 4))
                colours = sns.color_palette("Reds", n_colors=len(res["top_tokens"]))
                ax.barh(range(len(res["top_tokens"])), res["top_scores"][::-1], color=colours[::-1])
                ax.set_yticks(range(len(res["top_tokens"])))
                ax.set_yticklabels(res["top_tokens"][::-1], fontsize=9)
                ax.set_xlabel("Importance")
                ax.set_xlim(0, max(res["top_scores"]) * 1.15)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with exp_col3:
            st.markdown("#### 💡 Explanation Summary")
            if res["decision"] == "Consistent":
                st.success("The image clearly shows content matching the description.")
                st.info(f"Key terms like **{', '.join(res['top_tokens'][:4])}** strongly support consistency.")
                st.success("No conflicting or suspicious elements detected.")
            else:
                st.error("Significant mismatch detected between image and description.")
                st.warning("Key visual elements do not align with the provided text.")
                st.info("Manual review is recommended.")

        with st.expander("🛡 Risk Interpretation Guide", expanded=False):
            st.markdown("""
            | Score Range              | Decision       | Risk Level     | Action                          |
            |--------------------------|----------------|----------------|---------------------------------|
            | **Score ≥ 0.70**         | Consistent     | Low risk       | Image and description match     |
            | **0.40 ≤ Score < 0.70**  | Review         | Medium risk    | Manual review recommended       |
            | **Score < 0.40**         | Inconsistent   | High risk      | Image and description do not match |
            """)

elif page == "History":
    st.title("Recent Verifications")
    if not st.session_state.history:
        st.info("No verifications yet. Go to **New Verification** to start.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Analytics":
    st.title("Analytics")
    st.info("Coming soon – consistency rate, score distribution, domain statistics.")

elif page == "Settings":
    st.title("Settings")
    st.write(f"Current threshold: **{THRESHOLD}**")
    st.info("To change the threshold permanently, edit `config.yaml`.")

elif page == "About":
    st.title("About")
    st.markdown("""
    ### Multimodal Consistency Verification Demo

    This interactive demo showcases a **two-tower multimodal model** (ViT + BERT)
    trained with contrastive learning (TMCL / Align11 framework).

    **Features**
    - Image–text consistency scoring
    - Visual explanation via ViT attention maps
    - Textual explanation via BERT token importance
    - Risk interpretation guide

    Developed as part of ongoing research on multimodal AI for risk detection.
    """)