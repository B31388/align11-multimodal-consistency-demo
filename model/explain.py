# model/explain.py
# ------------------------------------------------------------------------------
# Stage-2 Explainability utilities for TMCL / Align11
# - ViT attention map extraction
# - BERT token importance (from [CLS])
# - Full explain_pair pipeline
# ------------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer


def get_vit_attention_map(
    attentions: Optional[Tuple[torch.Tensor, ...]],
    head_fusion: str = "mean"
) -> Optional[np.ndarray]:
    """
    Extract the CLS → patch attention map from the last ViT layer.

    Args:
        attentions: tuple of attention tensors from ViT (one per layer)
        head_fusion: "mean" or "max" across attention heads

    Returns:
        2-D numpy array of shape (14, 14) for ViT-B/16, or None
    """
    if attentions is None or len(attentions) == 0:
        print("⚠ ViT attentions not available")
        return None

    # Last layer: (batch=1, heads, seq, seq)
    att = attentions[-1][0]

    if head_fusion == "mean":
        att = att.mean(dim=0)          # (seq, seq)
    else:
        att = att.max(dim=0)[0]

    # Attention from CLS token to all patches (drop CLS itself)
    cls_att = att[0, 1:]               # (196,) for 14×14 patches
    cls_att = cls_att / (cls_att.sum() + 1e-8)

    side = int(cls_att.shape[0] ** 0.5)  # 14
    return cls_att.reshape(side, side).detach().cpu().numpy()


def get_bert_token_importance(
    attentions: Optional[Tuple[torch.Tensor, ...]],
    attention_mask: torch.Tensor
) -> Optional[np.ndarray]:
    """
    Compute token importance scores from the last BERT layer
    using attention weights from the [CLS] token.
    """
    if attentions is None or len(attentions) == 0:
        print("⚠ BERT attentions not available")
        return None

    # Last layer, average over heads → (seq, seq)
    att = attentions[-1][0].mean(dim=0)
    cls_att = att[0]                   # attention from [CLS] to all tokens

    mask = attention_mask[0].detach().cpu().numpy()
    scores = cls_att.detach().cpu().numpy() * mask

    if scores.sum() > 0:
        scores = scores / scores.sum()

    return scores


def explain_pair(
    image_path: Union[str, Path, Image.Image],
    document: str,
    model: torch.nn.Module,
    tokenizer: BertTokenizer,
    transform,
    threshold: float = 0.40,
    device: torch.device = None,
    max_length: int = 128,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    top_k_tokens: int = 15
) -> Dict:
    """
    Run a full consistency check + visual & textual explanation.
    """
    if device is None:
        device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
    if isinstance(image_path, (str, Path)):
        image_pil = Image.open(image_path).convert("RGB")
    else:
        image_pil = image_path.convert("RGB")

    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    encoding = tokenizer(
        document,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ------------------------------------------------------------------
    # Forward pass with attentions
    # ------------------------------------------------------------------
    with torch.no_grad():
        out = model(
            image_tensor,
            input_ids,
            attention_mask,
            output_attentions=True
        )

    similarity = out["similarity"].item()
    decision = "Consistent" if similarity >= threshold else "Inconsistent"

    # ------------------------------------------------------------------
    # Extract explanations
    # ------------------------------------------------------------------
    vit_map = get_vit_attention_map(out["image_attentions"])
    token_scores = get_bert_token_importance(out["text_attentions"], attention_mask)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    real_len = int(attention_mask[0].sum().item())
    tokens = tokens[:real_len]

    if token_scores is not None:
        token_scores = token_scores[:real_len]
    else:
        token_scores = np.ones(real_len) / real_len

    # Top-k tokens
    order = np.argsort(token_scores)[::-1]
    top_k = min(top_k_tokens, len(tokens))
    top_tokens = [tokens[i] for i in order[:top_k]]
    top_scores = [float(token_scores[i]) for i in order[:top_k]]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original image
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Attention heatmap overlay
    if vit_map is not None:
        att_img = Image.fromarray((vit_map * 255).astype(np.uint8)).resize(
            image_pil.size, resample=Image.BILINEAR
        )
        att_np = np.array(att_img) / 255.0
        axes[1].imshow(image_pil)
        axes[1].imshow(att_np, cmap="jet", alpha=0.45)
        axes[1].set_title("ViT Attention (CLS → patches)", fontsize=12)
    else:
        axes[1].imshow(image_pil)
        axes[1].set_title("ViT Attention (not available)", fontsize=12)
    axes[1].axis("off")

    # Token importance bar chart
    colours = sns.color_palette("Reds", n_colors=top_k)
    axes[2].barh(range(top_k), top_scores[::-1], color=colours[::-1])
    axes[2].set_yticks(range(top_k))
    axes[2].set_yticklabels(top_tokens[::-1], fontsize=9)
    axes[2].set_xlabel("Attention weight (from [CLS])")
    axes[2].set_title(f"Top-{top_k} Token Importance", fontsize=12)
    axes[2].set_xlim(0, max(top_scores) * 1.15 if top_scores else 1)

    fig.suptitle(
        f"Decision: {decision}  |  Similarity = {similarity:.4f}  |  Threshold = {threshold}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved explanation → {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "similarity": similarity,
        "decision": decision,
        "vit_attention_map": vit_map,
        "tokens": tokens,
        "token_scores": token_scores,
        "top_tokens": top_tokens,
        "top_scores": top_scores,
        "image_pil": image_pil,
        "figure": fig if show_plot else None
    }