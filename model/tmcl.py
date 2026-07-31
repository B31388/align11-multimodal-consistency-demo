# model/tmcl.py
# ------------------------------------------------------------------------------
# TMCL Stage-1 Model Definition
# Multimodal Consistency Learning (ViT + BERT + Projection Heads)
# Fully compatible with Stage-2 explainability (eager attention)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel
from huggingface_hub import hf_hub_download


class ProjectionHead(nn.Module):
    """
    Projection head that maps 768-dim encoder outputs to a lower-dimensional
    normalized embedding space (default 256).
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512,
                 output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), p=2, dim=-1)


class TMCLStage1(nn.Module):
    """
    Two-tower multimodal model:
      - Vision encoder  : google/vit-base-patch16-224
      - Text encoder    : bert-base-uncased
      - Projection heads: map both modalities into a shared embedding space

    Returns cosine similarity between image and text embeddings.
    Supports output_attentions=True for explainability.
    """
    def __init__(
        self,
        vision_encoder: ViTModel,
        text_encoder: BertModel,
        image_projection: ProjectionHead,
        text_projection: ProjectionHead
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.text_projection = text_projection

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False
    ) -> dict:
        # Vision tower
        vision_out = self.vision_encoder(
            pixel_values=images,
            output_attentions=output_attentions
        )
        image_cls = vision_out.pooler_output          # (B, 768)

        # Text tower
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        text_cls = text_out.pooler_output             # (B, 768)

        # Project into shared space
        image_emb = self.image_projection(image_cls)  # (B, 256)
        text_emb  = self.text_projection(text_cls)    # (B, 256)

        # Cosine similarity (already L2-normalized)
        similarity = (image_emb * text_emb).sum(dim=-1)  # (B,)

        return {
            "image_emb": image_emb,
            "text_emb": text_emb,
            "similarity": similarity,
            "image_attentions": vision_out.attentions if output_attentions else None,
            "text_attentions": text_out.attentions if output_attentions else None,
        }


def build_tmcl_model(
    device: torch.device = None,
    attn_implementation: str = "eager"
) -> TMCLStage1:
    """
    Convenience factory that builds a complete TMCLStage1 model
    with frozen-ready ViT + BERT + projection heads.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_encoder = ViTModel.from_pretrained(
        "google/vit-base-patch16-224",
        attn_implementation=attn_implementation
    )
    text_encoder = BertModel.from_pretrained(
        "bert-base-uncased",
        attn_implementation=attn_implementation
    )

    model = TMCLStage1(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        image_projection=ProjectionHead(),
        text_projection=ProjectionHead()
    ).to(device)

    return model


def load_tmcl_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
    freeze: bool = True,
    attn_implementation: str = "eager"
) -> TMCLStage1:
    """
    Build the model and load a Stage-1 checkpoint from a local path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_tmcl_model(device=device, attn_implementation=attn_implementation)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model


def load_tmcl_from_hub(
    repo_id: str,
    filename: str = "TMCL_Stage1_Pretrained.pt",
    token: str = None,
    device: torch.device = None,
    freeze: bool = True,
    attn_implementation: str = "eager"
) -> TMCLStage1:
    """
    Load TMCL Stage-1 model from a (private) Hugging Face repository.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download the checkpoint (works with private repos when token is provided)
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token
    )

    return load_tmcl_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        freeze=freeze,
        attn_implementation=attn_implementation
    )