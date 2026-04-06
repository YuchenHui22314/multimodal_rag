"""
asymmetric_biencoder_model.py
Asymmetric bi-encoder model for multimodal retrieval.

Query encoder : BAAI/bge-base-zh-v1.5 (768-dim, BERT-based)
Document side : pre-computed Qwen3-VL-Embedding-2B embeddings (2048-dim)
                optionally MRL-truncated to 768 at load time.

Five training variants (controlled by config["model"]["variant"]):
  fullft   - Full fine-tune BGE; no projection
  lora     - LoRA (r=8, attn+FFN) on BGE; no projection
  query_mlp- BGE frozen + trainable MLPx4 or GLUx4 on query side (768→768)
  doc_mlp  - BGE frozen + trainable MLPx4 or GLUx4 on doc side  (2048→768)
  both_mlp - BGE frozen + projection on both sides

Projection flavours (config["model"]["proj_type"]): "mlp" | "glu"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Projection layers (MLPx4 / GLUx4)
# ---------------------------------------------------------------------------

class MLPx4(nn.Module):
    """
    Two-layer MLP with hidden_dim = 4 × input_dim (from SAIL paper, arXiv:2412.04616).
    Activation: GELU.
    Output: L2-normalized.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = 4 * input_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)


class GLUx4(nn.Module):
    """
    Gated Linear Unit with hidden_dim = 4 × input_dim (from SAIL paper, arXiv:2412.04616).
    gate  = ReLU(gate_proj(x))
    value = value_proj(x)
    hidden = gate * value
    out   = out_proj(hidden)
    Output: L2-normalized.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = 4 * input_dim
        self.gate_proj  = nn.Linear(input_dim, hidden)
        self.value_proj = nn.Linear(input_dim, hidden)
        self.out_proj   = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate   = F.relu(self.gate_proj(x))
        value  = self.value_proj(x)
        hidden = gate * value
        out    = self.out_proj(hidden)
        return F.normalize(out, p=2, dim=-1)


def build_projection(proj_type: str, input_dim: int, output_dim: int) -> nn.Module:
    """Factory for projection layers."""
    if proj_type == "mlp":
        return MLPx4(input_dim, output_dim)
    elif proj_type == "glu":
        return GLUx4(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown proj_type='{proj_type}'. Choose 'mlp' or 'glu'.")


# ---------------------------------------------------------------------------
# Main asymmetric bi-encoder model
# ---------------------------------------------------------------------------

class AsymmetricBiEncoderModel(nn.Module):
    """
    Asymmetric bi-encoder.

    - Query side  : BGE-base-zh-v1.5 (BERT) + optional projection
    - Document side: pre-encoded embeddings (numpy, loaded outside this class)
                     + optional projection

    The model only handles the query forward pass.
    Doc embeddings are passed as plain tensors from the DataLoader.
    """

    # Dimensions
    BGE_DIM  = 768   # BGE-base-zh-v1.5 hidden size
    QWEN_DIM = 2048  # Qwen3-VL-Embedding-2B output dim

    def __init__(self, config: dict):
        super().__init__()
        model_cfg  = config["model"]
        self.variant   = model_cfg["variant"]       # fullft | lora | query_mlp | doc_mlp | both_mlp
        self.proj_type = model_cfg.get("proj_type", "mlp")  # mlp | glu

        # doc_dim: dimension of the pre-encoded doc embeddings loaded from disk
        #   - variants that truncate to 768 at load time  → doc_dim = 768
        #   - variants that keep 2048 and project later   → doc_dim = 2048
        self.doc_dim   = model_cfg.get("doc_dim", 768)

        # ---- Build BGE query encoder ----
        bge_path = model_cfg["model_name_or_path"]
        self.bge = AutoModel.from_pretrained(bge_path, trust_remote_code=True)

        # Freeze or add LoRA depending on variant
        if self.variant == "fullft":
            # All BGE parameters trainable
            pass

        elif self.variant == "lora":
            # LoRA on attention + FFN layers; everything else frozen
            lora_config = LoraConfig(
                r=model_cfg.get("lora_r", 8),
                lora_alpha=model_cfg.get("lora_alpha", 16),
                lora_dropout=model_cfg.get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                # Matches: self-attn q/k/v, attn output dense, FFN intermediate + output
                target_modules=["query", "key", "value",
                                 "intermediate.dense", "output.dense"],
            )
            self.bge = get_peft_model(self.bge, lora_config)
            self.bge.print_trainable_parameters()

        elif self.variant in ("query_mlp", "doc_mlp", "both_mlp"):
            # BGE fully frozen
            for p in self.bge.parameters():
                p.requires_grad = False

        else:
            raise ValueError(f"Unknown variant='{self.variant}'")

        # ---- Optional query-side projection (query_mlp, both_mlp) ----
        self.query_proj: nn.Module | None = None
        if self.variant in ("query_mlp", "both_mlp"):
            # 768 → 768 (maintain query embedding dimension)
            self.query_proj = build_projection(self.proj_type, self.BGE_DIM, self.BGE_DIM)

        # ---- Optional doc-side projection (doc_mlp, both_mlp) ----
        self.doc_proj: nn.Module | None = None
        if self.variant in ("doc_mlp", "both_mlp"):
            # doc_dim (2048) → 768
            self.doc_proj = build_projection(self.proj_type, self.doc_dim, self.BGE_DIM)

    # ------------------------------------------------------------------
    def encode_queries(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        """
        Encode queries through BGE + optional query projection.
        Returns L2-normalized embeddings of shape [B, 768].
        """
        bge_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            bge_kwargs["token_type_ids"] = token_type_ids

        outputs = self.bge(**bge_kwargs)

        # Mean pooling over non-padding tokens (standard for BGE/BERT)
        last_hidden = outputs.last_hidden_state          # [B, L, 768]
        mask = attention_mask.unsqueeze(-1).float()      # [B, L, 1]
        query_emb = (last_hidden * mask).sum(1) / mask.sum(1)  # [B, 768]

        if self.query_proj is not None:
            query_emb = self.query_proj(query_emb)  # already L2-normed inside proj
        else:
            query_emb = F.normalize(query_emb, p=2, dim=-1)

        return query_emb

    def project_docs(self, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Apply optional doc projection and L2-normalize.
        doc_embs : [N, doc_dim]
        Returns  : [N, 768]
        """
        if self.doc_proj is not None:
            return self.doc_proj(doc_embs)  # L2-normed inside proj
        else:
            return F.normalize(doc_embs, p=2, dim=-1)

    def forward(self, input_ids, attention_mask, doc_embs,
                token_type_ids=None):
        """
        Full forward pass used during training.

        Args:
            input_ids      : [B, L]
            attention_mask : [B, L]
            doc_embs       : [B * (1 + N_neg), doc_dim]  (pos first per query)
            token_type_ids : [B, L] (optional, BERT-style)

        Returns:
            query_emb : [B, 768]
            doc_emb   : [B * (1 + N_neg), 768]
        """
        query_emb = self.encode_queries(input_ids, attention_mask, token_type_ids)
        doc_emb   = self.project_docs(doc_embs)
        return query_emb, doc_emb

    # ------------------------------------------------------------------
    def save(self, save_dir: str):
        """Save model weights to save_dir."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        if self.variant == "lora":
            # Save only LoRA adapters (small)
            self.bge.save_pretrained(save_dir)
        else:
            # Save BGE state dict (full FT or frozen doesn't matter – save all)
            torch.save(self.bge.state_dict(), f"{save_dir}/bge_state_dict.pt")

        # Save projection layers if any
        if self.query_proj is not None:
            torch.save(self.query_proj.state_dict(), f"{save_dir}/query_proj.pt")
        if self.doc_proj is not None:
            torch.save(self.doc_proj.state_dict(), f"{save_dir}/doc_proj.pt")

    @classmethod
    def load(cls, config: dict, load_dir: str):
        """Load a saved model from load_dir."""
        model = cls(config)
        variant = config["model"]["variant"]

        if variant == "lora":
            from peft import PeftModel
            # Load LoRA adapters on top of base BGE
            model.bge = PeftModel.from_pretrained(model.bge.base_model, load_dir)
        else:
            bge_weights = torch.load(f"{load_dir}/bge_state_dict.pt", map_location="cpu")
            model.bge.load_state_dict(bge_weights)

        import os
        if model.query_proj is not None and os.path.exists(f"{load_dir}/query_proj.pt"):
            model.query_proj.load_state_dict(
                torch.load(f"{load_dir}/query_proj.pt", map_location="cpu")
            )
        if model.doc_proj is not None and os.path.exists(f"{load_dir}/doc_proj.pt"):
            model.doc_proj.load_state_dict(
                torch.load(f"{load_dir}/doc_proj.pt", map_location="cpu")
            )
        return model
