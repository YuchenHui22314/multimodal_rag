"""
asymmetric_biencoder_dataset.py
Dataset for asymmetric bi-encoder training.

Key design:
  - Query text is returned raw (tokenized in collate_fn by the caller)
  - Document embeddings are fetched from pre-computed numpy mmap files
    produced by encode_qwen3vl.py (2048-dim per note)
  - Positive/negative sampling follows the same strategy as the original
    Qilin DenseRetrievalTrainingDataProcessor:
      positive  : one randomly-sampled clicked note (click == 1)
      negatives : N randomly-sampled non-clicked notes from the impression
                  padded with random corpus samples when not enough

Usage example (see train_asymmetric_biencoder.py):
    dataset = AsymmetricBiEncoderDataset(config)
    loader  = DataLoader(dataset, batch_size=B, collate_fn=dataset.collate_fn, ...)
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class AsymmetricBiEncoderDataset(Dataset):
    """
    PyTorch Dataset for asymmetric bi-encoder training.

    Each sample returns:
        {
          "query"   : str,
          "pos_emb" : np.ndarray [doc_dim],      # positive doc embedding
          "neg_embs": np.ndarray [N_neg, doc_dim] # negative doc embeddings
        }
    """

    def __init__(self, config: dict, split: str = "train"):
        """
        Args:
            config : full experiment config dict
            split  : "train" (future: "val")
        """
        dset_cfg   = config["datasets"]
        model_cfg  = config["model"]

        self.N_neg      = dset_cfg["negative_samples"]       # negatives per query
        self.use_title   = dset_cfg.get("use_title", True)
        self.use_content = dset_cfg.get("use_content", True)

        # doc_dim loaded from disk (might be 768 or 2048 depending on variant)
        self.doc_dim   = model_cfg.get("doc_dim", 768)
        # Slice doc embeddings to doc_dim at load time (MRL truncation)
        # If doc_dim == QWEN_DIM (2048) we keep the full vector.
        self.truncate_dim = self.doc_dim  # truncate mmap slice to this dim

        # ---- Load Qilin search_train split ----
        data_path = dset_cfg["dataset_name_or_path"]
        print(f"[Dataset] Loading Qilin search_train from {data_path} …")
        self.dataset = load_dataset("THUIR/Qilin", "search_train")["train"]
        print(f"[Dataset] search_train size: {len(self.dataset)}")

        # Corpus size (for random negative sampling)
        # We get this from the mmap shape below; store as attribute.
        self.corpus_size = None  # set after loading mmaps

        # ---- Load pre-computed doc embeddings (memory-mapped) ----
        emb_dir = dset_cfg["doc_emb_dir"]
        gpu0_path = f"{emb_dir}/passage_gpu_0.npy"
        gpu1_path = f"{emb_dir}/passage_gpu_1.npy"
        print(f"[Dataset] Loading doc embeddings (mmap) from {emb_dir} …")
        self._emb_gpu0 = np.load(gpu0_path, mmap_mode="r")  # [shard0_size, 2048]
        self._emb_gpu1 = np.load(gpu1_path, mmap_mode="r")  # [shard1_size, 2048]
        self._shard_size = len(self._emb_gpu0)               # 991969
        self.corpus_size  = len(self._emb_gpu0) + len(self._emb_gpu1)
        print(f"[Dataset] Corpus size: {self.corpus_size} "
              f"(shard0={len(self._emb_gpu0)}, shard1={len(self._emb_gpu1)})")

        # ---- Optional: limit to queries whose pos+neg are already encoded ----
        # Set max_encoded_note_idx in config to restrict debug runs.
        # Only note_idxs < max_encoded_note_idx are guaranteed to have real embeddings.
        self.max_encoded_idx = dset_cfg.get("max_encoded_note_idx", None)

        # Pre-filter dataset if max_encoded_idx is set
        if self.max_encoded_idx is not None:
            print(f"[Dataset] Filtering to note_idx < {self.max_encoded_idx} …")
            self._valid_indices = [
                i for i, item in enumerate(self.dataset)
                if self._sample_has_encoded_docs(item)
            ]
            print(f"[Dataset] {len(self._valid_indices)} / {len(self.dataset)} "
                  f"samples have all docs encoded")
        else:
            self._valid_indices = list(range(len(self.dataset)))

    # ------------------------------------------------------------------
    def _sample_has_encoded_docs(self, item: dict) -> bool:
        """Check that at least one positive and N negatives are within encoded range."""
        max_idx = self.max_encoded_idx
        impressions = item.get("search_result_details_with_idx", [])
        positives   = [x["note_idx"] for x in impressions if x["click"] == 1
                       and x["note_idx"] < max_idx]
        negatives   = [x["note_idx"] for x in impressions if x["click"] == 0
                       and x["note_idx"] < max_idx]
        return len(positives) >= 1 and len(negatives) >= 1  # at least 1 neg; pad rest randomly

    # ------------------------------------------------------------------
    def _get_emb(self, note_idx: int) -> np.ndarray:
        """Fetch pre-computed embedding for note_idx (shape [truncate_dim])."""
        if note_idx < self._shard_size:
            raw = self._emb_gpu0[note_idx, :self.truncate_dim]
        else:
            raw = self._emb_gpu1[note_idx - self._shard_size, :self.truncate_dim]
        return raw.copy()  # copy from mmap to avoid stale references

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self._valid_indices[idx]
        item = self.dataset[real_idx]

        # ---- Query text ----
        query = item["query"]

        # ---- Positive note ----
        impressions = item["search_result_details_with_idx"]

        # Filter to encoded range if needed
        if self.max_encoded_idx is not None:
            pos_pool = [x["note_idx"] for x in impressions if x["click"] == 1
                        and x["note_idx"] < self.max_encoded_idx]
        else:
            pos_pool = [x["note_idx"] for x in impressions if x["click"] == 1]

        positive_idx = random.choice(pos_pool)
        pos_emb = self._get_emb(positive_idx)

        # ---- Negative notes ----
        if self.max_encoded_idx is not None:
            neg_pool = [x["note_idx"] for x in impressions if x["click"] == 0
                        and x["note_idx"] < self.max_encoded_idx]
        else:
            neg_pool = [x["note_idx"] for x in impressions if x["click"] == 0]

        if len(neg_pool) < self.N_neg:
            # Pad with random corpus samples (within encoded range if needed)
            max_rand = self.max_encoded_idx if self.max_encoded_idx else self.corpus_size
            extra = random.sample(range(max_rand), self.N_neg - len(neg_pool))
            neg_pool = neg_pool + extra
        else:
            neg_pool = random.sample(neg_pool, self.N_neg)

        neg_embs = np.stack([self._get_emb(nidx) for nidx in neg_pool])  # [N, doc_dim]

        return {
            "query":    query,
            "pos_emb":  pos_emb,    # np [doc_dim]
            "neg_embs": neg_embs,   # np [N_neg, doc_dim]
        }

    # ------------------------------------------------------------------
    def get_dataloader(self, tokenizer_path: str, max_length: int,
                       batch_size: int, shuffle: bool = True,
                       num_workers: int = 4,
                       query_instruction: str = "") -> DataLoader:
        """
        Return a DataLoader with a collate_fn that tokenizes queries.

        Args:
            tokenizer_path   : path to BGE tokenizer
            max_length       : max token length for queries
            batch_size       : batch size (number of queries per step)
            shuffle          : shuffle training data
            num_workers      : DataLoader workers
            query_instruction: BGE-style instruction prefix (can be empty)
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        instruction = query_instruction  # captured in closure

        def collate_fn(samples):
            # Prepend instruction if provided
            queries = [
                (instruction + s["query"]) if instruction else s["query"]
                for s in samples
            ]
            B = len(samples)
            N = self.N_neg

            # Tokenize queries
            queries_tok = tokenizer(
                queries, truncation=True, padding=True,
                max_length=max_length, return_tensors="pt"
            )

            # Stack doc embeddings: [B * (1 + N), doc_dim]
            # Layout: pos_0, neg_0_1..neg_0_N, pos_1, neg_1_1..neg_1_N, ...
            doc_embs = []
            for s in samples:
                doc_embs.append(torch.from_numpy(s["pos_emb"]).float())
                doc_embs.extend([
                    torch.from_numpy(s["neg_embs"][i]).float()
                    for i in range(N)
                ])
            doc_embs = torch.stack(doc_embs)  # [B*(1+N), doc_dim]

            return {
                "input_ids":       queries_tok["input_ids"],
                "attention_mask":  queries_tok["attention_mask"],
                "token_type_ids":  queries_tok.get("token_type_ids"),
                "doc_embs":        doc_embs,
                "batch_size":      B,
                "n_neg":           N,
            }

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
