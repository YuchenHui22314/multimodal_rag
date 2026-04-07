"""
asymmetric_biencoder_dataset.py
Dataset for asymmetric bi-encoder training.

Key design:
  - Query text is returned raw (tokenized in collate_fn by the caller)
  - Document embeddings are fetched from pre-computed numpy mmap files
    produced by encode_qwen3vl.py (2048-dim per note)
  - Every clicked note becomes an independent positive training sample.
  - Negatives are sampled from non-clicked notes in the same impression,
    padded with random corpus samples when not enough.

Usage example (see train_asymmetric_biencoder.py):
    dataset = AsymmetricBiEncoderDataset(config)
    loader  = DataLoader(dataset, batch_size=B, collate_fn=dataset.collate_fn, ...)
"""

import glob
import os
import random
import numpy as np
import pandas as pd
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
        local_parquet = os.path.join(data_path, "search_train", "train-00000-of-00001.parquet")
        print(f"[Dataset] Loading Qilin search_train from {data_path} …")
        if os.path.exists(local_parquet):
            self.dataset = pd.read_parquet(local_parquet)
            print(f"[Dataset] Loaded local parquet cache: {local_parquet}")
        else:
            self.dataset = load_dataset("THUIR/Qilin", "search_train")["train"]
            print("[Dataset] Loaded search_train from HuggingFace cache/hub")
        print(f"[Dataset] search_train size: {len(self.dataset)}")

        # Corpus size (for random negative sampling)
        # We get this from the mmap shape below; store as attribute.
        self.corpus_size = None  # set after loading mmaps

        # ---- Load pre-computed doc embeddings (memory-mapped) ----
        emb_dir = dset_cfg["doc_emb_dir"]
        print(f"[Dataset] Loading doc embeddings (mmap) from {emb_dir} …")
        shard_paths = sorted(glob.glob(os.path.join(emb_dir, "passage_gpu_*.npy")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No passage shards found under {emb_dir}. "
                "Expected files like passage_gpu_0.npy"
            )

        self._emb_shards = [np.load(path, mmap_mode="r") for path in shard_paths]
        self._shard_sizes = [len(shard) for shard in self._emb_shards]
        self._shard_offsets = []

        offset = 0
        for shard_size in self._shard_sizes:
            self._shard_offsets.append(offset)
            offset += shard_size

        self.corpus_size = offset
        shard_desc = ", ".join(
            f"shard{i}={size}" for i, size in enumerate(self._shard_sizes)
        )
        print(f"[Dataset] Corpus size: {self.corpus_size} ({shard_desc})")

        # ---- Optional: limit to queries whose pos+neg are already encoded ----
        # Set max_encoded_note_idx in config to restrict debug runs.
        # Only note_idxs < max_encoded_note_idx are guaranteed to have real embeddings.
        self.max_encoded_idx = dset_cfg.get("max_encoded_note_idx", None)
        self.sample_cache_dir = dset_cfg.get(
            "sample_cache_dir",
            "/data/rech/zhangyan/multimodal_rag/baselines/cache/asym_dataset",
        )

        self._build_samples()

    # ------------------------------------------------------------------
    def _get_item(self, idx: int):
        return self.dataset.iloc[idx] if hasattr(self.dataset, "iloc") else self.dataset[idx]

    def _get_positive_pool(self, item) -> list[int]:
        impressions = item.get("search_result_details_with_idx", [])
        if self.max_encoded_idx is not None:
            return [
                x["note_idx"] for x in impressions
                if x["click"] == 1 and x["note_idx"] < self.max_encoded_idx
            ]
        return [x["note_idx"] for x in impressions if x["click"] == 1]

    def _get_negative_pool(self, item) -> list[int]:
        impressions = item.get("search_result_details_with_idx", [])
        if self.max_encoded_idx is not None:
            return [
                x["note_idx"] for x in impressions
                if x["click"] == 0 and x["note_idx"] < self.max_encoded_idx
            ]
        return [x["note_idx"] for x in impressions if x["click"] == 0]

    def _build_samples(self):
        """
        Expand each (query, positive) pair into an independent training sample.

        This uses all clicked notes instead of randomly sampling one positive
        per query, which better matches Qilin's multi-positive supervision.
        """
        os.makedirs(self.sample_cache_dir, exist_ok=True)
        max_idx_tag = self.max_encoded_idx if self.max_encoded_idx is not None else "all"
        cache_path = os.path.join(
            self.sample_cache_dir,
            f"{self.__class__.__name__}_{len(self.dataset)}q_max{max_idx_tag}.npz",
        )

        if os.path.exists(cache_path):
            cache = np.load(cache_path)
            item_indices = cache["item_indices"].astype(np.int64)
            positive_indices = cache["positive_indices"].astype(np.int64)
            self.samples = list(zip(item_indices.tolist(), positive_indices.tolist()))
            print(f"[Dataset] Loaded expanded sample cache: {cache_path}")
            print(f"[Dataset] Expanded training samples: {len(self.samples)} "
                  f"from {len(self.dataset)} queries")
            return

        self.samples = []
        for item_idx in range(len(self.dataset)):
            item = self._get_item(item_idx)
            positives = self._get_positive_pool(item)
            negatives = self._get_negative_pool(item)
            if not positives or not negatives:
                continue
            for positive_idx in positives:
                self.samples.append((item_idx, positive_idx))

        if self.samples:
            item_indices = np.asarray([sample[0] for sample in self.samples], dtype=np.int32)
            positive_indices = np.asarray([sample[1] for sample in self.samples], dtype=np.int32)
            np.savez_compressed(
                cache_path,
                item_indices=item_indices,
                positive_indices=positive_indices,
            )
            print(f"[Dataset] Saved expanded sample cache: {cache_path}")

        print(f"[Dataset] Expanded training samples: {len(self.samples)} "
              f"from {len(self.dataset)} queries")

    # ------------------------------------------------------------------
    def _get_emb(self, note_idx: int) -> np.ndarray:
        """Fetch pre-computed embedding for note_idx (shape [truncate_dim])."""
        if note_idx < 0 or note_idx >= self.corpus_size:
            raise IndexError(f"note_idx {note_idx} out of range for corpus size {self.corpus_size}")

        shard_id = len(self._shard_offsets) - 1
        for idx, offset in enumerate(self._shard_offsets):
            next_offset = self.corpus_size if idx + 1 == len(self._shard_offsets) else self._shard_offsets[idx + 1]
            if offset <= note_idx < next_offset:
                shard_id = idx
                break

        local_idx = note_idx - self._shard_offsets[shard_id]
        raw = self._emb_shards[shard_id][local_idx, :self.truncate_dim]
        return raw.copy()  # copy from mmap to avoid stale references

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        real_idx, positive_idx = self.samples[idx]
        item = self._get_item(real_idx)

        # ---- Query text ----
        query = item["query"]

        # ---- Positive note ----
        pos_emb = self._get_emb(positive_idx)

        # ---- Negative notes ----
        neg_pool = self._get_negative_pool(item)

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
