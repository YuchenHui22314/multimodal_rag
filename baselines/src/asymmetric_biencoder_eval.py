import csv
import glob
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


def load_qrels(qrels_path: str) -> dict[int, set[int]]:
    qrels = defaultdict(set)
    with open(qrels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qrels[int(row["qid"])].add(int(row["pid"]))
    return dict(qrels)


def average_precision(ranked_pids: list[int], relevant_pids: set[int]) -> float:
    if not relevant_pids:
        return 0.0

    num_hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(ranked_pids, start=1):
        if pid in relevant_pids:
            num_hits += 1
            precision_sum += num_hits / rank
    return precision_sum / len(relevant_pids)


def reciprocal_rank_at_k(ranked_pids: list[int], relevant_pids: set[int], k: int) -> float:
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_pids:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_pids: list[int], relevant_pids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for pid in ranked_pids[:k] if pid in relevant_pids)
    return hits / k


def recall_at_k(ranked_pids: list[int], relevant_pids: set[int], k: int) -> float:
    if not relevant_pids:
        return 0.0
    hits = sum(1 for pid in ranked_pids[:k] if pid in relevant_pids)
    return hits / len(relevant_pids)


def ndcg_at_k(ranked_pids: list[int], relevant_pids: set[int], k: int) -> float:
    if not relevant_pids:
        return 0.0

    dcg = 0.0
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_pids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_pids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_metrics(sorted_results: dict[int, list[int]], qrels: dict[int, set[int]]) -> dict[str, float]:
    ndcg_ks = [1, 3, 5, 10]
    p_ks = [1, 3, 5, 10]
    recall_ks = [5, 50, 100, 1000]
    mrr_ks = [1, 3, 5, 10]

    metrics = {
        "map": 0.0,
        **{f"ndcg_cut.{k}": 0.0 for k in ndcg_ks},
        **{f"P.{k}": 0.0 for k in p_ks},
        **{f"recall.{k}": 0.0 for k in recall_ks},
        **{f"MRR@{k}": 0.0 for k in mrr_ks},
    }

    valid_queries = 0
    for qid, relevant_pids in qrels.items():
        ranked_pids = sorted_results.get(qid)
        if not ranked_pids:
            continue

        valid_queries += 1
        metrics["map"] += average_precision(ranked_pids, relevant_pids)

        for k in ndcg_ks:
            metrics[f"ndcg_cut.{k}"] += ndcg_at_k(ranked_pids, relevant_pids, k)
        for k in p_ks:
            metrics[f"P.{k}"] += precision_at_k(ranked_pids, relevant_pids, k)
        for k in recall_ks:
            metrics[f"recall.{k}"] += recall_at_k(ranked_pids, relevant_pids, k)
        for k in mrr_ks:
            metrics[f"MRR@{k}"] += reciprocal_rank_at_k(ranked_pids, relevant_pids, k)

    if valid_queries == 0:
        return metrics

    for key in metrics:
        metrics[key] /= valid_queries
    return metrics


def load_search_test_queries(data_dir: str, max_queries: int | None = None) -> pd.DataFrame:
    path = os.path.join(data_dir, "search_test", "train-00000-of-00001.parquet")
    queries = pd.read_parquet(path, columns=["query"])
    if max_queries is not None:
        queries = queries.iloc[: int(max_queries)].reset_index(drop=True)
    return queries


def load_search_test_dataframe(data_dir: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = os.path.join(data_dir, "search_test", "train-00000-of-00001.parquet")
    return pd.read_parquet(path, columns=columns)


def iter_doc_chunks(doc_emb_dir: str, doc_dim: int, doc_chunk_size: int):
    shard_paths = sorted(glob.glob(os.path.join(doc_emb_dir, "passage_gpu_*.npy")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No passage shards found under {doc_emb_dir}. "
            "Expected files like passage_gpu_0.npy"
        )

    global_offset = 0
    for shard_path in shard_paths:
        shard = np.load(shard_path, mmap_mode="r")
        shard_size = shard.shape[0]
        for start in range(0, shard_size, doc_chunk_size):
            end = min(start + doc_chunk_size, shard_size)
            yield shard[start:end, :doc_dim], global_offset + start
        global_offset += shard_size


class AsymmetricBiEncoderEvaluator:
    def __init__(self, accelerator, model, config: dict, project_dir: str):
        self.accelerator = accelerator
        self.model = model
        self.config = config
        self.project_dir = project_dir

        self.eval_cfg = config.get("evaluation", {})
        self.data_dir = config["datasets"]["dataset_name_or_path"]
        self.model_path = config["model"]["model_name_or_path"]
        self.query_instruction = config["datasets"].get("query_instruction", "")
        self.doc_emb_dir = config["datasets"]["doc_emb_dir"]
        self.doc_dim = config["model"].get("doc_dim", 768)

        self.top_k = int(self.eval_cfg.get("top_k", 1000))
        self.query_batch_size = int(self.eval_cfg.get("query_batch_size", 256))
        self.doc_chunk_size = int(self.eval_cfg.get("doc_chunk_size", 16384))
        self.max_queries = self.eval_cfg.get("max_queries")

        self.qrels = load_qrels(self.eval_cfg["qrels_data_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        self._queries = None

    def _load_queries(self) -> pd.DataFrame:
        if self._queries is None:
            path = os.path.join(self.data_dir, "search_test", "train-00000-of-00001.parquet")
            self._queries = pd.read_parquet(path, columns=["query"])
            if self.max_queries is not None:
                self._queries = self._queries.iloc[: int(self.max_queries)].reset_index(drop=True)
        return self._queries

    def _iter_doc_chunks(self):
        shard_paths = sorted(glob.glob(os.path.join(self.doc_emb_dir, "passage_gpu_*.npy")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No passage shards found under {self.doc_emb_dir}. "
                "Expected files like passage_gpu_0.npy"
            )

        global_offset = 0
        for shard_path in shard_paths:
            shard = np.load(shard_path, mmap_mode="r")
            shard_size = shard.shape[0]
            for start in range(0, shard_size, self.doc_chunk_size):
                end = min(start + self.doc_chunk_size, shard_size)
                yield shard[start:end, :self.doc_dim], global_offset + start
            global_offset += shard_size

    def _encode_local_queries(self, local_queries: list[str]) -> torch.Tensor:
        raw_model = self.accelerator.unwrap_model(self.model)
        all_embs = []

        with torch.no_grad():
            for start in range(0, len(local_queries), self.query_batch_size):
                batch_queries = local_queries[start:start + self.query_batch_size]
                if self.query_instruction:
                    batch_queries = [
                        self.query_instruction + query for query in batch_queries
                    ]
                tokenized = self.tokenizer(
                    batch_queries,
                    truncation=True,
                    padding=True,
                    max_length=self.config["datasets"].get("max_length", 256),
                    return_tensors="pt",
                )
                tokenized = {
                    key: value.to(self.accelerator.device)
                    for key, value in tokenized.items()
                }
                query_embs = raw_model.encode_queries(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    token_type_ids=tokenized.get("token_type_ids"),
                )
                all_embs.append(query_embs)

        return torch.cat(all_embs, dim=0)

    def evaluate(self, epoch: int, global_step: int) -> dict[str, float] | None:
        queries_df = self._load_queries()
        all_indices = np.arange(len(queries_df))
        local_indices = np.array_split(all_indices, self.accelerator.num_processes)[
            self.accelerator.process_index
        ]
        local_queries = queries_df.iloc[local_indices]["query"].tolist()

        raw_model = self.accelerator.unwrap_model(self.model)
        raw_model.eval()

        eval_dir = os.path.join(self.project_dir, "eval", f"epoch_{epoch:03d}")
        os.makedirs(eval_dir, exist_ok=True)

        with torch.no_grad():
            query_embs = self._encode_local_queries(local_queries)
            local_top_scores = torch.full(
                (len(local_queries), self.top_k),
                -float("inf"),
                device=self.accelerator.device,
            )
            local_top_pids = torch.full(
                (len(local_queries), self.top_k),
                -1,
                dtype=torch.long,
                device=self.accelerator.device,
            )

            for doc_chunk_np, global_offset in self._iter_doc_chunks():
                doc_chunk = torch.from_numpy(np.array(doc_chunk_np, copy=True)).to(
                    self.accelerator.device,
                    dtype=torch.float32,
                )
                doc_embs = raw_model.project_docs(doc_chunk)

                sim = torch.matmul(query_embs, doc_embs.T)
                chunk_top_k = min(self.top_k, sim.shape[1])
                cand_scores, cand_idx = torch.topk(sim, k=chunk_top_k, dim=1)
                cand_pids = cand_idx + global_offset

                combined_scores = torch.cat([local_top_scores, cand_scores], dim=1)
                combined_pids = torch.cat([local_top_pids, cand_pids], dim=1)
                new_scores, new_idx = torch.topk(combined_scores, k=self.top_k, dim=1)
                new_pids = torch.gather(combined_pids, 1, new_idx)

                local_top_scores = new_scores
                local_top_pids = new_pids

                del doc_chunk, doc_embs, sim, cand_scores, cand_idx, cand_pids

        rank_file = os.path.join(
            eval_dir,
            f"rank_{self.accelerator.process_index:02d}.npz",
        )
        np.savez_compressed(
            rank_file,
            qids=local_indices.astype(np.int32),
            pids=local_top_pids.detach().cpu().numpy().astype(np.int32),
            scores=local_top_scores.detach().cpu().numpy().astype(np.float32),
        )

        self.accelerator.wait_for_everyone()

        metrics = None
        if self.accelerator.is_main_process:
            expected_rank_files = [
                os.path.join(eval_dir, f"rank_{rank:02d}.npz")
                for rank in range(self.accelerator.num_processes)
            ]
            deadline = time.time() + 60
            missing = [path for path in expected_rank_files if not os.path.exists(path)]
            while missing and time.time() < deadline:
                time.sleep(1)
                missing = [path for path in expected_rank_files if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    "Missing rank files after distributed eval synchronization: "
                    + ", ".join(missing)
                )

            sorted_results = {}
            for rank in range(self.accelerator.num_processes):
                part = np.load(os.path.join(eval_dir, f"rank_{rank:02d}.npz"))
                qids = part["qids"]
                pids = part["pids"]
                for qid, ranked_pids in zip(qids.tolist(), pids.tolist()):
                    sorted_results[int(qid)] = [int(pid) for pid in ranked_pids if pid >= 0]

            metrics = compute_metrics(sorted_results, self.qrels)
            metrics_path = os.path.join(eval_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "metrics": metrics,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

        self.accelerator.wait_for_everyone()
        raw_model.train()
        return metrics
