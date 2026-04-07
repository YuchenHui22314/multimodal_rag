import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from asymmetric_biencoder_eval import (
    compute_metrics,
    iter_doc_chunks,
    load_qrels,
    load_search_test_dataframe,
    load_search_test_queries,
)
from asymmetric_biencoder_model import AsymmetricBiEncoderModel


def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = yaml.safe_load(v)
    return cfg


def evaluate_model(
    model,
    cfg: dict,
    output_dir: str,
    max_queries: int | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    queries_df = load_search_test_queries(cfg["datasets"]["dataset_name_or_path"], max_queries=max_queries)
    qrels = load_qrels(cfg["evaluation"]["qrels_data_path"])
    if max_queries is not None:
        qrels = {qid: pids for qid, pids in qrels.items() if qid < int(max_queries)}

    top_k = int(cfg.get("evaluation", {}).get("top_k", 1000))
    query_batch_size = int(cfg.get("evaluation", {}).get("query_batch_size", 256))
    doc_chunk_size = int(cfg.get("evaluation", {}).get("doc_chunk_size", 16384))
    query_instruction = cfg["datasets"].get("query_instruction", "")

    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        cfg["model"]["model_name_or_path"],
        trust_remote_code=True,
    )

    model.eval()
    model.to(device)

    all_query_embs = []
    with torch.no_grad():
        queries = queries_df["query"].tolist()
        for start in tqdm(range(0, len(queries), query_batch_size), desc="Encoding queries"):
            batch_queries = queries[start:start + query_batch_size]
            if query_instruction:
                batch_queries = [query_instruction + q for q in batch_queries]
            tokenized = tokenizer(
                batch_queries,
                truncation=True,
                padding=True,
                max_length=cfg["datasets"].get("max_length", 256),
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            query_embs = model.encode_queries(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                token_type_ids=tokenized.get("token_type_ids"),
            )
            all_query_embs.append(query_embs.cpu())

    query_embs = torch.cat(all_query_embs, dim=0).to(device)
    top_scores = torch.full((len(queries_df), top_k), -float("inf"), device=device)
    top_pids = torch.full((len(queries_df), top_k), -1, dtype=torch.long, device=device)

    for doc_chunk_np, global_offset in tqdm(
        iter_doc_chunks(cfg["datasets"]["doc_emb_dir"], cfg["model"]["doc_dim"], doc_chunk_size),
        desc="Scoring doc shards",
    ):
        doc_chunk = torch.from_numpy(np.asarray(doc_chunk_np)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            doc_embs = model.project_docs(doc_chunk)
            sim = torch.matmul(query_embs, doc_embs.T)
            chunk_top_k = min(top_k, sim.shape[1])
            cand_scores, cand_idx = torch.topk(sim, k=chunk_top_k, dim=1)
            cand_pids = cand_idx + global_offset
            combined_scores = torch.cat([top_scores, cand_scores], dim=1)
            combined_pids = torch.cat([top_pids, cand_pids], dim=1)
            new_scores, new_idx = torch.topk(combined_scores, k=top_k, dim=1)
            new_pids = torch.gather(combined_pids, 1, new_idx)
            top_scores = new_scores
            top_pids = new_pids

    sorted_results = {
        qid: [int(pid) for pid in ranked_pids if pid >= 0]
        for qid, ranked_pids in enumerate(top_pids.cpu().tolist())
    }
    metrics = compute_metrics(sorted_results, qrels)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "rankings.json"), "w") as f:
        json.dump(sorted_results, f)
    return metrics


def evaluate_dpr_baseline(cfg: dict, output_dir: str, max_queries: int | None = None) -> dict[str, float]:
    df = load_search_test_dataframe(
        cfg["datasets"]["dataset_name_or_path"],
        columns=["dpr_results"],
    )
    if max_queries is not None:
        df = df.iloc[: int(max_queries)].reset_index(drop=True)

    qrels = load_qrels(cfg["evaluation"]["qrels_data_path"])
    if max_queries is not None:
        qrels = {qid: pids for qid, pids in qrels.items() if qid < int(max_queries)}

    sorted_results = {}
    for qid, results in enumerate(df["dpr_results"].tolist()):
        ranked = []
        for item in results:
            if isinstance(item, np.ndarray):
                pid = int(item[0])
            else:
                pid = int(item[0])
            ranked.append(pid)
        sorted_results[qid] = ranked

    metrics = compute_metrics(sorted_results, qrels)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate asymmetric bi-encoder checkpoints and baselines")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--baseline", choices=["none", "dpr", "frozen_bge_qwen"], default="none")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    overrides = dict(kv.split("=", 1) for kv in (args.override or []))
    cfg = load_config(args.config, overrides)

    if args.baseline == "dpr":
        metrics = evaluate_dpr_baseline(cfg, args.output_dir, max_queries=args.max_queries)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    if args.baseline == "frozen_bge_qwen":
        cfg["model"]["variant"] = "fullft"
        cfg["model"]["proj_type"] = "mlp"
        cfg["model"]["doc_dim"] = 768

    model = AsymmetricBiEncoderModel(cfg)
    if args.checkpoint_dir:
        model = AsymmetricBiEncoderModel.load(cfg, args.checkpoint_dir)
    metrics = evaluate_model(
        model,
        cfg,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
