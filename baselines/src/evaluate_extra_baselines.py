import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from asymmetric_biencoder_eval import compute_metrics, load_qrels, load_search_test_queries
from encode_qwen3vl import encode_batch, load_qwen3vl_model
from utils import mean_token_pool


NOTES_GLOB = "/data/rech/huiyuche/multimodal_rag/datasets/qilin/notes/train-*.parquet"
SEARCH_TEST_PATH = "/data/rech/huiyuche/multimodal_rag/datasets/qilin/search_test/train-00000-of-00001.parquet"
BGE_MODEL_PATH = "/data/rech/huiyuche/huggingface/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"
QWEN_MODEL_PATH = "/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-VL-Embedding-2B"
DEFAULT_QWEN_DOC_DIR = "/part/01/Tmp/zhangyan/qilin_qwen3vl"


def build_note_text(row: pd.Series) -> str:
    title = str(row.get("note_title") or "").strip()
    content = str(row.get("note_content") or "").strip()
    if title and content:
        return f"{title}\n{content}"
    return title or content or ""


def encode_bge_texts(texts: list[str], tokenizer, model, device: str, max_length: int, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding BGE texts"):
            batch = texts[start:start + batch_size]
            tokenized = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            hidden = model(**tokenized).last_hidden_state
            emb = mean_token_pool(hidden, tokenized["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
            outputs.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def ensure_bge_doc_cache(cache_dir: Path, device: str, batch_size: int, max_length: int) -> list[tuple[Path, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    note_files = [Path(p) for p in sorted(glob.glob(NOTES_GLOB))]
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(BGE_MODEL_PATH, trust_remote_code=True).to(device)

    for note_file in note_files:
        shard_id = note_file.stem
        emb_path = cache_dir / f"{shard_id}.emb.npy"
        pid_path = cache_dir / f"{shard_id}.pid.npy"
        shard_paths.append((emb_path, pid_path))
        if emb_path.exists() and pid_path.exists():
            continue

        df = pd.read_parquet(note_file, columns=["note_idx", "note_title", "note_content"])
        texts = [build_note_text(row) for _, row in df.iterrows()]
        embs = encode_bge_texts(texts, tokenizer, model, device, max_length, batch_size)
        np.save(emb_path, embs)
        np.save(pid_path, df["note_idx"].to_numpy(dtype=np.int64))

    return shard_paths


def encode_qwen_queries(queries: list[str], device: str, batch_size: int, max_length: int, dim: int | None) -> np.ndarray:
    model, processor = load_qwen3vl_model(QWEN_MODEL_PATH, use_int8=True, device=torch.device(device), logger=_DummyLogger())
    items = [{"text": q, "image": None} for q in queries]
    outputs = []
    for start in tqdm(range(0, len(items), batch_size), desc="Encoding Qwen queries"):
        batch_items = items[start:start + batch_size]
        arr = encode_batch(
            model,
            processor,
            batch_items,
            instruction="Represent this query for retrieving relevant Xiaohongshu notes.",
            max_length=max_length,
            device=torch.device(device),
            mrl_dim=dim,
            max_pixels=256 * 28 * 28,
        )
        outputs.append(arr.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def iter_qwen_doc_shards(doc_emb_dir: str, dim: int | None):
    shard_paths = sorted(Path(doc_emb_dir).glob("passage_gpu_*.npy"))
    offset = 0
    shard_offsets = {}
    for shard_path in shard_paths:
        shard_offsets[shard_path] = offset
        offset += len(np.load(shard_path, mmap_mode="r"))

    for shard_path in shard_paths:
        arr = np.load(shard_path, mmap_mode="r")
        if dim is not None:
            arr = arr[:, :dim]
        offset = shard_offsets[shard_path]
        pids = np.arange(offset, offset + len(arr), dtype=np.int64)
        yield arr, pids


def iter_bge_doc_shards(doc_dir: Path):
    shard_paths = sorted(doc_dir.glob("passage_gpu_*.npy"))
    if shard_paths:
        offset = 0
        for shard_path in shard_paths:
            arr = np.load(shard_path, mmap_mode="r")
            pids = np.arange(offset, offset + len(arr), dtype=np.int64)
            offset += len(arr)
            yield arr, pids
        return

    for emb_path in sorted(doc_dir.glob("*.emb.npy")):
        pid_path = doc_dir / emb_path.name.replace(".emb.npy", ".pid.npy")
        yield np.load(emb_path, mmap_mode="r"), np.load(pid_path, mmap_mode="r")


def search_topk(query_embs: np.ndarray, doc_iter, top_k: int, device: str, query_chunk_size: int = 512):
    query_embs_t = torch.from_numpy(query_embs).to(device=device, dtype=torch.float32)
    num_queries = query_embs_t.shape[0]
    top_scores = torch.full((num_queries, top_k), -float("inf"), device=device)
    top_pids = torch.full((num_queries, top_k), -1, dtype=torch.long, device=device)

    for doc_emb_np, pids_np in tqdm(doc_iter, desc="Scoring doc shards"):
        doc_emb_t = torch.from_numpy(np.asarray(doc_emb_np)).to(device=device, dtype=torch.float32)
        doc_emb_t = F.normalize(doc_emb_t, p=2, dim=-1)
        pids_t = torch.from_numpy(np.asarray(pids_np)).to(device=device, dtype=torch.long)
        for start in range(0, num_queries, query_chunk_size):
            end = min(start + query_chunk_size, num_queries)
            sim = torch.matmul(query_embs_t[start:end], doc_emb_t.T)
            chunk_top_k = min(top_k, sim.shape[1])
            cand_scores, cand_idx = torch.topk(sim, k=chunk_top_k, dim=1)
            cand_pids = pids_t[cand_idx]
            combined_scores = torch.cat([top_scores[start:end], cand_scores], dim=1)
            combined_pids = torch.cat([top_pids[start:end], cand_pids], dim=1)
            new_scores, new_idx = torch.topk(combined_scores, k=top_k, dim=1)
            new_pids = torch.gather(combined_pids, 1, new_idx)
            top_scores[start:end] = new_scores
            top_pids[start:end] = new_pids

    return {
        qid: [int(pid) for pid in ranked if pid >= 0]
        for qid, ranked in enumerate(top_pids.cpu().tolist())
    }


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate extra retrieval baselines.")
    parser.add_argument("--baseline", choices=["bge_bge", "qwen_qwen"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--doc_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--qrels_path", default="/data/rech/huiyuche/multimodal_rag/datasets/search.test.qrels.csv")
    parser.add_argument("--doc_emb_dir", default=DEFAULT_QWEN_DOC_DIR)
    parser.add_argument("--cache_dir", default="/data/rech/zhangyan/multimodal_rag/baselines/cache/extra_baselines")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries_df = load_search_test_queries("/data/rech/huiyuche/multimodal_rag/datasets/qilin")
    queries = queries_df["query"].tolist()
    qrels = load_qrels(args.qrels_path)

    if args.baseline == "bge_bge":
        tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(BGE_MODEL_PATH, trust_remote_code=True).to(args.device)
        query_embs = encode_bge_texts(queries, tokenizer, model, args.device, args.max_length, args.query_batch_size)
        doc_dir = Path(args.doc_emb_dir) if args.doc_emb_dir else (Path(args.cache_dir) / "bge_bge_docs")
        if not sorted(doc_dir.glob("passage_gpu_*.npy")) and not sorted(doc_dir.glob("*.emb.npy")):
            ensure_bge_doc_cache(doc_dir, args.device, args.doc_batch_size, args.max_length)
        rankings = search_topk(query_embs, iter_bge_doc_shards(doc_dir), args.top_k, args.device)
    else:
        query_embs = encode_qwen_queries(queries, args.device, args.query_batch_size, args.max_length, dim=2048)
        rankings = search_topk(query_embs, iter_qwen_doc_shards(args.doc_emb_dir, dim=2048), args.top_k, args.device)

    metrics = compute_metrics(rankings, qrels)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
