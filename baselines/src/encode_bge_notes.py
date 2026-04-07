import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from utils import mean_token_pool


def setup_logger(log_path: str, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"encode_bge_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(f"[rank{rank}][%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if rank == 0 and log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def build_note_text(note: dict, use_title: bool, use_content: bool) -> str:
    parts = []
    if use_title and note.get("note_title"):
        parts.append(str(note["note_title"]).strip())
    if use_content and note.get("note_content"):
        parts.append(str(note["note_content"]).strip())
    return "\n".join([p for p in parts if p]) or ""


def encode_batch(model, tokenizer, texts: list[str], device: torch.device, max_length: int) -> np.ndarray:
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        hidden = model(**tokenized).last_hidden_state
        emb = mean_token_pool(hidden, tokenized["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().float().numpy()


def find_best_batch_size(model, tokenizer, sample_texts: list[str], device: torch.device, max_length: int,
                         logger: logging.Logger) -> int:
    candidates = [64, 128, 256, 512, 1024, 1536, 2048]
    best_bs = 64
    best_tput = 0.0
    logger.info("=== Batch size benchmark ===")
    for bs in candidates:
        if bs > len(sample_texts):
            break
        batch = sample_texts[:bs]
        try:
            encode_batch(model, tokenizer, batch, device, max_length)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            logger.info(f"  bs={bs}: OOM on warm-up, stopping search")
            break

        times = []
        for _ in range(3):
            try:
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                encode_batch(model, tokenizer, batch, device, max_length)
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - t0)
            except torch.cuda.OutOfMemoryError:
                times = []
                break

        if not times:
            logger.info(f"  bs={bs}: OOM during timing, stopping search")
            break

        avg_t = sum(times) / len(times)
        throughput = bs / avg_t
        logger.info(f"  bs={bs}: {avg_t * 1000:.1f} ms/batch -> {throughput:.1f} notes/sec")
        if throughput > best_tput:
            best_tput = throughput
            best_bs = bs

    logger.info(f"==> Best batch size: {best_bs} ({best_tput:.1f} notes/sec)")
    return best_bs


def main():
    parser = argparse.ArgumentParser(description="Encode Qilin notes with BGE.")
    parser.add_argument("--model_path", default="/data/rech/huiyuche/huggingface/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65")
    parser.add_argument("--data_path", default="/data/rech/huiyuche/multimodal_rag/datasets/qilin")
    parser.add_argument("--output_dir", default="/data/rech/zhangyan/bge_qilin_embedding")
    parser.add_argument("--log_path", default="/data/rech/zhangyan/multimodal_rag/baselines/logs/encode_bge_qilin_20260406.log")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--find_batch_size", action="store_true", default=False)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_every_notes", type=int, default=20000)
    parser.add_argument("--sample_num", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--use_title", action="store_true", default=True)
    parser.add_argument("--use_content", action="store_true", default=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    logger = setup_logger(args.log_path, local_rank)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading notes corpus from parquet shards...")
    corpus = load_dataset(
        "parquet",
        data_files=sorted(glob.glob(os.path.join(args.data_path, "notes", "*.parquet"))),
        split="train",
    )
    logger.info(f"Corpus total size: {len(corpus)}")
    if args.sample_num > 0:
        corpus = corpus.select(range(min(args.sample_num, len(corpus))))
        logger.info(f"Sample mode enabled: {len(corpus)} notes")

    shard = corpus.shard(num_shards=world_size, index=local_rank, contiguous=True)
    shard_size = len(shard)
    logger.info(f"Rank {local_rank}: shard size = {shard_size}")

    device = torch.device(f"cuda:{local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    model.eval()

    batch_size = args.batch_size
    if args.find_batch_size and local_rank == 0:
        sample_texts = [
            build_note_text(shard[i], args.use_title, args.use_content)
            for i in range(min(2048, shard_size))
        ]
        batch_size = find_best_batch_size(model, tokenizer, sample_texts, device, args.max_length, logger)
        with open(os.path.join(args.output_dir, ".batch_size"), "w") as f:
            f.write(str(batch_size))
    elif args.find_batch_size and world_size > 1:
        bs_file = os.path.join(args.output_dir, ".batch_size")
        for _ in range(60):
            if os.path.exists(bs_file):
                with open(bs_file) as f:
                    batch_size = int(f.read())
                break
            time.sleep(1)
    logger.info(f"Rank {local_rank}: using batch_size={batch_size}")

    out_path = os.path.join(args.output_dir, f"passage_gpu_{local_rank}.npy")
    ckpt_path = os.path.join(args.output_dir, f"passage_gpu_{local_rank}.ckpt")
    resume_batch = 0
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        resume_batch = int(ckpt.get("batches_done", 0))
        logger.info(f"Rank {local_rank}: resuming from batch {resume_batch}")
    elif args.resume and os.path.exists(out_path) and not os.path.exists(ckpt_path):
        logger.info(f"Rank {local_rank}: output already complete, skipping.")
        return

    mmap_mode = "r+" if (resume_batch > 0 and os.path.exists(out_path)) else "w+"
    fp = np.lib.format.open_memmap(out_path, mode=mmap_mode, dtype="float32", shape=(shard_size, 768))
    logger.info(f"Rank {local_rank}: opened mmap {out_path} with shape={fp.shape}")

    total_batches = (shard_size + batch_size - 1) // batch_size
    start_idx = resume_batch * batch_size
    notes_since_flush = 0
    pbar = tqdm(range(start_idx, shard_size, batch_size), disable=(local_rank != 0), desc=f"Rank{local_rank} encode")
    start_time = time.time()

    for batch_idx, idx in enumerate(pbar, start=resume_batch):
        end = min(idx + batch_size, shard_size)
        texts = [build_note_text(shard[i], args.use_title, args.use_content) for i in range(idx, end)]
        embs = encode_batch(model, tokenizer, texts, device, args.max_length)
        fp[idx:end] = embs
        notes_since_flush += (end - idx)

        if notes_since_flush >= args.save_every_notes or end == shard_size:
            fp.flush()
            with open(ckpt_path, "w") as f:
                json.dump(
                    {
                        "batches_done": batch_idx + 1,
                        "encoded_notes": end,
                        "shard_size": shard_size,
                        "batch_size": batch_size,
                        "time": time.time(),
                    },
                    f,
                )
            notes_since_flush = 0

        if local_rank == 0 and (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            done = end
            rate = done / max(elapsed, 1e-6)
            eta = (shard_size - done) / max(rate, 1e-6)
            logger.info(
                f"Progress: {done}/{shard_size} notes on rank0, "
                f"{rate:.1f} notes/sec, ETA(rank0)={eta/60:.1f} min"
            )

    fp.flush()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    logger.info(f"Rank {local_rank}: finished encoding -> {out_path}")

    if dist.is_initialized():
        dist.barrier()
        if local_rank == 0:
            logger.info("All ranks finished.")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
