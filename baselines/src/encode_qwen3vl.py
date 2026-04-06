"""
encode_qwen3vl.py

Encode Qilin notes and/or queries using Qwen3-VL-Embedding-2B.

Features:
  - Multi-GPU support via torchrun (each rank encodes a contiguous shard of corpus)
  - Matryoshka Representation Learning: slice first --dim dimensions then L2-renorm
    (default 768 dims; pass --dim 0 for full 2048)
  - int8 inference via bitsandbytes load_in_8bit (default ON, ~halves GPU memory)
  - Output format compatible with DenseRetrievalEvaluator:
      notes   → {output_dir}/passage_gpu_{rank}.npy   shape [shard_size, dim]
      queries → {output_dir}/question.npy             shape [num_queries, dim]
  - Batch size auto-benchmark: --find_batch_size scans candidate sizes and
    picks the fastest one (run on first rank only, single small batch)
  - RAM is not a concern: 4 GPU shards × ~500k × 768 × 4 bytes ≈ 1.5 GB each

Usage — encode notes (4 GPUs):
    torchrun --nproc_per_node=4 src/encode_qwen3vl.py \\
        --mode notes \\
        --model_path /data/rech/huiyuche/huggingface/models--Qwen--Qwen3-VL-Embedding-2B \\
        --data_path  /data/rech/huiyuche/multimodal_rag/datasets/qilin \\
        --image_root /data/rech/huiyuche/qilin_image \\
        --output_dir /data/rech/huiyuche/multimodal_rag/embeddings/qilin_qwen3vl \\
        --dim 768 --int8 --find_batch_size

Usage — encode queries:
    torchrun --nproc_per_node=4 src/encode_qwen3vl.py \\
        --mode queries --split search_test ...same flags...

Usage — smoke test (encode first 200 notes, single GPU):
    torchrun --nproc_per_node=1 src/encode_qwen3vl.py \\
        --mode notes --sample_num 200 --batch_size 4 ...

Author: Qilin multimodal search project, 2026.
"""

import os
import sys
import glob
import time
import argparse
import logging
import unicodedata
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: str, rank: int) -> logging.Logger:
    """rank-0 writes to file; all ranks log to stdout."""
    logger = logging.getLogger(f"encode_qwen3vl_rank{rank}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        f"[rank{rank}][%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if rank == 0 and log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen3vl_model(model_path: str, use_int8: bool, device: torch.device,
                       logger: logging.Logger):
    """
    Load Qwen3-VL-Embedding model.
    use_int8=True  → bitsandbytes 8-bit (halves GPU memory, small accuracy loss)
    use_int8=False → fp16
    Returns (model, processor).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.modeling_outputs import ModelOutput

    @dataclass
    class EmbeddingOutput(ModelOutput):
        last_hidden_state: Optional[torch.FloatTensor] = None
        attention_mask: Optional[torch.Tensor] = None

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False
        config: Qwen3VLConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, pixel_values=None,
                    pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                    cache_position=None, logits_to_keep=0, **kwargs):
            outputs = self.model(
                input_ids=input_ids, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, position_ids=position_ids,
                attention_mask=attention_mask, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs
            )
            return EmbeddingOutput(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask
            )

    # For newer transformers, quantization must be passed via BitsAndBytesConfig
    from transformers import BitsAndBytesConfig
    load_kwargs = {"trust_remote_code": True, "device_map": {"": device}}
    if use_int8:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info(f"Loading model in int8 on {device}")
    else:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info(f"Loading model in fp16 with flash_attention_2 on {device}")

    model = Qwen3VLForEmbedding.from_pretrained(model_path, **load_kwargs)
    processor = Qwen3VLProcessor.from_pretrained(model_path, padding_side='right')
    logger.info("Model and processor loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """EOS-token pooling: embedding at last non-padding position."""
    flipped = attention_mask.flip(dims=[1])
    last_positions = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_positions - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


def format_conversation(text: Optional[str], image: Optional[Image.Image],
                        instruction: str,
                        max_pixels: int = 256 * 28 * 28) -> List[Dict]:
    """Build Qwen3-VL chat-format conversation for one item.

    max_pixels controls image resolution fed to the vision encoder.
    Default 256*28*28 ≈ 200k pixels → ~196 image tokens per image.
    (1280*28*28 default → ~980 tokens, too slow; 512*28*28 → ~392 tokens, still CPU-bound)
    """
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
        instruction = instruction + '.'

    content = []
    if image is not None:
        content.append({
            'type': 'image', 'image': image,
            'min_pixels': 4 * 32 * 32,
            'max_pixels': max_pixels,
        })
    if text:
        content.append({'type': 'text', 'text': text})
    if not content:
        content.append({'type': 'text', 'text': 'NULL'})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user",   "content": content}
    ]


def prepare_inputs_cpu(processor, items: List[Dict], instruction: str,
                       max_length: int,
                       max_pixels: int = 256 * 28 * 28) -> dict:
    """
    Full CPU preprocessing for one batch: build conversations, run process_vision_info,
    tokenize. Returns a dict of CPU tensors (not yet on GPU).

    Designed to run in a background thread while the GPU executes the previous batch.
    """
    from qwen_vl_utils.vision_process import process_vision_info

    conversations = [
        format_conversation(item.get('text'), item.get('image'), instruction,
                            max_pixels=max_pixels)
        for item in items
    ]
    text_inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    try:
        images, _, _ = process_vision_info(
            conversations, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True
        )
    except Exception:
        images = None

    inputs = processor(
        text=text_inputs, images=images,
        truncation=True, max_length=max_length,
        padding=True, do_resize=False, return_tensors='pt'
    )
    return inputs  # CPU tensors


@torch.no_grad()
def gpu_forward(model, inputs_cpu: dict, device: torch.device,
                mrl_dim: Optional[int]) -> np.ndarray:
    """
    H2D transfer + model forward + EOS pooling + MRL renorm.
    Returns float32 numpy array [batch, dim].
    """
    inputs = {k: v.to(device) for k, v in inputs_cpu.items()}
    outputs = model(**inputs)
    embeddings = _pooling_last(outputs.last_hidden_state, outputs.attention_mask)
    if mrl_dim is not None:
        embeddings = embeddings[:, :mrl_dim]
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().float().numpy()


@torch.no_grad()
def encode_batch(model, processor, items: List[Dict], instruction: str,
                 max_length: int, device: torch.device,
                 mrl_dim: Optional[int],
                 max_pixels: int = 256 * 28 * 28) -> np.ndarray:
    """
    Encode a list of {'text': str, 'image': PIL or None} items.
    Returns float32 numpy array [len(items), dim].
    Used directly in encode_queries; encode_notes uses the pipelined version.
    """
    inputs_cpu = prepare_inputs_cpu(processor, items, instruction, max_length, max_pixels)
    return gpu_forward(model, inputs_cpu, device, mrl_dim)


# ---------------------------------------------------------------------------
# Image loading (path remapping: dataset stores relative paths under image/)
# ---------------------------------------------------------------------------

def load_note_image(image_path_list: List[str], image_root: str,
                    logger: logging.Logger, note_idx: int) -> Optional[Image.Image]:
    """
    image_path_list entries look like: "image/part_0/9998/42995.jpg"
    Full path = image_root / image_path_list[0]
    Returns PIL Image or None if missing/corrupt.
    """
    if not image_path_list:
        return None
    full_path = os.path.join(image_root, image_path_list[0])
    try:
        return Image.open(full_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Cannot load image for note {note_idx} at {full_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# DataLoader: worker globals + Dataset + collate_fn
# ---------------------------------------------------------------------------
# These are set in the main process before DataLoader creation and inherited
# by worker processes via fork (Linux default). Workers do all CPU work
# (image load + process_vision_info + tokenize) in separate processes,
# bypassing the GIL entirely.
_DL_PROCESSOR = None
_DL_INSTRUCTION: str = ""
_DL_MAX_LENGTH: int = 2048
_DL_MAX_PIXELS: int = 256 * 28 * 28


class NoteDataset(torch.utils.data.Dataset):
    """Returns one raw {'text', 'image'} item per note — IO only, no heavy CPU work."""

    def __init__(self, corpus_shard, args):
        self.corpus = corpus_shard
        self.args = args

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Dict:
        note = self.corpus[idx]
        text = ""
        if self.args.use_title and note['note_title']:
            text += note['note_title']
        if self.args.use_content and note['note_content']:
            text += note['note_content']
        image = None
        if self.args.use_image and note['image_path']:
            full_path = os.path.join(self.args.image_root, note['image_path'][0])
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception:
                image = None
        return {'text': text, 'image': image}


def _note_collate_fn(items: List[Dict]) -> dict:
    """Runs inside worker processes — calls prepare_inputs_cpu on a batch."""
    return prepare_inputs_cpu(
        _DL_PROCESSOR, items, _DL_INSTRUCTION, _DL_MAX_LENGTH, _DL_MAX_PIXELS
    )


# ---------------------------------------------------------------------------
# Batch size benchmark
# ---------------------------------------------------------------------------

def find_best_batch_size(model, processor, corpus_sample, args,
                         device: torch.device, logger: logging.Logger) -> int:
    """
    Try candidate batch sizes on a tiny warm-up set and return the fastest one.
    Runs only on rank-0 before main encoding starts.
    """
    candidates = [2, 4, 8, 16, 32, 64]
    # Build a fixed set of 64 sample items (text + image)
    sample_size = min(64, len(corpus_sample))
    sample_items = []
    for i in range(sample_size):
        text = ""
        if args.use_title and corpus_sample['note_title'][i]:
            text += corpus_sample['note_title'][i]
        if args.use_content and corpus_sample['note_content'][i]:
            text += corpus_sample['note_content'][i]
        image = None
        if args.use_image:
            image = load_note_image(corpus_sample['image_path'][i],
                                    args.image_root, logger, i)
        sample_items.append({'text': text, 'image': image})

    logger.info("=== Batch size benchmark ===")
    best_bs = 2
    best_throughput = 0.0

    for bs in candidates:
        if bs > sample_size:
            break
        # warm-up pass
        try:
            encode_batch(model, processor, sample_items[:bs],
                         args.note_instruction, args.max_length, device, args.dim,
                         max_pixels=args.max_pixels)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            logger.info(f"  bs={bs}: OOM on warm-up, stopping search")
            break

        # timed pass (3 runs)
        times = []
        for _ in range(3):
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                encode_batch(model, processor, sample_items[:bs],
                             args.note_instruction, args.max_length, device, args.dim,
                             max_pixels=args.max_pixels)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            except torch.cuda.OutOfMemoryError:
                times = []
                break

        if not times:
            logger.info(f"  bs={bs}: OOM during timing, stopping search")
            break

        avg_t = sum(times) / len(times)
        throughput = bs / avg_t  # notes/sec
        logger.info(f"  bs={bs}: {avg_t*1000:.1f} ms/batch  →  {throughput:.1f} notes/sec")

        if throughput > best_throughput:
            best_throughput = throughput
            best_bs = bs

    logger.info(f"==> Best batch size: {best_bs}  ({best_throughput:.1f} notes/sec)")
    return best_bs


# ---------------------------------------------------------------------------
# Main encoding routines
# ---------------------------------------------------------------------------

def encode_notes(args, local_rank: int, world_size: int, logger: logging.Logger):
    """
    Encode all notes (or first sample_num notes if set) sharded across GPUs.
    Output: {output_dir}/passage_gpu_{local_rank}.npy  shape [shard_size, dim]

    Design:
    - Embeddings are written directly to a memory-mapped .npy file each batch
      (zero in-memory accumulation — no OOM risk on long runs).
    - A checkpoint file tracks the last flushed batch so a crashed run can resume
      from that point rather than restarting from scratch.
    - Each rank is fully independent: no dist.barrier() needed here.
    """
    import json
    import psutil

    from datasets import load_dataset

    os.makedirs(args.output_dir, exist_ok=True)
    out_path  = os.path.join(args.output_dir, f"passage_gpu_{local_rank}.npy")
    ckpt_path = os.path.join(args.output_dir, f"passage_gpu_{local_rank}.ckpt")
    dim = args.dim if args.dim is not None else 2048

    # ---- resume logic ----
    # Case 1: fully done (no checkpoint left) → skip
    if args.resume and os.path.exists(out_path) and not os.path.exists(ckpt_path):
        logger.info(f"Rank {local_rank}: output already complete, skipping.")
        return
    # Case 2: partial run (checkpoint exists) → resume from last saved batch
    resume_batch = 0
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        resume_batch = ckpt["batches_done"]
        logger.info(f"Rank {local_rank}: resuming from batch {resume_batch} "
                    f"({resume_batch * args.batch_size} notes already encoded)")

    logger.info("Loading notes corpus from HuggingFace parquet files...")
    corpus = load_dataset(
        "parquet",
        data_files=sorted(glob.glob(
            os.path.join(args.data_path, "notes", "*.parquet")
        )),
        split="train"
    )
    logger.info(f"Corpus total size: {len(corpus)}")

    if args.sample_num > 0:
        corpus = corpus.select(range(min(args.sample_num, len(corpus))))
        logger.info(f"Smoke test: limited to {len(corpus)} notes")

    corpus_shard = corpus.shard(num_shards=world_size, index=local_rank, contiguous=True)
    shard_size = len(corpus_shard)
    logger.info(f"Rank {local_rank}: shard size = {shard_size}")

    device = torch.device(f"cuda:{local_rank}")
    model, processor = load_qwen3vl_model(args.model_path, args.int8, device, logger)

    # ---- batch size ----
    batch_size = args.batch_size
    if args.find_batch_size and local_rank == 0:
        sample_slice = corpus_shard[:min(64, shard_size)]
        batch_size = find_best_batch_size(
            model, processor, sample_slice, args, device, logger
        )
    if args.find_batch_size and world_size > 1:
        # Use file-based sync instead of NCCL broadcast (avoids P2P NCCL issues on PCIe-only setups)
        bs_file = os.path.join(args.output_dir, ".batch_size")
        if local_rank == 0:
            with open(bs_file, 'w') as f:
                f.write(str(batch_size))
        else:
            import time as _time
            for _ in range(30):
                if os.path.exists(bs_file):
                    with open(bs_file) as f:
                        batch_size = int(f.read())
                    break
                _time.sleep(1)
    logger.info(f"Rank {local_rank}: using batch_size={batch_size}")

    # ---- open memory-mapped output file ----
    # np.lib.format.open_memmap writes a valid .npy header so the file is
    # directly loadable by np.load() at any point during/after encoding.
    mmap_mode = 'r+' if (resume_batch > 0 and os.path.exists(out_path)) else 'w+'
    fp = np.lib.format.open_memmap(
        out_path, mode=mmap_mode, dtype='float32', shape=(shard_size, dim)
    )
    logger.info(f"Rank {local_rank}: output mmap opened ({out_path}), shape={fp.shape}")

    # ---- DataLoader pipeline (multi-process, bypasses GIL) ----
    proc = psutil.Process()
    n_batches = (shard_size + batch_size - 1) // batch_size
    save_every_notes = args.save_every_notes
    notes_since_flush = 0

    # Set module-level globals inherited by worker processes via fork
    global _DL_PROCESSOR, _DL_INSTRUCTION, _DL_MAX_LENGTH, _DL_MAX_PIXELS
    _DL_PROCESSOR = processor
    _DL_INSTRUCTION = args.note_instruction
    _DL_MAX_LENGTH = args.max_length
    _DL_MAX_PIXELS = args.max_pixels

    full_dataset = NoteDataset(corpus_shard, args)
    dataset = (torch.utils.data.Subset(full_dataset, range(resume_batch * batch_size, shard_size))
               if resume_batch > 0 else full_dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=_note_collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        pin_memory=False,
        drop_last=False,
    )

    pbar = tqdm(
        loader,
        total=n_batches - resume_batch,
        desc=f"[rank{local_rank}] Encoding notes",
        unit="batch",
        dynamic_ncols=True,
        disable=(local_rank != 0),
    )
    t_start = time.perf_counter()
    encoded_total = 0
    note_idx = resume_batch * batch_size  # absolute write position in shard mmap

    for inputs_cpu in pbar:
        actual_bs = inputs_cpu['input_ids'].shape[0]
        start = note_idx
        end = note_idx + actual_bs

        embs = gpu_forward(model, inputs_cpu, device, args.dim)
        fp[start:end] = embs
        note_idx += actual_bs
        notes_since_flush += actual_bs
        encoded_total += actual_bs

        if notes_since_flush >= save_every_notes:
            fp.flush()
            with open(ckpt_path, 'w') as cf:
                json.dump({"batches_done": note_idx // batch_size,
                           "notes_done":   note_idx}, cf)
            notes_since_flush = 0
            logger.info(f"Rank {local_rank}: checkpoint saved ({note_idx}/{shard_size} notes)")

        elapsed = time.perf_counter() - t_start
        notes_per_sec = encoded_total / max(elapsed, 1e-6)
        pbar.set_postfix({
            'speed': f'{notes_per_sec:.1f} n/s',
            'RAM':   f'{proc.memory_info().rss/1e9:.1f}G',
            'GPU':   f'{torch.cuda.memory_allocated(device)/1e9:.1f}G',
            'ETA':   f'{(shard_size - note_idx) / max(notes_per_sec, 1):.0f}s',
        })

        if note_idx % (500 * batch_size) < actual_bs:
            logger.info(
                f"Rank {local_rank}: {note_idx}/{shard_size} notes | "
                f"{notes_per_sec:.1f} notes/sec | "
                f"RAM {proc.memory_info().rss/1e9:.1f} GB | "
                f"GPU {torch.cuda.memory_allocated(device)/1e9:.1f} GB"
            )

    # ---- finalise ----
    fp.flush()
    elapsed_total = time.perf_counter() - t_start
    logger.info(
        f"Rank {local_rank}: encoding done. shape={fp.shape}, "
        f"total_time={elapsed_total:.1f}s, "
        f"avg={shard_size/elapsed_total:.1f} notes/sec"
    )
    # Remove checkpoint now that the file is complete
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    logger.info(f"Rank {local_rank}: saved → {out_path}")


def encode_queries(args, local_rank: int, world_size: int, logger: logging.Logger):
    """
    Encode queries from args.split (e.g. search_test), sharded by rank.
    Rank-0 merges shards into question.npy.
    Output: {output_dir}/question.npy  shape [num_queries, dim]
    """
    import glob as _glob
    from datasets import load_dataset

    logger.info(f"Loading split '{args.split}' for query encoding...")
    data_files = sorted(_glob.glob(
        os.path.join(args.data_path, args.split, "*.parquet")
    ))
    dataset = load_dataset("parquet", data_files=data_files, split="train")

    if args.sample_num > 0:
        dataset = dataset.select(range(min(args.sample_num, len(dataset))))

    dataset_shard = dataset.shard(num_shards=world_size, index=local_rank, contiguous=True)
    shard_size = len(dataset_shard)
    logger.info(f"Rank {local_rank}: {shard_size} queries to encode")

    device = torch.device(f"cuda:{local_rank}")
    model, processor = load_qwen3vl_model(args.model_path, args.int8, device, logger)

    batch_size = args.batch_size
    all_embeddings = []

    pbar = tqdm(
        range(0, shard_size, batch_size),
        total=(shard_size + batch_size - 1) // batch_size,
        desc=f"[rank{local_rank}] Encoding queries",
        unit="batch",
        dynamic_ncols=True,
        disable=(local_rank != 0)
    )
    for start in pbar:
        end = min(start + batch_size, shard_size)
        batch = dataset_shard[start:end]
        items = [{'text': q} for q in batch['query']]
        embs = encode_batch(
            model, processor, items,
            instruction=args.query_instruction,
            max_length=args.max_length,
            device=device,
            mrl_dim=args.dim,
            max_pixels=args.max_pixels,
        )
        all_embeddings.append(embs)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Rank {local_rank}: query shard shape={all_embeddings.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_path = os.path.join(args.output_dir, f"question_gpu_{local_rank}.npy")
    np.save(tmp_path, all_embeddings)

    # merge all shards into question.npy on rank-0
    if world_size > 1:
        dist.barrier()
    if local_rank == 0:
        shards = []
        for i in range(world_size):
            shard_path = os.path.join(args.output_dir, f"question_gpu_{i}.npy")
            shards.append(np.load(shard_path))
            os.remove(shard_path)
        merged = np.concatenate(shards, axis=0)
        out_path = os.path.join(args.output_dir, "question.npy")
        np.save(out_path, merged)
        logger.info(f"Rank 0: question.npy saved, shape={merged.shape}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode Qilin notes or queries with Qwen3-VL-Embedding-2B"
    )

    # mode
    parser.add_argument("--mode", choices=["notes", "queries"], required=True)
    parser.add_argument("--split", type=str, default="search_test",
                        help="Dataset split for query encoding")

    # paths
    parser.add_argument("--model_path", type=str,
                        default="/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-VL-Embedding-2B")
    parser.add_argument("--data_path", type=str,
                        default="/data/rech/huiyuche/multimodal_rag/datasets/qilin")
    parser.add_argument("--image_root", type=str,
                        default="/data/rech/huiyuche/qilin_image")
    parser.add_argument("--output_dir", type=str,
                        default="/data/rech/huiyuche/multimodal_rag/embeddings/qilin_qwen3vl")
    parser.add_argument("--log_path", type=str,
                        default="/data/rech/huiyuche/TREC_iKAT_2024/logs/encode_qwen3vl.log")

    # model options
    parser.add_argument("--int8", action="store_true", default=True,
                        help="int8 inference via bitsandbytes (default ON)")
    parser.add_argument("--no_int8", dest="int8", action="store_false",
                        help="Disable int8, use fp16")
    parser.add_argument("--dim", type=int, default=0,
                        help="Matryoshka output dim (64-2048; 0=full 2048). Default 0 (full 2048).")

    # data options
    parser.add_argument("--use_title",   action="store_true", default=True)
    parser.add_argument("--use_content", action="store_true", default=True)
    parser.add_argument("--use_image",   action="store_true", default=True,
                        help="Include note image in encoding (default ON)")
    parser.add_argument("--no_image", dest="use_image", action="store_false")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max token length for truncation. "
                             "With max_pixels=512*28*28, worst case is ~600 tokens; 2048 is safe.")
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28,
                        help="Max image pixels fed to vision encoder. "
                             "Default 256*28*28=200704 → ~196 image tokens. "
                             "Original Qwen3 default is 1280*28*28 but that was too slow. "
                             "Use 512*28*28 for higher quality at 2x slower.")
    parser.add_argument("--sample_num", type=int, default=0,
                        help="If >0, encode only first N notes/queries (smoke test)")

    # batch size
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size. Overridden if --find_batch_size is set.")
    parser.add_argument("--find_batch_size", action="store_true", default=False,
                        help="Benchmark candidate batch sizes and pick the fastest")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes for CPU preprocessing (default 4)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from checkpoint if one exists, or skip complete ranks")
    parser.add_argument("--save_every_notes", type=int, default=50000,
                        help="Flush mmap + write checkpoint every N notes (default 50000 ≈ 45 min)")

    # instructions
    parser.add_argument("--query_instruction", type=str,
                        default="Retrieve relevant notes matching the search query.")
    parser.add_argument("--note_instruction", type=str,
                        default="Represent the multimodal note for retrieval.")

    args = parser.parse_args()
    if args.dim == 0:
        args.dim = None   # None = use full 2048 dims
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    logger = setup_logger(args.log_path, rank)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Qwen3-VL-Embedding Encoding Pipeline")
        logger.info(f"  mode         : {args.mode}")
        logger.info(f"  model_path   : {args.model_path}")
        logger.info(f"  data_path    : {args.data_path}")
        logger.info(f"  output_dir   : {args.output_dir}")
        logger.info(f"  int8         : {args.int8}")
        logger.info(f"  mrl_dim      : {args.dim if args.dim else 'full 2048'}")
        logger.info(f"  use_image    : {args.use_image}")
        logger.info(f"  max_pixels   : {args.max_pixels} (~{args.max_pixels//1024} img tokens max)")
        logger.info(f"  max_length   : {args.max_length}")
        logger.info(f"  batch_size   : {args.batch_size}"
                    + (" (will benchmark)" if args.find_batch_size else ""))
        logger.info(f"  world_size   : {world_size}")
        logger.info(f"  sample_num   : {args.sample_num if args.sample_num else 'all'}")
        logger.info("=" * 60)

    if args.mode == "notes":
        encode_notes(args, local_rank, world_size, logger)
    elif args.mode == "queries":
        encode_queries(args, local_rank, world_size, logger)

    if world_size > 1:
        # queries mode: rank-0 needs all shards present before merging
        # notes mode: each rank writes its own file independently — no barrier needed
        if args.mode == "queries":
            dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        logger.info("Encoding complete.")


if __name__ == "__main__":
    main()
