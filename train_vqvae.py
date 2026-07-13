"""train_vqvae.py — Phase 3/4 training loop.

Milestone-aware: flags toggle EMA codebook, stop-token, length penalty, etc.
Supports single-GPU and multi-node DDP via torchrun.

DDP launch (P4 M6):
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \\
      --rdzv_endpoint=spark1:29500 --rdzv_backend=c10d train_vqvae.py [args]
  # on the second machine, set --node_rank=1 with the same rdzv_endpoint.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from vqvae.data import (
    Opus100Dataset, ParallelDataset, combined_split, load_corpus,
    make_collate, make_streaming_collate,
)
from vqvae.losses import (
    length_penalty, reconstruction_loss, semantic_loss, usage_entropy,
)
from vqvae.model import VQVAE


def init_ddp() -> tuple[bool, int, int]:
    """If torchrun env vars are set, init the process group. Returns
    (is_ddp, rank, world_size)."""
    if "WORLD_SIZE" not in os.environ:
        return False, 0, 1
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False, rank, world_size
    # gloo backend works across machines on plain Ethernet (no NCCL/IB needed)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"[DDP] rank={rank} world_size={world_size} backend=gloo")
    return True, rank, world_size


def is_main_rank(rank: int) -> bool:
    return rank == 0


def short_to_indices(short_codes: list[str], names: list[str]) -> list[int]:
    return [short_codes.index(n) for n in names]


@torch.no_grad()
def embed_sentences(st_model, input_ids: torch.Tensor,
                    pad_token_id: int) -> torch.Tensor:
    """Mean-pooled sentence embedding from a frozen HF encoder (MiniLM).

    input_ids are already tokenized with the SAME tokenizer (incl. special
    tokens), so we feed them directly — no re-tokenization. Returns (B, D_sem),
    detached. Mean-pooling over non-pad tokens matches the sentence-transformers
    pooling for paraphrase-multilingual-MiniLM-L12-v2.
    """
    attn = (input_ids != pad_token_id).long()
    out = st_model(input_ids=input_ids, attention_mask=attn)
    hidden = out.last_hidden_state                      # (B, T, D_sem)
    mask = attn.unsqueeze(-1).to(hidden.dtype)          # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (summed / counts).detach()


class _MixedLoader:
    """Round-robin mix of two dataloaders with a sampling weight on the first.
    Used for --corpus both: 80% opus, 20% flores per batch."""

    def __init__(self, primary, secondary, opus_fraction: float, rng):
        self.primary = primary
        self.secondary = secondary
        self.opus_fraction = opus_fraction
        self.rng = rng

    def __iter__(self):
        p_iter = iter(self.primary)
        s_iter = iter(self.secondary)
        while True:
            if self.rng.random() < self.opus_fraction:
                try:
                    yield next(p_iter)
                except StopIteration:
                    p_iter = iter(self.primary)
                    yield next(p_iter)
            else:
                try:
                    yield next(s_iter)
                except StopIteration:
                    s_iter = iter(self.secondary)
                    yield next(s_iter)


def compute_target_len(src_ids: torch.Tensor, pad_token_id: int,
                       compression_ratio: float, slack: int,
                       m_max: int) -> torch.Tensor:
    """Per-example hard length cap: target_len[i] = ceil(src_len[i] * ratio + slack).
    src_ids: (B, T) int64 with pad_token_id padding. Returns (B,) int64.
    """
    src_lens = (src_ids != pad_token_id).sum(dim=1).float()      # (B,)
    target = (src_lens * compression_ratio + slack).ceil().long()
    return target.clamp(min=1, max=m_max)


@torch.no_grad()
def evaluate(model, loader, device, pad_token_id, n_batches: int = 8,
             st_model=None) -> dict:
    model.eval()
    total_recon = 0.0
    total_correct = 0
    total_count = 0
    sem_cos_sum = 0.0
    sem_cos_n = 0
    code_counts = torch.zeros(model.quantizer.k, device=device)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_lang_id = batch["tgt_lang_id"].to(device)
        out = model(src_ids, tgt_ids, tgt_lang_id)
        recon = reconstruction_loss(out["logits"], tgt_ids[:, 1:], pad_token_id)
        total_recon += recon.item()
        preds = out["logits"].argmax(dim=-1)
        mask = tgt_ids[:, 1:] != pad_token_id
        total_correct += ((preds == tgt_ids[:, 1:]) & mask).sum().item()
        total_count += mask.sum().item()
        code_counts += out["usage"]
        # Phase 5: how well does the bottleneck align with true sentence meaning?
        if st_model is not None and out["sem_pred"] is not None:
            tgt_emb = embed_sentences(st_model, src_ids, pad_token_id)
            sem_cos_sum += F.cosine_similarity(
                out["sem_pred"].float(), tgt_emb.float(), dim=-1).mean().item()
            sem_cos_n += 1
    model.train()
    p = code_counts / (code_counts.sum() + 1e-8)
    used = (code_counts > 0).sum().item()
    perplexity = torch.exp(-(p * (p + 1e-8).log()).sum()).item()
    return {
        "val_recon": total_recon / max(1, min(n_batches, i + 1)),
        "val_token_acc": total_correct / max(1, total_count),
        "codes_used": used,
        "code_perplexity": perplexity,
        "val_sem_cos": (sem_cos_sum / sem_cos_n) if sem_cos_n else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-path", default="data/parallel_corpus.npz",
                        help="Path to FLORES parallel_corpus.npz (used for eval and "
                             "for --corpus flores/both training).")
    parser.add_argument("--embedding-table", default="data/embedding_table.pt")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--checkpoint-name", default="vqvae")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-decay", choices=["none", "cosine"], default="none",
                        help="Phase 6: 'cosine' decays LR from peak to lr-min-ratio "
                             "after warmup (fixes long-run divergence from constant LR)")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1,
                        help="Cosine decay floor as a fraction of peak LR")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=1000)

    parser.add_argument("--src-langs", default="en",
                        help="Comma-separated list of source langs (or 'all')")
    parser.add_argument("--tgt-langs", default="en",
                        help="Comma-separated list of target langs (or 'all')")
    parser.add_argument("--combine-splits", action="store_true",
                        help="Merge dev+devtest, then split 90/10 for train/val "
                             "(more data; original split unused)")
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--d-code", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--m-max", type=int, default=64)
    parser.add_argument("--n-enc-layers", type=int, default=4)
    parser.add_argument("--n-dec-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--beta-commit", type=float, default=0.25)

    parser.add_argument("--corpus", default="flores",
                        choices=["flores", "opus100", "both"],
                        help="flores=in-memory parallel; opus100=streaming sharded; "
                             "both=mix (80% opus, 20% flores per step)")
    parser.add_argument("--opus100-dir", default="data/opus100")

    parser.add_argument("--use-stop-mask", action="store_true",
                        help="Enable <stop>-token cross-attention masking (M6)")
    parser.add_argument("--lambda-len", type=float, default=0.0,
                        help="Soft length penalty weight (M6, legacy). 0 disables.")
    parser.add_argument("--lambda-len-warmup", type=int, default=0,
                        help="Linearly ramp soft lambda_len from 0 over this many steps")

    # M2 hard cap + Lagrangian self-tuning
    parser.add_argument("--compression-ratio", type=float, default=0.0,
                        help="Hard cap target_len = ceil(src_len*ratio + slack). "
                             "0 disables the hard cap.")
    parser.add_argument("--length-slack", type=int, default=4)
    parser.add_argument("--target-avg-len", type=float, default=0.0,
                        help="If > 0, use Lagrangian λ_len update toward this "
                             "target average bottleneck length.")
    parser.add_argument("--lambda-len-lr", type=float, default=0.001)
    parser.add_argument("--lambda-len-max", type=float, default=2.0)
    parser.add_argument("--lambda-use", type=float, default=0.0,
                        help="Usage entropy bonus weight (M4+)")
    parser.add_argument("--use-ema", action="store_true",
                        help="EMA codebook updates (M4)")
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--dead-threshold", type=float, default=0.01,
                        help="Reset codes with cluster_size below this fraction of mean")
    parser.add_argument("--reset-dead-every", type=int, default=500)

    # Phase 5: semantic-target loss. Pool the quantized bottleneck and regress
    # it (cosine) toward the frozen MiniLM sentence embedding of the source, so
    # codes encode meaning instead of high-frequency token/punctuation glue.
    parser.add_argument("--use-semantic-head", action="store_true",
                        help="Add the pooled-bottleneck → sentence-embedding head (P5)")
    parser.add_argument("--lambda-sem", type=float, default=0.0,
                        help="Weight on the semantic cosine loss. 0 disables. "
                             "Key knob balancing token-CE vs meaning into the bottleneck.")
    parser.add_argument("--semantic-model",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Frozen sentence encoder used to produce targets (P5)")

    # Phase 5b: frequency-weighted token-CE. Downweights punctuation/function
    # tokens in the reconstruction loss so it stops rewarding boilerplate and
    # pulls the same direction as the semantic loss.
    parser.add_argument("--use-token-weights", action="store_true",
                        help="Apply per-token frequency weights to reconstruction CE (P5b)")
    parser.add_argument("--token-weights", default="data/token_weights.pt",
                        help="Path to (vocab,) weight vector from build_token_weights.py")

    # Phase 5c: word dropout (force code reliance) + length head (stop fix).
    parser.add_argument("--word-dropout", type=float, default=0.0,
                        help="Fraction of teacher-forced decoder input tokens replaced "
                             "with <unk> during training (attacks posterior collapse).")
    parser.add_argument("--word-dropout-token", type=int, default=3,
                        help="Token id used to replace dropped inputs (3=<unk> for MiniLM)")
    parser.add_argument("--use-length-head", action="store_true",
                        help="Predict bottleneck length from pooled codes (stop fix / NAT prereq)")
    parser.add_argument("--lambda-lenpred", type=float, default=0.1,
                        help="Weight on the length-prediction MSE loss")
    parser.add_argument("--no-vq", action="store_true",
                        help="Phase 6 B1: skip quantization, pass continuous z_e to "
                             "the decoder (reconstruction upper bound / VQ's cost)")
    parser.add_argument("--no-code", action="store_true",
                        help="Phase 6 B0: zero the bottleneck (LM lower bound)")
    parser.add_argument("--decoder-type", choices=["ar", "nat"], default="ar",
                        help="Phase 6 B3: 'nat' = non-autoregressive decoder (codes are "
                             "the only info source; removes AR exposure bias)")
    parser.add_argument("--unfreeze-embeddings", action="store_true",
                        help="Phase 6: train the token embeddings (input+tied output) "
                             "instead of keeping the frozen XLM-R table")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 for forward/backward")
    args = parser.parse_args()

    is_ddp, rank, world_size = init_ddp()
    # IMPORTANT: torch seed must be the SAME on every rank so all ranks build
    # an identical model + identical EMA buffers (codebook, cluster_size,
    # cluster_sum). DDP syncs Parameters but NOT buffers — the EMA codebook
    # is a buffer, so divergent init causes silent drift and eventually
    # rank-divergent collective shapes (n_dead differing across ranks in
    # reset_dead_codes), which deadlocks the run. We re-seed with the
    # rank offset further down, after the model is built, so per-rank data
    # sampling and dropout still differ.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)

    def log(msg: str):
        if is_main_rank(rank):
            print(msg, flush=True)

    log("=" * 60)
    if is_ddp:
        log(f"DDP active: rank {rank}/{world_size}")
    log(f"Loading FLORES from {args.flores_path}...")
    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.flores_path)
    print(f"  dev: {dev_ids.shape}, devtest: {devtest_ids.shape}")
    print(f"  langs: {meta.short_codes}")

    if args.combine_splits:
        train_ids, val_ids, train_lens, val_lens = combined_split(
            dev_ids, devtest_ids, dev_lens, devtest_lens,
            val_fraction=args.val_fraction, seed=args.seed)
        print(f"  combined→split: train {train_ids.shape}, val {val_ids.shape}")
    else:
        train_ids, val_ids = dev_ids, devtest_ids
        train_lens, val_lens = dev_lens, devtest_lens

    def _parse_langs(arg: str) -> list[int]:
        if arg.strip() == "all":
            return list(range(len(meta.short_codes)))
        return short_to_indices(meta.short_codes, arg.split(","))

    src_langs = _parse_langs(args.src_langs)
    tgt_langs = _parse_langs(args.tgt_langs)
    print(f"  src lang indices: {src_langs}, tgt lang indices: {tgt_langs}")

    print(f"\nLoading embedding table from {args.embedding_table}...")
    emb = torch.load(args.embedding_table, map_location="cpu")
    vocab_size = emb.shape[0]
    print(f"  shape: {tuple(emb.shape)} -> vocab_size={vocab_size}")

    print(f"\nBuilding VQVAE...")
    model = VQVAE(
        vocab_size=vocab_size,
        n_langs=len(meta.short_codes),
        d_model=args.d_model, d_code=args.d_code, k=args.k, m_max=args.m_max,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        beta_commit=args.beta_commit, pad_token_id=meta.pad_token_id,
        embedding_table=emb,
        use_stop_mask=args.use_stop_mask,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        dead_threshold=args.dead_threshold,
        use_semantic_head=args.use_semantic_head,
        use_length_head=args.use_length_head,
        no_vq=args.no_vq,
        no_code=args.no_code,
        decoder_type=args.decoder_type,
    )
    if args.unfreeze_embeddings:
        # Phase 6: the frozen XLM-R *input* embeddings double as the tied output
        # projection, which caps generation quality. Unfreeze both enc+dec token
        # embeddings (must happen before the optimizer is built so they're included).
        model.encoder.token_emb.weight.requires_grad = True
        model.decoder.token_emb.weight.requires_grad = True
        log("  embeddings UNFROZEN (encoder + decoder token_emb trainable)")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_train/1e6:.2f}M (total {n_total/1e6:.2f}M)")

    device = torch.device(args.device)
    model = model.to(device)

    # Cap CUDA memory on the GB10 unified-memory architecture. The default
    # caching allocator will reserve nearly the full system pool (we measured
    # 124 GB reserved on a 121 GB system), starving the rest of the OS and
    # eventually forcing a lockup as the kernel dumps page cache and swaps.
    # Training only needs ~4 GB of working set; capping at 30% leaves >80 GB
    # for the OS, page cache (mmap'd opus shards), and any other processes.
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("CUDA_MEM_FRACTION", "0.30")), device.index or 0)

    # Phase 5: load the frozen sentence encoder that produces semantic targets.
    # Kept OUTSIDE the VQVAE module so DDP never tries to sync/train it. Every
    # rank holds its own copy (targets are computed locally, no collectives).
    st_model = None
    if args.use_semantic_head and args.lambda_sem > 0:
        from transformers import AutoModel
        log(f"\nLoading frozen semantic encoder {args.semantic_model}...")
        st_model = AutoModel.from_pretrained(args.semantic_model)
        st_model.eval().to(device)
        for p in st_model.parameters():
            p.requires_grad = False
        log(f"  semantic target dim: {st_model.config.hidden_size}")

    # Phase 5b: per-token frequency weights for the reconstruction loss.
    token_weight = None
    if args.use_token_weights:
        log(f"\nLoading token weights {args.token_weights}...")
        token_weight = torch.load(args.token_weights, map_location="cpu").to(device)
        log(f"  nonzero weights: {int((token_weight > 0).sum())}/{token_weight.numel()}  "
            f"mean_over_seen~{float(token_weight[token_weight > 0].mean()):.3f}")

    # Now that model + buffers are built identically on every rank, re-seed
    # torch with the rank offset so dropout and any stochastic ops differ.
    if is_ddp:
        torch.manual_seed(args.seed + rank)

    if is_ddp:
        # find_unused_parameters=False: every fwd uses every param (both encoder
        # and decoder are touched). gradient_as_bucket_view: small memory win.
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=(args.no_vq or args.no_code),
                    gradient_as_bucket_view=True)
        # Helper to peek at the unwrapped module (for ema_update, codebook lookup)
        unwrapped = model.module
    else:
        unwrapped = model

    log(f"\nDevice: {device}")

    print(f"\nBuilding dataloaders (corpus={args.corpus})...")
    val_ds = ParallelDataset(val_ids, val_lens)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate(meta.pad_token_id, len(meta.short_codes),
                                src_langs, tgt_langs, np.random.default_rng(0)),
        drop_last=True, num_workers=0,
    )

    if args.corpus == "flores":
        train_ds = ParallelDataset(train_ids, train_lens)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=make_collate(meta.pad_token_id, len(meta.short_codes),
                                    src_langs, tgt_langs, rng),
            drop_last=True, num_workers=0,
        )
        print(f"  train batches per epoch: {len(train_loader)}")
    elif args.corpus == "opus100":
        opus_ds = Opus100Dataset(args.opus100_dir, meta.short_codes, seed=args.seed)
        train_loader = DataLoader(
            opus_ds, batch_size=args.batch_size,
            collate_fn=make_streaming_collate(meta.pad_token_id),
            num_workers=0,
        )
        print(f"  opus shards: {[(l, n) for l, n, _ in opus_ds.shard_meta]}")
    elif args.corpus == "both":
        # Mix at the iterator level: alternate per-batch
        flores_ds = ParallelDataset(train_ids, train_lens)
        flores_loader = DataLoader(
            flores_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=make_collate(meta.pad_token_id, len(meta.short_codes),
                                    src_langs, tgt_langs, rng),
            drop_last=True, num_workers=0,
        )
        opus_ds = Opus100Dataset(args.opus100_dir, meta.short_codes, seed=args.seed)
        opus_loader = DataLoader(
            opus_ds, batch_size=args.batch_size,
            collate_fn=make_streaming_collate(meta.pad_token_id),
            num_workers=0,
        )
        train_loader = _MixedLoader(opus_loader, flores_loader, opus_fraction=0.8,
                                    rng=np.random.default_rng(args.seed + 7))
        print(f"  mixed: 80% opus + 20% flores")
    else:
        raise ValueError(f"unknown corpus {args.corpus}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        # Linear warmup to 1.0
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.lr_decay == "cosine":
            # Cosine decay from 1.0 -> lr_min_ratio over the remaining steps.
            # Phase 6: the warmup-only constant-LR schedule diverged over long
            # (100k) horizons (VQ + unfrozen both crashed in the 2nd half); decay
            # to a floor keeps late training stable.
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            progress = min(1.0, progress)
            return args.lr_min_ratio + 0.5 * (1 - args.lr_min_ratio) * \
                (1 + math.cos(math.pi * progress))
        return 1.0  # constant (legacy default)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    autocast_dtype = torch.bfloat16 if args.bf16 else None

    # Lagrangian λ_len state (for M2 self-tuning length penalty)
    lagrangian_lambda = 0.0
    use_lagrangian = args.target_avg_len > 0
    use_hard_cap = args.compression_ratio > 0
    bits_per_code = float(np.log2(args.k))

    print("\n" + "=" * 60)
    print(f"Training for {args.steps} steps...")
    if use_hard_cap:
        print(f"  hard length cap: target_len = ceil(src_len*{args.compression_ratio} + {args.length_slack})")
    if use_lagrangian:
        print(f"  Lagrangian λ_len: target_avg_len={args.target_avg_len}, "
              f"lr={args.lambda_len_lr}, max={args.lambda_len_max}")
    print("=" * 60)
    model.train()
    step = 0
    epoch = 0
    t_start = time.time()
    log_buf = {"recon": 0.0, "commit": 0.0, "codebook": 0.0, "sem": 0.0,
               "lenpred": 0.0, "len_frac": 0.0, "avg_len": 0.0, "lagrangian": 0.0,
               "src_len": 0.0, "n": 0}
    while step < args.steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.steps:
                break

            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            tgt_lang_id = batch["tgt_lang_id"].to(device)

            if use_hard_cap:
                target_len = compute_target_len(
                    src_ids, meta.pad_token_id, args.compression_ratio,
                    args.length_slack, args.m_max)
            else:
                target_len = None

            if autocast_dtype is not None:
                cm = torch.autocast(device_type=device.type, dtype=autocast_dtype)
            else:
                from contextlib import nullcontext
                cm = nullcontext()

            with cm:
                out = model(src_ids, tgt_ids, tgt_lang_id, target_len=target_len,
                            word_dropout=args.word_dropout,
                            mask_token_id=args.word_dropout_token)
                recon = reconstruction_loss(
                    out["logits"], tgt_ids[:, 1:], meta.pad_token_id,
                    token_weight=token_weight)
                vq_l = out["vq_losses"]
                loss = recon + vq_l["commit"]
                if "codebook" in vq_l:
                    loss = loss + vq_l["codebook"]

                # Phase 5: semantic cosine loss on the pooled bottleneck
                sem_l = torch.tensor(0.0, device=device)
                if st_model is not None and out["sem_pred"] is not None:
                    tgt_emb = embed_sentences(st_model, src_ids, meta.pad_token_id)
                    sem_l = semantic_loss(out["sem_pred"], tgt_emb)
                    loss = loss + args.lambda_sem * sem_l

                # Phase 5c: length-prediction loss (train length head vs the cap)
                len_pred_l = torch.tensor(0.0, device=device)
                if out["len_pred"] is not None and target_len is not None:
                    len_pred_l = F.mse_loss(out["len_pred"].float(), target_len.float())
                    loss = loss + args.lambda_lenpred * len_pred_l

                # Bottleneck length stats — needed for both reporting and Lagrangian
                first_stop = unwrapped.quantizer.first_stop_position(out["indices"]).float()
                avg_len_batch = first_stop.mean()

                # Soft length penalty (legacy / fixed-λ variant)
                len_l = torch.tensor(0.0, device=device)
                if args.lambda_len > 0:
                    ramp = min(1.0, step / args.lambda_len_warmup) if args.lambda_len_warmup > 0 else 1.0
                    len_l = length_penalty(out["indices"], stop_index=0, m_max=args.m_max)
                    loss = loss + args.lambda_len * ramp * len_l

                # Lagrangian self-tuning λ_len
                if use_lagrangian:
                    loss = loss + lagrangian_lambda * (avg_len_batch / args.m_max)

                if args.lambda_use > 0:
                    use_l = -usage_entropy(out["usage"])
                    loss = loss + args.lambda_use * use_l

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()

            if args.use_ema:
                unwrapped.quantizer.ema_update(out["z_e"], out["indices"],
                                                ddp_world_size=world_size)
                if step > 0 and step % args.reset_dead_every == 0:
                    n_reset = unwrapped.quantizer.reset_dead_codes(
                        out["z_e"], ddp_world_size=world_size)
                    if n_reset > 0:
                        log(f"  ── reset {n_reset} dead codes at step {step}")

            # Lagrangian λ_len update (after backward, on detached scalars)
            if use_lagrangian:
                avg_len_now = avg_len_batch.detach().item()
                lagrangian_lambda += args.lambda_len_lr * (avg_len_now - args.target_avg_len)
                lagrangian_lambda = max(0.0, min(args.lambda_len_max, lagrangian_lambda))

            src_len_avg = float((src_ids != meta.pad_token_id).sum(dim=1).float().mean())
            log_buf["recon"] += recon.item()
            log_buf["commit"] += vq_l["commit"].item()
            if "codebook" in vq_l:
                log_buf["codebook"] += vq_l["codebook"].item()
            log_buf["sem"] += sem_l.item()
            log_buf["lenpred"] += len_pred_l.item()
            log_buf["len_frac"] += len_l.item()
            log_buf["avg_len"] += avg_len_batch.detach().item()
            log_buf["lagrangian"] += lagrangian_lambda
            log_buf["src_len"] += src_len_avg
            log_buf["n"] += 1

            step += 1

            if step % args.log_every == 0:
                n = log_buf["n"]
                elapsed = time.time() - t_start
                steps_per_s = step / elapsed
                avg_len = log_buf["avg_len"] / n
                src_len = log_buf["src_len"] / n
                bits_per_sentence = avg_len * bits_per_code
                bits_per_src_token = bits_per_sentence / max(1.0, src_len)
                # Memory health check — unified-memory GB10 will lock up if
                # the training process crowds out the kernel page cache, so we
                # log GPU + system RSS at every log step. CUDA reserved memory
                # is the more useful number than allocated (allocator pool).
                mem_str = ""
                if device.type == "cuda":
                    cuda_alloc_gb = torch.cuda.memory_allocated() / 1e9
                    cuda_reserved_gb = torch.cuda.memory_reserved() / 1e9
                    mem_str += f"  cuda={cuda_alloc_gb:.1f}/{cuda_reserved_gb:.1f}GB"
                try:
                    import psutil
                    rss_gb = psutil.Process().memory_info().rss / 1e9
                    sys_avail_gb = psutil.virtual_memory().available / 1e9
                    mem_str += f"  rss={rss_gb:.1f}GB  sys_avail={sys_avail_gb:.0f}GB"
                except ImportError:
                    pass

                log(f"step {step:5d}  recon={log_buf['recon']/n:.4f}  "
                    f"commit={log_buf['commit']/n:.4f}  "
                    f"codebook={log_buf['codebook']/n:.4f}  "
                    f"sem={log_buf['sem']/n:.4f}  "
                    f"lenpred={log_buf['lenpred']/n:.3f}  "
                    f"avg_len={avg_len:.1f}/{args.m_max}  "
                    f"λ={log_buf['lagrangian']/n:.3f}  "
                    f"bits/sent={bits_per_sentence:.1f}  "
                    f"bits/srctok={bits_per_src_token:.2f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}  "
                    f"({steps_per_s:.1f} steps/s)"
                    f"{mem_str}")
                log_buf = {k: 0.0 for k in log_buf}; log_buf["n"] = 0

            if step % args.eval_every == 0:
                # Eval only on rank 0 to avoid duplicate work; other ranks wait
                if is_main_rank(rank):
                    metrics = evaluate(unwrapped, val_loader, device,
                                       meta.pad_token_id, st_model=st_model)
                    log(f"  ── val: recon={metrics['val_recon']:.4f}  "
                        f"acc={metrics['val_token_acc']:.4f}  "
                        f"codes_used={metrics['codes_used']}/{args.k}  "
                        f"perplexity={metrics['code_perplexity']:.1f}  "
                        f"sem_cos={metrics['val_sem_cos']:.3f}")
                if is_ddp:
                    dist.barrier()

            if step % args.ckpt_every == 0 or step == args.steps:
                if is_main_rank(rank):
                    ckpt_path = os.path.join(
                        args.output_dir, f"{args.checkpoint_name}_step{step}.pt")
                    torch.save({
                        "step": step,
                        "args": vars(args),
                        "model_state": unwrapped.state_dict(),
                        "optim_state": optim.state_dict(),
                        "lagrangian_lambda": lagrangian_lambda,
                    }, ckpt_path)
                if is_ddp:
                    dist.barrier()
                if is_main_rank(rank):
                    log(f"  ── checkpoint saved: {ckpt_path}")

    log(f"\nTraining done in {time.time() - t_start:.1f}s")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
