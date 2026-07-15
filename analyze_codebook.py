"""analyze_codebook.py — Output-side semantic discovery, batched.

For each codebook entry k, we want to discover its "meaning" without ever having
labeled it. Method:

  1. Generate "test" latent sequences containing k at a random position, with
     other slots filled with random non-stop codes.
  2. Generate matched "control" sequences with k swapped for a random non-stop
     code, otherwise identical.
  3. Decode each in every supported language (greedy autoregressive).
  4. Compute log-odds-ratio per output token: how much more likely is each
     token in test decodings vs. control? Tokens with high ratio are the
     semantic signature of k.

The Phase-3 implementation processed (code × lang) sequentially. This version
batches across (code × sample × lang × role) to fill the GPU. Chunks are sized
to fit decoder logits (B × T × |vocab|) in bf16 within ~10 GB.

Output: results/codebook_semantics.txt.
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

from vqvae.data import load_corpus, ParallelDataset, make_collate
from vqvae.model import VQVAE


@torch.no_grad()
def gather_active_codes(model, val_ids, val_lens, meta, device,
                        max_batches: int = 16, batch_size: int = 32) -> list[int]:
    """Run val data through the encoder, return codes used at least once."""
    val_ds = ParallelDataset(val_ids, val_lens)
    rng = np.random.default_rng(0)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=make_collate(meta.pad_token_id, len(meta.short_codes), rng=rng),
        drop_last=True)

    counts = torch.zeros(model.quantizer.k, device=device)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model(batch["src_ids"].to(device),
                    batch["tgt_ids"].to(device),
                    batch["tgt_lang_id"].to(device))
        counts += out["usage"]
    active = torch.where(counts > 0)[0].tolist()
    # Always exclude stop (index 0)
    return [k for k in active if k != model.quantizer.stop_index]


def build_test_control_pairs(codes: list[int], K: int, M_max: int,
                             n_samples: int, seq_len_range: tuple[int, int],
                             stop_index: int = 0,
                             seed: int = 0,
                             device: str = "cuda",
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each (code, sample) pair, build a (test, control) sequence that
    differs only at one position.

    Returns:
      test_indices : (C * n_samples, M_max) int64 — codebook index per slot
      ctrl_indices : (C * n_samples, M_max) int64
      Each row: random fill in [1, K), <stop> at L. test[..., p] = code,
      control[..., p] = a random non-stop, non-code index at the same position.
    """
    L_min, L_max = seq_len_range
    C = len(codes)
    N = C * n_samples
    rng = torch.Generator(device=device); rng.manual_seed(seed)

    # Random lengths and target positions
    lengths = torch.randint(L_min, L_max + 1, (N,), device=device, generator=rng)
    positions = (lengths.float() * torch.rand((N,), device=device, generator=rng)).long()
    positions = positions.clamp(min=0, max=M_max - 1)

    # Random non-stop fill
    test_idx = torch.randint(1, K, (N, M_max), device=device, generator=rng)
    ctrl_idx = test_idx.clone()

    # Mask positions >= L to <stop>
    pos_arange = torch.arange(M_max, device=device).unsqueeze(0)            # (1, M)
    mask = pos_arange >= lengths.unsqueeze(1)                                # (N, M)
    test_idx = torch.where(mask, torch.full_like(test_idx, stop_index), test_idx)
    ctrl_idx = torch.where(mask, torch.full_like(ctrl_idx, stop_index), ctrl_idx)

    # Place target code at chosen position (test) and a fresh random non-code
    # non-stop value at the same position (control)
    arange_n = torch.arange(N, device=device)
    code_per_row = torch.tensor([c for c in codes for _ in range(n_samples)],
                                device=device, dtype=torch.long)
    test_idx[arange_n, positions] = code_per_row

    # Pick control replacement: any value in [1, K) that differs from code_per_row
    ctrl_codes = torch.randint(1, K, (N,), device=device, generator=rng)
    same = ctrl_codes == code_per_row
    while same.any():
        ctrl_codes[same] = torch.randint(1, K, (int(same.sum().item()),),
                                          device=device, generator=rng)
        same = ctrl_codes == code_per_row
    ctrl_idx[arange_n, positions] = ctrl_codes

    return test_idx, ctrl_idx, code_per_row


def rvq_lookup(quantizer, idx_multi: torch.Tensor) -> torch.Tensor:
    """Sum the per-level codebook selections for a multi-level index tensor.
    idx_multi: (..., n_levels) int64. Returns (..., d_code)."""
    z = None
    for level, cb in enumerate(quantizer.codebooks):
        sel = cb[idx_multi[..., level]]
        z = sel if z is None else z + sel
    return z


def build_rvq_test_control(codes, K, M_max, n_levels, target_level,
                           n_samples, seq_len_range, seed=0, device="cuda"):
    """RVQ analogue of build_test_control_pairs. Each slot has n_levels codes;
    the SUM is its vector. We vary the target code at `target_level` of one slot
    (test) vs a different code (control), holding all other (slot, level) entries
    fixed. Index 0 is a real code in RVQ (no reserved <stop>); validity is a
    separate length mask, returned as `lengths`.

    Returns test_idx, ctrl_idx : (N, M, n_levels); code_per_row (N,); lengths (N,).
    """
    L_min, L_max = seq_len_range
    C = len(codes)
    N = C * n_samples
    g = torch.Generator(device=device); g.manual_seed(seed)

    lengths = torch.randint(L_min, L_max + 1, (N,), device=device, generator=g)
    positions = (lengths.float() * torch.rand((N,), device=device, generator=g)).long()
    positions = positions.clamp(min=0, max=M_max - 1)

    test_idx = torch.randint(0, K, (N, M_max, n_levels), device=device, generator=g)
    ctrl_idx = test_idx.clone()

    arange_n = torch.arange(N, device=device)
    code_per_row = torch.tensor([c for c in codes for _ in range(n_samples)],
                                device=device, dtype=torch.long)
    test_idx[arange_n, positions, target_level] = code_per_row

    ctrl_codes = torch.randint(0, K, (N,), device=device, generator=g)
    same = ctrl_codes == code_per_row
    while same.any():
        ctrl_codes[same] = torch.randint(0, K, (int(same.sum().item()),),
                                          device=device, generator=g)
        same = ctrl_codes == code_per_row
    ctrl_idx[arange_n, positions, target_level] = ctrl_codes

    return test_idx, ctrl_idx, code_per_row, lengths


@torch.no_grad()
def batched_greedy_decode(model, z_q: torch.Tensor, mem_mask: torch.Tensor,
                          lang_ids: torch.Tensor, bos_id: int, eos_id: int,
                          pad_id: int, max_len: int) -> torch.Tensor:
    """Greedy decode (B, T+1) token IDs (BOS + up to max_len)."""
    B = z_q.size(0)
    device = z_q.device
    out = torch.full((B, 1), bos_id, dtype=torch.int64, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model.decoder(z_q, mem_mask, out, lang_ids)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        # Once finished, keep emitting pad (will be filtered out at counting time)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
        out = torch.cat([out, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_id) | (next_tok == pad_id)
        if finished.all():
            break
    return out


@torch.no_grad()
def decode_chunked(model, z_q: torch.Tensor, mem_mask: torch.Tensor,
                   lang_ids: torch.Tensor, bos_id: int, eos_id: int,
                   pad_id: int, max_len: int, chunk: int) -> torch.Tensor:
    """Run batched_greedy_decode in chunks, concat results.
    Pads each chunk's output to max_len + 1 columns."""
    outputs = []
    for start in range(0, z_q.size(0), chunk):
        end = min(start + chunk, z_q.size(0))
        out = batched_greedy_decode(
            model, z_q[start:end], mem_mask[start:end], lang_ids[start:end],
            bos_id, eos_id, pad_id, max_len)
        # Pad to max_len + 1
        T = out.size(1)
        if T < max_len + 1:
            pad = torch.full((out.size(0), max_len + 1 - T), pad_id,
                             dtype=torch.int64, device=out.device)
            out = torch.cat([out, pad], dim=1)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flores-path", default="data/parallel_corpus.npz")
    parser.add_argument("--embedding-table", default="data/embedding_table.pt")
    parser.add_argument("--tokenizer",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--n-samples", type=int, default=64,
                        help="Test/control samples per (code, lang)")
    parser.add_argument("--decode-max-len", type=int, default=15)
    parser.add_argument("--seq-len-min", type=int, default=3)
    parser.add_argument("--seq-len-max", type=int, default=16)
    parser.add_argument("--top-tokens", type=int, default=8)
    parser.add_argument("--active-only", action="store_true",
                        help="Restrict to codes that appear in val encoder outputs")
    parser.add_argument("--rvq-level", type=int, default=0,
                        help="For RVQ models: which residual level's codes to analyze "
                             "(0 = coarsest/primary radical)")
    parser.add_argument("--max-codes", type=int, default=None,
                        help="Hard cap on number of codes (for debugging)")
    parser.add_argument("--decode-chunk", type=int, default=1024,
                        help="Sequences per decoder forward pass")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output", default="results/codebook_semantics.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"  trained for {ckpt['step']} steps")

    print(f"Loading FLORES metadata: {args.flores_path}")
    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.flores_path)

    print(f"Loading embedding table: {args.embedding_table}")
    emb = torch.load(args.embedding_table, map_location="cpu")
    vocab_size = emb.shape[0]

    print("Building model from checkpoint config...")
    model = VQVAE(
        vocab_size=vocab_size, n_langs=len(meta.short_codes),
        d_model=cfg["d_model"], d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"],
        n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=meta.pad_token_id,
        embedding_table=emb, use_stop_mask=cfg["use_stop_mask"],
        use_ema=cfg.get("use_ema", False), ema_decay=cfg.get("ema_decay", 0.99),
        use_semantic_head=cfg.get("use_semantic_head", False),
        use_length_head=cfg.get("use_length_head", False),
        no_vq=cfg.get("no_vq", False), no_code=cfg.get("no_code", False),
        decoder_type=cfg.get("decoder_type", "ar"),
        use_rvq=cfg.get("use_rvq", False),
        n_rvq_levels=cfg.get("n_rvq_levels", 4),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(args.device).eval()

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    K = model.quantizer.k
    M = model.encoder.m_max
    is_rvq = getattr(model, "use_rvq", False)
    label = f"L{args.rvq_level} Char" if is_rvq else "Char"

    # Pick which codes to analyze. In RVQ index 0 is a real code (no reserved
    # <stop>); in single-VQ index 0 is <stop> and skipped.
    if args.active_only and not is_rvq:
        print("Identifying active codes from val encoder pass...")
        codes = gather_active_codes(model, devtest_ids, devtest_lens, meta, args.device)
        print(f"  active codes: {len(codes)}/{K}")
    else:
        codes = list(range(0, K)) if is_rvq else list(range(1, K))
    if args.max_codes is not None:
        codes = codes[: args.max_codes]
    n_codes = len(codes)
    print(f"\nAnalyzing {n_codes} codes ({'RVQ level ' + str(args.rvq_level) if is_rvq else 'single-VQ'}) "
          f"with n_samples={args.n_samples}, chunk={args.decode_chunk}")

    # Build all test/control latent index sequences in one pass
    print("Building test/control latent sequences...")
    if is_rvq:
        n_levels = model.quantizer.n_levels
        assert 0 <= args.rvq_level < n_levels, f"rvq-level must be in [0,{n_levels})"
        test_idx, ctrl_idx, code_per_row, lengths = build_rvq_test_control(
            codes, K, M, n_levels, args.rvq_level, args.n_samples,
            (args.seq_len_min, args.seq_len_max), device=args.device)
        n_seqs_per_role = test_idx.size(0)
        z_q_test = rvq_lookup(model.quantizer, test_idx)   # (N, M, D)
        z_q_ctrl = rvq_lookup(model.quantizer, ctrl_idx)
        pos_arange = torch.arange(M, device=args.device).unsqueeze(0)
        mask_test = pos_arange < lengths.unsqueeze(1)        # (N, M) validity mask
        mask_ctrl = mask_test.clone()
    else:
        test_idx, ctrl_idx, code_per_row = build_test_control_pairs(
            codes, K, M, args.n_samples, (args.seq_len_min, args.seq_len_max),
            stop_index=model.quantizer.stop_index, device=args.device)
        n_seqs_per_role = test_idx.size(0)
        cb = model.quantizer.codebook
        z_q_test = cb[test_idx]                   # (N, M, D)
        z_q_ctrl = cb[ctrl_idx]
        mask_test = model.quantizer.get_stop_mask(test_idx)
        mask_ctrl = model.quantizer.get_stop_mask(ctrl_idx)

    n_lang = len(meta.short_codes)

    # Replicate per language: full attribution matrix has shape
    # (n_codes * n_samples * n_lang) for each role
    # Index layout: [code, sample, lang] flattened in that order
    print(f"Decoding {n_seqs_per_role * n_lang * 2} total sequences "
          f"({n_codes} codes × {args.n_samples} samples × {n_lang} langs × 2 roles)")

    autocast_cm = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if args.bf16 and args.device == "cuda" else
                   __import__("contextlib").nullcontext())

    # Do test then control in two passes (keeps memory bounded)
    role_token_counts = {role: defaultdict(Counter) for role in ("test", "ctrl")}
    for role, z_q_role, mask_role in [("test", z_q_test, mask_test),
                                       ("ctrl", z_q_ctrl, mask_ctrl)]:
        # Replicate each (code, sample) sequence n_lang times, with lang_id 0..n_lang-1
        # interleaved so attribution by row is consistent.
        # New shape: (n_seqs_per_role * n_lang, M, D)
        z_q_rep = z_q_role.repeat_interleave(n_lang, dim=0)
        mask_rep = mask_role.repeat_interleave(n_lang, dim=0)
        lang_ids = torch.arange(n_lang, device=args.device).repeat(n_seqs_per_role)

        with autocast_cm:
            decoded = decode_chunked(
                model, z_q_rep, mask_rep, lang_ids,
                meta.bos_token_id, meta.eos_token_id, meta.pad_token_id,
                args.decode_max_len, args.decode_chunk)

        decoded = decoded.cpu().numpy()
        # Aggregate per (code, lang): for sequence i, code = codes[i // (n_samples * n_lang)],
        # lang = i % n_lang
        special = {meta.pad_token_id, meta.bos_token_id, meta.eos_token_id}
        for i, row in enumerate(decoded):
            code_pos = i // n_lang
            lang = i % n_lang
            code = code_per_row[code_pos].item()
            for t in row:
                if t not in special:
                    role_token_counts[role][(code, lang)][int(t)] += 1
        print(f"  {role}: aggregated {len(decoded)} decodings")

    # Compute log-odds + write report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nWriting report to {args.output}...")
    with open(args.output, "w") as f:
        f.write("VQ-VAE Codebook Semantics (output-side discovery, batched)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"K={K}, codes_analyzed={n_codes}, samples={args.n_samples}, "
                f"decode_max_len={args.decode_max_len}"
                f"{', RVQ level ' + str(args.rvq_level) if is_rvq else ''}\n")
        f.write("=" * 72 + "\n\n")

        for code in codes:
            f.write(f"{label} {code:3d}\n")
            cross = Counter()
            for lang in range(n_lang):
                test_c = role_token_counts["test"][(code, lang)]
                ctrl_c = role_token_counts["ctrl"][(code, lang)]
                all_toks = set(test_c) | set(ctrl_c)
                lo = {t: float(np.log((test_c[t] + 1) / (ctrl_c[t] + 1)))
                      for t in all_toks}
                top = sorted(lo.items(), key=lambda x: -x[1])[:args.top_tokens]
                short = meta.short_codes[lang]
                top_str = []
                for tid, score in top:
                    word = tok.decode([tid]).strip().replace("\n", " ")
                    top_str.append(f"{word!r}({score:+.2f})")
                    if score > 0:
                        cross[word] += score
                f.write(f"  {short}: {', '.join(top_str)}\n")
            top_cross = sorted(cross.items(), key=lambda x: -x[1])[:args.top_tokens]
            f.write(f"  AGGREGATE: {', '.join(f'{w}({s:+.2f})' for w, s in top_cross)}\n\n")

    print(f"Done. {n_codes} codes analyzed.")


if __name__ == "__main__":
    main()
