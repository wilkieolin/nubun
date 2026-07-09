"""roundtrip_eval.py — Phase 6 headline metric: round-trip meaning preservation.

The Phase 5 diagnostics were all TEACHER-FORCED: the decoder is fed the gold
prefix, so it can ignore the codes and still look accurate. This script measures
the regime the thesis actually lives in:

    src --encode-->quantize--> codes --FREE GENERATE(tgt lang)--> y_hat
    cos( MiniLM(y_hat), MiniLM(src) )            = round-trip meaning
    cos_real - cos_shuffle                       = MEANING shuffle-gap  (the KPI)

MiniLM (paraphrase-multilingual-MiniLM-L12-v2) is XLM-R based and shares our
vocabulary, so generated ids re-embed directly; it aligns translations
cross-lingually, so a faithful reconstruction in any target language scores high
against the source.

Conditions (eval-time baselines — no training needed):
  real    : the model's own quantized codes
  shuffle : each target generated from ANOTHER sentence's codes (roll batch by 1)
  zero    : bottleneck zeroed  == the B0 no-code lower bound at eval time

Interpretation:
  real >> shuffle ~ zero   -> codes carry meaning under free generation (thesis OK)
  real ~ shuffle ~ zero    -> decoder ignores codes even when generating freely
                              (posterior collapse is real, not a teacher-forcing
                               artifact) -> NAT / bottleneck pivot needed
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from vqvae.data import combined_split, load_corpus
from vqvae.model import VQVAE


def compute_target_len(src_ids, pad, ratio, slack, m_max):
    src_lens = (src_ids != pad).sum(dim=1).float()
    return (src_lens * ratio + slack).ceil().long().clamp(min=1, max=m_max)


@torch.no_grad()
def embed(st_model, ids, pad):
    """Mean-pool MiniLM last_hidden_state over non-pad tokens. (B, H)."""
    attn = (ids != pad).long()
    out = st_model(input_ids=ids, attention_mask=attn).last_hidden_state
    m = attn.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1.0)


@torch.no_grad()
def bottleneck(model, src_t, pad, ratio, slack):
    """Encode+quantize+length-cap, mirroring VQVAE.forward. Returns z_q, mem_mask."""
    z_e = model.encoder(src_t)
    z_q, indices, _, _ = model.quantizer(z_e)
    if ratio > 0:
        tl = compute_target_len(src_t, pad, ratio, slack, model.encoder.m_max)
        indices = model.quantizer.force_stop_at(indices, tl)
        z_q_forced = model.quantizer.codebook[indices]
        mb = (indices != model.quantizer.stop_index).unsqueeze(-1)
        z_q = torch.where(mb, z_q, z_q_forced)
    if model.use_stop_mask:
        mem_mask = model.quantizer.get_stop_mask(indices)
    else:
        mem_mask = torch.ones_like(indices, dtype=torch.bool)
    return z_q, mem_mask


@torch.no_grad()
def generate(model, z_q, mem_mask, lang_t, bos, eos, pad, max_len, device):
    """Greedy autoregressive free generation. Returns ids (B, <=max_len)."""
    B = z_q.size(0)
    ids = torch.full((B, 1), bos, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        logits = model.decoder(z_q, mem_mask, ids, lang_t)   # (B, t, V)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        finished = finished | (nxt == eos)
        if bool(finished.all()):
            break
    return ids


@torch.no_grad()
def gold_ceiling(st_model, val_ids, langs, meta, device, batch_size):
    """Cosine between src and the GOLD parallel target — the translation ceiling
    that calibrates the (otherwise uninterpretable) free-generation cosines."""
    pad = meta.pad_token_id
    agg = {"same": [0.0, 0], "cross": [0.0, 0]}
    n = val_ids.shape[0]
    for s in langs:
        for t in langs:
            bucket = "same" if s == t else "cross"
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                def pack(l):
                    seqs = [val_ids[i, l] for i in range(start, end)]
                    T = max(len(x) for x in seqs)
                    bt = torch.full((len(seqs), T), pad, dtype=torch.long)
                    for i, x in enumerate(seqs):
                        bt[i, :len(x)] = torch.as_tensor(x)
                    return bt.to(device)
                e_s = F.normalize(embed(st_model, pack(s), pad).float(), dim=-1)
                e_t = F.normalize(embed(st_model, pack(t), pad).float(), dim=-1)
                cos = (e_s * e_t).sum(-1)
                agg[bucket][0] += float(cos.sum())
                agg[bucket][1] += int(cos.numel())
    return {b: (agg[b][0] / agg[b][1] if agg[b][1] else float("nan"))
            for b in ("same", "cross")}


@torch.no_grad()
def roundtrip_condition(model, st_model, val_ids, langs, meta, device,
                        ratio, slack, cond, max_gen_len, batch_size):
    """Mean round-trip cosine for one condition, split same/cross lang."""
    pad, bos, eos = meta.pad_token_id, meta.bos_token_id, meta.eos_token_id
    agg = {"same": [0.0, 0], "cross": [0.0, 0]}   # sum_cos, count
    n = val_ids.shape[0]
    for s in langs:
        for t in langs:
            bucket = "same" if s == t else "cross"
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                if end - start < 2:
                    continue
                src = [val_ids[i, s] for i in range(start, end)]
                Ts = max(len(x) for x in src)
                src_t = torch.full((len(src), Ts), pad, dtype=torch.long)
                for i, x in enumerate(src):
                    src_t[i, :len(x)] = torch.as_tensor(x)
                src_t = src_t.to(device)
                lang_t = torch.full((src_t.size(0),), t, dtype=torch.long, device=device)

                z_q, mem_mask = bottleneck(model, src_t, pad, ratio, slack)
                if cond == "zero":
                    z_q = torch.zeros_like(z_q)
                elif cond == "shuffle":
                    perm = torch.roll(torch.arange(z_q.size(0), device=device), 1)
                    z_q, mem_mask = z_q[perm], mem_mask[perm]
                elif cond != "real":
                    raise ValueError(cond)

                gen = generate(model, z_q, mem_mask, lang_t, bos, eos, pad,
                               max_gen_len, device)
                e_gen = F.normalize(embed(st_model, gen, pad).float(), dim=-1)
                e_src = F.normalize(embed(st_model, src_t, pad).float(), dim=-1)
                cos = (e_gen * e_src).sum(-1)                 # (B,)
                agg[bucket][0] += float(cos.sum())
                agg[bucket][1] += int(cos.numel())
    out = {}
    for b in ("same", "cross"):
        out[b] = agg[b][0] / agg[b][1] if agg[b][1] else float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", default="data/parallel_corpus.npz")
    p.add_argument("--embedding-table", default="data/embedding_table.pt")
    p.add_argument("--semantic-model",
                   default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--langs", default="en,zh,es,fr,ar,ru")
    p.add_argument("--combine-splits", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-gen-len", type=int, default=48)
    p.add_argument("--conditions", default="real,shuffle,zero")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"  trained for {ckpt['step']} steps")

    dev_ids, devtest_ids, dev_lens, devtest_lens, meta = load_corpus(args.corpus)
    if args.combine_splits:
        _, val_ids, _, _ = combined_split(dev_ids, devtest_ids, dev_lens,
                                          devtest_lens, args.val_fraction, args.seed)
    else:
        val_ids = devtest_ids
    print(f"  val set: {val_ids.shape}")

    emb = torch.load(args.embedding_table, map_location="cpu")
    model = VQVAE(
        vocab_size=emb.shape[0], n_langs=len(meta.short_codes),
        d_model=cfg["d_model"], d_code=cfg["d_code"], k=cfg["k"], m_max=cfg["m_max"],
        n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        beta_commit=cfg["beta_commit"], pad_token_id=meta.pad_token_id,
        embedding_table=emb, use_stop_mask=cfg["use_stop_mask"],
        use_ema=cfg.get("use_ema", False), ema_decay=cfg.get("ema_decay", 0.99),
        use_semantic_head=cfg.get("use_semantic_head", False),
        use_length_head=cfg.get("use_length_head", False),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(args.device).eval()

    ratio = float(cfg.get("compression_ratio", 0.0) or 0.0)
    slack = int(cfg.get("length_slack", 4))

    from transformers import AutoModel
    print(f"Loading semantic encoder: {args.semantic_model}")
    st = AutoModel.from_pretrained(args.semantic_model).eval().to(args.device)
    for prm in st.parameters():
        prm.requires_grad = False

    langs = [meta.short_codes.index(c) for c in args.langs.split(",")]
    conds = args.conditions.split(",")

    gc = gold_ceiling(st, val_ids, langs, meta, args.device, args.batch_size)
    print(f"\nGOLD ceiling (src vs gold parallel target):")
    print(f"  same={gc['same']:.4f}  cross={gc['cross']:.4f}   "
          f"(same=identity~1.0 sanity; cross=translation ceiling)")

    print(f"\nFree-generation round-trip meaning (langs={args.langs}, "
          f"max_gen_len={args.max_gen_len})")
    res = {}
    for c in conds:
        res[c] = roundtrip_condition(model, st, val_ids, langs, meta, args.device,
                                     ratio, slack, c, args.max_gen_len, args.batch_size)
        print(f"  {c:8s}  same={res[c]['same']:.4f}  cross={res[c]['cross']:.4f}")

    if "real" in res:
        base = "shuffle" if "shuffle" in res else ("zero" if "zero" in res else None)
        if base:
            print(f"\nMEANING shuffle-gap (real - {base}):")
            print(f"  same : {res['real']['same']  - res[base]['same']:+.4f}")
            print(f"  cross: {res['real']['cross'] - res[base]['cross']:+.4f}")
            print("  (KPI. >= ~0.10 => codes carry meaning under free generation;")
            print("   ~0 => posterior collapse is real, not a teacher-forcing artifact.)")


if __name__ == "__main__":
    main()
