"""code_prior.py — PHASE9 A5: bits per sentence under a learned prior.

Trains a small causal transformer on the sentences dumped by dump_codes.py and
reports held-out bits per sentence, so the code and the text it encodes can be
compared under the same model and budget:

  --mode codes --depth D   code stacks, levels 0..D-1, slot-major:
                           slot0 L0..L(D-1), slot1 L0.., ...  (64*D tokens,
                           fixed length, so no end token is needed)
  --mode text              the XLM-R ids of the same sentences, scored through
                           EOS. The vocabulary is compacted to the types seen
                           in training; a test token never seen in training
                           costs log2 p(UNK) + log2(#unseen types), so bits stay
                           an honest code length.

Neither stream is told the language. Reports bits/sentence on the opus100
held-out pairs and on FLORES (per language too), against the raw size of the
code (64 x D x log2 k bits).

Usage:
    python code_prior.py --data data/codes_cs3_500k --mode codes --depth 2
    python code_prior.py --data data/codes_cs3_500k --mode text
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.h, self.drop = h, dropout

    def forward(self, x, pad_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.ln2(x))


class CausalLM(nn.Module):
    def __init__(self, vocab, max_len, d=384, layers=6, heads=6, ff=1536, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids):
        x = self.drop(self.emb(ids) + self.pos.weight[: ids.size(1)])
        for b in self.blocks:
            x = b(x)
        return F.linear(self.ln(x), self.emb.weight)   # tied output


# ---------------------------------------------------------------- data views
class CodeStream:
    """Sentence i -> [BOS] + slot-major code tokens; token = level*k + code + 1."""

    def __init__(self, codes, k, depth):
        self.codes, self.k, self.depth = codes, k, depth
        self.vocab = 1 + depth * k
        self.max_len = 1 + codes.shape[1] * depth
        self.level_off = (np.arange(depth) * k + 1).astype(np.int64)

    def batch(self, idx):
        c = self.codes[np.sort(idx)][:, :, : self.depth].astype(np.int64) + self.level_off
        seq = np.concatenate([np.zeros((len(idx), 1), np.int64), c.reshape(len(idx), -1)], 1)
        t = torch.from_numpy(seq)
        return t, torch.ones_like(t[:, 1:], dtype=torch.bool)   # every target counts

    def unseen_bits(self, idx):
        return np.zeros(len(idx))


class TextStream:
    """Sentence i -> compacted XLM-R ids (BOS ... EOS); id 0 = PAD, 1 = UNK."""

    def __init__(self, tokens, offsets, train_idx, n_types_full, max_len):
        self.tokens, self.offsets, self.max_len = tokens, offsets, max_len
        seen = np.zeros(n_types_full, bool)
        for i in train_idx:
            seen[tokens[offsets[i]:offsets[i + 1]]] = True
        self.map = np.full(n_types_full, 1, np.int64)            # unseen -> UNK
        self.map[seen] = np.arange(2, 2 + seen.sum())
        self.vocab = int(2 + seen.sum())
        self.n_unseen_types = int(n_types_full - seen.sum())
        self.seen = seen

    def _rows(self, idx):
        return [self.tokens[self.offsets[i]:self.offsets[i + 1]][: self.max_len] for i in idx]

    def batch(self, idx):
        rows = self._rows(idx)
        T = max(len(r) for r in rows)
        seq = np.zeros((len(rows), T), np.int64)
        for j, r in enumerate(rows):
            seq[j, : len(r)] = self.map[r]
        t = torch.from_numpy(seq)
        return t, t[:, 1:] != 0

    def unseen_bits(self, idx):
        """Extra bits to name which unseen type each UNK was (uniform)."""
        per = math.log2(max(self.n_unseen_types, 1))
        return np.array([(~self.seen[r[1:]]).sum() * per for r in self._rows(idx)])


# ---------------------------------------------------------------- train/eval
@torch.no_grad()
def sentence_bits(model, stream, idx, device, batch):
    model.eval()
    out = np.zeros(len(idx))
    order = np.argsort(idx)
    idx_sorted = np.asarray(idx)[order]
    for s in range(0, len(idx_sorted), batch):
        b = idx_sorted[s:s + batch]
        ids, mask = stream.batch(b)
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(ids[:, :-1])
        nll = F.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")
        bits = (nll * mask).sum(1) / math.log(2)
        out[order[s:s + len(b)]] = bits.cpu().numpy() + stream.unseen_bits(b)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=["codes", "text"], required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=20000, help="opus test sentences scored")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/phase9")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    split = np.load(os.path.join(a.data, "split.npy"))
    lang = np.load(os.path.join(a.data, "lang.npy"))
    train_idx = np.where(split == 0)[0]
    test_idx = np.where(split == 1)[0]
    test_idx = rng.choice(test_idx, min(a.n_eval, len(test_idx)), replace=False)
    flores_idx = np.where(split == 2)[0]

    if a.mode == "codes":
        codes = np.load(os.path.join(a.data, "codes.npy"), mmap_mode="r")
        stream = CodeStream(codes, meta["k"], a.depth)
        name = f"prior_codes_d{a.depth}"
        raw_bits = meta["M"] * a.depth * math.log2(meta["k"])
    else:
        tokens = np.load(os.path.join(a.data, "tokens.npy"), mmap_mode="r")
        offsets = np.load(os.path.join(a.data, "offsets.npy"))
        max_len = int(np.diff(offsets).max())   # no truncation: every token is scored
        stream = TextStream(tokens, offsets, train_idx, meta["vocab_size"], max_len)
        name = "prior_text"
        raw_bits = float("nan")
        print(f"text vocab: {stream.vocab} types seen in train "
              f"({stream.n_unseen_types} unseen, charged uniformly)")

    model = CausalLM(stream.vocab, stream.max_len, d=a.d_model, layers=a.layers,
                     heads=a.d_model // 64, ff=4 * a.d_model).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.emb.weight.numel() + model.pos.weight.numel()
    print(f"{name}: vocab {stream.vocab}, max_len {stream.max_len}, params {n_params / 1e6:.1f}M "
          f"({(n_params - n_emb) / 1e6:.1f}M non-embedding); train {len(train_idx)} sentences")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) *
        0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        ids, mask = stream.batch(rng.choice(train_idx, a.batch_size, replace=False))
        ids, mask = ids.to(a.device), mask.to(a.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            logits = model(ids[:, :-1])
        nll = F.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")
        loss = (nll * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step}  loss {loss.item():.4f} nats/token  "
                  f"{(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            tb = sentence_bits(model, stream, test_idx[:5000], a.device, a.batch_size)
            print(f"  eval step {step}: opus test {tb.mean():.1f} bits/sentence", flush=True)
            log.append((step, float(tb.mean())))

    test_bits = sentence_bits(model, stream, test_idx, a.device, a.batch_size)
    fl_bits = sentence_bits(model, stream, flores_idx, a.device, a.batch_size)
    per_lang = {meta["short_codes"][l]: float(fl_bits[lang[flores_idx] == l].mean())
                for l in np.unique(lang[flores_idx])}
    res = dict(name=name, mode=a.mode, depth=a.depth if a.mode == "codes" else None,
               params=n_params, raw_bits=raw_bits, steps=a.steps,
               opus_test_bits=float(test_bits.mean()), flores_bits=float(fl_bits.mean()),
               flores_bits_per_lang=per_lang, curve=log, data=a.data)
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(res, open(os.path.join(a.outdir, f"{name}.json"), "w"), indent=1)
    print(f"\n{name}: opus test {res['opus_test_bits']:.1f} bits/sentence, "
          f"FLORES {res['flores_bits']:.1f} (raw code {raw_bits:.0f})")
    print("FLORES per language: " + ", ".join(f"{k} {v:.1f}" for k, v in per_lang.items()))


if __name__ == "__main__":
    main()
