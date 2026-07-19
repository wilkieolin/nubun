"""Minimal standalone cstorch training loop — DIAGNOSTIC ONLY.

This is the canonical Cerebras custom-loop example (a 2-layer MLP) with synthetic
data, run as a *standalone script* (not cszoo fit / the modelzoo Trainer). It
exists to answer one question:

    Does a raw standalone cstorch loop compile+execute on this CS-3 cluster,
    or does only the modelzoo Trainer path work?

Our real model (train_cstorch.py) fails with "Cannot compile empty CIRH module".
The modelzoo gpt3 example (cszoo fit) compiles fine on the same venv+cluster.
The difference is Trainer vs raw loop. If THIS minimal raw loop also fails empty,
the problem is the standalone-loop mechanism (nothing model-specific). If it
compiles+runs, the problem is specific to our model and we bisect that.

Nothing here imports from vqvae/ — it depends only on cstorch + torch.

Run via:  bash cerebras/submit_alcf.sh minprobe        # real torch DataLoader
          bash cerebras/submit_alcf.sh minprobe-raw    # raw generator (train_cstorch style)

--raw-input swaps the proper torch DataLoader for a raw generator yielding
PRE-BATCHED tuples — exactly how train_cstorch feeds data. Everything else stays
identical (same MLP). If plain minprobe compiles but minprobe-raw fails empty,
the input pipeline (raw generator vs torch DataLoader) is the culprit.
"""
import argparse
import os

import torch
import torch.nn.functional as F

import cerebras.pytorch as cstorch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-input", action="store_true",
                    help="Feed a raw generator of pre-batched tuples (as "
                         "train_cstorch does) instead of a torch DataLoader.")
    ap.add_argument("--opt", choices=["sgd", "adamw"], default="sgd",
                    help="sgd = constant lr (canonical). adamw = AdamW + the "
                         "train_cstorch warmup(from 0)+cosine schedule, stepped "
                         "in-loop. Tests whether LR=0 on the compiled first step "
                         "zeros the weight updates and empties the graph.")
    ap.add_argument("--model", choices=["mlp", "attn", "attn2", "vqvae"], default="mlp",
                    help="mlp = canonical net. attn = bare nn.TransformerEncoder "
                         "(torch's fused attention). attn2 = SAME shape but a "
                         "hand-written attention (explicit Q/K/V + softmax(QKt)V, "
                         "no SDPA) — tests the fix. vqvae = our real model.")
    ap.add_argument("--no-rvq", action="store_true",
                    help="vqvae only: use plain VQ instead of residual VQ.")
    ap.add_argument("--no-tie", action="store_true",
                    help="vqvae only: untie enc/dec/output embeddings.")
    args = ap.parse_args()

    if args.model == "vqvae":
        return run_vqvae(args)
    if args.model == "attn":
        return run_attn(args)
    if args.model == "attn2":
        return run_attn2(args)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 256)
            self.fc2 = torch.nn.Linear(256, 10)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)          # raw logits (CrossEntropyLoss expects them)

    # Same cluster-config shape train_cstorch uses (mgmt/creds come from the
    # ALCF user node's /opt/cerebras/config_v2).
    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", cluster_config=cluster_config)

    model = MLP()
    compiled_model = cstorch.compile(model, backend)

    lr_scheduler = None
    if args.opt == "sgd":
        optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        # Mirror train_cstorch exactly: AdamW + LinearLR warmup from 0.0 -> lr,
        # then cosine. Warmup starting at 0 means step 1 runs at LR=0.
        optimizer = cstorch.optim.AdamW(model.parameters(), lr=3e-4)
        lr = 3e-4
        warmup = cstorch.optim.lr_scheduler.LinearLR(
            optimizer, initial_learning_rate=0.0, end_learning_rate=lr, total_iters=500)
        cosine = cstorch.optim.lr_scheduler.CosineDecayLR(
            optimizer, initial_learning_rate=lr, end_learning_rate=lr * 0.1,
            total_iters=1000)
        lr_scheduler = cstorch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[500])
    print(f"minimal_probe: optimizer = {args.opt}")

    def input_fn_dl():
        # A real torch DataLoader (as the canonical example uses), over a
        # synthetic in-memory dataset — no downloads, no data/ files.
        n = 640
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, 10, (n,), dtype=torch.int32)
        ds = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    def input_fn_raw():
        # Raw generator yielding PRE-BATCHED tuples — exactly train_cstorch's
        # style. The only thing that differs from input_fn_dl.
        def gen():
            for _ in range(7):
                x = torch.randn(64, 1, 28, 28)
                y = torch.randint(0, 10, (64,), dtype=torch.int32)
                yield (x, y)
        return gen()

    input_fn = input_fn_raw if args.raw_input else input_fn_dl
    print(f"minimal_probe: input = {'RAW generator' if args.raw_input else 'torch DataLoader'}")
    dataloader = cstorch.utils.data.DataLoader(input_fn)
    loss_fn = torch.nn.CrossEntropyLoss()

    @cstorch.trace
    def training_step(inputs, targets):
        outputs = compiled_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return loss

    @cstorch.step_closure
    def print_loss(loss, step):
        print(f"step {step}: loss={loss.item():.4f}")

    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=5)
    model.train()
    step = 0
    print("minimal_probe: starting standalone cstorch loop...")
    for inputs, targets in executor:
        loss = training_step(inputs, targets)
        print_loss(loss, step)
        step += 1
    print("minimal_probe: DONE — standalone loop compiled and executed.")


def run_attn(args):
    """Bare nn.TransformerEncoder — the ONE thing the VQVAE has that the MLP
    doesn't (our encoder/decoder use torch's built-in transformer + MHA, which
    dispatch to scaled_dot_product_attention). Same known-good loop otherwise.
    If this fails empty, torch's transformer is what won't lower on cstorch."""
    vocab, d_model, T, B, n_cls = 512, 128, 16, 8, 10

    class AttnNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d_model)
            layer = torch.nn.TransformerEncoderLayer(
                d_model, nhead=4, dim_feedforward=256, dropout=0.0,
                batch_first=True, norm_first=True)      # matches vqvae/encoder.py
            self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.head = torch.nn.Linear(d_model, n_cls)

        def forward(self, ids):
            h = self.enc(self.emb(ids))
            return self.head(h.mean(dim=1))

    model = AttnNet()
    model.train()
    print("minimal_probe: model=attn (bare nn.TransformerEncoder)")

    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe-attn"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", cluster_config=cluster_config)
    compiled_model = cstorch.compile(model, backend)
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def input_fn():
        def gen():
            for _ in range(7):
                yield (torch.randint(0, vocab, (B, T)),
                       torch.randint(0, n_cls, (B,), dtype=torch.int32))
        return gen()
    dataloader = cstorch.utils.data.DataLoader(input_fn)
    loss_fn = torch.nn.CrossEntropyLoss()

    @cstorch.trace
    def training_step(ids, targets):
        loss = loss_fn(compiled_model(ids), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @cstorch.step_closure
    def print_loss(loss, step):
        print(f"step {step}: loss={loss.item():.4f}")

    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=5)
    step = 0
    print("minimal_probe: starting attn loop...")
    for ids, targets in executor:
        loss = training_step(ids, targets)
        print_loss(loss, step)
        step += 1
    print("minimal_probe: DONE — nn.TransformerEncoder compiled and executed.")


def run_attn2(args):
    """Same shape as run_attn, but attention written with EXPLICIT ops (no
    nn.MultiheadAttention / scaled_dot_product_attention). If this compiles
    while run_attn hits the aamatmul assertion, the fix is to hand-roll
    attention in vqvae/encoder.py + decoder.py."""
    vocab, d_model, T, B, n_cls, H = 512, 128, 16, 8, 10, 4
    dh = d_model // H

    class ManualMHA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(d_model, d_model)
            self.k = torch.nn.Linear(d_model, d_model)
            self.v = torch.nn.Linear(d_model, d_model)
            self.o = torch.nn.Linear(d_model, d_model)

        def forward(self, x):
            b, t, _ = x.shape
            q = self.q(x).view(b, t, H, dh).transpose(1, 2)   # (b,H,t,dh)
            k = self.k(x).view(b, t, H, dh).transpose(1, 2)
            v = self.v(x).view(b, t, H, dh).transpose(1, 2)
            att = torch.matmul(q, k.transpose(-2, -1)) / (dh ** 0.5)
            att = torch.softmax(att, dim=-1)
            out = torch.matmul(att, v).transpose(1, 2).reshape(b, t, d_model)
            return self.o(out)

    class Block(torch.nn.Module):        # pre-norm, matches norm_first=True
        def __init__(self):
            super().__init__()
            self.n1 = torch.nn.LayerNorm(d_model)
            self.attn = ManualMHA()
            self.n2 = torch.nn.LayerNorm(d_model)
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(d_model, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, d_model))

        def forward(self, x):
            x = x + self.attn(self.n1(x))
            x = x + self.ff(self.n2(x))
            return x

    class AttnNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d_model)
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.head = torch.nn.Linear(d_model, n_cls)

        def forward(self, ids):
            h = self.emb(ids)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h.mean(dim=1))

    model = AttnNet()
    model.train()
    print("minimal_probe: model=attn2 (hand-written attention, no SDPA)")

    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe-attn2"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", cluster_config=cluster_config)
    compiled_model = cstorch.compile(model, backend)
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def input_fn():
        def gen():
            for _ in range(7):
                yield (torch.randint(0, vocab, (B, T)),
                       torch.randint(0, n_cls, (B,), dtype=torch.int32))
        return gen()
    dataloader = cstorch.utils.data.DataLoader(input_fn)
    loss_fn = torch.nn.CrossEntropyLoss()

    @cstorch.trace
    def training_step(ids, targets):
        loss = loss_fn(compiled_model(ids), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @cstorch.step_closure
    def print_loss(loss, step):
        print(f"step {step}: loss={loss.item():.4f}")

    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=5)
    step = 0
    print("minimal_probe: starting attn2 (manual) loop...")
    for ids, targets in executor:
        loss = training_step(ids, targets)
        print_loss(loss, step)
        step += 1
    print("minimal_probe: DONE — manual attention compiled and executed.")


def run_vqvae(args):
    """Run OUR VQVAE (small synthetic config) through the same known-good loop
    the MLP just passed. Isolates the model as the empty-CIRH cause and, via
    --no-rvq / --no-tie, which component. Uses SGD + torch DataLoader (both
    already proven fine) so the model is the only new variable."""
    import sys
    sys.path.insert(0, REPO_ROOT)
    from vqvae.model import VQVAE

    vocab, d_model, T, B, n_langs = 512, 128, 16, 8, 4
    emb = torch.randn(vocab, d_model)
    model = VQVAE(
        vocab_size=vocab, n_langs=n_langs, d_model=d_model, d_code=64, k=32,
        m_max=16, n_enc_layers=2, n_dec_layers=2, n_heads=4, d_ff=256,
        beta_commit=0.25, pad_token_id=1, embedding_table=emb,
        use_semantic_head=False, use_rvq=not args.no_rvq, n_rvq_levels=4,
        decoder_type="ar", tie_embeddings=not args.no_tie,
    )
    model.train()
    print(f"minimal_probe: model=vqvae rvq={not args.no_rvq} tie={not args.no_tie}")

    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe-vqvae"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", cluster_config=cluster_config)
    compiled_model = cstorch.compile(model, backend)
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def input_fn():
        def gen():
            for _ in range(7):
                # Host-side AR shift: yield dec_in and labels pre-shifted so the
                # graph never slices (cstorch's slice_filter kernel asserts).
                tgt = torch.randint(4, vocab, (B, T + 1))
                yield {
                    "src_ids": torch.randint(4, vocab, (B, T)),
                    "dec_in": tgt[:, :-1].contiguous(),   # (B, T)
                    "labels": tgt[:, 1:].contiguous(),    # (B, T)
                    "tgt_lang_id": torch.randint(0, n_langs, (B,)),
                }
        return gen()
    dataloader = cstorch.utils.data.DataLoader(input_fn)

    @cstorch.trace
    def training_step(batch):
        out = compiled_model(batch["src_ids"], batch["dec_in"], batch["tgt_lang_id"],
                             pre_shifted=True)
        logits = out["logits"]
        # Explicit CE (log_softmax + gather) — the fused F.cross_entropy op does
        # not lower on cstorch (wgth.cast placement error over the vocab dim).
        l2d = logits.reshape(-1, logits.size(-1))
        t1d = batch["labels"].reshape(-1)
        logp = torch.log_softmax(l2d, dim=-1)
        nll = -logp.gather(1, t1d.unsqueeze(1)).squeeze(1)
        m = (t1d != 1).to(logp.dtype)
        recon = (nll * m).sum() / m.sum().clamp(min=1.0)
        loss = recon + out["vq_losses"]["commit"] + out["vq_losses"]["codebook"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @cstorch.step_closure
    def print_loss(loss, step):
        print(f"step {step}: loss={loss.item():.4f}")

    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=5)
    step = 0
    print("minimal_probe: starting VQVAE loop...")
    for batch in executor:
        loss = training_step(batch)
        print_loss(loss, step)
        step += 1
    print("minimal_probe: DONE — VQVAE compiled and executed.")


if __name__ == "__main__":
    main()
