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
    ap.add_argument("--model", choices=["mlp", "attn", "attn2", "vqvae", "cehead"],
                    default="mlp",
                    help="mlp = canonical net. attn = bare nn.TransformerEncoder "
                         "(torch's fused attention). attn2 = SAME shape but a "
                         "hand-written attention (explicit Q/K/V + softmax(QKt)V, "
                         "no SDPA) — tests the fix. vqvae = our real model.")
    ap.add_argument("--no-rvq", action="store_true",
                    help="vqvae only: use plain VQ instead of residual VQ.")
    ap.add_argument("--no-tie", action="store_true",
                    help="vqvae only: untie enc/dec/output embeddings.")
    # Capacity probing at REAL dims. The full-vocab logits tensor is
    # (B, T, V) = (32, 63, 250037) ~= 1 GB in f16 regardless of how the CE is
    # computed, so whether it fits at all is a prior question to the CE design.
    ap.add_argument("--vocab", type=int, default=512,
                    help="vqvae only: vocab size (real model: 250037).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="vqvae only: batch size (real model: 32).")
    ap.add_argument("--seq-len", type=int, default=16,
                    help="vqvae only: sequence length (real model: 64).")
    ap.add_argument("--loss", choices=["ce", "logsumexp"], default="ce",
                    help="vqvae only. ce = full cross-entropy via a host-side "
                         "one-hot (only viable at small vocab: the one-hot is "
                         "B*T*V*2 bytes/step, ~1 GB at the real vocab). "
                         "logsumexp = loss is logits.logsumexp(-1).mean(), which "
                         "needs no labels at all — it exercises the full-vocab "
                         "logits tensor and its backward through the tied output "
                         "projection, and nothing else. Use it to ask 'do the "
                         "logits fit?' without entangling the CE design.")
    ap.add_argument("--compile-only", action="store_true",
                    help="Compile without executing — cheap capacity probing.")
    args = ap.parse_args()

    if args.model == "vqvae":
        return run_vqvae(args)
    if args.model == "cehead":
        return run_cehead(args)
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


def run_cehead(args):
    """Probe ONLY the output head + CE for the real (full-vocab) path.

    Does NOT build the VQVAE — `h` comes from the dataloader as a random float
    activation, so this isolates the three primitives the real CE needs and
    nothing else:

      1. token_emb(labels)  — (V,d) embedding lookup by int32 id, giving
         emb[label] for the target-logit term.
      2. logsumexp with a DETACHED max (torch.logsumexp does not lower).
      3. token_weight and the pad mask arriving from the HOST as float (B,T)
         activations rather than as wafer-side tables.

    THE FORWARD INDEX MUST BE A RAW DATALOADER TENSOR. Every F.embedding lowers
    to a cirh.Gather, which a peephole rewrites to `ws_km.onehot(input) @ weight`
    ONLY when the index operand is a load_input. In the working graph the ws_km
    op is literally `ws_km.onehot %input_N {depth=250037}` with zero gathers
    surviving. Any op on the index first (labels.reshape(N), a cast, arithmetic)
    breaks the match; the gather stays `acth.gather` on the weight and fails
    "Unable to insert transfer WGT_HOST->ACT_HOST". So: look up with the raw
    tensor, reshape the RESULT. Width and rank are irrelevant — earlier failures
    at (V,1)/(V,8)/(V,384) were all this one bug.

    THE TABLE MUST BE FROZEN. A frozen embedding's lookup has a no-op backward. A
    TRAINABLE one's backward is a ScatterNd into the weight, which has no WAF
    kernel. token_emb is the frozen XLM-R table, so it is fine. This is why there
    is NO output bias: a trainable per-vocab bias scatters at any width, and it
    cannot be folded into the frozen token_emb (partial-freeze = slice, Class B).
    Reading the target logit out of the full logits instead — (onehot*logits) —
    avoids the scatter but needs a second (N,V) tensor beside logits, and the two
    do not co-fit at real dims ("did not fit in memory"). So the bias is dropped;
    a tied softmax without an output bias is standard.

    Cost: the rewrite materializes a tensor<...x250037xf16> one-hot on the
    wafer, ~1 GB, but it fits (this probe reaches micro-batch 16). What the
    lookup CE saves is ~1 GB/step of HOST->wafer bandwidth vs shipping a
    host-built one-hot.

    CE identity (exact, not an approximation): with a tied projection and no
    output bias, logits[i,label] == h[i]·emb[label], so
        nll = logsumexp(logits) - h·emb[label]
    with no (N,V) intermediate besides logits itself. Verified vs a plain gather
    CE on CPU (§4): CE diff 0.0, grads ~1e-9.
    """
    V, d, T, B = args.vocab, 384, args.seq_len, args.batch_size
    print(f"minimal_probe: model=cehead vocab={V} d={d} B={B} T={T} "
          f"compile_only={args.compile_only}")
    print(f"minimal_probe: logits will be ({B}, {T}, {V}) = "
          f"{B*T*V*2/2**30:.3f} GiB in f16")

    class CEHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = torch.nn.Embedding(V, d)
            self.token_emb.weight.requires_grad = False   # frozen, like the real model
            # Something trainable must remain, or there is no backward at all
            # and the step DCEs to an empty module.
            self.proj = torch.nn.Linear(d, d)

        def forward(self, h, labels, tok_w, mask):
            N = h.size(0) * h.size(1)
            h2d = self.proj(h.reshape(N, -1))
            logits = F.linear(h2d, self.token_emb.weight)          # (N, V)
            # logsumexp with a DETACHED max — torch.logsumexp backprops through
            # amax, whose grad is (x==max)/mask.sum(dim): an ADD-reduce over i1,
            # and ReduceOp only supports f16/f32/bf16/cb16.
            x = logits.float()
            mx = x.amax(dim=-1, keepdim=True).detach()
            lse = mx.squeeze(-1) + torch.log(torch.exp(x - mx).sum(dim=-1))
            # Target logit by LOOKUP on the RAW `labels` input, then reshape the
            # RESULT. The gather->onehot rewrite is a peephole match on
            # Gather(weight, load_input): in the working graph the ws_km op is
            # literally `ws_km.onehot %input_N {depth=250037}`, zero gathers
            # surviving. Reshaping labels BEFORE the lookup breaks the match and
            # leaves an acth.gather that cannot be placed. token_emb is frozen so
            # its lookup backward is a no-op; a TRAINABLE embedding would instead
            # emit a ScatterNd into the weight (no WAF kernel) — which is why the
            # output bias cannot be a trainable per-vocab table and is dropped.
            emb_t = self.token_emb(labels).reshape(N, -1)          # (N, d)
            logit_t = (h2d * emb_t).sum(dim=-1)
            nll = lse - logit_t.float()
            # token weight comes from the HOST (frozen constant, host knows the
            # labels) — no wafer-side (V,1) table.
            w = tok_w.reshape(N).float() * mask.reshape(N).float()
            return (nll * w).sum() / w.sum().clamp(min=1.0)

    model = CEHead()
    model.train()

    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe-cehead"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", compile_only=args.compile_only,
                              cluster_config=cluster_config)
    compiled_model = cstorch.compile(model, backend)
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Frozen per-token weights, as in the real pipeline. Looked up HOST-side.
    token_weight = torch.rand(V) + 0.5

    def input_fn():
        def gen():
            for _ in range(7):
                lab = torch.randint(4, V, (B, T))
                yield {
                    "h": torch.randn(B, T, d),
                    "labels": lab.to(torch.int32),      # int32: no i64 on wafer
                    "tok_w": token_weight[lab].to(torch.float16),   # host lookup
                    "mask": (lab != 1).to(torch.float16),
                }
        return gen()
    dataloader = cstorch.utils.data.DataLoader(input_fn)

    @cstorch.trace
    def training_step(batch):
        loss = compiled_model(batch["h"], batch["labels"],
                              batch["tok_w"], batch["mask"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @cstorch.step_closure
    def fetch(step, loss):
        print(f"step {step}: loss={loss.item():.4f}")

    print("minimal_probe: starting CEHEAD loop...")
    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=5)
    step = 0
    for batch in executor:
        fetch(step, training_step(batch))
        step += 1
    print("minimal_probe: DONE — CEHEAD compiled.")


def run_vqvae(args):
    """Run OUR VQVAE (small synthetic config) through the same known-good loop
    the MLP just passed. Isolates the model as the empty-CIRH cause and, via
    --no-rvq / --no-tie, which component. Uses SGD + torch DataLoader (both
    already proven fine) so the model is the only new variable."""
    import sys
    sys.path.insert(0, REPO_ROOT)
    from vqvae.model import VQVAE

    vocab, T, B = args.vocab, args.seq_len, args.batch_size
    d_model, n_langs = 128, 4
    emb = torch.randn(vocab, d_model)
    model = VQVAE(
        vocab_size=vocab, n_langs=n_langs, d_model=d_model, d_code=64, k=32,
        m_max=16, n_enc_layers=2, n_dec_layers=2, n_heads=4, d_ff=256,
        beta_commit=0.25, pad_token_id=1, embedding_table=emb,
        use_semantic_head=False, use_rvq=not args.no_rvq, n_rvq_levels=4,
        decoder_type="ar", tie_embeddings=not args.no_tie,
        # Exact positional-table sizes: src_ids and dec_in are both (B, T) here.
        src_len=T, tgt_len=T,
    )
    model.train()
    print(f"minimal_probe: model=vqvae rvq={not args.no_rvq} tie={not args.no_tie} "
          f"vocab={vocab} B={B} T={T} loss={args.loss} "
          f"compile_only={args.compile_only}")
    print(f"minimal_probe: logits will be ({B}, {T}, {vocab}) = "
          f"{B*T*vocab*2/2**30:.3f} GiB in f16")

    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe-vqvae"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", compile_only=args.compile_only,
                              cluster_config=cluster_config)
    compiled_model = cstorch.compile(model, backend)
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def input_fn():
        def gen():
            for _ in range(7):
                # Host-side AR shift: yield dec_in and labels pre-shifted so the
                # graph never slices (cstorch's slice_filter kernel asserts).
                tgt = torch.randint(4, vocab, (B, T + 1))
                b = {
                    # int32, never int64. These ids feed BOTH the embedding
                    # lookup (which lowers to ws_km.onehot and would tolerate
                    # i64) and the pad-mask compare `ids == pad_token_id`, which
                    # needs the raw ids on-wafer — and the wafer has no 64-bit
                    # datapath. F.embedding accepts int32 indices, so this costs
                    # nothing. The embedding path masks the compare path, so the
                    # i64 only surfaces once other i64 tensors are removed.
                    "src_ids": torch.randint(4, vocab, (B, T), dtype=torch.int32),
                    "dec_in": tgt[:, :-1].contiguous().to(torch.int32),
                    # Labels arrive ALREADY ONE-HOT as a float activation, not as
                    # ids. Selecting the target logit on-wafer needs either a
                    # gather (i64 index -> no valid layout) or a compare against
                    # arange (materializes an (N,V) i64/i32 tensor plus a large
                    # fan of intermediates through the backward -> did not fit in
                    # memory). Doing it host-side makes the CE a plain multiply.
                    # The pad mask comes over too: deriving it on-wafer as
                    # onehot[:, pad_id] would be a slice, and slice_filter asserts.
                    "tgt_lang_id": torch.randint(0, n_langs, (B,), dtype=torch.int32),
                }
                if args.loss == "ce":
                    # B*T*V*2 bytes per step — fine at vocab 512, ~1 GB at the
                    # real vocab, which is why --loss logsumexp exists.
                    b["labels_onehot"] = F.one_hot(tgt[:, 1:], vocab).to(torch.float16)
                    b["label_mask"] = (tgt[:, 1:] != 1).to(torch.float16)
                yield b
        return gen()
    dataloader = cstorch.utils.data.DataLoader(input_fn)

    @cstorch.trace
    def training_step(batch):
        out = compiled_model(batch["src_ids"], batch["dec_in"], batch["tgt_lang_id"],
                             pre_shifted=True)
        logits = out["logits"]
        # Explicit CE — the fused F.cross_entropy op does not lower on cstorch
        # (wgth.cast placement error over the vocab dim). With the labels already
        # one-hot from the host (see input_fn), picking out the target logit is
        # just a multiply-and-sum: no gather, no compare, no integer tensor ever
        # reaches the wafer.
        l2d = logits.reshape(-1, logits.size(-1))
        if args.loss == "logsumexp":
            # Capacity probe only, NOT a real objective: no labels, no CE. This
            # exercises exactly the full-vocab logits tensor and its backward
            # through the tied output projection, so a failure here is a
            # statement about logits capacity and nothing else.
            #
            # NOT torch.logsumexp: its backward runs through amax, whose gradient
            # distributes to the max positions via (x == max) / mask.sum(dim) —
            # an ADD-reduce over an i1 tensor, and ReduceOp only supports
            # f16/f32/bf16/cb16. Detaching the max gives the identical value and
            # identical gradient (the max term cancels analytically) while
            # keeping amax out of the backward graph entirely. The real CE needs
            # this same treatment.
            x = l2d.float()
            mx = x.amax(dim=-1, keepdim=True).detach()
            recon = (mx.squeeze(-1) + torch.log(torch.exp(x - mx).sum(dim=-1))).mean()
        else:
            oh2d = batch["labels_onehot"].reshape(-1, logits.size(-1)).to(l2d.dtype)
            logp = torch.log_softmax(l2d, dim=-1)
            nll = -(logp * oh2d).sum(dim=1)
            m = batch["label_mask"].reshape(-1).to(logp.dtype)
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
