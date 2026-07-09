"""Top-level VQ-VAE assembly + smoke test."""

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import Decoder
from .encoder import Encoder
from .nat_decoder import NATDecoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_langs: int = 10,
        d_model: int = 384,
        d_code: int = 256,
        k: int = 256,
        m_max: int = 64,
        n_enc_layers: int = 4,
        n_dec_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        beta_commit: float = 0.25,
        pad_token_id: int = 1,
        embedding_table: torch.Tensor | None = None,
        use_stop_mask: bool = False,  # enabled in M6
        use_ema: bool = False,         # enabled in M4
        ema_decay: float = 0.99,
        dead_threshold: float = 0.01,
        use_semantic_head: bool = False,  # Phase 5: semantic-target loss
        d_semantic: int = 384,            # dim of frozen sentence-embedding target
        use_length_head: bool = False,    # Phase 5c: predict length (stop fix / NAT prereq)
        no_vq: bool = False,              # Phase 6 B1: skip quantization (continuous z_e -> decoder)
        no_code: bool = False,            # Phase 6 B0: zero the bottleneck (LM lower bound)
        decoder_type: str = "ar",         # Phase 6 B3: "ar" | "nat" (non-autoregressive)
    ):
        super().__init__()
        self.use_stop_mask = use_stop_mask
        self.use_semantic_head = use_semantic_head
        self.use_length_head = use_length_head
        self.no_vq = no_vq
        self.no_code = no_code
        self.decoder_type = decoder_type
        self.pad_token_id = pad_token_id
        self.encoder = Encoder(
            vocab_size=vocab_size, d_model=d_model, d_code=d_code,
            n_enc_layers=n_enc_layers, n_heads=n_heads, d_ff=d_ff,
            m_max=m_max, dropout=dropout, pad_token_id=pad_token_id,
            embedding_table=embedding_table,
        )
        self.quantizer = VectorQuantizer(
            k=k, d_code=d_code, beta_commit=beta_commit,
            use_ema=use_ema, ema_decay=ema_decay,
            dead_threshold=dead_threshold)
        if decoder_type == "nat":
            self.decoder = NATDecoder(
                vocab_size=vocab_size, n_langs=n_langs, d_model=d_model,
                d_code=d_code, n_dec_layers=n_dec_layers, n_heads=n_heads,
                d_ff=d_ff, dropout=dropout, pad_token_id=pad_token_id,
                embedding_table=embedding_table,
            )
        else:
            self.decoder = Decoder(
                vocab_size=vocab_size, n_langs=n_langs, d_model=d_model,
                d_code=d_code, n_dec_layers=n_dec_layers, n_heads=n_heads,
                d_ff=d_ff, dropout=dropout, pad_token_id=pad_token_id,
                embedding_table=embedding_table,
            )
        # Phase 5: project the (masked-mean-pooled) quantized bottleneck to the
        # frozen sentence-embedding space, so a cosine loss can pressure the
        # codes toward meaning rather than high-frequency token boilerplate.
        if use_semantic_head:
            self.semantic_head = nn.Sequential(
                nn.Linear(d_code, d_code),
                nn.GELU(),
                nn.Linear(d_code, d_semantic),
            )
        # Phase 5c: predict bottleneck length from the pooled codes. Fixes the
        # (untrained) self-stop and gives content-determined length; also the
        # prerequisite for a future non-autoregressive decoder.
        if use_length_head:
            self.length_head = nn.Sequential(
                nn.Linear(d_code, d_code),
                nn.GELU(),
                nn.Linear(d_code, 1),
            )

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_lang_id: torch.Tensor,
        target_len: torch.Tensor | None = None,
        word_dropout: float = 0.0,
        mask_token_id: int = 3,
    ) -> dict:
        """If target_len is provided (per-example, B-shaped), positions
        >= target_len[i] are forced to <stop> after quantization. Used
        with the M2 hard length cap."""
        z_e = self.encoder(src_ids)
        if self.no_vq:
            # Phase 6 B1 upper bound: continuous bottleneck, no quantization, no
            # length cap / stop mask (uses all m_max slots). Isolates the cost of
            # quantization vs the decoder's ability to translate at all.
            z_q = z_e
            indices = torch.zeros(z_e.shape[:2], dtype=torch.long, device=z_e.device)
            zero = z_e.new_zeros(())
            vq_losses = {"commit": zero, "codebook": zero}
            usage = torch.zeros(self.quantizer.k, device=z_e.device)
            mem_mask = torch.ones_like(indices, dtype=torch.bool)
        else:
            z_q, indices, vq_losses, usage = self.quantizer(z_e)
            if target_len is not None:
                indices = self.quantizer.force_stop_at(indices, target_len)
                # When we force stop, the quantized vectors at the forced positions
                # should be the stop codebook entry, not the encoder's free choice.
                # Look them up:
                z_q_forced = self.quantizer.codebook[indices]
                # Preserve the straight-through path on positions before the cap
                mask_before = (indices != self.quantizer.stop_index).unsqueeze(-1)
                z_q = torch.where(mask_before, z_q, z_q_forced)
            if self.use_stop_mask:
                mem_mask = self.quantizer.get_stop_mask(indices)
            else:
                mem_mask = torch.ones_like(indices, dtype=torch.bool)

        # Phase 6 B0 lower bound: zero the bottleneck content (decoder sees only
        # the lang tag + zeroed memory) -> what the AR decoder does with 0 bits.
        if self.no_code:
            z_q = torch.zeros_like(z_q)

        # Phase 5/5c: masked-mean-pool the quantized bottleneck for the auxiliary
        # heads. Gradients flow through z_q (straight-through) into encoder + codebook.
        sem_pred = None
        len_pred = None
        if self.use_semantic_head or self.use_length_head:
            mask_f = mem_mask.unsqueeze(-1).to(z_q.dtype)          # (B, M, 1)
            pooled = (z_q * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            if self.use_semantic_head:
                sem_pred = self.semantic_head(pooled)              # (B, d_semantic)
            if self.use_length_head:
                len_pred = self.length_head(pooled).squeeze(-1)    # (B,)

        if self.decoder_type == "nat":
            # Phase 6 B3: non-autoregressive. No teacher-forced tokens — the decoder
            # predicts all positions in parallel from the codes. Output grid aligns
            # to the target body (drop BOS); train length = gold non-pad count.
            tgt_body = tgt_ids[:, 1:]
            w_full = tgt_body.shape[1]
            out_len = (tgt_body != self.pad_token_id).sum(dim=1).clamp(min=1)
            logits = self.decoder(z_q, mem_mask, out_len, tgt_lang_id)  # (B, T, V), T=out_len.max()
            if logits.shape[1] < w_full:
                # Pad time dim to full target width so the existing loss alignment
                # holds; padded positions map to pad targets and are ignored by CE.
                logits = F.pad(logits, (0, 0, 0, w_full - logits.shape[1]))
        else:
            # Phase 5c: word dropout — randomly replace teacher-forced decoder input
            # tokens with <unk> so the decoder can't lean on the gold prefix and must
            # read the codes (directly attacks the measured posterior collapse).
            dec_in = tgt_ids[:, :-1]
            if self.training and word_dropout > 0.0:
                drop = (torch.rand_like(dec_in, dtype=torch.float) < word_dropout) & \
                       (dec_in != self.pad_token_id)
                dec_in = torch.where(drop, torch.full_like(dec_in, mask_token_id), dec_in)
            logits = self.decoder(z_q, mem_mask, dec_in, tgt_lang_id)
        return {
            "logits": logits,
            "indices": indices,
            "z_e": z_e,
            "z_q": z_q,
            "vq_losses": vq_losses,
            "usage": usage,
            "mem_mask": mem_mask,
            "sem_pred": sem_pred,
            "len_pred": len_pred,
        }


def smoke_test():
    """Forward-pass smoke test: build a small VQVAE, run a fake batch."""
    import numpy as np

    print("Loading parallel corpus metadata...")
    d = np.load("data/parallel_corpus.npz", allow_pickle=True)
    vocab_size = int(d["vocab_size"])
    pad = int(d["pad_token_id"])
    print(f"  vocab_size={vocab_size}, pad={pad}")

    print("\nBuilding VQVAE (smoke config: K=64, m_max=16, d_model=128)...")
    model = VQVAE(
        vocab_size=vocab_size, n_langs=10,
        d_model=128, d_code=64, k=64, m_max=16,
        n_enc_layers=2, n_dec_layers=2, n_heads=4, d_ff=256,
        pad_token_id=pad,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params/1e6:.2f}M")

    # Build a tiny batch from the real corpus
    B, T_in, T_out = 4, 32, 32
    # Pull 4 sentences from dev for english (idx 0)
    sents_en = [d["dev_token_ids"][i, 0] for i in range(B)]
    sents_zh = [d["dev_token_ids"][i, 1] for i in range(B)]

    def pad_batch(seqs, T, pad_id):
        out = np.full((len(seqs), T), pad_id, dtype=np.int64)
        for i, s in enumerate(seqs):
            n = min(len(s), T)
            out[i, :n] = s[:n]
        return torch.from_numpy(out)

    src_ids = pad_batch(sents_en, T_in, pad)
    tgt_ids = pad_batch(sents_zh, T_out, pad)
    tgt_lang_id = torch.tensor([1] * B, dtype=torch.int64)  # zh

    print(f"\nRunning forward pass: src {src_ids.shape}, tgt {tgt_ids.shape}")
    out = model(src_ids, tgt_ids, tgt_lang_id)

    print(f"  logits:  {out['logits'].shape}  (expect B, T-1, V)")
    print(f"  indices: {out['indices'].shape}  (expect B, M_max)")
    print(f"  usage:   {out['usage'].shape}    (expect K)")
    print(f"  mem_mask:{out['mem_mask'].shape} (expect B, M_max)")
    print(f"  vq commit loss: {out['vq_losses']['commit'].item():.4f}")
    print(f"  vq codebook loss: {out['vq_losses']['codebook'].item():.4f}")

    # Compute a recon loss for the smoke test
    recon = nn.functional.cross_entropy(
        out["logits"].reshape(-1, vocab_size),
        tgt_ids[:, 1:].reshape(-1),
        ignore_index=pad,
    )
    print(f"  recon loss: {recon.item():.4f}")
    total = recon + out["vq_losses"]["commit"] + out["vq_losses"]["codebook"]
    print(f"  total loss: {total.item():.4f}")

    print("\nBackward pass...")
    total.backward()
    print("  backward OK")

    # Verify gradients flow to codebook
    cb_grad = model.quantizer.codebook.grad
    print(f"  codebook grad norm: {cb_grad.norm().item():.4f} "
          f"(nonzero: {(cb_grad.abs() > 0).any().item()})")

    print("\nSmoke test PASSED")


if __name__ == "__main__":
    smoke_test()
