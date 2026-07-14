"""Vector quantizer with straight-through estimator.

Two modes:
  - gradient (default): codebook is an nn.Parameter, updated by SGD.
    Loss has 'codebook' term that pulls codebook → encoder outputs.
  - EMA (van den Oord 2017): codebook is a buffer, updated via exponential
    moving average of encoder outputs assigned to each code. More stable.
    Loss only has 'commit' term.

EMA mode also enables dead-code reset: every reset_every steps, codes whose
EMA cluster size has fallen below a threshold are reinitialized from random
encoder outputs in the current batch.
"""

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """K codebook entries in d_code-dim space. Index 0 is reserved for <stop>.

    Forward returns:
      z_q       : (B, M, D) quantized vectors (with straight-through gradient)
      indices   : (B, M) int64 codebook indices
      losses    : dict with 'commit' (always) and 'codebook' (gradient mode only)
      usage     : (K,) per-step usage count for monitoring
    """

    def __init__(
        self,
        k: int = 256,
        d_code: int = 256,
        beta_commit: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_threshold: float = 0.01,
    ):
        super().__init__()
        self.k = k
        self.d_code = d_code
        self.beta_commit = beta_commit
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.dead_threshold = dead_threshold
        self.stop_index = 0  # convention

        init = torch.randn(k, d_code) * (1.0 / (d_code ** 0.5))
        if use_ema:
            # Buffers: not optimized by SGD, updated manually via ema_update()
            self.register_buffer("codebook", init.clone())
            self.register_buffer("cluster_size", torch.ones(k))
            self.register_buffer("cluster_sum", init.clone())
        else:
            self.codebook = nn.Parameter(init)

    def forward(self, z_e: torch.Tensor) -> tuple:
        """z_e: (B, M, D). Returns z_q, indices, losses_dict, usage."""
        B, M, D = z_e.shape
        assert D == self.d_code, f"z_e last dim {D} != d_code {self.d_code}"

        flat = z_e.reshape(-1, D)

        z_sq = (flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook ** 2).sum(dim=1).unsqueeze(0)
        dist = z_sq + e_sq - 2 * flat @ self.codebook.t()

        indices = dist.argmin(dim=1)
        z_q_flat = self.codebook[indices]

        commit_loss = (flat - z_q_flat.detach()).pow(2).mean()
        losses = {"commit": self.beta_commit * commit_loss}
        if not self.use_ema:
            losses["codebook"] = (z_q_flat - flat.detach()).pow(2).mean()

        # Straight-through estimator
        z_q_flat = flat + (z_q_flat - flat).detach()

        usage = torch.bincount(indices, minlength=self.k).float()

        z_q = z_q_flat.view(B, M, D)
        idx = indices.view(B, M)
        return z_q, idx, losses, usage

    @torch.no_grad()
    def ema_update(self, z_e: torch.Tensor, indices: torch.Tensor,
                   ddp_world_size: int = 1) -> None:
        """Update codebook via EMA on encoder outputs assigned to each code.
        Call this after each forward pass when use_ema=True.

        DDP NOTE: codebook is a buffer, so DDP doesn't sync it. We compute the
        per-batch sums locally, then all-reduce them across ranks before the
        EMA blend, so all ranks compute the same updated codebook.
        """
        if not self.use_ema:
            return
        # EMA buffers live in fp32; cast inputs to match for numerical stability
        flat = z_e.reshape(-1, self.d_code).detach().float()
        idx = indices.reshape(-1)

        # Per-code count and sum of assigned vectors in this batch
        one_hot = torch.zeros(idx.size(0), self.k, device=flat.device, dtype=flat.dtype)
        one_hot.scatter_(1, idx.unsqueeze(1), 1)
        batch_size = one_hot.sum(dim=0)                            # (K,)
        batch_sum = one_hot.t() @ flat                              # (K, D)

        if ddp_world_size > 1:
            import torch.distributed as dist
            dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)

        d = self.ema_decay
        self.cluster_size.mul_(d).add_(batch_size, alpha=1 - d)
        self.cluster_sum.mul_(d).add_(batch_sum, alpha=1 - d)

        # Laplace smoothing for numerical stability
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.ema_eps) / (n + self.k * self.ema_eps) * n
        self.codebook.copy_(self.cluster_sum / smoothed.unsqueeze(1))

    @torch.no_grad()
    def reset_dead_codes(self, z_e_pool: torch.Tensor,
                         ddp_world_size: int = 1) -> int:
        """Replace codes with cluster_size < dead_threshold (relative to mean)
        with random vectors from z_e_pool (a recent batch of encoder outputs).

        DDP NOTE: cluster_size is already synced via ema_update's all-reduce, so
        all ranks see the same set of dead codes. To keep replacements identical
        across ranks, we broadcast rank-0's chosen replacements.

        Returns the number of codes reset.
        """
        if not self.use_ema:
            return 0
        flat = z_e_pool.reshape(-1, self.d_code).detach().float()

        mean_size = self.cluster_size.mean()
        dead = self.cluster_size < self.dead_threshold * mean_size

        # DDP determinism: even though cluster_size is all-reduced in
        # ema_update, FP precision in mean()/comparison can produce different
        # boolean masks across ranks at the threshold boundary. all_reduce the
        # mask with MAX (logical OR) so every rank agrees on which codes are
        # dead — without this, ranks can call broadcast() with different
        # tensor shapes and deadlock.
        if ddp_world_size > 1:
            import torch.distributed as dist
            dead_int = dead.to(torch.int32)
            dist.all_reduce(dead_int, op=dist.ReduceOp.MAX)
            dead = dead_int.bool()

        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        # Sample replacement vectors from the pool (rank 0 picks; broadcast
        # to all so every rank applies the same update).
        n_pool = flat.size(0)
        idx = torch.randint(0, n_pool, (n_dead,), device=flat.device)
        replacements = flat[idx] + 0.01 * torch.randn_like(flat[idx])

        if ddp_world_size > 1:
            import torch.distributed as dist
            dist.broadcast(replacements, src=0)

        self.codebook[dead] = replacements
        self.cluster_sum[dead] = replacements * mean_size
        self.cluster_size[dead] = mean_size
        return n_dead

    def get_stop_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """For each row, return a (B, M) bool mask: True for positions BEFORE
        the first <stop>, False for <stop> itself and everything after.
        If no <stop> appears, the entire row is True."""
        is_stop = indices == self.stop_index                      # (B, M)
        cum_stop = torch.cumsum(is_stop.long(), dim=1) > 0        # (B, M)
        return ~cum_stop                                          # True = keep

    def force_stop_at(self, indices: torch.Tensor,
                      target_len: torch.Tensor) -> torch.Tensor:
        """Hard-cap each row at target_len[i]: positions >= target_len[i]
        become <stop>. Positions before are unchanged.

        indices: (B, M) int64
        target_len: (B,) int64 — per-example position at which to force stop.
                    Values are clamped to [1, M].
        Returns a new (B, M) tensor (does not modify in-place).
        """
        B, M = indices.shape
        target = target_len.clamp(min=1, max=M).to(indices.device)
        positions = torch.arange(M, device=indices.device).unsqueeze(0)  # (1, M)
        mask = positions >= target.unsqueeze(1)                          # (B, M)
        return torch.where(mask, torch.full_like(indices, self.stop_index), indices)

    def first_stop_position(self, indices: torch.Tensor) -> torch.Tensor:
        """Return (B,) the index of the first <stop> per row, or M if none."""
        B, M = indices.shape
        is_stop = indices == self.stop_index                              # (B, M)
        # argmax returns first True index, but is 0 if all-False — fix that
        any_stop = is_stop.any(dim=1)
        first = is_stop.float().argmax(dim=1)
        return torch.where(any_stop, first,
                           torch.full_like(first, M, dtype=torch.long))


class ResidualVectorQuantizer(nn.Module):
    """Phase 7: residual VQ — each slot is quantized by n_levels codebooks in
    sequence, each fitting the residual of the previous. The bottleneck vector is
    the SUM of the per-level selections, giving additive/combinatorial capacity
    (n_levels codes per slot instead of one) — and a natural "stack of radicals"
    per character for the composable-logography goal.

    Gradient mode only (no EMA), so the training loop's ema_update/reset_dead
    calls (guarded by args.use_ema) are simply not used. Returns level-0 indices
    as the (B, M) `indices` for logging/stop-mask compatibility; z_q is the full
    multi-level sum. Run without stop-mask / length-cap (fixed m_max slots).
    """

    def __init__(self, k: int = 128, d_code: int = 256, n_levels: int = 4,
                 beta_commit: float = 0.25):
        super().__init__()
        self.k = k
        self.d_code = d_code
        self.n_levels = n_levels
        self.beta_commit = beta_commit
        self.use_ema = False
        self.stop_index = 0  # convention (unused when run without stop-mask)
        scale = 1.0 / (d_code ** 0.5)
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(k, d_code) * scale) for _ in range(n_levels)])

    def forward(self, z_e: torch.Tensor) -> tuple:
        B, M, D = z_e.shape
        assert D == self.d_code, f"z_e last dim {D} != d_code {self.d_code}"
        flat = z_e.reshape(-1, D)

        residual = flat
        quantized = torch.zeros_like(flat)
        commit = flat.new_zeros(())
        codebook = flat.new_zeros(())
        idx0 = None
        for level, cb in enumerate(self.codebooks):
            r_sq = (residual ** 2).sum(dim=1, keepdim=True)
            e_sq = (cb ** 2).sum(dim=1).unsqueeze(0)
            dist = r_sq + e_sq - 2 * residual @ cb.t()
            idx = dist.argmin(dim=1)
            zq = cb[idx]
            # commit binds the residual to its code; codebook pulls the code to it
            commit = commit + (residual - zq.detach()).pow(2).mean()
            codebook = codebook + (zq - residual.detach()).pow(2).mean()
            quantized = quantized + zq
            residual = residual - zq.detach()   # detach: encoder grad flows only via final STE
            if level == 0:
                idx0 = idx

        # Straight-through on the full multi-level sum
        z_q_flat = flat + (quantized - flat).detach()
        losses = {"commit": self.beta_commit * commit, "codebook": codebook}
        usage = torch.bincount(idx0, minlength=self.k).float()
        return z_q_flat.view(B, M, D), idx0.view(B, M), losses, usage

    # --- compatibility shims (operate on level-0 indices; unused w/o stop-mask) ---
    def get_stop_mask(self, indices: torch.Tensor) -> torch.Tensor:
        is_stop = indices == self.stop_index
        return ~(torch.cumsum(is_stop.long(), dim=1) > 0)

    def force_stop_at(self, indices, target_len):
        B, M = indices.shape
        target = target_len.clamp(min=1, max=M).to(indices.device)
        positions = torch.arange(M, device=indices.device).unsqueeze(0)
        mask = positions >= target.unsqueeze(1)
        return torch.where(mask, torch.full_like(indices, self.stop_index), indices)

    def first_stop_position(self, indices: torch.Tensor) -> torch.Tensor:
        B, M = indices.shape
        is_stop = indices == self.stop_index
        any_stop = is_stop.any(dim=1)
        first = is_stop.float().argmax(dim=1)
        return torch.where(any_stop, first,
                           torch.full_like(first, M, dtype=torch.long))
