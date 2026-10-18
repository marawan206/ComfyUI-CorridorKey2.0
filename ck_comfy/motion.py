"""Motion analysis and mask tracking with RVM features.

RVM (Recurrent Video MAE, the keyer's temporal foundation model) encodes each
frame into 16x16-patch tokens and then mixes them into a recurrent state that
is carried from frame to frame. Its output features are therefore causal
(frame t only sees earlier frames in processing order) and they never settle
on a still shot: the state keeps cycling with a period of about four frames.
So:

* ``motion_energy`` compares the per-frame encoder tokens (zero on a still shot);
* ``identity_pca`` shows the recurrent features with each frame's mean token
  removed, which takes out most of that cycle;
* ``track`` propagates a reference mask through feature affinity (label
  propagation with a memory of the reference frame and the previous
  prediction, in the spirit of space-time memory readout). The encoder tokens
  track best (the recurrent core also emits a few outlier tokens that can
  punch holes in the mask); the recurrent state, or both, can be used too.
  Recurrent features are causal, so frames before the reference are then
  tracked on a separate reversed-time pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .semantic import Check, Tick, autocast, load_frames, mask_to_grid, pca_video, snap_size, tokens, upsample

PATCH = 16
ENERGY_PERCENTILE = 0.99
# Smallest divisor for motion energy (a cosine distance that clearly means motion), so a
# still clip stays near zero instead of having its tiny changes stretched to full range.
ENERGY_FLOOR = 0.1
_QUERY_BLOCK = 4096
TRACK_FEATURES = ("encoder", "recurrent", "both")


@torch.no_grad()
def features(rvm, frames: torch.Tensor, resolution: int = 512, chunk: int = 8, device=None, *,
             order: list[int] | None = None, encoded: bool = False, tick: Tick = None,
             check: Check = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RVM features of a (T,H,W,3) clip, in processing ``order``.

    One continuous recurrent pass in chunks of at most 8 frames, long side
    about ``resolution`` (multiples of 16). ``order`` defaults to all frames
    forwards; ``backward_order(r)`` gives the reversed-time pass. Returns
    ``(recurrent, encoder)`` as (L,384,h,w) float16 CPU tensors; ``encoder``
    (per-frame tokens before the recurrent core) is None unless ``encoded``.
    ``rvm`` (``foundations.rvm``) must already be on ``device``.
    """
    device = torch.device(device or "cpu")
    T, H, W = frames.shape[:3]
    order = list(range(T)) if order is None else [int(i) for i in order]
    hw = snap_size(H, W, resolution, PATCH)
    h, w = hw[0] // PATCH, hw[1] // PATCH
    D = rvm.embed_dim
    out = torch.empty((len(order), D, h, w), dtype=torch.float16)
    enc_out = torch.empty_like(out) if encoded else None
    captured: list[torch.Tensor] = []
    hook = rvm.encoder_final_norm.register_forward_hook(lambda _m, _i, o: captured.append(o)) if encoded else None
    chunk = max(1, min(int(chunk), 8))
    state = None
    try:
        for s in range(0, len(order), chunk):
            if check:
                check()
            idx = order[s:s + chunk]
            n = len(idx)
            x = load_frames(frames, idx, hw, device).unsqueeze(0)
            with autocast(device):
                feat, state = rvm(x, state=state)
            out[s:s + n] = feat[0].to(torch.float16).cpu()
            if encoded:
                # (n, 1 + h*w, D) with the CLS token first.
                tok = captured.pop()[:, 1:].float().view(n, h, w, D).permute(0, 3, 1, 2)
                enc_out[s:s + n] = tok.to(torch.float16).cpu()
            del x, feat
            if tick:
                tick(n)
    finally:
        if hook is not None:
            hook.remove()
    return out, enc_out


def backward_order(reference_frame: int) -> list[int]:
    """Frame order of the reversed-time pass that tracks frames before ``reference_frame``."""
    return list(range(int(reference_frame), -1, -1))


def joint(recurrent: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
    """Concatenated unit-length recurrent and encoder features (L,2C,h,w) float16.

    Their cosine similarity is the mean of the two similarities: the recurrent
    state brings temporal context, the encoder tokens anchor the current frame
    and outvote the few outlier tokens the recurrent core produces.
    """
    out = torch.empty((recurrent.shape[0], recurrent.shape[1] + encoder.shape[1], *recurrent.shape[2:]),
                      dtype=torch.float16)
    c = recurrent.shape[1]
    for s in range(0, recurrent.shape[0], 16):
        out[s:s + 16, :c] = F.normalize(recurrent[s:s + 16].float(), dim=1).to(torch.float16)
        out[s:s + 16, c:] = F.normalize(encoder[s:s + 16].float(), dim=1).to(torch.float16)
    return out


def tracking_features(kind: str, recurrent: torch.Tensor, encoder: torch.Tensor | None) -> torch.Tensor:
    """The features ``track`` matches on: one of ``TRACK_FEATURES`` (``encoder`` may be None for "recurrent")."""
    if kind == "encoder":
        return encoder
    if kind == "recurrent":
        return recurrent
    if kind == "both":
        return joint(recurrent, encoder)
    raise ValueError(f"track features must be one of {TRACK_FEATURES}, not '{kind}'")


def identity_pca(feats: torch.Tensor, frames_hw: tuple[int, int], *, device=None, batch: int = 8,
                 check: Check = None) -> torch.Tensor:
    """(T,C,h,w) recurrent features -> (T,H,W,3) float32 CPU PCA colors."""
    centered = torch.empty_like(feats)
    for s in range(0, feats.shape[0], batch):
        f = feats[s:s + batch].float()
        centered[s:s + batch] = (f - f.mean(dim=(2, 3), keepdim=True)).to(feats.dtype)
    return pca_video(centered, frames_hw, device=device, batch=batch, check=check)


def motion_energy(encoded: torch.Tensor, frames_hw: tuple[int, int], *, percentile: float = ENERGY_PERCENTILE,
                  floor: float = ENERGY_FLOOR, device=None, batch: int = 16, check: Check = None) -> torch.Tensor:
    """Per-patch change 1 - cos(e_t, e_{t-1}) of encoder tokens (T,C,h,w) -> (T,H,W) float32 CPU.

    Frame 0 is 0. Values are divided by the clip's ``percentile`` (at least
    ``floor``) and clipped to [0, 1].
    """
    device = torch.device(device or "cpu")
    T, _, h, w = encoded.shape
    raw = torch.zeros((T, h, w), dtype=torch.float32)
    prev = None
    for s in range(0, T, batch):
        if check:
            check()
        cur = F.normalize(encoded[s:s + batch].to(device).float(), dim=1)
        seq = cur if prev is None else torch.cat((prev, cur))
        dist = (1.0 - (seq[1:] * seq[:-1]).sum(dim=1)).clamp(min=0.0)
        end = s + cur.shape[0]
        raw[end - dist.shape[0]:end] = dist.cpu()
        prev = cur[-1:]
    scale = float(floor)
    if T > 1:
        # torch.quantile is limited to 16M elements; a strided subsample is plenty.
        moving = raw[1:].flatten()
        moving = moving[::max(1, moving.numel() // 4_000_000)]
        scale = max(scale, float(torch.quantile(moving, percentile)))
    out = torch.empty((T, *frames_hw), dtype=torch.float32)
    for s in range(0, T, batch):
        e = (raw[s:s + batch].to(device) / scale).clamp(0.0, 1.0).unsqueeze(1)
        out[s:s + e.shape[0]] = upsample(e, frames_hw)[:, 0].clamp(0.0, 1.0).cpu()
    return out


def propagate(seq: torch.Tensor, coverage: torch.Tensor, *, top_k: int = 8, temperature: float = 0.05,
              device=None, check: Check = None) -> torch.Tensor:
    """Carry a soft (h,w) mask from ``seq[0]`` along features (L,C,h,w) -> (L,h,w) float32 CPU.

    Every frame reads labels from a memory of the first frame and the previous
    frame's prediction: for each patch, the ``top_k`` most similar memory
    patches (cosine) are averaged with softmax(similarity / temperature) weights.
    """
    device = torch.device(device or "cpu")
    L, _, h, w = seq.shape
    out = torch.empty((L, h, w), dtype=torch.float32)
    out[0] = coverage
    ref_k = tokens(seq[:1].to(device))
    ref_v = coverage.reshape(-1).to(device)
    prev_k, prev_v = ref_k, ref_v
    temperature = max(float(temperature), 1e-4)
    for t in range(1, L):
        if check:
            check()
        q = tokens(seq[t:t + 1].to(device))
        keys = torch.cat((ref_k, prev_k))
        vals = torch.cat((ref_v, prev_v))
        k = max(1, min(int(top_k), keys.shape[0]))
        pred = torch.empty(q.shape[0], device=device)
        for b in range(0, q.shape[0], _QUERY_BLOCK):
            sim, idx = (q[b:b + _QUERY_BLOCK] @ keys.T).topk(k, dim=1)
            weight = torch.softmax(sim / temperature, dim=1)
            pred[b:b + _QUERY_BLOCK] = (weight * vals[idx]).sum(dim=1)
        out[t] = pred.view(h, w).cpu()
        prev_k, prev_v = q, pred
    return out


def track(feats: torch.Tensor, reference_mask: torch.Tensor, reference_frame: int = 0, *,
          backward_feats: torch.Tensor | None = None, top_k: int = 8, temperature: float = 0.05,
          device=None, batch: int = 16, check: Check = None) -> torch.Tensor:
    """Propagate ``reference_mask`` (H,W, 1 = subject) from ``reference_frame`` to every frame.

    ``feats`` (T,C,h,w) come from a forward pass. ``backward_feats`` are the
    features of ``backward_order(reference_frame)`` from a reversed-time pass;
    without them the forward features are reused for the earlier frames, which
    is exact for per-frame encoder tokens.
    Returns a (T,H,W) float32 CPU soft mask at the mask's resolution; the
    reference frame keeps the given mask.
    """
    device = torch.device(device or "cpu")
    T, _, h, w = feats.shape
    r = int(reference_frame)
    if not 0 <= r < T:
        raise ValueError(f"reference_frame {r} is outside 0..{T - 1}")
    reference_mask = reference_mask.float().clamp(0.0, 1.0)
    frames_hw = tuple(reference_mask.shape[-2:])
    coverage = mask_to_grid(reference_mask, (h, w))
    grid = torch.empty((T, h, w), dtype=torch.float32)
    grid[r:] = propagate(feats[r:], coverage, top_k=top_k, temperature=temperature, device=device, check=check)
    if r > 0:
        seq = backward_feats if backward_feats is not None else feats[:r + 1].flip(0)
        if seq.shape[0] != r + 1:
            raise ValueError(f"backward_feats must hold {r + 1} frames (reference back to frame 0)")
        grid[:r + 1] = propagate(seq, coverage, top_k=top_k, temperature=temperature,
                                 device=device, check=check).flip(0)
    out = torch.empty((T, *frames_hw), dtype=torch.float32)
    for s in range(0, T, batch):
        g = grid[s:s + batch].to(device).unsqueeze(1)
        out[s:s + g.shape[0]] = upsample(g, frames_hw)[:, 0].clamp(0.0, 1.0).cpu()
    out[r] = reference_mask
    return out
