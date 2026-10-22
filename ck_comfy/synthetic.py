"""Procedural green/blue-screen test clips with exact ground truth.

The subject is drawn analytically (soft-edged body, fine hair-like strands,
a semi-transparent veil) over an unevenly lit screen, with screen-colored
spill on the subject and sensor noise. Compositing happens in linear light,
so ``alpha`` and ``fg`` are the exact answer a keyer should recover. Useful
for trying the nodes without footage, for tests, and as fine-tuning data.
"""

from __future__ import annotations

import math

import torch

from ..corridorkey import color_utils as cu
from .semantic import Check, Tick


def _smooth_edge(signed_dist: torch.Tensor, softness: float) -> torch.Tensor:
    """Coverage from a signed distance (negative inside), anti-aliased over ``softness`` px."""
    return (0.5 - signed_dist / max(softness, 1e-3)).clamp(0.0, 1.0)


def _segment_distance(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    t = ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby + 1e-9)
    t = t.clamp(0.0, 1.0)
    dx, dy = px - (ax + t * abx), py - (ay + t * aby)
    return torch.sqrt(dx * dx + dy * dy)


@torch.no_grad()
def make_clip(frames: int = 48, width: int = 1280, height: int = 720, *, seed: int = 0,
              screen: str = "green", spill: float = 0.35, noise: float = 0.01,
              strands: int = 70, device=None, tick: Tick = None, check: Check = None) -> dict[str, torch.Tensor]:
    """Return ``images`` (T,H,W,3) sRGB, ``alpha`` (T,H,W) and ``fg`` (T,H,W,3) sRGB, on the CPU.

    Frames are drawn on ``device`` (default: CUDA when available). Every random
    number comes from a seeded CPU generator, so a seed gives the same clip on
    any device. ``check`` is called before each frame and ``tick(1)`` after it.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rnd = lambda *shape: torch.rand(*shape, generator=gen)  # noqa: E731

    H, W = int(height), int(width)
    s = min(H, W) / 720.0
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=device, dtype=torch.float32) + 0.5, indexing="ij")

    # Unevenly lit screen in linear light.
    screen_rgb = torch.tensor([0.08, 0.55, 0.12] if screen == "green" else [0.06, 0.16, 0.6], device=device)
    light = 0.75 + 0.35 * torch.exp(-(((xx - W * 0.35) / (W * 0.6)) ** 2 + ((yy - H * 0.3) / (H * 0.8)) ** 2))
    light = light - 0.18 * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2)
    background = screen_rgb.view(1, 1, 3) * light.unsqueeze(-1)

    skin = torch.tensor([0.62, 0.42, 0.32], device=device)
    cloth = torch.tensor([0.18, 0.2, 0.42], device=device)
    hair_col = torch.tensor([0.10, 0.07, 0.05], device=device)
    veil_col = torch.tensor([0.85, 0.85, 0.9], device=device)

    roots = rnd(strands, 2)
    lengths = 0.6 + 0.6 * rnd(strands)
    phases = rnd(strands) * math.tau
    widths = 0.5 + 1.0 * rnd(strands)
    opac = 0.55 + 0.45 * rnd(strands)

    images = torch.empty((frames, H, W, 3), dtype=torch.float32)
    alphas = torch.empty((frames, H, W), dtype=torch.float32)
    fgs = torch.empty((frames, H, W, 3), dtype=torch.float32)
    g_noise = torch.Generator(device="cpu").manual_seed(int(seed) + 1)

    for t in range(frames):
        if check:
            check()
        ph = t / max(frames - 1, 1)
        cx = W * (0.38 + 0.22 * ph) + 12 * s * math.sin(ph * 6.0)
        cy = H * 0.42 + 10 * s * math.sin(ph * 4.0 + 1.0)
        sway = math.sin(ph * math.tau * 1.5)

        # Head (ellipse) and torso (rounded box), soft-edged.
        hr = 95 * s
        head_d = torch.sqrt(((xx - cx) / 0.82) ** 2 + (yy - cy) ** 2) - hr
        head = _smooth_edge(head_d, 1.2)
        bx, by, bw, bh, br = cx, cy + hr * 2.6, 170 * s, 190 * s, 60 * s
        qx = (xx - bx).abs() - (bw - br)
        qy = (yy - by).abs() - (bh - br)
        box_d = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2) + torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx)) - br
        torso = _smooth_edge(box_d, 1.2)
        shading = (0.75 + 0.25 * ((cx - xx) / hr).clamp(-1, 1)).unsqueeze(-1)
        body_a = torch.maximum(head, torso)
        body_f = torch.where((head >= torso).unsqueeze(-1), skin * shading, cloth * (0.8 + 0.2 * shading))

        # Hair strands from the top of the head, each a short polyline.
        hair_cov = torch.zeros((H, W), device=device)
        for k in range(strands):
            ang = math.pi * (1.05 + 0.9 * float(roots[k, 0]))
            ax = cx + math.cos(ang) * hr * 0.8
            ay = cy + math.sin(ang) * hr * 0.95
            length = float(lengths[k]) * hr
            px, py = ax, ay
            direction = ang + (0.3 * sway + 0.25 * math.sin(float(phases[k]) + ph * 5.0))
            x0, x1 = int(max(0, min(ax, ax + math.cos(direction) * length) - 40 * s)), int(min(W, max(ax, ax + math.cos(direction) * length) + 40 * s))
            y0, y1 = int(max(0, min(ay, ay + math.sin(direction) * length) - 40 * s)), int(min(H, max(ay, ay + math.sin(direction) * length) + 40 * s))
            if x1 <= x0 or y1 <= y0:
                continue
            sx, sy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
            dist = torch.full_like(sx, 1e9)
            for seg in range(8):
                bend = direction + 0.08 * seg * math.sin(float(phases[k]) + seg + ph * 3.0)
                qx_ = px + math.cos(bend) * length / 8
                qy_ = py + math.sin(bend) * length / 8
                dist = torch.minimum(dist, _segment_distance(sx, sy, px, py, qx_, qy_))
                px, py = qx_, qy_
            cov = (float(widths[k]) * 0.5 + 0.5 - dist).clamp(0, 1) * float(opac[k])
            hair_cov[y0:y1, x0:x1] = 1 - (1 - hair_cov[y0:y1, x0:x1]) * (1 - cov)

        # Semi-transparent veil drifting across the torso.
        vx = cx + 120 * s + 30 * s * sway
        veil_d = torch.sqrt(((xx - vx) / 1.6) ** 2 + (yy - (cy + hr * 2.2)) ** 2) - 110 * s
        veil = _smooth_edge(veil_d, 18 * s) * (0.35 + 0.15 * torch.sin(xx / (9 * s) + ph * 8.0))

        # "over" composite of veil over hair over body (all in linear light).
        a = body_a
        f = body_f
        a_new = hair_cov + a * (1 - hair_cov)
        f = (hair_col * hair_cov.unsqueeze(-1) + f * (a * (1 - hair_cov)).unsqueeze(-1)) / a_new.clamp(min=1e-6).unsqueeze(-1)
        a = a_new
        a_new = veil + a * (1 - veil)
        f = (veil_col * veil.unsqueeze(-1) + f * (a * (1 - veil)).unsqueeze(-1)) / a_new.clamp(min=1e-6).unsqueeze(-1)
        a = a_new
        f = torch.where(a.unsqueeze(-1) > 1e-6, f, torch.zeros_like(f))

        # Spill: the screen tints the subject, strongest on thin / transparent parts.
        edge = (1 - a).clamp(0, 1) + 0.25
        spilled = f + spill * edge.unsqueeze(-1) * (screen_rgb * light.unsqueeze(-1) - f) * 0.6
        observed = spilled * a.unsqueeze(-1) + background * (1 - a.unsqueeze(-1))
        if noise > 0:
            observed = observed + noise * torch.randn(observed.shape, generator=g_noise).to(device)

        images[t] = cu.linear_to_srgb(observed.clamp(min=0)).clamp(0, 1).cpu()
        alphas[t] = a.clamp(0, 1).cpu()
        fgs[t] = cu.linear_to_srgb(f.clamp(min=0)).clamp(0, 1).cpu()
        if tick:
            tick(1)
    return {"images": images, "alpha": alphas, "fg": fgs}
