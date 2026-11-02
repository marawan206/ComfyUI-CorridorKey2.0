"""Norm output dtypes under autocast (issue #1).

PyTorch 2.12 put ``rms_norm`` on autocast's float32 list (pytorch/pytorch#174824), so
``F.rms_norm`` returns float32 for a bfloat16 input under bf16 autocast. CorridorKey's
``RMSNorm`` and ``LayerNorm2d`` were written against PyTorch 2.9, where the output stayed
bfloat16. Without the cast back to the input dtype, the float32 queries from ``q_norm``
meet the bfloat16 result of scaled_dot_product_attention in ``_sdpa_with_region_groups``
and keying stops with "index_copy_(): self and source expected to have the same dtype,
but got (self) Float and (source) BFloat16" (the same fix as upstream pull request
nikopueringer/CorridorKey2.0#6).

The simulated policy reproduces PyTorch 2.12's behaviour on any PyTorch, so the test
fails without the fix on this Mac too; with PyTorch 2.12 or newer the unsimulated run
checks the real thing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ck2pack.corridorkey import blocks


def _bf16_autocast_devices() -> list[str]:
    devices = []
    for name in ("cpu", "cuda", "mps"):
        if name == "cuda" and not torch.cuda.is_available():
            continue
        if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            continue
        try:
            with torch.autocast(name, dtype=torch.bfloat16):
                a = torch.ones(8, 8, device=name)
                if F.linear(a, a).dtype == torch.bfloat16:
                    devices.append(name)
        except Exception:  # noqa: BLE001 - no bf16 autocast on this device: nothing to test there
            pass
    return devices


DEVICES = _bf16_autocast_devices()


@pytest.fixture(params=[False, True], ids=["this torch", "torch 2.12 policy"])
def rms_norm_policy(request, monkeypatch):
    """True: F.rms_norm runs in float32 under autocast and returns float32, as PyTorch 2.12 and newer do."""
    if request.param:
        real = F.rms_norm

        def fp32_under_autocast(x, normalized_shape, weight=None, eps=None):
            if torch.is_autocast_enabled(x.device.type):
                return real(x.float(), normalized_shape, None if weight is None else weight.float(), eps)
            return real(x, normalized_shape, weight, eps)

        monkeypatch.setattr(F, "rms_norm", fp32_under_autocast)
    return request.param


@pytest.mark.parametrize("device", DEVICES)
def test_norms_keep_the_input_dtype_under_autocast(device, rms_norm_policy):
    torch.manual_seed(0)
    rms, ln = blocks.RMSNorm(64).to(device), blocks.LayerNorm2d(8).to(device)
    torch.nn.init.normal_(rms.weight, mean=1.0, std=0.1)
    x = torch.randn(4, 16, 64, device=device, dtype=torch.bfloat16)
    with torch.autocast(device, dtype=torch.bfloat16):
        y = rms(x)
        z = ln(torch.randn(1, 8, 4, 4, device=device, dtype=torch.bfloat16))
    assert y.dtype == torch.bfloat16 and z.dtype == torch.bfloat16
    # The values are the float32 normalisation rounded to bfloat16, as the fused kernel gives.
    reference = F.rms_norm(x.float(), (64,), rms.weight.float(), rms.eps)
    torch.testing.assert_close(y.float(), reference, atol=2e-2, rtol=2e-2)
    # Outside autocast nothing changes: float32 in, float32 out.
    assert rms(torch.randn(4, 16, 64, device=device)).dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_shifted_window_attention_runs_under_autocast(device, rms_norm_policy):
    """q_norm's output meets the attention result in _sdpa_with_region_groups: the call that failed in issue #1."""
    torch.manual_seed(0)
    block = blocks.SwinBlock2D(dim=16, num_heads=2, window_size=4, shift=True).to(device).eval()
    torch.nn.init.normal_(block.attn.proj.weight, std=0.05)   # zero-initialised upstream; make the branch count
    # 6x6 is padded to 8x8 and shifted, so partial windows take the region-group path.
    x = torch.randn(1, 6, 6, 16, device=device)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        y = block(x)
    assert y.shape == x.shape and torch.isfinite(y.float()).all()
    assert not torch.equal(y.float(), x)
