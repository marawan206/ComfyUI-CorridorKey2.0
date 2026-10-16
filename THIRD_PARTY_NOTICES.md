# Notices

## CorridorKey 2.0

This pack is built on CorridorKey 2.0 by Corridor Digital LLC. The CorridorKey
2.0 model was designed and trained by Zachary Harris, in direct collaboration
with Niko Pueringer and Corridor Digital.

- Source: https://github.com/nikopueringer/CorridorKey2.0 (vendored at commit `485f5e9`)
- Weights: https://huggingface.co/nikopueringer/CorridorKey2.0 (downloaded, not redistributed)
- License: Corridor Key Licence 2.0, a variation of CC BY-NC-SA 4.0 with additional
  terms (full text in [LICENSE](LICENSE)). The material is provided as-is, without
  warranties of any kind.

This is an unofficial integration. It is not affiliated with, sponsored by or
endorsed by Corridor Digital. "CorridorKey" is used here only to identify the
model and software this pack is built on, as the licence requires.

The ComfyUI integration code in this repository is © 2026 Marwan Mostafa and is
released under the same Corridor Key Licence.

### Changes made to the licensed material

- `corridorkey/` and `third_party/MoGe/` are copied from upstream commit `485f5e9`
  **without modification**.
- `ck_comfy/pipeline.py` is adapted from `corridorkey/runtime.py`
  (`run_inference`, `hann_stage1_stream` and the foundation pre-passes): frames
  are read from and written to in-memory tensors, hints can be placed on any
  frame with forward/backward propagation, stage-1 auxiliary mattes are
  returned, outputs can be rescaled after processing, and memory decisions use
  ComfyUI's model management.
- `ck_comfy/loader.py` replaces `corridorkey/runtime.py:load_model`: no
  process-wide offline flags, CPU construction, per-phase device placement.
- `ck_comfy/download.py` is adapted from `scripts/download_weights.py`: ComfyUI
  model folder layout, progress reporting, interruption and resume.
- `ck_comfy/train.py` builds the foundation and RVM features for training the
  same way as `hann_stage1_stream` and `precompute_rvm_to_memmap` in
  `corridorkey/runtime.py`, and `ck_comfy/semantic.py` repeats the per-channel
  feature normalization of `corridorkey/foundations.py`.
- `ck_comfy/env.py` patches the MoGe and C-RADIO loaders of
  `corridorkey/foundations.py` at runtime (private MoGe import, a small
  `utils3d` replacement, eager C-RADIO weight loading).

### Models you train with this pack

LoRA adapters produced by the fine-tune nodes are adaptations of the
CorridorKey 2.0 weights. If you share them, they must stay freely available,
under the same licence, and keep the CorridorKey name.

## Foundation models

The three frozen foundation models are not redistributed in this repository.
They are downloaded from their original publishers at pinned revisions and
remain under their own licenses.

### C-RADIOv4-SO400M

- Publisher: NVIDIA Corporation
- Source: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- License: [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

### MoGe-2 ViT-B

- Publisher: Microsoft Corporation
- Source: https://github.com/microsoft/MoGe, weights at https://huggingface.co/Ruicheng/moge-2-vitb-normal
- License: MIT (code and weights). The model code is vendored in
  `third_party/MoGe/` with its license at `third_party/MoGe/LICENSE`. Parts of
  `ck_comfy/geometry.py` are adapted from MoGe's `MoGeModel.infer`.

The vendored MoGe code includes DINOv2 (Meta Platforms, Inc.), licensed under
the Apache License 2.0, also reproduced in `third_party/MoGe/LICENSE`.

### RVM (Recurrent Video Masked Autoencoders), small

- Publisher: Google DeepMind
- Source: https://github.com/google-deepmind/representations4d
- Paper: Zoran et al., "Recurrent Video Masked Autoencoders", arXiv:2512.13684
- License: Apache License 2.0 (code), Creative Commons Attribution 4.0 (weights)

`corridorkey/rvm.py` is Corridor Digital's PyTorch port of the reference
architecture, distributed under the Apache License 2.0 terms of the original
code (the Apache 2.0 text is included in `third_party/MoGe/LICENSE`).
