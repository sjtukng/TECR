# TECR V2 core

Minimal MMDetection 2.x implementation of the TECR V2 neck used in the
AI-TOD-v2 Cascade R-CNN experiments. This repository contains the method and
one integration config. It does not bundle MMDetection, AI-TOD-v2 images,
annotations, pretrained weights, trained checkpoints, or experiment outputs.

## Files

- `tecr_v2_core/neck.py`: evidence estimation, local P2/P3 refinement, and
  consistency-gated adjacent-scale fusion over P2–P6.
- `configs/aitodv2_cascade_r50_tecr.py`: minimal overlay for the AI-TOD-v2
  MMDetection 2.24.1 fork's Cascade R50 baseline.

The neck is derived from the source snapshot saved with the completed TECR V2
training. Its computation and state-dict keys are unchanged. The public plugin
uses absolute MMDetection imports and registers the type name `TECRV2FPN` to
avoid colliding with any in-tree `TECRFPN` registration. Checkpoints trained
with the in-tree class load with the same neck parameter names.

## Integration

Use the AI-TOD-v2 MMDetection 2.24.1 fork and install its dependencies. The
recorded training environment used Python 3.8, PyTorch 1.13.1, MMCV 1.7.2,
and MMDetection 2.24.1. This plugin targets the MMDetection 2.x API.

Place the config in the fork so that its relative `_base_` path resolves:

```bash
cp configs/aitodv2_cascade_r50_tecr.py \
  /path/to/mmdet-nwdrka/configs_nwdrka/tecr/
export PYTHONPATH=/path/to/tecr-v2-core:$PYTHONPATH
cd /path/to/mmdet-nwdrka
python tools/train.py configs_nwdrka/tecr/aitodv2_cascade_r50_tecr.py
```

The dataset and pretrained ResNet-50 path must be configured in the host
AI-TOD-v2 fork. For the recorded two-GPU experiment, use 2 images per GPU
(global batch 4), 800 × 800 maximum input, SGD with learning rate 0.02, and
the inherited 12-epoch schedule. The exported config preserves the neck and
optimizer settings; the remaining detector, dataset, and runtime settings
come from the host baseline config.

## Scope

This is the core method implementation. It is not a standalone training
framework or a distribution of the AI-TOD-v2 dataset. MMDetection and the
AI-TOD-v2 fork retain their own licenses.
