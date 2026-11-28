# 06: Execute Track B patient-disjoint synthetic generation and model training on Colab L4

**What to build:** Run the first stage of the Track B pipeline on a Colab Pro L4 GPU via local Colab CLI: construct and verify the zero-overlap four-way Patient-Disjoint Partition, complete multi-batch FID-gated Synthetic Augmentation Batch generation from the new training split (checkpointed to Google Drive), train Tier 1 (MobileNetV2), both independent Tier 2 backbones (EfficientNet-B4 and Ark+ Swin), and all ablation trainings (A8, A9, A11, A12, A15), and persist all trained checkpoints, synthetic artifacts, and training logs to Google Drive.

**Blocked by:** 03: Add Python 3.13 Colab dependency spec and set up local Colab CLI workflow; 04: Fix multi-batch FID-gated synthetic generation with resume and pre-training wiring; 05: Fix disjoint temperature-before-conformal calibration order and end-to-end provenance verification

**Status:** resolved

- [x] A verified four-way Patient-Disjoint Partition manifest confirms zero pairwise patient overlap across training, validation, calibration, and test splits, with exact patient and image counts recorded.
- [x] Multi-batch FID-gated Synthetic Augmentation Batch generation completes on the new training split with exact seed count, generated count, FID-accepted count, batch FID log, and qualitative real-vs-synthetic grid saved to Drive.
- [x] Validation-only routing threshold fitting, Tier 1 training, both Tier 2 backbone trainings (EfficientNet-B4 and Ark+ Swin), and all required ablation trainings (A8, A9, A11, A12, A15) complete cleanly on L4 GPU with wall-clock training times logged.
- [x] All newly trained checkpoints and training histories belong strictly to the clean patient-disjoint protocol and are durably synced to Google Drive.

## Comments

- Added pre-flight Track B training and synthetic pipeline upgrades (`d94dc4a`, `6195c7f`): `HistoryObserver` persisting per-epoch metrics, wall-clock duration, and peak CUDA VRAM (`outputs/models/<run_name>/training_history.json`), cached Inception-v3 real reference statistics in `PyTorchFIDAdapter`, batched multi-variation FP16 synthesis preserving native `512x512` grayscale-consistent RGB (eliminating +19.6 FID 224px resize bias and color tint) with unbiased `batch_size=200` (`n=600`/batch) and `strength=0.15` defaults, plus `--drive-sync-dir` batch checkpointing in `scripts/generate_synthetics.py`.
- Executed Track B Stage 1 on Colab Pro NVIDIA L4 (23.7 GB VRAM, Python 3.13.16, `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, `numpy 2.4.6`) syncing into isolated Drive tree `/content/drive/MyDrive/xray_outputs_track_b` (leaving legacy Track A `/content/drive/MyDrive/xray_outputs` intact as fallback):
  - **Four-way Patient-Disjoint Partition** (built from full 112,120-image NIH ChestX-ray14, 31,812 radiographs / 15,071 unique patients, 5:1 No Finding cap, zero pairwise patient overlap across all 6 pairs):
    - `train`: 22,234 images (10,549 patients; 3,609 Pneumothorax [1,516 isolated + 2,093 comorbid], 18,625 No Finding)
    - `val`: 3,259 images (1,582 patients; 546 Pneumothorax [227 isolated + 319 comorbid], 2,713 No Finding)
    - `calibration`: 1,469 images (679 patients; 229 Pneumothorax [82 isolated + 147 comorbid], 1,240 No Finding)
    - `test`: 4,850 images (2,261 patients; 918 Pneumothorax [369 isolated + 549 comorbid], 3,932 No Finding)
    - Cached all 31,812 selected radiographs in `/content/drive/MyDrive/xray_outputs_track_b/nih_31812_selected.tar` (12.0 GB) for fast Ticket 07 restore.
  - **Multi-batch FID-gated Synthetic Augmentation Batches** (`seed_scope=all`, `batch_size=200`, `num_variations=3`, `strength=0.15`, `fid_threshold=50.0`, `512x512` grayscale RGB, 8,143.43s wall-clock):
    - 3,609 training pneumothorax seeds -> 10,827 generated candidates across 19 batches; **14/19 batches accepted (8,400 images; accepted FID 39.24–49.56, mean 44.53)**, 5/19 rejected (2,427 images; batches 4 [52.59], 11 [52.82], 17 [64.08], 18 [68.05], 19 [154.46, 9-seed tail]).
    - Zero non-training patient overlap (`unique_synthetic_patients=853`); `fid_log.csv`, `synthetic_metadata.csv`, `generation_complete.json`, `resume_state.json`, and `outputs/figures/synthetic_quality_grid.png` saved to Drive.
  - **Model Trainings & Validation-Only Routing Threshold** (protocol `patient-disjoint-v1`, total Stage 1C wall-clock 27,516.13s; all 8 checkpoints + `training_history.json` + 12 SHA-256 hashes verified on Drive):
    - `Tier1_MobileNetV2_Colab` (real-only, `n=22,234`): best ep 5/12, `val_auc=0.8892`, 978.18s, 1,289.42 MB VRAM
    - Validation-only Youden J routing threshold (`val.csv`, `n=3,259`): `optimal_threshold=0.5482`, `val_auc=0.8892`
    - `Tier2_EfficientNetB4_Colab` (synthetic-enabled, `n=30,634`): best ep 8/15, `val_auc=0.8844`, 3,413.32s, 2,018.56 MB VRAM
    - `Tier2_ArkPlus_Colab` (Swin-Base ImageNet fallback, synthetic-enabled, `n=30,634`): best ep 11/19, `val_auc=0.8999`, 6,558.43s, 2,938.97 MB VRAM
    - `A8_NoSynthetic` (EfficientNet-B4, real-only, `n=22,234`): best ep 6/13, `val_auc=0.8777`, 1,528.95s, 3,763.58 MB VRAM
    - `A9_NoAugmentation` (EfficientNet-B4, real-only no-aug, `n=22,234`): best ep 3/10, `val_auc=0.8603`, 1,169.84s, 3,763.58 MB VRAM
    - `A11_ArkPlus_Only_NoMCTTA` (Ark+ Swin, synthetic-enabled, `n=30,634`): best ep 10/17, `val_auc=0.8930`, 3,821.38s, 5,781.52 MB VRAM
    - `A12_ArkPlus_Only_MC_TTA` (Ark+ Swin, synthetic-enabled, `n=30,634`): best ep 13/20, `val_auc=0.8987`, 4,616.83s, 5,781.52 MB VRAM
    - `A15_Mixup_Cutmix` (Ark+ Swin + Mixup/CutMix, synthetic-enabled, `n=30,634`): best ep 15/22, `val_auc=0.8979`, 5,189.47s, 5,777.17 MB VRAM
- Flushed Google Drive, terminated Colab L4 session cleanly (`0` active assignments, `369.48` compute units remaining out of `398.39`), and verified all local quality gates green.
