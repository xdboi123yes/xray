# 06: Execute Track B patient-disjoint synthetic generation and model training on Colab L4

**What to build:** Run the first stage of the Track B pipeline on a Colab Pro L4 GPU via local Colab CLI: construct and verify the zero-overlap four-way Patient-Disjoint Partition, complete multi-batch FID-gated Synthetic Augmentation Batch generation from the new training split (checkpointed to Google Drive), train Tier 1 (MobileNetV2), both independent Tier 2 backbones (EfficientNet-B4 and Ark+ Swin), and all ablation trainings (A8, A9, A11, A12, A15), and persist all trained checkpoints, synthetic artifacts, and training logs to Google Drive.

**Blocked by:** 03: Add Python 3.13 Colab dependency spec and set up local Colab CLI workflow; 04: Fix multi-batch FID-gated synthetic generation with resume and pre-training wiring; 05: Fix disjoint temperature-before-conformal calibration order and end-to-end provenance verification

**Status:** open

- [ ] A verified four-way Patient-Disjoint Partition manifest confirms zero pairwise patient overlap across training, validation, calibration, and test splits, with exact patient and image counts recorded.
- [ ] Multi-batch FID-gated Synthetic Augmentation Batch generation completes on the new training split with exact seed count, generated count, FID-accepted count, batch FID log, and qualitative real-vs-synthetic grid saved to Drive.
- [ ] Validation-only routing threshold fitting, Tier 1 training, both Tier 2 backbone trainings (EfficientNet-B4 and Ark+ Swin), and all required ablation trainings (A8, A9, A11, A12, A15) complete cleanly on L4 GPU with wall-clock training times logged.
- [ ] All newly trained checkpoints and training histories belong strictly to the clean patient-disjoint protocol and are durably synced to Google Drive.
