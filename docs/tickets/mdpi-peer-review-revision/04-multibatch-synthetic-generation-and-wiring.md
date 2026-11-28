# 04: Fix multi-batch FID-gated synthetic generation with resume and pre-training wiring

**What to build:** Upgrade Synthetic Augmentation Batch generation from the hardcoded tiny five-seed stub into a configurable, multi-batch, checkpoint-and-resume-capable workflow iterating across all Patient-Disjoint Partition training pneumothorax seeds (three variations per seed, Inception-v3 FID <= 50.0 batch gate against real training pneumothorax references, persisted batch FID log, accepted metadata, and qualitative comparison grid); reorder the production notebook so synthetic generation executes before Tier 2 training and feeds synthetic-enabled configurations while keeping the no-synthetic control strictly real-only; prove batching, FID gating, and crash-resume idempotency via CPU unit tests with fake generator and FID adapters.

**Blocked by:** 02: Deepen and simplify research package architecture

**Status:** resolved

- [x] Synthetic generation supports configurable seed scope, batch size, variations per seed, random seed, and FID threshold, defaulting to processing all Patient-Disjoint Partition training pneumothorax seeds (both isolated and comorbid positive cases) rather than a hardcoded small sample.
- [x] Each generated batch is evaluated with Inception-v3 FID against real training pneumothorax radiographs, logs batch index, seed count, generated count, FID score, and accept/reject decision, and routes images accordingly.
- [x] Interrupting and restarting generation resumes cleanly from the last completed batch without duplicating already-processed seeds or corrupting metadata and FID logs.
- [x] The production notebook runs synthetic generation before Tier 2 training and passes accepted synthetics to synthetic-enabled Tier 2 training while keeping the no-synthetic ablation (A8) strictly real-only.
- [x] CPU unit tests with fake generator and fake FID adapters verify full multi-batch iteration, accept/reject FID gating, FID log recording, and resume idempotency.

## Comments

- Added `SyntheticGenerationConfig`, `MultiBatchGenerationSummary`, `select_training_pneumothorax_seeds`, `build_qualitative_comparison_grid`, and `SyntheticBatchPipeline.run_multi_batch` in `src/tcrp/data/synthetic.py` (exported in `src/tcrp/data/__init__.py`) with deterministic per-seed RNG seeding, atomic persistence of `fid_log.csv`, `synthetic_metadata.csv`, `resume_state.json`, and `generation_complete.json`, stale-file cleanup on retry, and split-plan hash mismatch protection.
- Upgraded `scripts/generate_synthetics.py` (and `Makefile` `generate-synthetic` target) with configurable CLI arguments (`--seed-scope`, `--batch-size`, `--num-variations`, `--seed`, `--fid-threshold`, `--strength`, `--max-seeds`, `--fid-reference-count`, `--resume`/`--no-resume`, `--qualitative-grid-path`, `--qualitative-grid-samples`), defaulting to all isolated and comorbid training pneumothorax seeds.
- Reordered `notebooks/xray_colab_produce_all.ipynb` so `## 7.5 Stable Diffusion Synthetic Augmentation Batches (Pre-Training)` executes before `## 8. Model Training`, enabled `synthetic_generation` in `STEPS` and `NIH_CONSUMERS` with `SYNTH_HP` and `generation_complete.json` skip-safety, enabled `use_synthetic=True` for Tier 2 base models (`train_tier2_efficientnet`, `train_tier2_arkplus`) and synthetic-enabled ablations (`A11`, `A12`, `A15`), and kept `train_tier1`, `A8`, and `A9` strictly real-only.
- Verified with CPU seam unit tests in `tests/unit/test_synthetic_pipeline.py` using fake generator and fake FID adapters, plus full repo quality gates (`ruff`, `mypy`, `lint-imports`, `check_comment_language`, and full `pytest` suite).
