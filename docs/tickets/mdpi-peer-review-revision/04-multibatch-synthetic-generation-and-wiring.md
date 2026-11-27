# 04: Fix multi-batch FID-gated synthetic generation with resume and pre-training wiring

**What to build:** Upgrade Synthetic Augmentation Batch generation from the hardcoded tiny five-seed stub into a configurable, multi-batch, checkpoint-and-resume-capable workflow iterating across all Patient-Disjoint Partition training pneumothorax seeds (three variations per seed, Inception-v3 FID <= 50.0 batch gate against real training pneumothorax references, persisted batch FID log, accepted metadata, and qualitative comparison grid); reorder the production notebook so synthetic generation executes before Tier 2 training and feeds synthetic-enabled configurations while keeping the no-synthetic control strictly real-only; prove batching, FID gating, and crash-resume idempotency via CPU unit tests with fake generator and FID adapters.

**Blocked by:** 02: Deepen and simplify research package architecture

**Status:** open

- [ ] Synthetic generation supports configurable seed scope, batch size, variations per seed, random seed, and FID threshold, defaulting to processing all Patient-Disjoint Partition training pneumothorax seeds (both isolated and comorbid positive cases) rather than a hardcoded small sample.
- [ ] Each generated batch is evaluated with Inception-v3 FID against real training pneumothorax radiographs, logs batch index, seed count, generated count, FID score, and accept/reject decision, and routes images accordingly.
- [ ] Interrupting and restarting generation resumes cleanly from the last completed batch without duplicating already-processed seeds or corrupting metadata and FID logs.
- [ ] The production notebook runs synthetic generation before Tier 2 training and passes accepted synthetics to synthetic-enabled Tier 2 training while keeping the no-synthetic ablation (A8) strictly real-only.
- [ ] CPU unit tests with fake generator and fake FID adapters verify full multi-batch iteration, accept/reject FID gating, FID log recording, and resume idempotency.
