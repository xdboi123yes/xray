# 07: Execute Track B calibration, full ablation/revision analyses, CheXpert, and L4 latency benchmarks

**What to build:** Run the second stage of the Track B pipeline on Colab Pro L4 via local Colab CLI: execute Validation Split temperature scaling followed by temperature-scaled Calibration Split conformal calibration, full ablations (A1–A15 including A5b TTA-only), zero-shot CheXpert evaluation on both the 202-image validation cohort and expanded patient-disjoint cohort (A14 and A14b with Wilson CIs and threshold recalibration), 2,000-iteration patient-clustered bootstrap with Holm-Bonferroni correction and Cohen's h (including direct Tier 1 vs. TCR-P), the full revision analysis suite (validation routing grid search, 10-seed permutation and adversarial drift arrival-order sensitivity, 15-bin pre/post-temperature reliability diagrams with ECE/class-ECE/Brier/NLL, clinical FNR/threshold sweep, synthetic consistency audit), CUDA-synchronized L4 latency/VRAM/FLOPs benchmarks, frozen library versions, and a green SHA-256 provenance manifest pulled back locally.

**Blocked by:** 06: Execute Track B patient-disjoint synthetic generation and model training on Colab L4

**Status:** claimed

- [ ] Disjoint temperature calibration on Validation Split and temperature-scaled conformal calibration on Calibration Split succeed and log singleton, two-class, empty-fallback, and per-class coverage rates.
- [ ] All ablation configurations (including A5b TTA-only and decomposed epistemic vs. TTA variance) and both CheXpert cohorts (A14 and A14b with 95% Wilson CIs) complete with persisted metrics and predictions.
- [ ] Patient-clustered 2,000-iteration bootstrap 95% CIs, Holm-Bonferroni adjusted p-values, and Cohen's h effect sizes are generated for Tier 2 backbone comparison and direct Tier 1 vs. TCR-P comparison.
- [ ] All revision analyses complete and produce publication-ready figures and tables: validation routing Pareto grid, 10-seed + drift arrival-order sensitivity, 15-bin pre/post-T reliability diagrams, clinical sensitivity/FNR sweep, synthetic quality/consistency audit, and print-legible 300-DPI Figure 2.
- [ ] CUDA-synchronized L4 latency, preprocessing/H2D overhead, single-pass vs. MC/TTA/vectorized MC+TTA timing, peak VRAM, parameter counts, and GFLOPs are recorded across all primary configurations.
- [ ] The scientific integrity and provenance gate passes with exact git commit, frozen library lockfile, and SHA-256 artifact hashes synced to Drive and downloaded locally.
