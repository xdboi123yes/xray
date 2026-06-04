<!--
  WORKING TRACKER for finishing the Chest X-Ray thesis project.
  This file is the SINGLE SOURCE OF TRUTH for remaining work and survives chat
  switches. A fresh assistant session should READ THIS FIRST, then continue from
  the first unchecked item. Update the checkboxes + "Done log" as work completes.
  (This file is temporary scaffolding and can be deleted once the project ships.)
-->

# Project Completion Tracker — Chest X-Ray Tiered Classifier (Thesis)

## 0. Read this first (context for a cold start)

A bachelor thesis project: a confidence-routed tiered chest X-ray pneumothorax
classifier (MobileNetV2 → EfficientNet-B4 / Ark+-Swin) with MC-Dropout, TTA,
conformal prediction, and Stable-Diffusion synthetic augmentation. FastAPI +
React app, Docker, MLflow, full CI.

**State:** Engineering is DONE. CI is green (4 workflows: CI Quality Gate, UI
Language, Docker Images, Documentation). Docker works. The app works. **All 15
ablations (A1–A15) are genuinely evaluated** (`outputs/results/ablation.json` =
15/15 `evaluation_json`). Stats, ONNX, 4/5 analysis notebooks done.

### KEY DECISIONS (2026-06-03)
1. **The whole project + thesis are now ENGLISH-ONLY.** This reverses PLAN.md §1
   (which mandated Turkish for thesis/README/UI/notebook-markdown). Everything
   user-facing becomes English.
2. **Frontend stays BILINGUAL (EN/TR toggle).** `web/frontend/src/i18n/tr.json`
   is the Turkish translation and STAYS. English is already the default/fallback
   (`i18n/index.ts` `fallbackLng:'en'`). The EN locale was audited and is already
   100% English — the only "Turkish" in EN is the switch-button label `"Türkçe"`,
   which is correct. **=> No frontend language change needed.**
3. **The 7 thesis chapters' PROSE is the USER's job** (written last, in English).
   I produce the English scaffold (main.tex, abstract, per-chapter section
   skeletons, all figures/tables slotted in) so the user only writes prose.
4. **Do NOT delete PLAN.md.** It is the design/decision record (valuable thesis
   supplementary material). It gets translated to English and may move to `docs/`.

### DO-NOT-TOUCH Turkish (intentional)
- `web/frontend/src/i18n/tr.json` — the Turkish locale (translation).
- `en.json` keys `toggleLanguage`/`langTr` (= "Türkçe", the switch labels).
- `scripts/check_comment_language.py`, `.github/workflows/ui-language.yml`,
  `tests/unit/test_ci_gates_probe.py` — these are the Turkish DETECTORS/fixtures.

## Conventions
- **Language:** English everywhere except the DO-NOT-TOUCH list above.
- **CI faithful env:** `/Users/alperen/anaconda3/envs/xray-ci` (Python 3.10,
  mypy 1.10.0, ruff 0.4.5). Full gate = ruff (`core application infrastructure
  web tests scripts`) + mypy (`core application infrastructure`) + `lint-imports`
  + `python scripts/check_comment_language.py` + `python scripts/check_i18n_parity.py`
  + `pytest tests/ --cov=core --cov=application --cov-fail-under=72` + no-print
  grep + frontend `npm run build`.
- **Never fabricate metrics.** Results are genuine; only surface computed numbers.
- **Commits:** Conventional Commits, English, end body with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Push:** `gh auth setup-git` already done; use `GIT_TERMINAL_PROMPT=0 git push
  origin main` (plain push hangs on osxkeychain). Token has `repo`+`workflow`.
- Local results data IS present (`outputs/results/*.json|*.csv|*.tex`,
  `tiered_predictions*.csv`) so figures/tables can be generated locally without a GPU.

---

## ⛔ BLOCKERS — regeneration the USER must run on Colab/GPU
These were discovered while building the thesis tables/figures. They do **not**
block the genuine work already done, but they gate the per-sample figures and
the tiered/MobileNet statistical tests. **Until fixed, do NOT put numbers from
the stale artifacts into the thesis** — only `ablation.json` (15/15) and the
EfficientNet-vs-Ark+ rows of `statistical_comparison.csv` are trusted.

1. **Stale per-sample tiered predictions.** `outputs/results/tiered_predictions.csv`
   (Jun 1) has a BROKEN Tier-1 column: its `tier1_prob` scores AUC ≈ **0.78** and
   `tiered_prob` ≈ 0.80, contradicting the genuine `ablation.json` A1 (MobileNet
   Tier-1) AUC ≈ **0.92**. Its `tier2_prob` (Ark+, AUC 0.916) IS fine. Everything
   derived from the file inherits the break: `val_predictions.csv` (AUC 0.796),
   `threshold_sweep.csv`, `model_comparison_bootstrap.csv`, the MobileNet/Tiered
   rows of `statistical_comparison.csv`, and **all 4 analysis notebooks**.
   → **FIX:** re-export per-sample tiered predictions with the current pipeline
   (`scripts/evaluate_tiered.py`, same models as `ablation.json`) so the CSV's
   Tier-1/tiered columns match A1/A13. Then 1.5 + reliability diagram + the
   tiered/MobileNet stat tests become valid.
2. **A14 CheXpert eval is a small preliminary probe.** The current A14 result was
   computed on only **33 images** (7 pneumothorax / 26 normal — exact from the
   metric denominators 6/7, 21/26), far smaller than the NIH evaluation. USER will
   **re-run A14 on a larger CheXpert cohort** later (the result is genuine but
   under-powered at N=33). The thesis now frames A14 as *preliminary* everywhere
   (ch4/ch5 prose, the cross-dataset figure footnote, the ablation-table caption).
   → **FIX (USER, later):** build a larger CheXpert subset (extend
   `scripts/download_chexpert_meta.py`), re-run A14, then re-run
   `build_thesis_tables.py` + `build_thesis_figures.py` (numbers update
   automatically; no prose edits needed beyond removing the "preliminary" wording).

## Phase 0 — Make everything English  ✅ DONE
- [x] 0.1 `notebooks/xray_colab_produce_all.ipynb` — translate Turkish markdown
      cells + Turkish code comments to English (≈213 Turkish chars). Keep all code
      logic identical; re-validate the notebook still parses + cells compile.
- [x] 0.2 4 analysis notebooks (`error_analysis`, `tier_disagreement`,
      `subgroup_analysis`, `decision_curve_analysis`) — Turkish markdown/comments → English.
- [x] 0.3 `PLAN.md` — full English rewrite (≈676 Turkish chars; it is entirely
      Turkish). Also update its §1 language policy to "English-only". Consider moving to `docs/`.
- [x] 0.4 Thesis scaffold → English: `thesis/main.tex` (title page, Özet→Abstract,
      TOC, list of figures/tables) and English **section skeletons** for the 7
      chapters (`thesis/chapters/*.tex`) — headings/structure only; USER writes prose.
- [x] 0.5 Frontend EN locale — VERIFIED already 100% English (no action). ✅
- [x] 0.6 README / CHANGELOG / CONTRIBUTING — VERIFIED already English (0 Turkish chars). ✅
- [x] 0.7 Verify CI gates still green after Phase 0 (comment-lang, i18n-parity, ui-language, build).

## Phase 1 — Results: figures & tables (auto-generated)  🟡 PARTIAL
All tables/figures are produced by `scripts/build_thesis_tables.py` and
`scripts/build_thesis_figures.py` (genuine artifacts only). See the **BLOCKERS**
section below for why the per-sample items are deferred.
- [x] 1.1 **main A1–A15 ablation table** → `thesis/tables/ablation_main.tex`
      (AUC/Acc/ECE from `ablation.json`; best AUC bolded). NOTE: per-row
      bootstrap CI / DeLong are NOT in `ablation.json`, so the table reports the
      genuine AUC/Acc/ECE only; pairwise CIs live in the comparison table (1.2).
- [x] 1.2 Tier-2 backbone comparison → `thesis/tables/tier2_backbone_comparison.tex`
      (EfficientNet-B4 vs Ark+ Swin, bootstrap 95% CI + DeLong/McNemar/bootstrap
      p-values, from `statistical_comparison.csv`). **Did NOT move the old
      `model_comparison_table.tex` / `statistical_comparison_table.tex`**: their
      MobileNet/Tiered rows derive from the stale `tiered_predictions.csv` (see
      BLOCKERS). Only the EfficientNet-vs-Ark+ rows (consistent) are used.
- [x] 1.3 Dataset-statistics table → `thesis/tables/dataset_stats.tex` (NIH splits,
      genuine counts) + hyperparameter table → `thesis/tables/hyperparameters.tex`
      (genuine `config/base.yaml`). Fixed the fabricated inline ch4 table.
- [x] 1.4 (partial) Genuine figures → `thesis/figures/`: `ablation_overview.png`
      (AUC/ECE bars A1–A15) and `cross_dataset_generalization.png` (NIH A13 vs
      zero-shot CheXpert A14). **BLOCKED (need per-sample regen):** reliability/
      calibration diagram, DeLong pairwise heatmap, conformal-coverage histogram.
- [ ] 1.5 **BLOCKED** — the 4 analysis notebooks all read the stale
      `tiered_predictions.csv` (16 refs). Regenerate per-sample predictions first
      (see BLOCKERS), then re-run headless and save figures to `thesis/figures/`.
- [ ] 1.6 (USER) hand-drawn diagrams: system overview, routing flowchart, SD
      pipeline. I write the spec in `thesis/figures/SPEC.md` (TODO next).

## Phase 2 — Calibration metrics (Table 5.3)  ✅ DONE
- [x] 2.1 Added `brier_score` + `calibration_slope_intercept` (Newton/IRLS, pure
      numpy) to `core/evaluation/metrics.py` (typed, English docstrings) + 6 unit
      tests. ruff/mypy/pytest green.
- [x] 2.2 Wired into the **genuine** evaluator `scripts/evaluate_ablation.py`
      (not the stale-data stats export), so per-row JSONs carry `brier` /
      `calibration_slope` / `calibration_intercept` on the next eval run.

## Phase 3 — Bibliography  ✅ DONE
- [x] 3.1 Expanded `thesis/bibliography.bib` 5 → **45** genuine entries across
      backbones, datasets, uncertainty/calibration, conformal, generative/synthetic,
      augmentation, explainability, optimization and frameworks. The 5 cited keys
      are preserved. (unsrt prints only \cite'd entries; use \nocite{*} when drafting.)

## Phase 4 — Reproducibility & repo hygiene  🟡 PARTIAL
- [x] 4.1 Commit the genuine results (`ablation.json` 15/15). ✅ (see Done log)
- [x] 4.2 Reproducibility: added a local CI-faithful env recipe + table/figure
      regeneration steps to `docs/reproducibility.md`. (Skipped the
      expected-output checksum manifest — low value, GPU runs are non-deterministic.)
- [x] 4.3 Untracked `experiments/mlruns/` (205 files; dir was already gitignored)
      + removed `.coverage*` junk. `mlflow.db` left on disk (gitignored). (commit e471de4)
- [ ] 4.4 `git tag v1.0-thesis-defense` — **LAST**, once everything else (incl. the
      Colab regen blockers) is done. NOT yet.

## Phase 5 — Showcase polish  🟡 PARTIAL
- [x] 5.1 README: badges present; added a genuine **Results** block (headline
      AUCs + highlights + embedded ablation figure). **USER:** live-demo link +
      app screenshots still needed.
- [x] 5.2 `CHANGELOG.md` — added an `[Unreleased]` entry for the thesis work
      (the project is already at 2.0.0; version-number/tag decision left to USER).
- [ ] 5.3 HuggingFace Spaces deploy config (`scripts/deploy_huggingface.py` exists);
      USER provides HF token. I prep the Space.
- [ ] 5.4 (USER) record demo video + capture app screenshots.

---

## What the USER does (not me)
- Write the 7 thesis chapters' PROSE (English) into the scaffolds I provide.
- Hand-drawn diagrams (system overview, routing flowchart, SD pipeline) — to my spec.
- Provide HF token (deploy); record demo video; capture screenshots.
- Run the final LaTeX compile (`latexmk`) on their machine; reopen the Colab
  notebook after pulls (a running .ipynb won't refresh from git).

## Done log (append with commit SHAs)
- 2026-06-03 — Created this tracker; committed genuine 15/15 `ablation.json`. (commit: TBD)
- 2026-06-03 — Phase 0 English-only conversion complete. (commit: d12fa68)
- 2026-06-04 — Phase 3.1: bibliography 5→45 entries. (commit: d45fd1f)
- 2026-06-04 — Phase 1.1/1.2: `scripts/build_thesis_tables.py` +
  `ablation_main.tex` + `tier2_backbone_comparison.tex`, wired into ch5;
  removed ch5 fabricated threshold table + fixed fabricated conformal coverage
  (genuine target 0.95, empirical 97.3%/100%). (commit: 4314103)
- 2026-06-04 — Phase 1.3: `dataset_stats.tex` + `hyperparameters.tex`, fixed the
  fabricated ch4 hyperparameter/version/hardware claims. (commit: 0a01d1d)
- 2026-06-04 — Phase 2: `brier_score` + `calibration_slope_intercept` + tests,
  wired into `evaluate_ablation.py`. (commit: de33bc1)
- 2026-06-04 — Phase 1.4 (partial): `scripts/build_thesis_figures.py` +
  `ablation_overview.png` + `cross_dataset_generalization.png`, wired into ch5.
  (commit: f5bce6d)
- 2026-06-04 — Phase 1.6: `thesis/figures/SPEC.md` + fig:architecture slot.
  A14 reframed as preliminary. (commits: 8684acf, 2b2235a)
- 2026-06-04 — Phase 4.3: untracked 205 `experiments/mlruns/` files + coverage
  cleanup. (commit: e471de4)
- 2026-06-04 — Phase 4.2 / 5.1 / 5.2: README Results block, CHANGELOG entry,
  CI-faithful repro steps. (commit: 0925044)
- 2026-06-04 — Colab MCP: connection probed (`open_colab_browser_connection`
  returned false — no live Colab browser session). Driving Colab needs a
  one-time PC-browser sign-in; see the note in KEY DECISIONS / blockers.
