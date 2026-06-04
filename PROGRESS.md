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

## Phase 0 — Make everything English  ⏳ IN PROGRESS
- [x] 0.1 `notebooks/xray_colab_produce_all.ipynb` — translate Turkish markdown
      cells + Turkish code comments to English (≈213 Turkish chars). Keep all code
      logic identical; re-validate the notebook still parses + cells compile.
- [x] 0.2 4 analysis notebooks (`error_analysis`, `tier_disagreement`,
      `subgroup_analysis`, `decision_curve_analysis`) — Turkish markdown/comments → English.
- [x] 0.3 `PLAN.md` — full English rewrite (≈676 Turkish chars; it is entirely
      Turkish). Also update its §1 language policy to "English-only". Consider moving to `docs/`.
- [ ] 0.4 Thesis scaffold → English: `thesis/main.tex` (title page, Özet→Abstract,
      TOC, list of figures/tables) and English **section skeletons** for the 7
      chapters (`thesis/chapters/*.tex`) — headings/structure only; USER writes prose.
- [x] 0.5 Frontend EN locale — VERIFIED already 100% English (no action). ✅
- [x] 0.6 README / CHANGELOG / CONTRIBUTING — VERIFIED already English (0 Turkish chars). ✅
- [ ] 0.7 Verify CI gates still green after Phase 0 (comment-lang, i18n-parity, ui-language, build).

## Phase 1 — Results: figures & tables (auto-generated)  ⬜ TODO
- [ ] 1.1 Generate the **main A1–A15 ablation table** (AUC + bootstrap CI +
      DeLong p-value where available) as LaTeX from `ablation.json` →
      `thesis/tables/ablation_main.tex`. Add a small generator script.
- [ ] 1.2 Move existing generated `.tex` (`outputs/results/model_comparison_table.tex`,
      `statistical_comparison_table.tex`) → `thesis/tables/` and `\input` them.
- [ ] 1.3 Dataset-statistics table (Tbl 4.1) + hyperparameter table (Tbl 4.2) → `thesis/tables/`.
- [ ] 1.4 Generate missing data-driven figures → `thesis/figures/`:
      conformal-coverage histogram, DeLong pairwise-significance heatmap,
      NIH-vs-CheXpert cross-dataset bars, reliability/calibration diagram.
- [ ] 1.5 Re-run the 4 executed analysis notebooks headless and SAVE their figures
      to `thesis/figures/` (currently the figures are trapped inside the .ipynb).
- [ ] 1.6 (USER) hand-drawn diagrams: system overview, routing flowchart, SD
      pipeline. I will write an exact spec for each in `thesis/figures/SPEC.md`.

## Phase 2 — Calibration metrics (Table 5.3)  ⬜ TODO
- [ ] 2.1 Add `brier_score` + `calibration_slope_intercept` to
      `core/evaluation/metrics.py` (typed, English docstrings, + unit tests).
- [ ] 2.2 Wire them into the stats/prediction export so the calibration table is complete.

## Phase 3 — Bibliography  ⬜ TODO
- [ ] 3.1 Expand `thesis/bibliography.bib` 5 → ~40 English entries (CheXNet,
      EfficientNet, MobileNetV2, Swin, Ark+, Gal&Ghahramani MC-Dropout, Angelopoulos/
      Romano conformal, Stable Diffusion, NIH ChestX-ray14, CheXpert, Grad-CAM/HiResCAM…).

## Phase 4 — Reproducibility & repo hygiene  ⬜ TODO
- [x] 4.1 Commit the genuine results (`ablation.json` 15/15). ✅ (see Done log)
- [ ] 4.2 Reproducibility appendix: exact `make` commands, conda env export,
      expected-output checksum manifest → `thesis/` appendix + `docs/reproducibility.md`.
- [ ] 4.3 Untrack `experiments/mlruns/` (469 committed log files) + gitignore;
      remove stray artifacts (`.coverage *`, `.DS_Store`, root `mlflow.db`).
- [ ] 4.4 `git tag v1.0-thesis-defense` — LAST, once everything else is done.

## Phase 5 — Showcase polish  ⬜ TODO
- [ ] 5.1 README: status/license/python badges, a results-highlight block (headline
      AUCs), the live-demo link, and app screenshots.
- [ ] 5.2 `CHANGELOG.md` — v1.0 release entry.
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
