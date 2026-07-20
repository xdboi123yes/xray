# Patient-disjoint rerun and manuscript completion checklist

## Why the existing outputs cannot simply be reused

The old checkpoints were trained from image-level splits. Patients with multiple radiographs occurred in both training and test data, and calibration was sampled from validation without removing those rows. Re-evaluating the old checkpoints will reproduce similar numbers, but it will not remove the information leakage. Training must restart from pretrained backbone initialization on the new patient-disjoint splits.

The old artifacts should be retained as an audit archive, not used as final paper evidence.

## 1. Decide the synthetic-data scope before spending GPU time

The current synthetic generator was designed as a small technical test (five source images and three variations each). That is not enough to support a substantive Stable-Diffusion augmentation claim. The old A8/A9/A15 training labels were also not fully enforced; the relevant training flags are now wired into the code.

Choose one of these protocols and document it before running:

1. **Recommended minimum-risk protocol:** remove Stable-Diffusion augmentation and A8 from the paper, retain classical augmentation and the correctly implemented Mixup/CutMix experiment.
2. **Full synthetic protocol:** define the number of generated images, prompts/conditioning, patient/source separation, FID reference population, acceptance rule, manual radiological quality review and a fixed synthetic-data manifest. Generate this set using training patients only before training the full models.

Do not call 15 test images a validated augmentation dataset.

AI prompt:

> Audit this chest-radiograph synthetic-data protocol for leakage, clinical plausibility and reproducibility. Require that all source images come only from the training split, define a fixed generation manifest, justify sample size and FID use, propose blinded manual review, and identify which claims must be removed if those requirements cannot be met. Do not invent experimental outcomes.

## 2. Prepare the university server

Minimum expectations:

- Linux or macOS, Python 3.10+ and Git.
- NVIDIA GPU with a CUDA-compatible PyTorch build is strongly recommended.
- Enough storage for NIH ChestX-Ray14, CheXpert validation data, checkpoints and predictions.
- Start Jupyter from the repository root, or set `XRAY_PROJECT_ROOT` explicitly.
- Put Kaggle credentials at `~/.kaggle/kaggle.json`, mode `600`, if downloads are needed.

Example:

```bash
git clone https://github.com/xdboi123yes/xray.git
cd xray
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-training.txt
python -m pip install jupyter nbconvert
jupyter lab notebooks/xray_colab_produce_all.ipynb
```

For an existing checkout:

```bash
export XRAY_PROJECT_ROOT=/absolute/path/to/xray
export XRAY_RUNTIME_ROOT=/fast/local/scratch/xray-runtime
jupyter lab
```

AI prompt:

> Given this server's `nvidia-smi`, Python, CUDA, disk and RAM output, recommend safe batch size, micro-batch, worker count and AMP settings for this notebook. Preserve MC=20 and TTA=10 unless memory prevents it. Do not change scientific settings merely to make the run faster.

## 3. Configure the notebook for the one-time clean rerun

Open `notebooks/xray_colab_produce_all.ipynb` and keep:

```python
PATIENT_DISJOINT_REBUILD = True
DRY_RUN = False
SKIP_EXISTING = True
```

`PATIENT_DISJOINT_REBUILD=True` automatically sets `FORCE_ALL=True`. The notebook may restore old files as backups, but every critical training/evaluation step must overwrite them during this session. The final integrity gate rejects a mixed old/new run.

For a local server, `USE_DRIVE` defaults to false. For Colab it defaults to true. The project root is detected from either the repository root or its `notebooks/` directory; environment variables override detection.

Run a dry-run only to validate plumbing. Never use dry-run checkpoints or metrics in the manuscript.

AI prompt:

> Review these notebook configuration values for a full patient-disjoint scientific rerun. Flag any option that skips training, reduces epochs, reduces MC/TTA passes, permits stale outputs or changes the fixed random seed. Return only concrete configuration corrections.

## 4. Run the notebook from top to bottom

The corrected flow is:

1. Detect Colab versus local/server paths.
2. Restore caches and old outputs for backup/resume infrastructure.
3. Download or locate NIH images.
4. Regenerate train, validation, calibration and test CSVs by patient.
5. Fail immediately if any patient overlaps partitions.
6. Write `outputs/provenance/split_manifest.json` with counts and SHA-256 hashes.
7. Retrain Tier 1, Tier 2 backbones and genuine training ablations.
8. Select the static Tier-1 routing threshold on validation only using Youden's J statistic.
9. Recompute conformal and temperature calibration using calibration only.
10. Re-evaluate A1-A15 and CheXpert.
11. Re-export paired per-image predictions.
12. Recompute statistics, figures and analysis notebooks.
13. Pass the scientific-integrity gate and write `outputs/provenance/rerun_manifest.json`.

If a training step fails, fix it and rerun the notebook. Its existing marker/restore machinery remains useful, but keep `PATIENT_DISJOINT_REBUILD=True` until the final integrity gate passes in one scientifically consistent production session.

AI prompt:

> Monitor this notebook log. Classify each failure as environment, data, GPU memory, checkpoint, training, calibration or evaluation. Preserve the patient-disjoint protocol and do not suggest reusing old checkpoints. Give the smallest safe recovery action and identify which downstream artifacts must be rerun.

## 5. Verify the new outputs before editing the paper

Do not rely only on the green production summary. Confirm:

- `outputs/provenance/split_manifest.json` exists and reports zero overlap.
- `outputs/provenance/rerun_manifest.json` exists and the integrity gate passed.
- All critical checkpoints and result JSONs were produced during the clean rerun.
- Prediction CSV row counts equal the new test-set row count.
- A1-A15 contain real metrics and no placeholder/null headline values except intentionally unavailable metrics.
- Statistical comparisons use paired predictions from identical test images.
- Confidence intervals are present in the manuscript tables, not merely mentioned in captions.
- CheXpert's positive sample count and uncertainty intervals are reported.

AI prompt:

> Audit these split and rerun manifests, A1-A15 JSONs and paired prediction CSVs. Recalculate headline metrics independently, verify row identities and hashes, detect leakage or stale artifacts, and produce a discrepancy report. Do not trust filenames or prose as provenance.

## 6. Rebuild the manuscript from the new results

After the integrity gate passes:

1. Regenerate dataset counts and prevalence.
2. Replace every AUC, accuracy, ECE, sensitivity, specificity, F1, MCC and coverage value.
3. Replace the abstract, results, discussion and conclusion numbers.
4. Replace tables and figures.
5. Add confidence intervals to the main performance table where feasible.
6. Update the CheXpert paragraph with the new positive count and uncertainty.
7. Remove the provisional leakage disclosure only after the corrected rerun is independently verified; replace it with the actual patient-disjoint procedure and a reproducibility statement.

AI prompt:

> Update this LaTeX manuscript strictly from the attached verified rerun artifacts. Preserve authors and affiliations. Replace all stale numerical claims, tables and figure captions; include confidence intervals and sample counts. Never infer or invent a metric. Return a claim-to-artifact traceability table with each edited claim.

## 7. Complete the reference revision

The imported Overleaf manuscript currently has one 2025 reference and no 2026 references. Before searching, confirm:

- Target journal.
- Which co-author papers are genuinely relevant.
- Whether preprints are acceptable.
- Whether the journal expects a particular citation style or reference limit.

Then add recent comparable chest-radiograph classifiers, uncertainty/calibration work, selective prediction/routing work and relevant target-journal articles. Do not pad the bibliography with unrelated citations to meet a quota.

AI prompt:

> Using only verifiable primary sources, find 3-5 relevant 2025 papers and 2-3 relevant 2026 papers on chest-radiograph classification, uncertainty calibration, selective prediction or efficient routing. Also identify relevant papers by these manuscript authors and 1-2 papers from [TARGET JOURNAL]. For each candidate give DOI, publication venue, exact relevance and the manuscript sentence it supports. Exclude unverifiable, unrelated or citation-padding candidates.

## 8. Technical and mathematical review

Review:

- Dynamic routing update direction and its intended escalation-rate objective.
- Conformal quantile finite-sample correction and exchangeability assumptions.
- Temperature-scaling split discipline.
- FID appropriateness for grayscale medical radiographs.
- Threshold-selection methodology.
- Whether all stated architectural modifications actually exist.

AI prompt:

> Act as a statistical reviewer. Check every formula and algorithm in this manuscript against its cited primary source and the implementation. Report symbol errors, mismatches between prose and code, violated assumptions, and unsupported optimization claims. Do not rewrite until each issue is evidenced.

## 9. Remaining editorial decisions

- Confirm who wrote “myself” in the corresponding-author annotation.
- Replace the placeholder repository URL.
- Confirm acknowledgements and funding.
- Create the graphical abstract only after final results stabilize.
- Regenerate Fig. 2 with larger embedded labels.
- Confirm the first affiliation's city.

AI prompt:

> Produce a submission-readiness checklist for this manuscript covering authorship approval, corresponding author, affiliations, funding, acknowledgements, repository archival link, data availability, graphical abstract, figure readability and journal-specific declarations. Mark items requiring human authorization separately from items an AI can draft.

## 10. After one successful clean rerun

Archive the entire `outputs/provenance/` directory with the submitted artifact set. Then set:

```python
PATIENT_DISJOINT_REBUILD = False
```

Future notebook runs may use the original skip-existing/resume behavior, provided the manifests and split hashes still match. If data, split code, seed, preprocessing or training configuration changes, create a new protocol version and perform another clean rebuild.
