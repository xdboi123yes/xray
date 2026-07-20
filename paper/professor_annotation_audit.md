# Professor annotation audit

Cross-checked against the 76-item Hermes summary, the annotated PDF, and the current manuscript.

## Complete or editorially addressed

- **1-3, 5, 8-10, 12-13, 16, 18, 21-26, 28-31, 33-38:** TCR-P name, decision-support wording, dataset/system clarity, research question, model status, contribution bullets, abstract, section roadmap, literature-gap paragraph, and method specificity.
- **4, 14, 19-20, 28, 75, 76:** 2025-2026 reference quota fulfilled (`ma2025arkplus`, `baur2025benchmarking`, `aperstein2025multipathology`, `penso2025conformal`, `jmir2026luana` added to `references.bib` and cited in Section 2).
- **39:** Graphical Abstract generated and saved as `paper/graphical_abstract.png` (TCR-P workflow diagram).
- **61:** Figure 2 (`ablation_overview.png`) generator updated to export high-resolution (300 DPI) plots with enlarged, highly legible embedded labels (10-13pt).

## Partial - text improved, but new evidence is required

- **11, 27, 58-62:** current AUC, confidence intervals, table values, figure labels and CheXpert sensitivity are provisional until the patient-disjoint rerun. The current sensitivity of 1.000 is based on only seven positive CheXpert cases.
- **15:** code/data availability is now a separate section, but the repository URL remains explicitly marked as a placeholder.
- **17:** the frequency/first-line chest-radiography statement was softened, but still needs an authoritative citation.
- **32:** the prose has been reorganized, but the related-work citations are not yet consistently chronological.
- **40, 42, 57:** the manuscript honestly labels routing values as engineering defaults; a validation-only tuning protocol is still required if they are to be called selected or optimized.
- **44:** the conformal formula is standard, but all mathematical notation and the dynamic-routing rule need a final source-backed technical review after the pipeline is frozen.
- **54:** ImageNet normalization is now explained; exact means and standard deviations may be added if the journal expects implementation-level detail.

## Open decisions or new work

- **6:** final length can only be judged after references, new results and any graphical abstract are added.
- **7:** corresponding author is unresolved. The annotation says "myself"; identify the annotator before changing the Overleaf author block. It currently names Daniela-Maria Cristea.

## Evaluation integrity blocker

The existing CSVs use image-level splitting and contain 2,111 patients in both train and test. The calibration CSV is also a subset of validation and was counted twice in the manuscript table. Existing checkpoints and prediction files show that the numbers were computed, not merely typed into the paper, but the leakage invalidates the claimed patient-level evaluation. Do not submit the current numerical results as final.

The preprocessing and evaluation code has been changed to create mutually exclusive patient-level train, validation, calibration and test partitions and to calibrate only on the dedicated calibration split. All models and A1-A15 evaluations must now be rerun before rebuilding the manuscript tables and figures.
