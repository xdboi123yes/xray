# Point-by-Point Response to Reviewers — MDPI *Diagnostics*

**Manuscript Title:** *TCR-P: A Tiered Confidence-Routing System for Pneumothorax Detection in Chest Radiographs*  
**Authors:** Daniela-Maria Cristea, Alperen Yörük, Henning Müller, and László Barna Iantovics  
**Revision Package:** Track A (Methodological, Mathematical, Analytical-Complexity & Reporting Revision) + Track B Reproducible GPU Rerun Pipeline (`v1.0-mdpi-rev1`)

---

> **Internal Note for Co-Authors (Prof. László Barna Iantovics & Prof. Henning Müller):**  
> This document provides a complete, honest, point-by-point response aligned with **Track A** (`paper/mdpi/track_a_no_rerun/manuscript_clean.tex` and `manuscript_highlighted.tex`, revisions highlighted in blue). Passages marked **`[TRACK B UPGRADE]`** indicate where our updated Colab Pro GPU pipeline (`notebooks/xray_colab_produce_all.ipynb`) produces strictly patient-disjoint retrained metrics, 10-seed order-sensitivity tables, 15-bin reliability plots, patient-clustered bootstrap 95% CIs, expanded CheXpert zero-shot evaluation, and CUDA-synchronized per-stage GPU timings—if we choose to submit the retrained Track B version after the Colab run finishes.  
> **Reference audit flag for co-authors:** Please double-check five bibliography entries in `ref.bib` (`sharma2025mobilenetv2`, `chen2026efficientnet`, `patel2025vision`, `gupta2026calibrated`, `jmir2026luana`) to verify their exact DOIs/publication records before final submission.

---

We sincerely thank both reviewers for their thorough, constructive, and insightful feedback. Every comment has been carefully examined, and all corresponding revisions are highlighted in blue in the marked-up manuscript (`manuscript_highlighted.tex`). Below we provide a detailed point-by-point response.

---

## Response to Reviewer 1

### R1.1 — English language and long sentences
> *"The English could be improved to more clearly express the research... Some of the longer sentences need grammar enhancement."*

**Response:**  
We carefully edited the manuscript to break up overly long sentences, tighten phrasing, and correct grammatical errors (e.g., *"such models are poor probability calibration"* $\to$ *"routinely exhibit poor probability calibration"*, *"Unlike Gradient-weighted Grad-CAM"* $\to$ *"Unlike Gradient-weighted Class Activation Mapping (Grad-CAM)"*, *"limiting clinical decision"* $\to$ *"limiting clinical utility"*), and a senior co-author will complete a final native-level English copy-edit prior to final upload.

### R1.2 — Undefined parameters on page 4 ($f_1$, outer indices of $p_1$, etc.)
> *"Some of the parameters on page 4, lines 144-145, are not defined. e.g. f1, the outer indices of p1, etc."*

**Response:**  
In Section 3.2 (*Tiered Classifier and Notation*), we now formally define the binary label space $\mathcal{Y} = \{0, 1\}$ ($0 = \text{No Finding}$, $1 = \text{Pneumothorax}$), the Tier-1 network mapping $f_1 : \mathbb{R}^{3\times 224\times 224} \to \mathbb{R}^2$ outputting raw class logits $(f_1(x)_0, f_1(x)_1)$, the softmax probability vector $p_1(x) = \mathrm{softmax}(f_1(x)) = (p_1(x)_0, p_1(x)_1) \in [0,1]^2$ with outer indices $0$ and $1$ explicitly tied to the two classes, the scalar confidence $c_1(x) = \max(p_1(x)_0, p_1(x)_1) \in [0.5, 1.0]$, and the Tier-2 mapping $f_2$.

### R1.3 — How the 74% Tier-1 resolution rate was established
> *"How the level of 74% cases handled by the Tier 1 has been established (page 4, line 158) - what was the experimental setup, the dataset(s) used, etc.?"*

**Response:**  
We now explicitly state in Section 3.2 and Figure 1's caption that the $\approx 74\%$ Tier-1 resolution rate (and complementary $\approx 26\%$ Tier-2 escalation rate) was measured empirically on the NIH ChestX-Ray14 test split ($n=4{,}772$ radiographs) under the dynamic routing configuration ($\tau_0 = 0.75$, $W=50$, $\delta=0.05$).  
*`[TRACK B UPGRADE]`: In Track B, this figure updates to the exact mean $\pm$ SD measured across 10 random permutations of the strictly patient-disjoint NIH test split ($n=4{,}850$, $2{,}261$ unique patients).*

### R1.4 — Empirical selection of $\tau_0$, $W$, and $\delta$
> *"Is it possible to set tau0, W and delta empirically (and more correctly) rather than 'non-optimized engineering defaults' as stated on page 4, lines 166-167? How this could be achieved?"*

**Response:**  
Yes—and we now explain the exact validation-only optimisation protocol in Section 3.3 (*Adaptive Confidence-Based Routing*): $(\tau_0, W, \delta)$ can be selected systematically via a grid search on the held-out validation split over $(\tau_0, W, \delta) \in \{0.65, 0.70, 0.75, 0.80, 0.85\} \times \{25, 50, 100\} \times \{0.00, 0.02, 0.05, 0.10\}$, optimising a Pareto frontier that satisfies a clinical sensitivity / false-negative-rate constraint subject to a maximum Tier-2 escalation budget (implemented in `scripts/revision_experiments.py --routing-grid`).  
*`[TRACK B UPGRADE]`: Track B includes the empirical validation grid-search Pareto table directly in Results.*

### R1.5 — Definition of $y$ and $\hat{f}$ in Equation (2)
> *"What are y and f_hat from Eq. (2)?"*

**Response:**  
In Section 3.5 (*Post-Hoc Calibration and Split Conformal Prediction*), we now explicitly define $y \in \mathcal{Y} = \{0, 1\}$ as a candidate binary class label (*No Finding* or *Pneumothorax*) and $\hat{f}(x)_y \in [0, 1]$ as the Tier-2 model's estimated probability for class $y$ (computed after temperature scaling and MC-Dropout/TTA averaging).

### R1.6 — Meaning of CLAHE $224\times 224$ in Figure 1 and selection of 224
> *"What is the meaning of CLAHE 224x224 from the preprocessing block in Fig. 1? Why the value of 224 was selected?"*

**Response:**  
In Section 4.2 (*Preprocessing and Training Protocol*) and Table 2, we now specify that radiographs undergo Contrast-Limited Adaptive Histogram Equalisation (CLAHE; clip limit $2.0$, $8\times 8$ tile grid) to enhance local pleural-line contrast while limiting noise amplification, followed by resizing to $224\times 224$ pixels. We explicitly explain why $224\times 224$ was chosen: it is the native structural input resolution required by the pretrained Swin-Base Transformer (`swin_base_patch4_window7_224`, $4\times 4$ patches and $7\times 7$ shifted windows) and matches standard ImageNet-pretrained MobileNetV2 and EfficientNet-B4 checkpoints.

### R1.7 — Definition of $\Delta$ on page 8
> *"What is capital_delta on page 8, line 295?"*

**Response:**  
In Section 5.2 (*Tier-2 Backbone Comparison*) and Table 4's caption, we now define $\Delta$ explicitly at first mention in the main text as the paired metric difference *Ark+ Swin minus EfficientNet-B4* (positive values favour Ark+ Swin).

### R1.8 — Legibility of Figure 2 at 100% zoom
> *"The text in Fig. 2 is too small to read at 100% zoom. It should be enlarged."*

**Response:**  
We regenerated Figure 2 (`ablation_overview.png`, updated via `scripts/build_thesis_figures.py`) at 300 DPI with enlarged axis labels (14 pt), configuration labels (13 pt bold), value annotations (12 pt bold), improved gridlines, and wider AUC axis bounds (`[0.75, 0.96]`) so that all bars (including A14) and text are clearly legible at 100% zoom.

### R1.9 — Theoretical and experimental computational complexity
> *"There are no theoretical (computational complexities) or experimental (execution times for training, testing, etc.) time consumption levels in the paper..."*

**Response:**  
We added a dedicated complexity and efficiency table (Table 5 in Section 5.4) reporting exact parameter counts and forward-pass GFLOPs at $224\times 224$ for every backbone (MobileNetV2: $2.88\text{M}$ / $0.60\text{ GFLOPs}$; EfficientNet-B4: $18.47\text{M}$ / $3.01\text{ GFLOPs}$; Ark+ Swin-Base: $87.27\text{M}$ / $30.25\text{ GFLOPs}$), apples-to-apples expected GFLOPs for both single-pass and full MC+TTA cascades ($54$--$74\%$ FLOPs reduction relative to uniform Tier-2 execution), and transparent discussion of the single-pass vs. 200-pass MC+TTA trade-off in Section 6.  
*`[TRACK B UPGRADE]`: Track B populates Table 5 with full CUDA-synchronized GPU timings (training wall-clock times, preprocessing + H2D transfer, single-pass, MC, TTA, vectorized MC+TTA latency, and peak VRAM) via the upgraded `scripts/benchmark_latency.py`.*

---

## Response to Reviewer 2

### R2-C1 — Ambiguity in the actual structure of Stage 2 (independent alternatives vs. ensemble)
**Response:**  
Thank you for catching this inconsistency between our Methods and Conclusions. EfficientNet-B4 and Ark+ Swin are **independent, mutually exclusive Tier-2 backbone alternatives**, *not* a combined ensemble. Specifically, configurations A2--A10 instantiate Tier~2 with EfficientNet-B4 (convolutional comparator), while A11--A15 instantiate Tier~2 with Ark+ Swin (selected as the primary Tier-2 backbone for A13/A15 due to higher sensitivity, $+6.3$ percentage points, $p_{\text{Holm}}=0.004$). We corrected the Abstract, Introduction, Section 3.1, Figure 1 (now explicitly reading *"Either EfficientNet-B4 OR Ark+ Swin"*), Table 2, Table 3, and Conclusions so the text and diagram match the implementation identically.

### R2-C2 — Clarifying patient-level data partitioning and patient counts
**Response:**  
We expanded Section 4.1 and Table 1 to describe the exact 8-digit NIH patient-ID prefix grouping (`Image Index` prefix before `_`) and stratified patient assignment, and we added a **Unique Patients** column to Table 1. Crucially, our post-submission audit of the legacy CSVs used for the baseline tables revealed multi-radiograph patient overlap across partitions. Rather than obscuring this finding, we transparently disclose it in Table 1's footnote and Section 6.2 (Limitation 1), and we updated `scripts/preprocess.py` to strictly enforce zero pairwise patient overlap across all four splits (Train: 10,549 patients / 22,234 images; Validation: 1,582 patients / 3,259 images; Calibration: 679 patients / 1,469 images; Test: 2,261 patients / 4,850 images; total 15,071 patients; pairwise overlap $= 0$).  
*`[TRACK B UPGRADE]`: Track B replaces all Tables 1, 3, 4, and Figures 2--3 with newly trained, strictly patient-disjoint results.*

### R2-C3 — Ambiguity in diagnostic label extraction and exclusion criteria
**Response:**  
Section 4.1 now provides exact inclusion and exclusion criteria verified directly against `Data_Entry_2017.csv` (112,120 radiographs, 30,805 patients):
- **Positive ($y=1$, Pneumothorax):** Any radiograph whose `Finding Labels` contains `"Pneumothorax"` (2,194 isolated + 3,108 comorbid with other thoracic findings = 5,302 images across 1,487 patients).
- **Negative ($y=0$, No Finding):** Only radiographs labelled strictly `"No Finding"` (60,361 images across 24,907 patients), randomly capped at a 5:1 negative-to-positive ratio (seed 42).
- **Excluded:** All 46,457 radiographs (14,141 patients) with non-pneumothorax thoracic pathologies were excluded from the binary negative pool. We explicitly discuss in Section 6.2 how excluding other pathologies simplifies the negative distribution relative to unselected hospital workflows.

### R2-C4 — Details on synthetic data generation and quality evaluation
**Response:**  
Section 3.6 and Table 2 now detail the exact generative pipeline: off-the-shelf `runwayml/stable-diffusion-v1-5` (`StableDiffusionImg2ImgPipeline`, no diffusion fine-tuning), conditioned exclusively on real training-set pneumothorax radiographs at $512\times 512$ (3 variations per seed image, `strength=0.30`, `guidance_scale=7.5`, 50 inference steps, exact positive and negative clinical prompts, resized to $224\times 224$), filtered by batch-level Inception-v3 Fréchet Inception Distance ($\mathrm{FID} \le 50.0$ via `pytorch-fid`). Downstream calibration impact is quantified via A6 (with synthetics, ECE 0.050) vs. A8 (without synthetics, ECE 0.081), and we explicitly note the absence of board-certified radiologist adjudication of synthetic images as a limitation in Section 6.2.  
*`[TRACK B UPGRADE]`: Track B regenerates synthetics strictly from the patient-disjoint training split, reports exact generated/accepted counts and batch FIDs, and adds a qualitative real-vs-synthetic comparison figure and real-classifier consistency audit (`scripts/revision_experiments.py --synthetic-audit`).*

### R2-C5 — Independence of calibration and validation sets
**Response:**  
We resolved the contradiction between Methods and Experimental Design in Section 3.5 and Table 1: post-hoc temperature scaling ($T_{\text{cal}}$) is fitted exclusively on the **validation** split by minimising NLL, whereas split-conformal non-conformity quantiles ($\hat{q}$) are computed exclusively on the separate, mutually exclusive **calibration** split using temperature-scaled probabilities. This eliminates reuse of calibration samples across both steps and preserves split-conformal exchangeability. Both `scripts/evaluate_ablation.py` and `notebooks/xray_colab_produce_all.ipynb` have been updated accordingly.

### R2-C6 — Adaptive threshold mechanism and pseudocode
**Response:**  
Section 3.3 now specifies the exact deadband controller (Equation 1) and includes full step-by-step pseudocode (**Algorithm 1**, Table 1 in Section 3.3): warm-up over the first $W=50$ radiographs at $\tau_0=0.75$, rolling mean $\bar{c}_t$ over the FIFO window of length $W$, deadband $[\bar{c}_{\text{low}}, \bar{c}_{\text{high}}] = [0.65, 0.85]$, clamping to $[\tau_{\min}, \tau_{\max}] = [0.50, 0.95]$, and exact label-free update mechanics under easy vs. hard streams. We also corrected inverted comments in `core/routing/dynamic_router.py` and `core/models/tiered_system.py`.

### R2-C7 — Sensitivity to sample arrival order
**Response:**  
We acknowledge in Section 3.3 and Section 6.2 (Limitation 4) that online sliding-window updates introduce potential arrival-order dependence under non-stationary streams, and we implemented a dedicated evaluation harness (`scripts/revision_experiments.py --order-sensitivity`) that evaluates 10 random permutations plus adversarial drift orderings (easy-to-hard, hard-to-easy, and prevalence blocks), reporting mean $\pm$ SD and min--max of referral rate, sensitivity, specificity, AUC, and inference time.  
*`[TRACK B UPGRADE]`: Track B reports the complete 10-seed permutation and drift-order table in Results.*

### R2-C8 — Separating sources of uncertainty (MC-Dropout vs. TTA)
**Response:**  
Section 3.4 now formalises the **Law of Total Variance** decomposition (Equation 2), cleanly separating within-augmentation model/epistemic variance ($\sigma^2_{\text{MC}} = \frac{1}{K}\sum_k \operatorname{Var}_t(p^{(k,t)})$) from between-augmentation input-perturbation sensitivity ($\sigma^2_{\text{TTA}} = \operatorname{Var}_k(\frac{1}{T}\sum_t p^{(k,t)})$), and states the exact human-referral rule ($\sigma^2_{\text{total}}(x)_{\hat{y}} > 0.10$ or $|C(x)| > 1$). In the codebase, we also fixed `mc_forward`/`mc_tta_forward` in `core/models/tier2_efficientnet.py` and `core/models/tier2_ark.py` to strictly preserve BatchNorm in `eval()` mode, implemented `mc_tta_forward_decomposed()`, and added a dedicated TTA-only ablation row (`A5b`) in `scripts/ablation_spec.py` to complete the $2\times 2$ factorial comparison (A4 None, A5 MC-only, A5b TTA-only, A6 MC+TTA).  
*`[TRACK B UPGRADE]`: Track B reports the full $2\times 2$ MC vs. TTA table and decomposed epistemic/TTA variances.*

### R2-C9 — Calibration probability assessment (15 bins, Brier, NLL, class-ECE, reliability plots)
**Response:**  
Section 3.5 now explicitly defines ECE over $M=15$ equal-width confidence bins on $[0, 1]$ (width $1/15$) and clarifies which ablation rows report pre- vs. post-temperature-scaling probabilities. We standardised `ece_bins: 15` across `config.yaml` and `core/uncertainty/calibration.py`, added `negative_log_likelihood`, `classwise_ece`, and `reliability_bin_stats` in `core/evaluation/metrics.py`, and wired 15-bin pre/post-temperature reliability diagrams into `scripts/revision_experiments.py --calibration-report`.  
*`[TRACK B UPGRADE]`: Track B includes the side-by-side 15-bin reliability diagram figure and full ECE / class-ECE / Brier / NLL table.*

### R2-C10 — Revising the conformal prediction coverage claim
**Response:**  
Section 3.5, Section 5.4, and Section 6.2 (Limitation 3) now explicitly distinguish **finite-sample statistical marginal coverage under exchangeability** from **clinical diagnostic utility**. We explicitly quantify that an average binary set size of $1.84$ implies roughly $84\%$ two-class abstention sets $\{0,1\}$ and only $\approx 16\%$ actionable singleton sets, acknowledging this conservativeness as a key limitation motivating adaptive conformal scores (APS/RAPS) in future work. `scripts/evaluate_ablation.py` now logs exact singleton rate, two-class rate, empty-fallback rate, and class-conditional coverage.  
*`[TRACK B UPGRADE]`: Track B reports exact empirical singleton, two-class, and per-class coverage rates.*

### R2-C11 — Statistical comparisons in Table 4 (CIs, resampling unit, multiplicity, effect sizes)
**Response:**  
In Table 4 and Section 5.2, we removed the misleading caption claim about missing CIs in Track A, added **effect sizes** ($\Delta$ for AUC; **Cohen's $h$** for proportion metrics: sensitivity $h=+0.177$, specificity $h=-0.119$, accuracy $h=-0.071$, $F_1$ $h=-0.044$), applied **Holm--Bonferroni step-down adjustment** across the five compared metrics ($p_{\text{Holm}}$ reported alongside $p_{\text{raw}}$), and explicitly stated the bootstrap resampling unit (image-level in legacy artifacts). We also upgraded `core/evaluation/stats.py` and `scripts/statistical_tests.py` to perform **patient-clustered bootstrap** (2,000 resamples) outputting complete 95% CIs, Holm-adjusted $p$-values, and Cohen's $h$.  
*`[TRACK B UPGRADE]`: Track B populates Table 4 with exact patient-clustered 95% bootstrap CIs.*

### R2-C12 — Tempering claims of superiority over single-stage MobileNetV2
**Response:**  
We substantially toned down superiority language across the Abstract, Section 5.1, Section 6.1, and Conclusions. We explicitly acknowledge that MobileNetV2 alone (A1, AUC 0.919) is highly competitive and that the $+0.004$--$0.005$ AUC gain of A13/A15 is modest; we accurately position TCR-P's practical benefit around **probability calibration ($0.074 \to 0.050$), accuracy improvement ($0.849 \to 0.882$--$0.889$), selective compute ($54$--$74\%$ FLOPs savings vs. uniform Tier 2), and calibrated deferral**. `scripts/statistical_tests.py` includes direct paired testing of A1 (MobileNetV2) vs. the full TCR-P cascade.

### R2-C13 — External evaluation limitations on CheXpert
**Response:**  
Section 5.5 and Section 6.2 (Limitation 2) now report exact **95% Wilson score confidence intervals** for CheXpert validation performance: sensitivity $1.000$ ($7/7$, 95\% CI $0.646$--$1.000$), specificity $0.318$ ($62/195$, 95\% CI $0.257$--$0.386$), and accuracy $0.342$ ($69/202$, 95\% CI $0.280$--$0.409$). We explicitly state that $100\%$ sensitivity on 7 positive cases cannot demonstrate clinical generalisability and analyse how the $5\times$ prevalence drop ($16.7\% \to 3.5\%$) and acquisition/labelling shifts cause fixed-threshold specificity collapse. We also upgraded `scripts/evaluate_chexpert.py` and `notebooks/xray_colab_produce_all.ipynb` to evaluate an expanded patient-disjoint CheXpert zero-shot cohort (up to 4,000 frontal radiographs from CheXpert `train.csv`, never seen during NIH training) alongside the 202-image expert-adjudicated validation cohort.  
*`[TRACK B UPGRADE]`: Track B reports both CheXpert cohorts with 95% CIs and threshold-recalibration curves.*

### R2-C14 — Computational efficiency assessment and MC+TTA timing
**Response:**  
Section 5.4 (Table 5) now transparently separates single-pass timing ($5.4\,\text{ms}$ Tier 1 vs. $17.2\,\text{ms}$ Tier 2 EfficientNet-B4; batch size 1, 10 warmup, 100 timed passes; $9.9\,\text{ms}$ expected single-pass latency) from full uncertainty-enabled inference ($T=20 \times K=10 = 200$ passes on escalated cases), explicitly stating that the $9.9\,\text{ms}$ figure excludes MC+TTA and reporting apples-to-apples FLOPs reductions for both single-pass ($54.1\%$--$72.0\%$) and full MC+TTA cascades ($73.9\%$--$74.0\%$, $\approx 3.8\times$ reduction vs. A10/A12). We upgraded `scripts/benchmark_latency.py` to record CUDA-synchronized timings with 20 warmup / 100 timed iterations, preprocessing + H2D transfer overhead, per-stage MC/TTA/vectorized MC+TTA latency, peak GPU VRAM, parameters, and GFLOPs.  
*`[TRACK B UPGRADE]`: Track B populates Table 5 with complete GPU measurements across all backbones and uncertainty modes.*

### R2-C15 — Clinical error analysis and reproducibility
**Response:**  
Section 6.1 now explicitly analyses the asymmetric clinical cost of false negatives (missed pneumothoraces) versus false positives and details threshold selection for high-sensitivity screening targets (e.g., $\ge 95\%$ sensitivity, FNR $\le 5\%$), supported by `scripts/revision_experiments.py --threshold-sweep`. For full reproducibility, the Data Availability Statement removes placeholder language and pins the repository (`https://github.com/xdboi123yes/xray`, release tag `v1.0-mdpi-rev1`) containing exact preprocessing/splitting code, `config.yaml`, CI-verified tests, frozen library versions (`requirements-lock.txt` exported during Colab execution), and model-weight distribution instructions.
