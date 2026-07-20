#!/usr/bin/env python3
"""Replace manuscript result-bearing sections from verified production artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "manuscript.tex"
RESULTS = ROOT / "outputs" / "results"
PROVENANCE = ROOT / "outputs" / "provenance"


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"could not uniquely replace {label} (matches={count})")
    return updated


def main() -> None:
    rerun = json.loads((PROVENANCE / "rerun_manifest.json").read_text())
    splits = json.loads((PROVENANCE / "split_manifest.json").read_text())
    if rerun.get("protocol") != "patient-disjoint-v1" or splits.get("protocol") != "patient-disjoint-v1":
        raise RuntimeError("refusing to update manuscript from an unverified protocol")

    rows = json.loads((RESULTS / "ablation.json").read_text())
    by_id = {row["ablation_id"]: row for row in rows}
    required = {f"A{i}" for i in range(1, 16)}
    if missing := sorted(required - set(by_id)):
        raise RuntimeError(f"missing ablation rows: {missing}")
    for aid in required:
        metrics = by_id[aid].get("metrics", {})
        if metrics.get("auc_roc") is None or metrics.get("accuracy") is None:
            raise RuntimeError(f"{aid} lacks real headline metrics")

    def m(aid: str, key: str) -> float | None:
        value = by_id[aid]["metrics"].get(key)
        return None if value is None else float(value)

    a15_raw = json.loads((RESULTS / "A15_Mixup_Cutmix.json").read_text())
    a14_raw = json.loads((RESULTS / "A14_CheXpert_ZeroShot.json").read_text())
    coverage = float(a15_raw["conformal_empirical_coverage"])
    set_size = float(a15_raw["conformal_avg_set_size"])
    tier2_pct = float(a15_raw["percent_tier2"])
    tier1_pct = 100.0 - tier2_pct

    text = PAPER.read_text()
    abstract = (
        r"\abstract{Deep-learning decision-support systems for chest radiography commonly apply one model "
        r"to every image and provide neither calibrated uncertainty nor explicit abstention. We introduce "
        r"the \emph{Tiered Confidence Routing for Pneumothorax} (TCR-P) system and evaluate fifteen "
        rf"configurations on mutually exclusive patient-level NIH ChestX-Ray14 splits, followed by zero-shot "
        rf"CheXpert evaluation. Tier~1 resolves approximately {tier1_pct:.1f}\% of cases. The uncertainty "
        rf"stack changes expected calibration error from {fmt(m('A4','ece'))} to {fmt(m('A6','ece'))}, "
        rf"and the best configuration obtains an area under the receiver operating characteristic curve "
        rf"(AUC) of {fmt(m('A15','auc_roc'))} with {100*coverage:.1f}\% empirical conformal coverage at a "
        r"95\% target. All estimates are derived from the verified patient-disjoint production protocol.}"
    )
    text = replace_once(text, r"\\abstract\{.*?\}\n\n\\keywords", abstract + "\n\n\\keywords", "abstract")

    split_order = [("Train", "train"), ("Validation", "val"), ("Calibration", "calibration"), ("Test", "test")]
    total_rows = total_pos = 0
    split_lines = []
    for label, key in split_order:
        info = splits["splits"][key]
        n, pos = int(info["rows"]), int(info["positives"])
        neg = n - pos
        total_rows += n; total_pos += pos
        split_lines.append(f"{label} & {n:,} & {pos:,} & {neg:,} & {100*pos/n:.1f}\\%\\\\")
    split_lines.append(r"\midrule")
    split_lines.append(f"Total & {total_rows:,} & {total_pos:,} & {total_rows-total_pos:,} & {100*total_pos/total_rows:.1f}\\%\\\\")
    dataset_table = "\n".join([
        r"\begin{table}[t]", r"\caption{Patient-disjoint NIH ChestX-Ray14 dataset splits.}\label{tab:dataset}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Split & Images & Pneumothorax & Normal & Prevalence\\", r"\midrule",
        *split_lines, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])
    text = replace_once(text, r"\\begin\{table\}\[t\]\n\\caption\{NIH ChestX-Ray14 dataset splits\.\}.*?\\end\{table\}", dataset_table, "dataset table")

    ablation_lines = []
    for aid in sorted(required, key=lambda x: int(x[1:])):
        row = by_id[aid]
        name = str(row["name"]).replace("_", r"\_")
        ablation_lines.append(f"{aid} & {name} & {fmt(m(aid,'auc_roc'))} & {fmt(m(aid,'accuracy'))} & {fmt(m(aid,'ece'))}\\\\")
    ablation_table = "\n".join([
        r"\begin{table}[t]", r"\caption{Patient-disjoint ablation study.}\label{tab:ablation}",
        r"\small", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"ID & Configuration & AUC & Acc. & ECE\\", r"\midrule",
        *ablation_lines, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])
    text = replace_once(text, r"\\begin\{table\}\[t\]\n\\caption\{Ablation study.*?\\end\{table\}", ablation_table, "ablation table")

    results_intro = (
        rf"\textbf{{Overall classification performance.}} Table~\ref{{tab:ablation}} reports the verified "
        rf"patient-disjoint results. The best AUC is {fmt(m('A15','auc_roc'))} (A15), compared with "
        rf"{fmt(m('A1','auc_roc'))} for Tier~1 alone and {fmt(m('A2','auc_roc'))} for EfficientNet-B4 alone."
    )
    text = replace_once(text, r"\\textbf\{Overall classification performance\.\}.*?\n\n(?=\\begin\{table\})", results_intro + "\n\n", "results introduction")

    discussion = (
        rf"\section{{Discussion}}\label{{sec:discussion}}\n\n"
        rf"First, TCR-P resolves {tier1_pct:.1f}\% of cases through Tier~1 and invokes Tier~2 for the remaining "
        rf"{tier2_pct:.1f}\%, providing selective computation under the verified patient-disjoint protocol.\n\n"
        rf"Second, the uncertainty stack changes ECE from {fmt(m('A4','ece'))} to {fmt(m('A6','ece'))}; "
        rf"the conformal component achieves {100*coverage:.1f}\% empirical coverage with average set size "
        rf"{set_size:.2f}.\n\n"
        rf"Third, zero-shot CheXpert evaluation yields AUC {fmt(m('A14','auc_roc'))}, sensitivity "
        rf"{fmt(a14_raw['metrics'].get('sensitivity'))} and specificity {fmt(a14_raw['metrics'].get('specificity'))}. "
        r"These operating-point estimates should be interpreted with their external-cohort sample counts and require site-specific validation.\n\n"
        r"\textbf{Limitations.} This remains a retrospective binary study using report-mined labels. The routing parameters are partly heuristic, conformal validity assumes exchangeability, and prospective multi-site clinical validation has not been performed.\n\n"
        r"Future work should validate learned routing policies, adaptive conformal sets, multi-label prediction and site-specific recalibration.\n\n"
    )
    text = replace_once(text, r"\\section\{Discussion\}\\label\{sec:discussion\}.*?(?=\\section\{Conclusion\})", discussion, "discussion")

    conclusion = (
        rf"\section{{Conclusion}}\label{{sec:conclusion}}\n\nWe proposed Tiered Confidence Routing for Pneumothorax "
        rf"(TCR-P). Under the verified patient-disjoint protocol, the best configuration reaches AUC "
        rf"{fmt(m('A15','auc_roc'))}, Tier~1 resolves {tier1_pct:.1f}\% of cases, and empirical conformal "
        rf"coverage is {100*coverage:.1f}\% at a 95\% target. These results support TCR-P as a research "
        r"prototype; prospective multi-site validation is required before clinical deployment.\n\n"
    )
    text = replace_once(text, r"\\section\{Conclusion\}\\label\{sec:conclusion\}.*?(?=\\section\*\{Acknowledgements\})", conclusion, "conclusion")

    PAPER.write_text(text)
    print(f"updated {PAPER} from verified patient-disjoint outputs")


if __name__ == "__main__":
    main()
