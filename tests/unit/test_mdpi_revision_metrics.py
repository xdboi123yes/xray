"""Unit tests for MDPI peer-review revision additions.

Covers:
- BatchNorm eval-mode preservation and Law-of-Total-Variance decomposition (C8)
- 15-bin equal-width ECE, class-wise ECE, NLL, and Wilson score CIs (C9, C13)
- Patient-clustered bootstrap, Holm-Bonferroni correction, and Cohen's h (C11)
- Dynamic routing simulation, order sensitivity, and clinical threshold sweep (C6, C7, C15)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.evaluation.metrics import (
    classwise_ece,
    negative_log_likelihood,
    reliability_bin_stats,
    wilson_ci,
)
from core.evaluation.stats import bootstrap_ci, cohens_h, holm_bonferroni
from core.models.tier2_efficientnet import Tier2EfficientNet


def _load_revision_experiments():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "revision_experiments.py"
    spec = importlib.util.spec_from_file_location("revision_experiments", str(script_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["revision_experiments"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tier2_mc_tta_keeps_batchnorm_eval_and_decomposes_variance() -> None:
    model = Tier2EfficientNet(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    # Verify BatchNorm stays in eval() when _enable_dropout_only is active
    model._enable_dropout_only()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            assert not m.training, "BatchNorm must remain in eval mode during MC-Dropout"
    model.eval()

    mean_p, total_v, epi_v, tta_v = model.mc_tta_forward_decomposed(x, T=3, n_augments=2)
    assert mean_p.shape == (1, 2)
    assert torch.allclose(total_v, epi_v + tta_v, atol=1e-6)
    assert (epi_v >= 0).all() and (tta_v >= 0).all()


def test_calibration_metrics_nll_classwise_ece_and_wilson() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.8])
    nll = negative_log_likelihood(y, p)
    assert 0.0 < nll < 0.3

    cw = classwise_ece(y, p, n_bins=15)
    assert 0.0 <= cw["ece_class0"] <= 1.0
    assert 0.0 <= cw["ece_class1"] <= 1.0
    assert np.isclose(cw["classwise_ece"], 0.5 * (cw["ece_class0"] + cw["ece_class1"]))

    bins = reliability_bin_stats(y, p, n_bins=15)
    assert len(bins) == 15
    assert sum(int(b["count"]) for b in bins) == len(y)

    lo, hi = wilson_ci(7, 7)
    assert 0.5 < lo < hi <= 1.0


def test_patient_clustered_bootstrap_holm_and_cohens_h() -> None:
    rng = np.random.default_rng(7)
    n = 120
    pids = np.array([f"{i // 3:08d}" for i in range(n)])
    y = (rng.random(n) < 0.3).astype(int)
    p1 = np.clip(y * 0.6 + (1 - y) * 0.2 + rng.normal(0, 0.15, n), 0.01, 0.99)
    p2 = np.clip(y * 0.7 + (1 - y) * 0.15 + rng.normal(0, 0.12, n), 0.01, 0.99)

    res = bootstrap_ci(y, p1, p2, metric="auc", n_iterations=100, seed=42, patient_ids=pids)
    assert res["resampling_unit"] == "patient"
    assert res["model1_ci"][0] <= res["model1_metric"] <= res["model1_ci"][1]

    adj = holm_bonferroni([0.001, 0.02, 0.20])
    assert np.allclose(adj, [0.003, 0.04, 0.20])

    h = cohens_h(0.80, 0.90)
    assert h > 0.25


def test_revision_experiments_order_sensitivity_and_threshold_sweep(tmp_path: Path) -> None:
    mod = _load_revision_experiments()
    rng = np.random.default_rng(123)
    n = 150
    y = (rng.random(n) < 0.2).astype(int)
    t1 = np.clip(y * 0.65 + (1 - y) * 0.18 + rng.normal(0, 0.18, n), 0.01, 0.99)
    t2 = np.clip(y * 0.78 + (1 - y) * 0.12 + rng.normal(0, 0.12, n), 0.01, 0.99)

    csv_path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "image_id": [f"{i:08d}_000.png" for i in range(n)],
            "y_true": y,
            "tier1_prob": t1,
            "tier2_prob": t2,
            "tiered_prob": np.where(np.maximum(t1, 1 - t1) >= 0.75, t1, t2),
            "prob_uncal": t2,
            "prob_cal": t2,
        }
    ).to_csv(csv_path, index=False)

    order_json = tmp_path / "order.json"
    rep = mod.run_order_sensitivity(str(csv_path), n_permutations=4, output_json=str(order_json))
    assert order_json.exists()
    assert len(rep["random_order_runs"]) == 4
    assert "easy_to_hard_sorted" in rep["drift_stress_tests"]

    sweep_csv = tmp_path / "sweep.csv"
    sdf = mod.run_threshold_sweep(str(csv_path), output_csv=str(sweep_csv))
    assert sweep_csv.exists()
    assert "fnr" in sdf.columns and len(sdf) >= 5

    cal_json = tmp_path / "cal.json"
    cal_png = tmp_path / "cal.png"
    crep = mod.run_calibration_report(str(csv_path), output_json=str(cal_json), output_fig=str(cal_png))
    assert cal_json.exists() and cal_png.exists()
    assert crep["after_temperature_scaling"]["n_bins"] == 15

    # Clean up matplotlib env if needed
    assert os.path.getsize(cal_png) > 1000
