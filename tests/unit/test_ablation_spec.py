"""Unit tests for the canonical A1-A15 ablation specification.

Guards the single source of truth shared by ``scripts/build_ablation_json.py``
and ``scripts/evaluate_ablation.py`` so the full thesis ablation table stays
internally consistent (unique ids/run names, valid evaluation modes, sane
uncertainty pass counts).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

import pytest

_SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
    "ablation_spec.py",
)


def _load_module() -> ModuleType:
    """Load scripts/ablation_spec.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("ablation_spec", _SPEC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses with `from __future__ import annotations`
    # resolve field annotations via sys.modules[cls.__module__] during class build.
    sys.modules["ablation_spec"] = module
    spec.loader.exec_module(module)
    return module


def test_spec_has_full_a1_to_a15_table() -> None:
    mod = _load_module()
    ids = [s.ablation_id for s in mod.ABLATION_SPECS]
    assert ids == [f"A{i}" for i in range(1, 16)], ids


def test_spec_ids_and_run_names_are_unique() -> None:
    mod = _load_module()
    ids = [s.ablation_id for s in mod.ABLATION_SPECS]
    run_names = [s.run_name for s in mod.ABLATION_SPECS]
    assert len(set(ids)) == len(ids)
    assert len(set(run_names)) == len(run_names)


def test_modes_are_valid_and_a13_a14_are_external() -> None:
    mod = _load_module()
    valid = {"single_tier1", "single_tier2", "tiered", "external"}
    for s in mod.ABLATION_SPECS:
        assert s.mode in valid, f"{s.ablation_id} has invalid mode {s.mode}"
    assert mod.ABLATION_BY_ID["A13"].mode == "external"
    assert mod.ABLATION_BY_ID["A14"].mode == "external"
    assert mod.ABLATION_BY_ID["A14"].dataset == "chexpert"


def test_uncertainty_pass_counts_collapse_when_disabled() -> None:
    mod = _load_module()
    a1 = mod.ABLATION_BY_ID["A1"]  # no MC/TTA
    a6 = mod.ABLATION_BY_ID["A6"]  # MC + TTA
    assert a1.mc_passes == 1 and a1.tta_passes == 1
    assert a6.mc_passes == mod.MC_PASSES_DEFAULT and a6.tta_passes == mod.TTA_PASSES_DEFAULT


def test_result_json_path_matches_run_name() -> None:
    mod = _load_module()
    a13 = mod.ABLATION_BY_ID["A13"]
    assert a13.result_json == "outputs/results/A13_Tiered_ArkPlus.json"


def test_per_ablation_rows_point_at_their_own_weights() -> None:
    mod = _load_module()
    # Retrained rows must resolve to their dedicated checkpoint, not the core weight.
    a11 = mod.ABLATION_BY_ID["A11"]
    assert a11.tier2_weight_candidates() == [
        "outputs/models/A11_ArkPlus_Only_NoMCTTA/best_model.pth"
    ]


def test_get_spec_rejects_unknown_id() -> None:
    mod = _load_module()
    with pytest.raises(KeyError):
        mod.get_spec("A999")
