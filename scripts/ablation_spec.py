"""Canonical specification of the A1-A15 thesis ablation matrix.

This is the single source of truth for the full ablation table defined in
``PLAN.md`` section 5.4. Both ``build_ablation_json.py`` (which compiles the
dashboard table) and ``evaluate_ablation.py`` (which computes the genuine
metrics on Colab) import these specs, so the display metadata and the
evaluation configuration never drift apart.

The module is intentionally dependency-free (standard library only) so it can
be imported from any script without pulling in torch / the project packages.

Evaluation modes:
    single_tier1  -- evaluate the MobileNetV2 screening model on its own.
    single_tier2  -- evaluate one deep backbone on its own (optionally MC+TTA).
    tiered        -- evaluate the full TieredSystem (Tier 1 routes to Tier 2).
    external      -- produced by a dedicated script (A13 evaluate_tiered.py,
                     A14 evaluate_chexpert.py); listed here only for the table.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default uncertainty pass counts. "No MC/TTA" configurations collapse to a
# single pass each, matching the AblationRunner's --no-mc-tta convention.
MC_PASSES_DEFAULT = 20
TTA_PASSES_DEFAULT = 10

# Core (shared) trained weight locations.
TIER1_WEIGHTS = "outputs/models/best_tier1_mobilenet.pth"
CORE_TIER2_WEIGHTS = {
    "efficientnet_b4": ["outputs/models/best_tier2_efficientnet.pth"],
    # Ark+ checkpoints have been saved under both spellings across runs.
    "ark_plus": [
        "outputs/models/best_tier2_ark_plus.pth",
        "outputs/models/best_tier2_arkplus.pth",
    ],
}


@dataclass(frozen=True)
class AblationSpec:
    """Display metadata and evaluation configuration for one ablation row."""

    ablation_id: str
    run_name: str
    name: str
    description: str
    tier1: str
    tier2: str
    routing: str
    uncertainty: str
    mode: str
    backbone: str
    # Explicit Tier 2 / single-model weight path; None means use the core weight
    # for ``backbone``. Per-ablation retrains point at outputs/models/<run>/best_model.pth.
    tier2_weights: str | None = None
    mc: bool = False
    tta: bool = False
    conformal: bool = False
    dynamic: bool = False
    dataset: str = "nih"

    @property
    def result_json(self) -> str:
        """Path of the per-row evaluation marker the compiler reads."""
        return f"outputs/results/{self.run_name}.json"

    @property
    def mc_passes(self) -> int:
        """Number of MC-dropout passes (1 when MC is disabled)."""
        return MC_PASSES_DEFAULT if self.mc else 1

    @property
    def tta_passes(self) -> int:
        """Number of TTA passes (1 when TTA is disabled)."""
        return TTA_PASSES_DEFAULT if self.tta else 1

    def tier2_weight_candidates(self) -> list[str]:
        """Ordered candidate paths for this row's deep-model weights."""
        if self.tier2_weights is not None:
            return [self.tier2_weights]
        return CORE_TIER2_WEIGHTS.get(self.backbone, [f"outputs/models/best_tier2_{self.backbone}.pth"])


# Full A1-A15 matrix (PLAN.md section 5.4). The A1-A7 progression is cumulative
# so every row is distinct: routing, then dynamic threshold, then MC, then TTA,
# then conformal. A8/A9 reuse the A6 configuration with retrained Tier 2 weights.
ABLATION_SPECS: list[AblationSpec] = [
    AblationSpec(
        ablation_id="A1",
        run_name="A1_Tier1_Only",
        name="Tier 1 Only",
        description="Baseline screening model (MobileNetV2) with no escalation to Tier 2.",
        tier1="MobileNetV2",
        tier2="None (Bypassed)",
        routing="None",
        uncertainty="None",
        mode="single_tier1",
        backbone="mobilenet_v2",
    ),
    AblationSpec(
        ablation_id="A2",
        run_name="A2_Tier2_EfficientNet_Only",
        name="Tier 2 Only (EfficientNet)",
        description="All cases run directly on the EfficientNet-B4 backbone, no MC/TTA.",
        tier1="None",
        tier2="EfficientNetB4",
        routing="All Escalated",
        uncertainty="None",
        mode="single_tier2",
        backbone="efficientnet_b4",
    ),
    AblationSpec(
        ablation_id="A3",
        run_name="A3_Tiered_Static",
        name="Tiered (Static Threshold)",
        description="Tiered cascade with a static escalation threshold; Tier 2 single forward.",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Static Threshold",
        uncertainty="None",
        mode="tiered",
        backbone="efficientnet_b4",
        dynamic=False,
    ),
    AblationSpec(
        ablation_id="A4",
        run_name="A4_Tiered_Dynamic",
        name="Tiered (Dynamic Threshold)",
        description="Tiered cascade with an adaptive escalation threshold; Tier 2 single forward.",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Dynamic Threshold",
        uncertainty="None",
        mode="tiered",
        backbone="efficientnet_b4",
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A5",
        run_name="A5_Tiered_MC_Only",
        name="Tiered + MC Dropout",
        description="Dynamic tiered cascade adding Monte Carlo dropout uncertainty (no TTA).",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Dynamic Threshold",
        uncertainty="MC Dropout",
        mode="tiered",
        backbone="efficientnet_b4",
        mc=True,
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A6",
        run_name="A6_Tiered_MC_TTA",
        name="Tiered + MC + TTA",
        description="Dynamic tiered cascade with MC dropout and test-time augmentation.",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA",
        mode="tiered",
        backbone="efficientnet_b4",
        mc=True,
        tta=True,
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A7",
        run_name="A7_Tiered_Conformal",
        name="Tiered + Conformal",
        description="Full EfficientNet tiered system with MC, TTA and conformal prediction sets.",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA + Conformal",
        mode="tiered",
        backbone="efficientnet_b4",
        mc=True,
        tta=True,
        conformal=True,
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A8",
        run_name="A8_NoSynthetic",
        name="Without Synthetic Augmentation",
        description="A6 configuration with a Tier 2 trained without Stable Diffusion synthetics.",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA",
        mode="tiered",
        backbone="efficientnet_b4",
        tier2_weights="outputs/models/A8_NoSynthetic/best_model.pth",
        mc=True,
        tta=True,
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A9",
        run_name="A9_NoAugmentation",
        name="Without Any Augmentation",
        description="A6 configuration with a Tier 2 trained without any augmentation.",
        tier1="MobileNetV2",
        tier2="EfficientNetB4",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA",
        mode="tiered",
        backbone="efficientnet_b4",
        tier2_weights="outputs/models/A9_NoAugmentation/best_model.pth",
        mc=True,
        tta=True,
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A10",
        run_name="A10_Always_Tier2",
        name="Always Tier 2 (No Routing)",
        description="Every case forced to the EfficientNet Tier 2 with MC + TTA, bypassing routing.",
        tier1="None",
        tier2="EfficientNetB4",
        routing="All Escalated",
        uncertainty="MC + TTA",
        mode="single_tier2",
        backbone="efficientnet_b4",
        mc=True,
        tta=True,
    ),
    AblationSpec(
        ablation_id="A11",
        run_name="A11_ArkPlus_Only_NoMCTTA",
        name="Tier 2 = Ark+ (No MC/TTA)",
        description="Ark+ Swin Tier 2 evaluated standalone with a single forward pass.",
        tier1="None",
        tier2="Ark+ Swin",
        routing="All Escalated",
        uncertainty="None",
        mode="single_tier2",
        backbone="ark_plus",
        tier2_weights="outputs/models/A11_ArkPlus_Only_NoMCTTA/best_model.pth",
    ),
    AblationSpec(
        ablation_id="A12",
        run_name="A12_ArkPlus_Only_MC_TTA",
        name="Tier 2 = Ark+ + MC + TTA",
        description="Ark+ Swin Tier 2 evaluated standalone with MC dropout and TTA.",
        tier1="None",
        tier2="Ark+ Swin",
        routing="All Escalated",
        uncertainty="MC + TTA",
        mode="single_tier2",
        backbone="ark_plus",
        tier2_weights="outputs/models/A12_ArkPlus_Only_MC_TTA/best_model.pth",
        mc=True,
        tta=True,
    ),
    AblationSpec(
        ablation_id="A13",
        run_name="A13_Tiered_ArkPlus",
        name="Proposed Tiered + Ark+",
        description="Tiered MobileNetV2 (T1) + Ark+ (T2) with dynamic routing, MC, TTA, conformal.",
        tier1="MobileNetV2",
        tier2="Ark+ Swin",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA + Conformal",
        mode="external",
        backbone="ark_plus",
        mc=True,
        tta=True,
        conformal=True,
        dynamic=True,
    ),
    AblationSpec(
        ablation_id="A14",
        run_name="A14_CheXpert_ZeroShot",
        name="Zero-Shot CheXpert",
        description="Out-of-domain validation of A13 evaluated zero-shot on the CheXpert cohort.",
        tier1="MobileNetV2",
        tier2="Ark+ Swin",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA + Conformal",
        mode="external",
        backbone="ark_plus",
        mc=True,
        tta=True,
        conformal=True,
        dynamic=True,
        dataset="chexpert",
    ),
    AblationSpec(
        ablation_id="A15",
        run_name="A15_Mixup_Cutmix",
        name="Mixup/CutMix Regularized",
        description="A13 configuration with an Ark+ Tier 2 trained using Mixup/CutMix regularization.",
        tier1="MobileNetV2",
        tier2="Ark+ Swin",
        routing="Dynamic Threshold",
        uncertainty="MC + TTA + Conformal",
        mode="tiered",
        backbone="ark_plus",
        tier2_weights="outputs/models/A15_Mixup_Cutmix/best_model.pth",
        mc=True,
        tta=True,
        conformal=True,
        dynamic=True,
    ),
]

ABLATION_BY_ID: dict[str, AblationSpec] = {spec.ablation_id: spec for spec in ABLATION_SPECS}


def get_spec(ablation_id: str) -> AblationSpec:
    """Return the spec for ``ablation_id`` or raise a helpful error."""
    if ablation_id not in ABLATION_BY_ID:
        available = ", ".join(ABLATION_BY_ID)
        raise KeyError(f"Unknown ablation id '{ablation_id}'. Available: {available}")
    return ABLATION_BY_ID[ablation_id]
