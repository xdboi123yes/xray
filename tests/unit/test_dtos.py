"""Unit tests for the Data Transfer Objects (DTOs) in the application layer.

Verifies strict type validation and defaults for training and ablation DTOs.
"""

from __future__ import annotations

from application.dto.ablation_dto import AblationConfigDTO, AblationResultDTO
from application.dto.training_config_dto import TrainingConfigDTO


def test_ablation_config_dto() -> None:
    """Test AblationConfigDTO initialization and field mappings."""
    dto = AblationConfigDTO(
        ablation_id="A11",
        name="Proposed Tiered (Static)",
        description="Static routing experiment config details.",
        script="scripts/train_tier2.py",
        args=["--backbone", "ark_plus"],
        overrides={"training": {"epochs": 20}},
    )
    assert dto.ablation_id == "A11"
    assert dto.name == "Proposed Tiered (Static)"
    assert dto.description == "Static routing experiment config details."
    assert dto.script == "scripts/train_tier2.py"
    assert dto.args == ["--backbone", "ark_plus"]
    assert dto.overrides == {"training": {"epochs": 20}}


def test_ablation_result_dto() -> None:
    """Test AblationResultDTO validation and field mappings."""
    dto = AblationResultDTO(
        ablation_id="A11",
        name="Proposed Tiered (Static)",
        description="Static routing experiment details.",
        started_at="2026-05-27T12:00:00Z",
        completed_at="2026-05-27T12:15:00Z",
        status="SUCCESS",
        metrics={"auc_roc": 0.885, "accuracy": 0.842},
        error=None,
    )
    assert dto.ablation_id == "A11"
    assert dto.name == "Proposed Tiered (Static)"
    assert dto.started_at == "2026-05-27T12:00:00Z"
    assert dto.completed_at == "2026-05-27T12:15:00Z"
    assert dto.status == "SUCCESS"
    assert dto.metrics == {"auc_roc": 0.885, "accuracy": 0.842}
    assert dto.error is None


def test_training_config_dto() -> None:
    """Test TrainingConfigDTO validation and dynamic field defaults."""
    dto = TrainingConfigDTO(
        backbone="ark_plus",
        run_name="ark_plus_ablation_v1",
        batch_size=16,
        epochs=30,
    )
    assert dto.backbone == "ark_plus"
    assert dto.run_name == "ark_plus_ablation_v1"
    assert dto.batch_size == 16
    assert dto.epochs == 30
    # Verify default parameter validation values
    assert dto.lr_backbone == 1e-4
    assert dto.lr_head == 1e-3
    assert dto.early_stopping_patience == 7
    assert dto.seed == 42
    assert dto.use_synthetic is True
    assert dto.num_workers == 0
