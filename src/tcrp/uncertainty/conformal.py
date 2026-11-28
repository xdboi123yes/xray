"""Split-conformal prediction calibrated on the dedicated Calibration Split.

Computes finite-sample nonconformity quantile ``q_hat`` on temperature-scaled
probabilities (with exchangeability preserved by fitting temperature on the
separate Validation Split), maps softmax distributions to predictive sets, and
logs marginal coverage, class-conditional coverage, singleton-set rate,
two-class abstention-set rate, and empty-fallback rate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import structlog
import torch

from tcrp.uncertainty.calibration import ModelWithTemperature

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ConformalEvaluationReport:
    """Conformal prediction set utility and coverage summary."""

    split: str
    n_samples: int
    q_hat: float
    alpha: float
    target_coverage: float
    temperature: float
    marginal_coverage: float
    coverage_class0: float
    coverage_class1: float
    n_class0: int
    n_class1: int
    avg_set_size: float
    singleton_rate: float
    two_class_abstention_rate: float
    empty_fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a plain dictionary suitable for JSON persistence."""
        return asdict(self)


class ConformalPredictor:
    """Split-conformal predictor mapping probability tensors to predictive label sets."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.q_hat: float | None = None
        self.temperature: float = 1.0
        self.classes = ["No Finding", "Pneumothorax"]
        self.last_calibration_report: ConformalEvaluationReport | None = None

    def _resolve_temperature(self, model: Any, temperature: float | None) -> tuple[float, bool]:
        """Validate temperature argument and determine whether logits need scaling.

        Returns:
            ``(effective_temperature, should_divide_logits)``
        """
        is_wrapped = isinstance(model, ModelWithTemperature)
        if is_wrapped:
            if temperature is not None and float(temperature) != 1.0:
                raise ValueError(
                    "Model is already temperature-scaled (ModelWithTemperature); "
                    "pass temperature=1.0 (or None) to avoid double temperature scaling."
                )
            wrapped_temp = float(model.temperature.clamp(min=1e-3).item())
            return wrapped_temp, False

        raw_temp = self.temperature if temperature is None else temperature
        if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float)):
            raise ValueError(f"temperature must be a finite positive number, got {raw_temp!r}.")
        effective = float(raw_temp)
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError(f"temperature must be finite and strictly positive, got {effective}.")
        return effective, effective != 1.0

    def calibrate(
        self,
        model: Any,
        cal_loader: Any,
        device: torch.device,
        temperature: float = 1.0,
    ) -> float:
        """Compute finite-sample nonconformity quantile ``q_hat`` on ``cal_loader``.

        Args:
            model: Classifier (or ``ModelWithTemperature``) returning logits.
            cal_loader: DataLoader over the held-out Calibration Split.
            device: Compute device.
            temperature: Post-hoc temperature scalar ``T_cal`` fitted strictly on
                the separate Validation Split (ignored when ``model`` is already a
                ``ModelWithTemperature`` with ``temperature=1.0``).

        Returns:
            The calibrated nonconformity quantile ``q_hat``.
        """
        effective_temp, should_scale = self._resolve_temperature(model, temperature)
        model.eval()
        scores: list[float] = []
        probs_chunks: list[np.ndarray[Any, Any]] = []
        labels_chunks: list[np.ndarray[Any, Any]] = []

        log.info(
            "Calibrating split-conformal predictor",
            alpha=self.alpha,
            temperature=round(effective_temp, 4),
        )
        with torch.no_grad():
            for batch in cal_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                logits = model(images)
                if should_scale:
                    logits = logits / effective_temp
                probs = torch.softmax(logits, dim=1)

                true_class_probs = probs[torch.arange(len(labels)), labels]
                nc_scores = 1.0 - true_class_probs
                scores.extend(nc_scores.cpu().numpy().tolist())
                probs_chunks.append(probs.cpu().numpy())
                labels_chunks.append(labels.cpu().numpy())

        scores_arr = np.asarray(scores, dtype=float)
        n = len(scores_arr)
        if n == 0:
            raise ValueError("Calibration split is empty; cannot compute q_hat.")

        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_hat = float(np.quantile(scores_arr, q_level, method="higher"))
        self.temperature = effective_temp

        all_probs = np.concatenate(probs_chunks, axis=0)
        all_labels = np.concatenate(labels_chunks, axis=0)
        self.last_calibration_report = self.evaluate_predictions(
            all_probs, all_labels, split_name="calibration"
        )
        log.info(
            "Split-conformal calibration complete",
            q_hat=round(self.q_hat, 4),
            temperature=round(self.temperature, 4),
            n=n,
        )
        return self.q_hat

    def predict_set(self, probs: torch.Tensor) -> list[str]:
        """Map ``[1, num_classes]`` softmax probabilities to a conformal prediction set."""
        if self.q_hat is None:
            raise ValueError("ConformalPredictor is not calibrated yet.")

        threshold = 1.0 - self.q_hat
        probs_np = probs.detach().cpu().numpy()[0]
        prediction_set = [self.classes[i] for i, prob in enumerate(probs_np) if prob >= threshold]

        if not prediction_set:
            prediction_set.append(self.classes[int(np.argmax(probs_np))])
        return prediction_set

    def evaluate_predictions(
        self,
        probs: Any,
        labels: Any,
        split_name: str = "evaluation",
        temperature: float | None = None,
    ) -> ConformalEvaluationReport:
        """Compute and log conformal coverage and set-size diagnostics over probability rows.

        Args:
            probs: Array/tensor of shape ``[N, 2]`` containing temperature-scaled softmax
                probabilities for ``[No Finding, Pneumothorax]``.
            labels: 1D array/tensor of shape ``[N]`` containing binary labels in ``{0, 1}``.
            split_name: Human-readable split identifier for structured logging.
            temperature: Optional temperature scalar recorded in the report (defaults to
                ``self.temperature``).

        Returns:
            :class:`ConformalEvaluationReport` with marginal coverage, class-conditional
            coverage, singleton-set rate, two-class abstention-set rate, and empty-fallback rate.
        """
        if self.q_hat is None:
            raise ValueError("ConformalPredictor is not calibrated yet.")

        reported_temp = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(reported_temp) or reported_temp <= 0.0:
            raise ValueError(
                f"temperature must be finite and strictly positive, got {reported_temp}."
            )

        probs_np = (
            probs.detach().cpu().numpy()
            if isinstance(probs, torch.Tensor)
            else np.asarray(probs, dtype=float)
        )
        labels_np = (
            labels.detach().cpu().numpy().astype(int)
            if isinstance(labels, torch.Tensor)
            else np.asarray(labels, dtype=int)
        )
        if probs_np.ndim != 2 or probs_np.shape[1] != len(self.classes):
            raise ValueError(
                f"Expected probs of shape [N, {len(self.classes)}], got {probs_np.shape}."
            )
        if labels_np.ndim != 1:
            raise ValueError(f"Expected 1D labels array, got shape {labels_np.shape}.")
        if len(probs_np) != len(labels_np) or len(probs_np) == 0:
            raise ValueError("probs and labels must be non-empty and have matching lengths.")
        invalid_labels = set(np.unique(labels_np).tolist()) - {0, 1}
        if invalid_labels:
            raise ValueError(f"Labels must be binary in {{0, 1}}, got {sorted(invalid_labels)}.")

        threshold = 1.0 - self.q_hat
        n_total = len(labels_np)
        hits = 0
        hits_c0 = 0
        n_c0 = 0
        hits_c1 = 0
        n_c1 = 0
        singletons = 0
        two_class = 0
        empty_fallbacks = 0
        set_size_sum = 0

        for i in range(n_total):
            row = probs_np[i]
            raw_set = [c for c, p in enumerate(row) if float(p) >= threshold]
            if not raw_set:
                empty_fallbacks += 1
                pred_indices = [int(np.argmax(row))]
            else:
                pred_indices = raw_set

            set_size = len(pred_indices)
            set_size_sum += set_size
            if set_size == 1:
                singletons += 1
            elif set_size >= 2:
                two_class += 1

            true_lbl = int(labels_np[i])
            hit = true_lbl in pred_indices
            if hit:
                hits += 1
            if true_lbl == 0:
                n_c0 += 1
                if hit:
                    hits_c0 += 1
            else:
                n_c1 += 1
                if hit:
                    hits_c1 += 1

        report = ConformalEvaluationReport(
            split=split_name,
            n_samples=n_total,
            q_hat=float(self.q_hat),
            alpha=float(self.alpha),
            target_coverage=float(1.0 - self.alpha),
            temperature=reported_temp,
            marginal_coverage=hits / n_total,
            coverage_class0=(hits_c0 / n_c0) if n_c0 > 0 else float("nan"),
            coverage_class1=(hits_c1 / n_c1) if n_c1 > 0 else float("nan"),
            n_class0=n_c0,
            n_class1=n_c1,
            avg_set_size=set_size_sum / n_total,
            singleton_rate=singletons / n_total,
            two_class_abstention_rate=two_class / n_total,
            empty_fallback_rate=empty_fallbacks / n_total,
        )
        log.info(
            "Conformal evaluation metrics",
            split=split_name,
            n_samples=report.n_samples,
            q_hat=round(report.q_hat, 4),
            temperature=round(report.temperature, 4),
            marginal_coverage=round(report.marginal_coverage, 4),
            coverage_class0=round(report.coverage_class0, 4) if n_c0 > 0 else None,
            coverage_class1=round(report.coverage_class1, 4) if n_c1 > 0 else None,
            singleton_rate=round(report.singleton_rate, 4),
            two_class_abstention_rate=round(report.two_class_abstention_rate, 4),
            empty_fallback_rate=round(report.empty_fallback_rate, 4),
            avg_set_size=round(report.avg_set_size, 4),
        )
        return report

    def evaluate(
        self,
        model: Any,
        dataloader: Any,
        device: torch.device,
        temperature: float | None = None,
        split_name: str = "evaluation",
    ) -> ConformalEvaluationReport:
        """Run ``model`` over ``dataloader`` and compute conformal diagnostics."""
        if self.q_hat is None:
            raise ValueError("ConformalPredictor is not calibrated yet.")

        effective_temp, should_scale = self._resolve_temperature(model, temperature)
        model.eval()
        probs_chunks: list[np.ndarray[Any, Any]] = []
        labels_chunks: list[np.ndarray[Any, Any]] = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(device)
                labels = batch[1]
                logits = model(images)
                if should_scale:
                    logits = logits / effective_temp
                probs = torch.softmax(logits, dim=1)
                probs_chunks.append(probs.cpu().numpy())
                labels_chunks.append(
                    labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
                )

        if not probs_chunks:
            raise ValueError(f"Evaluation loader for split '{split_name}' is empty.")

        return self.evaluate_predictions(
            np.concatenate(probs_chunks, axis=0),
            np.concatenate(labels_chunks, axis=0),
            split_name=split_name,
            temperature=effective_temp,
        )

    def save(self, path: str) -> None:
        """Persist ``q_hat``, ``alpha``, and ``temperature`` to disk."""
        torch.save(
            {"q_hat": self.q_hat, "alpha": self.alpha, "temperature": self.temperature},
            path,
        )

    def load(self, path: str) -> None:
        """Load saved conformal threshold state (tolerant of scalar or partial dicts)."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            self.q_hat = data.get("q_hat", self.q_hat)
            self.alpha = data.get("alpha", self.alpha)
            if "temperature" in data and data["temperature"] is not None:
                self.temperature = float(data["temperature"])
        else:
            self.q_hat = float(data)
