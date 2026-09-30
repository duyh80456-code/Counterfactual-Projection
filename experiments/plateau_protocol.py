"""State and decisions for plateau-triggered interventions."""

from __future__ import annotations

from dataclasses import dataclass, field

import math


@dataclass
class BestCheckpointStallDetector:
    """Declare a stall only after validation fails to set a new best."""

    patience: int = 100
    min_gain: float = 1e-3
    best_metric: float = float("-inf")
    best_epoch: int | None = None
    patience_reference_metric: float = float("-inf")
    last_meaningful_improvement_epoch: int | None = None
    observations: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.patience < 1 or self.min_gain < 0:
            raise ValueError("invalid best-checkpoint stall configuration")

    def update(self, epoch: int, metric: float) -> dict:
        metric = float(metric)
        improved = self.best_epoch is None or metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.best_epoch = int(epoch)
        meaningful = (
            self.last_meaningful_improvement_epoch is None or
            metric >= self.patience_reference_metric + self.min_gain)
        if meaningful:
            self.patience_reference_metric = metric
            self.last_meaningful_improvement_epoch = int(epoch)
        without_improvement = (
            int(epoch) - int(self.last_meaningful_improvement_epoch))
        row = {
            "epoch": int(epoch), "metric": metric, "improved": improved,
            "meaningful_improvement": meaningful,
            "best_metric": self.best_metric, "best_epoch": self.best_epoch,
            "patience_reference_metric": self.patience_reference_metric,
            "last_meaningful_improvement_epoch":
                self.last_meaningful_improvement_epoch,
            "epochs_without_improvement": without_improvement,
            "stalled": without_improvement >= self.patience,
        }
        self.observations.append(row)
        return row

    def state_dict(self) -> dict:
        return {
            "patience": self.patience, "min_gain": self.min_gain,
            "best_metric": self.best_metric, "best_epoch": self.best_epoch,
            "patience_reference_metric": self.patience_reference_metric,
            "last_meaningful_improvement_epoch":
                self.last_meaningful_improvement_epoch,
            "observations": list(self.observations),
        }

    def load_state_dict(self, state: dict) -> None:
        if (int(state["patience"]) != self.patience or
                float(state["min_gain"]) != self.min_gain):
            raise RuntimeError("best-checkpoint detector configuration mismatch")
        self.best_metric = float(state["best_metric"])
        self.best_epoch = int(state["best_epoch"])
        self.patience_reference_metric = float(
            state["patience_reference_metric"])
        self.last_meaningful_improvement_epoch = int(
            state["last_meaningful_improvement_epoch"])
        self.observations = [dict(row) for row in state["observations"]]


class ConstantCheckpointScheduler:
    """Serializable no-op scheduler that never mutates checkpoint LR."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.steps = 0
        self.learning_rates = tuple(
            float(group["lr"]) for group in optimizer.param_groups)

    def step(self, metric=None) -> None:
        current = tuple(float(group["lr"])
                        for group in self.optimizer.param_groups)
        if current != self.learning_rates:
            raise RuntimeError("constant convergence LR changed unexpectedly")
        self.steps += 1

    def sync_optimizer_groups(self) -> None:
        """Register newly added groups at their current constant LR."""
        if len(self.learning_rates) > len(self.optimizer.param_groups):
            raise RuntimeError("optimizer groups disappeared")
        if len(self.learning_rates) < len(self.optimizer.param_groups):
            self.learning_rates = tuple(
                float(group["lr"]) for group in self.optimizer.param_groups)

    def state_dict(self) -> dict:
        return {"kind": "constant_checkpoint_lr", "steps": self.steps,
                "learning_rates": self.learning_rates}

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "constant_checkpoint_lr":
            raise RuntimeError("not a constant checkpoint scheduler state")
        expected = tuple(float(value) for value in state["learning_rates"])
        current = tuple(float(group["lr"])
                        for group in self.optimizer.param_groups)
        if current != expected:
            raise RuntimeError("optimizer LR differs from scheduler state")
        self.learning_rates = expected
        self.steps = int(state["steps"])


class CosineFloorScheduler:
    """One predeclared cosine decay followed by a non-zero constant floor."""

    def __init__(self, optimizer, decay_epochs: int = 300,
                 eta_min: float = 0.002, base_lrs=None):
        if decay_epochs < 1 or eta_min < 0:
            raise ValueError("invalid cosine-floor scheduler configuration")
        self.optimizer = optimizer
        self.decay_epochs = int(decay_epochs)
        self.eta_min = float(eta_min)
        self.base_lrs = tuple(
            float(value) for value in (
                base_lrs if base_lrs is not None else
                [group["lr"] for group in optimizer.param_groups]))
        if len(self.base_lrs) != len(optimizer.param_groups):
            raise ValueError("base LR count differs from optimizer groups")
        self.steps = 0
        self._apply()

    def _apply(self):
        position = min(self.steps, self.decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(
            math.pi * position / self.decay_epochs))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.eta_min + (base_lr - self.eta_min) * cosine

    def step(self, metric=None):
        self.steps += 1
        self._apply()

    def sync_optimizer_groups(self) -> None:
        """Put extension groups on the same uninterrupted LR trajectory."""
        missing = len(self.optimizer.param_groups) - len(self.base_lrs)
        if missing < 0:
            raise RuntimeError("optimizer groups disappeared")
        if missing:
            if not self.base_lrs:
                raise RuntimeError("cannot infer base LR for a new group")
            self.base_lrs = self.base_lrs + (self.base_lrs[0],) * missing
            self._apply()

    def state_dict(self):
        return {
            "kind": "cosine_then_constant_floor", "steps": self.steps,
            "decay_epochs": self.decay_epochs, "eta_min": self.eta_min,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state):
        if state.get("kind") != "cosine_then_constant_floor":
            raise RuntimeError("not a cosine-floor scheduler state")
        if (int(state["decay_epochs"]) != self.decay_epochs or
                float(state["eta_min"]) != self.eta_min or
                tuple(float(value) for value in state["base_lrs"]) !=
                self.base_lrs):
            raise RuntimeError("cosine-floor scheduler configuration mismatch")
        self.steps = int(state["steps"])
        self._apply()


def scheduler_from_state(optimizer, state):
    """Recreate a supported plateau scheduler without changing its trajectory."""
    kind = state.get("kind")
    if kind == "constant_checkpoint_lr":
        scheduler = ConstantCheckpointScheduler(optimizer)
    elif kind == "cosine_then_constant_floor":
        scheduler = CosineFloorScheduler(
            optimizer, decay_epochs=int(state["decay_epochs"]),
            eta_min=float(state["eta_min"]), base_lrs=state["base_lrs"])
    elif kind == "significant_plateau_lr":
        scheduler = SignificantPlateauScheduler(
            optimizer, patience=int(state["patience"]),
            factor=float(state["factor"]), min_lr=float(state["min_lr"]),
            threshold=float(state["threshold"]))
    else:
        raise RuntimeError(f"unsupported plateau scheduler kind: {kind}")
    scheduler.load_state_dict(state)
    return scheduler


class SignificantPlateauScheduler:
    """Reduce LR after metric stalls; no predeclared epoch horizon."""

    def __init__(self, optimizer, patience: int = 20, factor: float = 0.2,
                 min_lr: float = 0.002, threshold: float = 1e-3):
        if (patience < 1 or not 0 < factor < 1 or min_lr < 0 or
                threshold <= 0):
            raise ValueError("invalid significant-plateau LR configuration")
        self.optimizer = optimizer
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.threshold = float(threshold)
        self.steps = 0
        self.bad_epochs = 0
        self.reference_metric = float("-inf")
        self.reductions = 0

    def step(self, metric=None):
        if metric is None:
            raise ValueError("significant-plateau scheduler requires a metric")
        metric = float(metric)
        self.steps += 1
        if (self.reference_metric == float("-inf") or
                metric >= self.reference_metric + self.threshold):
            self.reference_metric = metric
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return
        changed = False
        for group in self.optimizer.param_groups:
            old_lr = float(group["lr"])
            new_lr = max(self.min_lr, old_lr * self.factor)
            group["lr"] = new_lr
            changed = changed or new_lr < old_lr
        self.reductions += int(changed)
        self.bad_epochs = 0

    def sync_optimizer_groups(self):
        """New Bypass coordinates inherit the already-current group LR."""
        if not self.optimizer.param_groups:
            raise RuntimeError("optimizer has no parameter groups")
        inherited = float(self.optimizer.param_groups[0]["lr"])
        for group in self.optimizer.param_groups[1:]:
            group["lr"] = inherited

    def state_dict(self):
        return {
            "kind": "significant_plateau_lr", "steps": self.steps,
            "patience": self.patience, "factor": self.factor,
            "min_lr": self.min_lr, "threshold": self.threshold,
            "bad_epochs": self.bad_epochs,
            "reference_metric": self.reference_metric,
            "reductions": self.reductions,
            "learning_rates": tuple(
                float(group["lr"]) for group in self.optimizer.param_groups),
        }

    def load_state_dict(self, state):
        if state.get("kind") != "significant_plateau_lr":
            raise RuntimeError("not a significant-plateau scheduler state")
        config = (int(state["patience"]), float(state["factor"]),
                  float(state["min_lr"]), float(state["threshold"]))
        expected = (self.patience, self.factor, self.min_lr, self.threshold)
        if config != expected:
            raise RuntimeError("significant-plateau scheduler mismatch")
        learning_rates = tuple(float(value) for value in state["learning_rates"])
        if len(learning_rates) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler/optimizer group count mismatch")
        for group, learning_rate in zip(
                self.optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate
        self.steps = int(state["steps"])
        self.bad_epochs = int(state["bad_epochs"])
        self.reference_metric = float(state["reference_metric"])
        self.reductions = int(state["reductions"])


@dataclass
class PlateauDetector:
    window: int = 15
    accuracy_min_gain: float = 5e-4
    loss_ema_min_drop: float = 1e-3
    ema_alpha: float = 0.3
    minimum_epochs: int = 10
    records: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        if self.window < 2 or self.minimum_epochs < 1:
            raise ValueError("invalid plateau window/minimum epochs")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")

    def update(self, epoch: int, accuracy: float, loss: float) -> dict:
        previous_ema = (self.records[-1]["loss_ema"]
                        if self.records else float(loss))
        loss_ema = (self.ema_alpha * float(loss) +
                    (1.0 - self.ema_alpha) * previous_ema)
        self.records.append({
            "epoch": int(epoch), "accuracy": float(accuracy),
            "loss": float(loss), "loss_ema": float(loss_ema),
        })
        recent = self.records[-self.window:]
        enough = (len(self.records) >= self.minimum_epochs and
                  len(recent) == self.window)
        if enough:
            accuracy_gain = max(row["accuracy"] for row in recent[1:]) - recent[0]["accuracy"]
            loss_ema_drop = recent[0]["loss_ema"] - recent[-1]["loss_ema"]
            plateau = (accuracy_gain < self.accuracy_min_gain and
                       loss_ema_drop < self.loss_ema_min_drop)
        else:
            accuracy_gain = None
            loss_ema_drop = None
            plateau = False
        return {
            "plateau": plateau, "window": self.window,
            "observations_since_probe": len(self.records),
            "accuracy_gain": accuracy_gain,
            "loss_ema_drop": loss_ema_drop,
            "loss_ema": loss_ema,
        }

    def reset(self) -> None:
        self.records.clear()

    def state_dict(self) -> dict:
        return {"records": list(self.records)}

    def load_state_dict(self, state: dict) -> None:
        self.records = [dict(row) for row in state.get("records", [])]


@dataclass
class ConsecutiveWindowPlateauDetector:
    """Require multiple complete, non-overlapping plateau windows."""

    window: int = 20
    required_windows: int = 2
    accuracy_min_gain: float = 1e-3
    loss_ema_min_drop: float = 1e-3
    ema_alpha: float = 0.3
    current: list[dict[str, float]] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    consecutive_plateau_windows: int = 0
    loss_ema: float | None = None

    def update(self, epoch: int, accuracy: float, loss: float) -> dict:
        self.loss_ema = (float(loss) if self.loss_ema is None else
                         self.ema_alpha * float(loss) +
                         (1.0 - self.ema_alpha) * self.loss_ema)
        self.current.append({
            "epoch": int(epoch), "accuracy": float(accuracy),
            "loss": float(loss), "loss_ema": float(self.loss_ema),
        })
        completed = None
        if len(self.current) == self.window:
            accuracy_gain = (max(row["accuracy"] for row in self.current[1:]) -
                             self.current[0]["accuracy"])
            loss_ema_drop = (self.current[0]["loss_ema"] -
                             self.current[-1]["loss_ema"])
            qualifies = (accuracy_gain < self.accuracy_min_gain and
                         loss_ema_drop < self.loss_ema_min_drop)
            self.consecutive_plateau_windows = (
                self.consecutive_plateau_windows + 1 if qualifies else 0)
            completed = {
                "start_epoch": self.current[0]["epoch"],
                "end_epoch": self.current[-1]["epoch"],
                "accuracy_gain": accuracy_gain,
                "loss_ema_drop": loss_ema_drop,
                "qualifies": qualifies,
                "consecutive_plateau_windows":
                    self.consecutive_plateau_windows,
            }
            self.windows.append(completed)
            self.current.clear()
        return {
            "plateau": self.consecutive_plateau_windows >= self.required_windows,
            "window": self.window,
            "required_windows": self.required_windows,
            "consecutive_plateau_windows": self.consecutive_plateau_windows,
            "observations_in_current_window": len(self.current),
            "loss_ema": self.loss_ema,
            "completed_window": completed,
        }

    def state_dict(self) -> dict:
        return {
            "current": list(self.current), "windows": list(self.windows),
            "consecutive_plateau_windows": self.consecutive_plateau_windows,
            "loss_ema": self.loss_ema,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current = [dict(row) for row in state.get("current", [])]
        self.windows = [dict(row) for row in state.get("windows", [])]
        self.consecutive_plateau_windows = int(
            state.get("consecutive_plateau_windows", 0))
        self.loss_ema = state.get("loss_ema")
