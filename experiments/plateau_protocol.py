"""State and decisions for plateau-triggered interventions."""

from __future__ import annotations

from dataclasses import dataclass, field


class ConstantCheckpointScheduler:
    """Serializable no-op scheduler that never mutates checkpoint LR."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.steps = 0
        self.learning_rates = tuple(
            float(group["lr"]) for group in optimizer.param_groups)

    def step(self) -> None:
        current = tuple(float(group["lr"])
                        for group in self.optimizer.param_groups)
        if current != self.learning_rates:
            raise RuntimeError("constant convergence LR changed unexpectedly")
        self.steps += 1

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
