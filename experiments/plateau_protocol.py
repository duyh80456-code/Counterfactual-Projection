"""State and decisions for plateau-triggered interventions."""

from __future__ import annotations

from dataclasses import dataclass, field


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

