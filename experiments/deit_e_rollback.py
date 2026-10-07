"""E-only rollback: strict accuracy stall, lexicographic accuracy/loss anchor."""
from __future__ import annotations

import copy
import math

import torch

from experiments.shared_protocol import rng_state


def rollback_protocol(algorithm_patience=10, *, stall_on_anchor=False):
    if algorithm_patience < 1:
        raise ValueError("algorithm patience must be positive")
    return {"version": 1, "algorithm_patience": algorithm_patience,
            "stall_metric": "accuracy_then_lower_loss_on_exact_tie" if stall_on_anchor else "strict_validation_accuracy",
            "anchor_metric": "accuracy_then_lower_loss_on_exact_tie",
            "restore": ["model", "optimizer", "scheduler"],
            "preserve": ["current_rng", "current_loader_stream"], "retrigger": False}


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


class EAccuracyRollback:
    def __init__(self, model, optimizer, scheduler, loader, validation, epoch, algorithm_patience=10, *, stall_on_anchor=False):
        self.protocol = rollback_protocol(algorithm_patience, stall_on_anchor=stall_on_anchor)
        self.accuracy_stall_counter = 0
        self.rollback_events = []
        self.anchor = self._capture(model, optimizer, scheduler, loader, validation, epoch, 0, "fork_initial")

    @staticmethod
    def _capture(model, optimizer, scheduler, loader, validation, epoch, offset, reason):
        accuracy, loss = float(validation["accuracy"]), float(validation["loss"])
        if not (math.isfinite(accuracy) and math.isfinite(loss)):
            raise ValueError("controller metrics must be finite")
        return {"model": _cpu_copy(model.state_dict()), "optimizer": _cpu_copy(optimizer.state_dict()),
                "scheduler": _cpu_copy(scheduler.state_dict()), "rng": _cpu_copy(rng_state()),
                "train_loader_generator_state": loader.generator.get_state().clone(),
                "epoch": int(epoch), "post_fork_epoch": int(offset),
                "validation": {"accuracy": accuracy, "loss": loss}, "reason": reason}

    @classmethod
    def from_state(cls, state, algorithm_patience, completed_epochs, *, stall_on_anchor=False):
        expected = rollback_protocol(algorithm_patience, stall_on_anchor=stall_on_anchor)
        anchor = state["anchor"]
        counter = int(state["accuracy_stall_counter"])
        if (state["protocol"] != expected or not 0 <= counter < algorithm_patience or
                not 0 <= anchor["post_fork_epoch"] <= completed_epochs):
            raise ValueError("inconsistent E rollback resume state")
        obj = cls.__new__(cls)
        obj.protocol = expected
        obj.anchor = _cpu_copy(anchor)
        obj.accuracy_stall_counter = counter
        obj.rollback_events = copy.deepcopy(state["rollback_events"])
        return obj

    def state_dict(self):
        return {"protocol": self.protocol, "anchor": self.anchor,
                "accuracy_stall_counter": self.accuracy_stall_counter,
                "rollback_events": self.rollback_events}

    def metrics(self):
        return {"controller_anchor_accuracy": self.anchor["validation"]["accuracy"],
                "controller_anchor_loss": self.anchor["validation"]["loss"],
                "controller_anchor_epoch": self.anchor["epoch"],
                "controller_anchor_post_fork_epoch": self.anchor["post_fork_epoch"],
                "controller_anchor_update_reason": self.anchor["reason"],
                "accuracy_stall_counter": self.accuracy_stall_counter,
                "rollback_count": len(self.rollback_events)}

    def observe(self, model, optimizer, scheduler, loader, validation, epoch, offset, *, count_stall=True):
        accuracy, loss = float(validation["accuracy"]), float(validation["loss"])
        if not (math.isfinite(accuracy) and math.isfinite(loss)):
            raise ValueError("controller metrics must be finite")
        old = self.anchor["validation"]
        accuracy_improved = accuracy > old["accuracy"]
        anchor_improved = accuracy_improved or (accuracy == old["accuracy"] and loss < old["loss"])
        reason = ("higher_accuracy" if accuracy_improved else
                  "same_accuracy_lower_loss" if anchor_improved else "no_improvement")
        if anchor_improved:
            self.anchor = self._capture(model, optimizer, scheduler, loader, validation, epoch, offset, reason)
        if count_stall:
            reset = anchor_improved if self.protocol["stall_metric"] != "strict_validation_accuracy" else accuracy_improved
            self.accuracy_stall_counter = 0 if reset else self.accuracy_stall_counter + 1
        stalled = self.accuracy_stall_counter
        rollback = stalled >= self.protocol["algorithm_patience"]
        if rollback:
            # Parameter objects stay in place. RNG and loader generator are deliberately untouched.
            model.load_state_dict(self.anchor["model"])
            optimizer.load_state_dict(self.anchor["optimizer"])
            scheduler.load_state_dict(self.anchor["scheduler"])
            self.rollback_events.append({"epoch": int(epoch), "post_fork_epoch": int(offset),
                "anchor_epoch": self.anchor["epoch"], "anchor_post_fork_epoch": self.anchor["post_fork_epoch"],
                "accuracy_stall_before_rollback": stalled,
                "observed_validation_accuracy": accuracy, "observed_validation_loss": loss})
            self.accuracy_stall_counter = 0
        return {**self.metrics(), "controller_accuracy_improved": accuracy_improved,
                "controller_anchor_improved": anchor_improved, "controller_anchor_reason": reason,
                "accuracy_stall_before_rollback": stalled, "rollback_applied": rollback,
                "rollback_anchor_epoch": self.anchor["epoch"] if rollback else None,
                "state_validation_accuracy": self.anchor["validation"]["accuracy"] if rollback else accuracy,
                "state_validation_loss": self.anchor["validation"]["loss"] if rollback else loss,
                "model_state_epoch": self.anchor["epoch"] if rollback else int(epoch),
                "scheduler_epoch_after_controller": scheduler.last_epoch,
                "learning_rates_after_controller": [group["lr"] for group in optimizer.param_groups]}
