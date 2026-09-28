"""Optimizer-state-preserving parameter-group operations for relaxed Bypass."""

from __future__ import annotations


def add_extension_parameters_(optimizer, parameters) -> None:
    parameters = list(parameters)
    known = {id(parameter) for group in optimizer.param_groups
             for parameter in group["params"]}
    if any(id(parameter) in known for parameter in parameters):
        raise RuntimeError("extension parameter already belongs to optimizer")
    # Keep one SGD group so the already-loaded cosine scheduler remains valid.
    optimizer.param_groups[0]["params"].extend(parameters)


def remove_extension_parameters_(optimizer, parameters) -> None:
    removed = {id(parameter) for parameter in parameters}
    for group in optimizer.param_groups:
        group["params"] = [parameter for parameter in group["params"]
                           if id(parameter) not in removed]
    for parameter in list(optimizer.state):
        if id(parameter) in removed:
            optimizer.state.pop(parameter, None)
