"""Pilot-scaled relaxed Bypass (Jung & Lee, Algorithm 1) from shared theta_20."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baselines.bypass import (
    add_extension_parameters_, contraction_norm,
    embed_relaxed_bypass, extension_parameters, project_ready_activations,
    transition_from_opt2_)
from experiments.shared_protocol import (
    FORK_EPOCH, POST_FORK_EPOCHS, atomic_json_save, atomic_torch_save,
    build_cifar_gromo_resnet18, build_optimizer_scheduler,
    datasets_and_indices, evaluate, load_shared_checkpoint, make_eval_loader,
    make_train_loader, protocol, restore_rng, rng_state, seed_everything,
    train_epoch)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shared-checkpoint", required=True)
    parser.add_argument("--shared-checkpoint-hash", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--tuning-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--opt1-epochs", type=int, default=20)
    parser.add_argument("--max-opt2-epochs", type=int, default=10)
    parser.add_argument("--contraction-epsilon", type=float, default=0.002)
    parser.add_argument("--gamma-slope", type=float, default=3e-6)
    return parser.parse_args()


@torch.no_grad()
def evaluation_loss(model, batch):
    model.eval()
    logits = model(batch[0]).detach()
    return float(F.cross_entropy(logits.float(), batch[1]))


def save_checkpoint(path, *, model, optimizer, scheduler, history,
                    post_epoch, shared_hash, train_loader, run_protocol,
                    phase, opt1_epochs, opt2_epochs, train3_epochs,
                    opt2_steps, projection_loss_jump, contraction_at_projection,
                    elapsed, extension_paths, peak_train_params,
                    peak_gpu_memory, opt2_soft_cap_exceeded):
    atomic_torch_save({
        "format_version": 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "history": history, "post_epoch": post_epoch,
        "shared_checkpoint_hash": shared_hash, "protocol": run_protocol,
        "rng": rng_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "phase": phase, "opt1_epochs": opt1_epochs,
        "opt2_epochs": opt2_epochs, "train3_epochs": train3_epochs,
        "opt2_steps": opt2_steps,
        "projection_loss_jump": projection_loss_jump,
        "contraction_at_projection": contraction_at_projection,
        "extension_paths": extension_paths,
        "peak_train_params": peak_train_params,
        "training_seconds": elapsed,
        "peak_gpu_memory": peak_gpu_memory,
        "opt2_soft_cap_exceeded": opt2_soft_cap_exceeded,
    }, path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    import sys
    reference_root = Path(args.reference_root).resolve()
    sys.path.insert(0, str(reference_root))
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    (train_set, eval_set, train_indices, validation_indices,
     tuning_indices) = datasets_and_indices(
         args.data_root, args.validation_samples, args.tuning_samples)
    common_protocol = protocol(
        args.seed, train_indices, validation_indices, tuning_indices,
        args.batch_size, args.lr, args.weight_decay)
    run_protocol = {**common_protocol, "method": "bypass",
        "bypass_variant": "relaxed_resnet_learnable_activation",
        "bypass_schedule": {
            "opt1_epochs": args.opt1_epochs,
            "max_opt2_epochs": args.max_opt2_epochs,
            "max_opt2_epochs_semantics": (
                "soft cap; continue opt2 within the 60-epoch budget until "
                "contraction succeeds"),
            "contraction_epsilon": args.contraction_epsilon,
            "gamma_t": f"{args.gamma_slope} * opt2_step",
            "schedule_status": "pilot scaling, not paper hyperparameters",
        }}
    model = build_cifar_gromo_resnet18(device)
    optimizer, scheduler = build_optimizer_scheduler(
        model, args.lr, args.weight_decay)
    shared, shared_hash = load_shared_checkpoint(
        Path(args.shared_checkpoint), args.shared_checkpoint_hash,
        device=device, model=model, optimizer=optimizer, scheduler=scheduler)
    if shared["protocol"] != common_protocol:
        raise RuntimeError("shared checkpoint protocol differs from Bypass")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint_latest.pt"
    saved = (torch.load(checkpoint_path, map_location=device)
             if checkpoint_path.is_file() else None)
    phase = "opt1"
    if saved is not None:
        if saved["shared_checkpoint_hash"] != shared_hash:
            raise RuntimeError("Bypass checkpoint came from another theta_20")
        if saved["protocol"] != run_protocol:
            raise RuntimeError("Bypass resume protocol mismatch")
        phase = saved["phase"]
    extension_paths = []
    if phase in {"opt1", "opt2"}:
        extension_paths = embed_relaxed_bypass(model)
        add_extension_parameters_(optimizer, extension_parameters(model))
    generator_state = shared["train_loader_generator_state"]
    history = []
    start_epoch = opt1_done = opt2_done = train3_done = opt2_steps = 0
    projection_loss_jump = None
    contraction_at_projection = None
    prior_seconds = 0.0
    prior_peak_gpu_memory = 0
    opt2_soft_cap_exceeded = False
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        generator_state = saved["train_loader_generator_state"]
        history = saved["history"]
        start_epoch = int(saved["post_epoch"])
        opt1_done = int(saved["opt1_epochs"])
        opt2_done = int(saved["opt2_epochs"])
        train3_done = int(saved["train3_epochs"])
        opt2_steps = int(saved["opt2_steps"])
        projection_loss_jump = saved["projection_loss_jump"]
        contraction_at_projection = saved["contraction_at_projection"]
        prior_seconds = float(saved.get("training_seconds", 0.0))
        prior_peak_gpu_memory = int(saved.get("peak_gpu_memory", 0))
        opt2_soft_cap_exceeded = bool(
            saved.get("opt2_soft_cap_exceeded", False))
        extension_paths = saved.get("extension_paths", extension_paths)
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        generator_state, args.seed)
    validation_loader = make_eval_loader(
        eval_set, validation_indices, args.batch_size * 2, args.workers)
    projection_loader = make_eval_loader(
        eval_set, tuning_indices, len(tuning_indices), args.workers)
    projection_batch = tuple(value.to(device, non_blocking=True)
                             for value in next(iter(projection_loader)))
    base_params = sum(parameter.numel() for parameter in model.parameters())
    peak_train_params = int(saved.get("peak_train_params", base_params)
                            if saved is not None else base_params)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for post_epoch in range(start_epoch, POST_FORK_EPOCHS):
        epoch_phase = phase
        gamma = 0.0
        criterion_met = None
        if phase == "opt1":
            train = train_epoch(model, train_loader, optimizer, device)
            opt1_done += 1
            if opt1_done >= args.opt1_epochs:
                phase = "opt2"
        elif phase == "opt2":
            def penalty():
                nonlocal opt2_steps, gamma
                opt2_steps += 1
                gamma = args.gamma_slope * opt2_steps
                return contraction_norm(model) * gamma

            train = train_epoch(
                model, train_loader, optimizer, device, loss_extra=penalty,
                post_step=lambda: project_ready_activations(
                    model, args.contraction_epsilon))
            opt2_done += 1
            before_loss = evaluation_loss(model, projection_batch)
            transition = transition_from_opt2_(
                model, optimizer, epsilon=args.contraction_epsilon,
                opt2_done=opt2_done, soft_cap=args.max_opt2_epochs)
            criterion_met = transition.criterion_met
            if transition.phase == "train3":
                contraction_at_projection = transition.contraction_norm
                after_loss = evaluation_loss(model, projection_batch)
                projection_loss_jump = after_loss - before_loss
                phase = transition.phase
                if transition.projected_count != len(extension_paths):
                    raise RuntimeError("Bypass extension bookkeeping mismatch")
            elif transition.soft_cap_exceeded:
                # The nominal 10-epoch pilot split is only a warning boundary.
                # Dropping a non-contracted D would violate relaxed Bypass, so
                # opt2 consumes the remaining common budget until it succeeds.
                opt2_soft_cap_exceeded = True
        else:
            train = train_epoch(model, train_loader, optimizer, device)
            train3_done += 1
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": FORK_EPOCH + post_epoch + 1,
            "post_fork_epoch": post_epoch + 1, "phase": epoch_phase,
            "train_loss": train["task_loss"],
            "optimized_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "gamma": gamma,
            "contraction_norm": (float(contraction_norm(model).detach())
                                 if phase in {"opt1", "opt2"} else 0.0),
            "contraction_criterion_met": criterion_met,
            "opt2_soft_cap_exceeded": opt2_soft_cap_exceeded,
            "projection_loss_jump": projection_loss_jump,
        }
        history.append(row)
        elapsed = prior_seconds + time.perf_counter() - started
        peak_gpu_memory = max(
            prior_peak_gpu_memory,
            int(torch.cuda.max_memory_allocated(device)))
        save_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer,
            scheduler=scheduler, history=history, post_epoch=post_epoch + 1,
            shared_hash=shared_hash, train_loader=train_loader,
            run_protocol=run_protocol, phase=phase,
            opt1_epochs=opt1_done, opt2_epochs=opt2_done,
            train3_epochs=train3_done, opt2_steps=opt2_steps,
            projection_loss_jump=projection_loss_jump,
            contraction_at_projection=contraction_at_projection,
            elapsed=elapsed, extension_paths=extension_paths,
            peak_train_params=peak_train_params,
            peak_gpu_memory=peak_gpu_memory,
            opt2_soft_cap_exceeded=opt2_soft_cap_exceeded)
        atomic_json_save({"method": "bypass", "phase": phase,
                          "completed_post_fork_epochs": post_epoch + 1,
                          "latest": row}, output / "progress.json")
        print(json.dumps({"bypass": row}, sort_keys=True), flush=True)

    elapsed = (prior_seconds if start_epoch >= POST_FORK_EPOCHS else
               prior_seconds + time.perf_counter() - started)
    last = history[-1]
    result = {
        "method": "bypass", "shared_checkpoint_hash": shared_hash,
        "source_paper": (
            "https://www.donghunlee.com/papers/"
            "Jung_Lee_Bypass__IEEE_TNNLS.pdf"),
        "implementation": "relaxed Bypass for ResNet, Algorithm 1",
        "schedule_status": (
            "20 opt1 + 10-epoch opt2 soft cap; opt2 continues within budget "
            "until contraction, using shared SGD"),
        "fork_epoch": FORK_EPOCH, "post_fork_epochs": len(history),
        "final_validation_accuracy": last["validation_accuracy"],
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "final_validation_loss": last["validation_loss"],
        "training_seconds": elapsed,
        "peak_gpu_memory": max(
            prior_peak_gpu_memory,
            int(torch.cuda.max_memory_allocated(device))),
        "deploy_params": sum(parameter.numel() for parameter in model.parameters()),
        "peak_train_params": peak_train_params,
        "opt1_epochs": opt1_done, "opt2_epochs": opt2_done,
        "train3_epochs": train3_done,
        "contraction_norm": (0.0 if phase == "train3"
                             else float(contraction_norm(model).detach())),
        "contraction_norm_at_projection": contraction_at_projection,
        "contraction_criterion_met": bool(
            contraction_at_projection is not None and
            contraction_at_projection < args.contraction_epsilon),
        "bypass_completed": phase == "train3",
        "projection_performed": contraction_at_projection is not None,
        "opt2_soft_cap_exceeded": opt2_soft_cap_exceeded,
        "projection_loss_jump": projection_loss_jump,
        "learnable_activations": len(extension_paths),
        "shared_extension_paths": extension_paths,
        "checkpoint": str(checkpoint_path), "history": history,
        "protocol": run_protocol,
    }
    atomic_json_save(result, output / "result.json")


if __name__ == "__main__":
    main()
