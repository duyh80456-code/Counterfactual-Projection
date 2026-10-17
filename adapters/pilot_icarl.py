"""Task-window sampling and post-SGD interventions for native PyCIL."""
import math
import time
import torch
from adapters.icarl_hooks import Prediction, sites_for, loss_context
from experiments.pilot_gir_vna import inference_state
from methods.vna import dedicated_generator
from experiments.vna_support import isolated_rng


def activity_stats(model, batch, sites=None):
    sites = sites_for(model) if sites is None else sites
    modules = dict(model.named_modules())
    values, handles = {}, []
    for site in sites:
        name = site.module_name
        def capture(module, inputs, output, name=name):
            values[name] = output.detach().relu().clone()
        handles.append(modules[site.between[0]].register_forward_hook(capture))
    try:
        with inference_state(model), torch.no_grad(): model(batch[0])
    finally:
        for handle in handles: handle.remove()
    result = {}
    for name, feature in values.items():
        width = feature.shape[1]
        matrix = feature.movedim(1, -1).reshape(-1, width).double()
        score = matrix.abs().mean(0)
        dormant = score / score.mean().clamp_min(1e-30) <= .1
        singular = torch.linalg.svdvals(matrix)
        probability = singular / singular.sum().clamp_min(1e-30)
        effective = float(torch.exp(-(probability*probability.clamp_min(1e-30).log()).sum())) if singular.sum() else 0.
        result[name] = dict(dormant_fraction=float(dormant.double().mean()), effective_rank=effective,
                            dormant_indices=torch.where(dormant)[0])
    return result


class ICaRLPilot:
    def __init__(self, engine, *, window=128, calibrate=False, every_epoch=False,
                 auto_schedule=False):
        self.engine, self.window, self.calibrate = engine, window, calibrate
        self.every_epoch = every_epoch
        self.auto_schedule = auto_schedule
        self.batch = None
        self.calibration, self.iteration_times = [], []
        self.full_batches = []
        self.measured_tasks = set()

    def state_dict(self):
        return dict(engine=self.engine.state_dict(), batch=self.batch, calibration=self.calibration,
                    iteration_times=self.iteration_times, full_batches=self.full_batches,
                    measured_tasks=self.measured_tasks, auto_schedule=self.auto_schedule)

    def load_state_dict(self, state):
        self.engine.load_state_dict(state['engine'])
        for k,v in state.items():
            if k!='engine': setattr(self, k, v)

    def start_task(self, learner):
        started = time.monotonic()
        task = learner._cur_task+1
        self.engine.begin_task(task)
        model = Prediction(learner._network)
        device = next(model.parameters()).device
        dataset = learner.train_loader.dataset
        window_generator = dedicated_generator(self.engine.seed, f'pilot-window-task-{task}')
        with inference_state(model), isolated_rng(window_generator):
            indices = torch.randperm(len(dataset), generator=window_generator)[:self.window]
            records = [dataset[int(i)] for i in indices]
            self.batch = (torch.stack([r[1] for r in records]).to(device),
                          torch.as_tensor([r[2] for r in records], device=device))
            sites = sites_for(model)
            stats = activity_stats(model, self.batch, sites)
            self.engine.activities.append(dict(task=task, sites=stats))
            if self.engine.gir and task >= 2:
                spec, ctx = loss_context(learner, self.batch)
                dormant_sites = [s for s in sites if len(stats[s.module_name]['dormant_indices'])]
                if dormant_sites: self.engine.refresh_pool(model, dormant_sites, self.batch, spec, ctx, 0)
                for site in dormant_sites:
                    units = stats[site.module_name]['dormant_indices']
                    if not len(units): continue
                    first, second = site.first_layer.layer, site.second_layer.layer
                    # Pinned CifarResNet: N(0, sqrt(2/(kh*kw*out_channels))).
                    std = math.sqrt(2/(first.kernel_size[0]*first.kernel_size[1]*first.out_channels))
                    norm = std * math.sqrt(first.weight[0].numel())
                    a = self.engine.take(model, site, self.batch, spec, ctx, len(units), norm, 0)
                    used = len(a['weight'])
                    with torch.no_grad():
                        if used: first.weight[units[:used]] = a['weight'].to(first.weight)
                        if used < len(units):
                            first.weight[units[used:]] = torch.randn(first.weight[units[used:]].shape,
                                dtype=first.weight.dtype, device=device, generator=self.engine.generator(device))*std
                        if first.bias is not None: first.bias[units] = 0
                        second.weight[:, units] = 0
                        bn = dict(model.named_modules())[site.between[0]]
                        bn.running_mean[units] = 0; bn.running_var[units] = 1
                        if bn.affine: bn.weight[units] = 1; bn.bias[units] = 0
        self.engine.events.append(dict(kind='gir_task', task=task, seconds=time.monotonic()-started))
        self.full_batches.append(len(learner.train_loader))

    def after_step(self, hook, learner, optimizer):
        if self.batch is None: return
        model = Prediction(learner._network)
        device = next(model.parameters()).device
        self.batch = tuple(t.to(device) for t in self.batch)
        task = learner._cur_task+1
        # One calibration correction at each smoke task, independent of K.
        calibrating = (self.calibrate or self.auto_schedule) and task not in self.measured_tasks
        step = self.engine.every if calibrating else hook.steps
        epoch_due = self.every_epoch and hook.epoch_step == len(learner.train_loader)
        if not calibrating:
            if self.engine.coefficient==0: return
            if self.every_epoch and not epoch_due: return
            if not self.every_epoch and step % self.engine.every: return
        spec, ctx = loss_context(learner, self.batch)
        device = next(model.parameters()).device
        if device.type=='cuda': torch.cuda.synchronize(device)
        started = time.monotonic()
        saved_interval = self.engine.every
        if calibrating or epoch_due: self.engine.every = 1
        try:
            first_row = self.engine.correction(model, sites_for(model), self.batch, spec, ctx,
                                               hook.steps if calibrating else step,
                                               optimizer.param_groups[0]['lr'])
            second_row = None
            if calibrating:
                # The second measurement reuses A and isolates the 9/10
                # applications in the weighted preflight estimate.
                second_row = self.engine.correction(model, sites_for(model), self.batch, spec, ctx,
                                                    hook.steps, optimizer.param_groups[0]['lr'],
                                                    apply=False)
        finally: self.engine.every = saved_interval
        if device.type=='cuda': torch.cuda.synchronize(device)
        if calibrating:
            first_row = first_row or {}
            point = dict(task=task, vna_seconds=time.monotonic()-started,
                vna_seconds_with_A=first_row.get('seconds', time.monotonic()-started),
                vna_seconds_without_A=(second_row or {}).get('seconds', first_row.get('seconds', 0.)),
                n_out=len(self.batch[0])*learner._total_classes,
                iteration_seconds=started-hook.iteration_started)
            self.calibration.append(point)
            self.measured_tasks.add(task)
            if self.auto_schedule and len(self.calibration) >= 2:
                from experiments.pilot_gir_vna import choose_interval
                schedule = choose_interval(self.calibration[-2:], len(learner.train_loader))
                self.engine.every = schedule['K']
                self.engine.events.append(dict(kind='schedule', task=task, **schedule))
