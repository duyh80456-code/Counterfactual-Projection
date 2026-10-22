"""V3 interventions around native optimizer steps; checkpointed with native hooks."""
from contextlib import contextmanager
import time
import torch
from adapters.lop_hooks import Prediction as LoPPrediction, sites_for as lop_sites
from diagnostics.vna_align import _global_state
from methods.vna_losses import CE, MSE
from methods.vna import dedicated_generator
from methods.vna_v3 import diagnose


@contextmanager
def evaluation(model):
    modes = {m: m.training for m in model.modules()}
    with _global_state():
        try:
            model.eval()
            with torch.no_grad(): yield
        finally:
            for module, mode in modes.items(): module.training = mode


class Calibration:
    def __init__(self, preflight=False):
        self.preflight = preflight
        self.calibration, self.iteration_times, self.full_batches = [], [], []
        self.counts = {}

    def apply(self, engine, model, sites, batch, spec, ctx, optimizer, iteration, remaining):
        task = engine.task
        if self.preflight and self.counts.get(task,0) >= 3: return None
        calibrating = self.preflight and task in (1, 2) and self.counts.get(task, 0) < 3
        if calibrating:
            engine.preflight(model, sites, batch, spec, ctx, optimizer, iteration,
                                     remaining=remaining)
        event = engine.apply(model, sites, batch, spec, ctx, optimizer, iteration,
                             remaining=remaining, force=calibrating)
        if calibrating and event is not None:
            self.counts[task] = self.counts.get(task, 0)+1
            self.calibration.append(dict(task=task, classes=ctx.get('classes', 1),
                vna_seconds=event.get('seconds', 0.),
                gir_seconds=event.get('gir_seconds',0.), iteration=iteration))
        return event


class LoPV3(Calibration):
    preserve_fifo = True

    def __init__(self, engine, *, benchmark='regression', preflight=False):
        super().__init__(preflight)
        self.engine, self.benchmark = engine, benchmark
        self.hook = self.learner = None
        self.task_metrics, self.contributions, self.insertions = [], [], []
        self._event_index = 0

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k not in ('engine', 'hook', 'learner', '_started')} | {'engine': self.engine.state_dict()}

    def load_state_dict(self, state):
        self.engine.load_state_dict(state['engine'])
        for key, value in state.items():
            if key != 'engine': setattr(self, key, value)

    def context(self):
        batch = self.hook.buffer.batch()
        regression = self.benchmark == 'regression'
        spec = MSE(reduction='mean') if regression else CE()
        def loss(z, ctx):
            target = ctx['targets'].to(z) if regression else ctx['targets']
            return self.learner.loss_func(z, target)
        return LoPPrediction(self.learner.net), batch, spec, {'targets': batch[1], 'loss_fn': loss}

    def before(self, hook, learner, x, target):
        self.hook, self.learner = hook, learner
        task = hook.step//hook.task_steps+1
        changed = task != self.engine.task
        self.engine.begin_task(task)
        if changed:
            hook.buffer.samples.clear()
            self.probe_batch = None
        count = 0 if self.probe_batch is None else len(self.probe_batch[0])
        if count < 256:
            generator = dedicated_generator(self.engine.seed, f'task-probe:{task}:{count}', x.device)
            indices = torch.randperm(len(x), generator=generator, device=x.device)[:256-count]
            samples = (x[indices].detach().clone(), target[indices].detach().clone())
            self.probe_batch = samples if self.probe_batch is None else tuple(
                torch.cat((old, new)) for old, new in zip(self.probe_batch, samples))
        hook.buffer.update((x, target))
        self._started = time.monotonic()

    def after(self, hook, learner):
        elapsed = time.monotonic()-self._started
        elapsed -= sum(e.get('wall_seconds',e.get('seconds', 0.)) for e in self.engine.events
                       if e.get('iteration') == hook.step)
        if len(self.iteration_times) < 200: self.iteration_times.append(elapsed)
        self.engine.protocol['baseline_iteration_seconds'] = elapsed
        self.engine.protocol['task_iterations'] = hook.task_steps
        self.engine.protocol['iterations_per_epoch'] = hook.task_steps
        model, batch, spec, ctx = self.context()
        within = (hook.step-1) % hook.task_steps+1
        remaining = hook.task_steps-within
        if len(hook.buffer) >= 32 and self.engine.arm not in ('cbp_gir','cbp_gradmax'):
            self.apply(self.engine, model, lop_sites(model), batch, spec, ctx, learner.opt,
                       hook.step, remaining)
        if any(not torch.isfinite(parameter).all() for parameter in learner.net.parameters()):
            self.engine.diverged = True
            if not any(event.get('kind') == 'nonfinite_params' for event in self.engine.events):
                self.engine.events.append(dict(kind='nonfinite_params', step=hook.step,
                                               task=self.engine.task, nonfinite_params=True))
            if hook.status:
                hook.status.update(task=self.engine.task, step=hook.step, phase='DIVERGED', status='DIVERGED')
        self.full_batches = [hook.task_steps]
        for event in self.engine.events[self._event_index:]:
            for swap in event.get('swap_records', []):
                self.insertions.append(dict(task=self.engine.task, **swap))
        self._event_index = len(self.engine.events)
        if within != hook.task_steps: return
        with evaluation(model):
            output, features = learner.net.predict(batch[0])
            sites = lop_sites(model)
            dead = [float((f == 0).all(0).double().mean()) for f in features]
            learning = float((output.argmax(1) == batch[1]).double().mean()) if self.benchmark != 'regression' else None
            for insertion in self.insertions:
                age = self.engine.task-insertion['task']
                if age not in (1, 5, 10): continue
                site_index = next((i for i,s in enumerate(sites) if s.module_name == insertion.get('site')), None)
                if site_index is None: continue
                unit = insertion.get('unit')
                if unit is None: continue
                phi = features[site_index][:, unit]
                weight = sites[site_index].second_layer.layer.weight[:, unit]
                norm = float((phi[:, None]*weight[None]).norm())
                self.contributions.append(dict(**insertion, age_tasks=age, observed_task=self.engine.task, contribution_norm=norm))
                for event in self.engine.events:
                    for swap in event.get('swap_records', []):
                        if (swap.get('site') == insertion.get('site') and
                                swap.get('unit') == insertion.get('unit') and
                                swap.get('task') == insertion.get('task')):
                            swap.setdefault('contribution_norm', {})[age] = norm
        probe_ctx = {**ctx, 'targets': self.probe_batch[1]}
        diagnostic = diagnose(model, self.probe_batch, sites=sites, spec=spec, ctx=probe_ctx,
            mu=self.engine.mu_star if self.engine.mu_star_task == self.engine.task else None)
        self.engine.activities.append(dict(task=self.engine.task, dead_fraction=dead, **diagnostic))
        start = hook.step-hook.task_steps
        losses = hook.results.losses[start:hook.step].double()
        half = torch.where(losses <= losses[0]*.5)[0]
        accuracy = hook.results.accuracies
        metric = dict(task=self.engine.task, learning_accuracy=learning,
                      anytime_loss_auc=float(torch.trapezoid(losses)/(max(1,len(losses)-1))),
                      iterations_to_half_loss=int(half[0])+1 if len(half) else None,
                      dead_fraction=float(sum(dead)/max(1,len(dead))))
        if accuracy is not None:
            values = accuracy[start:hook.step].double()
            metric['anytime_accuracy_auc'] = float(torch.trapezoid(values)/max(1,len(values)-1))
        self.task_metrics.append(metric)

    @contextmanager
    def install(self):
        from lop.algos.gnt import GnT
        original = GnT.gen_new_features
        pilot = self
        def generate(gnt, features_to_replace, counts):
            active = (pilot.engine.coefficient != 0 and 'cbp' in pilot.engine.arm and
                      ('gir' in pilot.engine.arm or 'gradmax' in pilot.engine.arm) and
                      pilot.learner is not None and getattr(pilot.learner, 'gnt', None) is gnt)
            proposals = []
            started = time.monotonic()
            if active and any(counts):
                model, batch, spec, ctx = pilot.context()
                if len(pilot.hook.buffer) >= 32:
                    for site, units in zip(lop_sites(model), features_to_replace):
                        if len(units):
                            proposal = pilot.engine.prepare_gir(model, site, batch, spec, ctx,
                                gnt.opt, units, pilot.hook.step+1, gradmax='gradmax' in pilot.engine.arm)
                            proposals.append((site, units, proposal, pilot.engine.gir_fallback))
            native_started = time.monotonic()
            result = original(gnt, features_to_replace, counts)
            native_seconds = time.monotonic()-native_started
            if active and any(counts) and len(pilot.hook.buffer) < 32:
                for site, units in zip(lop_sites(model), features_to_replace):
                    if len(units):
                        pilot.engine.events.append(dict(kind='cbp_replace',task=pilot.engine.task,
                            iteration=pilot.hook.step+1,site=site.module_name,
                            selected_units=[int(u) for u in units],swaps=0,
                            candidate_missing=len(units),gir_fallback=True,
                            seconds=(time.monotonic()-started-native_seconds)/max(1, len(features_to_replace))))
            for site, units, proposal, gir_fallback in proposals:
                used = pilot.engine.install_gir(model, site, proposal, units, gnt.opt)
                for unit in units[:used]:
                    pilot.insertions.append(dict(task=pilot.engine.task,site=site.module_name,unit=int(unit)))
                pilot.engine.events.append(dict(kind='cbp_replace',task=pilot.engine.task,
                    iteration=pilot.hook.step+1,site=site.module_name,
                    selected_units=[int(u) for u in units],swaps=used,
                    candidate_missing=len(units)-used,
                    gir_fallback=gir_fallback,
                    seconds=(time.monotonic()-started-native_seconds)/max(1,len(proposals))))
            return result
        try:
            GnT.gen_new_features = generate
            yield self
        finally: GnT.gen_new_features = original


