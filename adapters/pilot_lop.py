"""GIR intercepts only upstream GnT generation; selection and bookkeeping stay native."""
from contextlib import contextmanager
import math
import time
import torch
from adapters.lop_hooks import Prediction, sites_for
from diagnostics.lop_measure import activity_stats
from methods.vna_losses import MSE


class LoPPilot:
    def __init__(self, engine):
        self.engine = engine
        self.learner = self.hook = None

    def state_dict(self): return self.engine.state_dict()
    def load_state_dict(self, state): self.engine.load_state_dict(state)

    def context(self):
        batch = self.hook.buffer.batch()
        return Prediction(self.learner.net), batch, MSE(reduction='mean'), {
            'targets': batch[1], 'loss_fn': lambda z, c: self.learner.loss_func(z, c['targets'].to(z))}

    def before(self, hook, learner, x, target):
        self.hook, self.learner = hook, learner
        self.engine.begin_task(hook.step//hook.task_steps+1)
        hook.buffer.update((x, target))

    def after(self, hook, learner):
        model, batch, spec, ctx = self.context()
        if len(hook.buffer) == hook.buffer.m:
            self.engine.correction(model, sites_for(model), batch, spec, ctx, hook.step,
                                   learner.opt.param_groups[0]['lr'])
        elif self.engine.coefficient != 0 and hook.step % self.engine.every == 0:
            self.engine.events.append(dict(kind='correction', task=self.engine.task, step=hook.step,
                rho=None, skipped=True, reason='window_warmup', seconds=0.))
        if hook.step % hook.task_steps == 0 and self.engine.task % 10 == 0:
            self.engine.activities.append(dict(task=self.engine.task, sites=activity_stats(model, batch)))

    @contextmanager
    def install(self):
        from lop.algos.gnt import GnT
        original = GnT.gen_new_features
        pilot = self
        def generate(gnt, features_to_replace, num_features_to_replace):
            if (not pilot.engine.gir or pilot.engine.upstream_random or pilot.learner is None
                    or getattr(pilot.learner, 'gnt', None) is not gnt):
                return original(gnt, features_to_replace, num_features_to_replace)
            proposals = {}
            if len(pilot.hook.buffer):
                model, batch, spec, ctx = pilot.context()
                sites = sites_for(model)
                for i, count in enumerate(num_features_to_replace):
                    if not count: continue
                    # Uniform(-b,b): sqrt(E ||w||²) = b sqrt(fan_in/3); bias init is zero.
                    norm = gnt.bounds[i] * math.sqrt(gnt.net[2*i].in_features/3)
                    proposals[i] = pilot.engine.take(model, sites[i], batch, spec, ctx,
                        int(count), norm, pilot.hook.step+1)
            # Consume precisely the upstream random draws and retain bias correction,
            # age/utility reset, outgoing zeros and optimizer reset in upstream order.
            result = original(gnt, features_to_replace, num_features_to_replace)
            with torch.no_grad():
                for i, a in proposals.items():
                    units = features_to_replace[i][:len(a['weight'])]
                    gnt.net[2*i].weight[units] = a['weight'].to(gnt.net[2*i].weight)
                    if a['bias'] is not None: gnt.net[2*i].bias[units] = a['bias'].to(gnt.net[2*i].bias)
            return result
        try:
            GnT.gen_new_features = generate
            yield
        finally: GnT.gen_new_features = original
