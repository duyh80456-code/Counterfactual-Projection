"""Class-level learn interception; no upstream training loop is copied."""
from contextlib import contextmanager
import copy
import inspect
import time
import numpy as np
import torch
from torch import nn
from diagnostics.vna_align import MeasurementBuffer, MeasurementHook, _global_state
from methods.vna_sites import HookSite
from methods.vna_losses import CE, MSE
from methods.vna_select import neuron_norms
from experiments.vna_resume import rng_state, restore_rng, log


def detached(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, dict): return {k: detached(v) for k, v in value.items()}
    if isinstance(value, list): return [detached(v) for v in value]
    if isinstance(value, tuple): return tuple(detached(v) for v in value)
    return copy.deepcopy(value)


class StepLog:
    """Packed replay of exactly the loss/argmax consumed by upstream loops.

    Prefix replay never trains: the loops only use their learn return to log
    MSE or softmax argmax accuracy. Store those sufficient statistics, rather
    than millions of Python objects or unused full-precision logits.
    """
    def __init__(self, capacity, task_steps, batch_size, classification):
        self.count, self.task_steps = 0, task_steps
        self.losses = torch.empty(capacity, dtype=torch.float32)
        self.accuracies = torch.empty(capacity, dtype=torch.float32) if classification else None
        self.predictions = torch.empty((capacity, batch_size), dtype=torch.uint8) if classification else None

    def append(self, result, target):
        i = self.count
        self.losses[i] = float(result[0] if isinstance(result, tuple) else result)
        if self.predictions is not None:
            predicted = result[1].detach().softmax(1).argmax(1)
            self.predictions[i] = predicted.cpu().to(torch.uint8)
            self.accuracies[i] = float((predicted == target).float().mean())
        self.count += 1

    def replay(self, index, learner):
        device = next(learner.net.parameters()).device
        loss = self.losses[index].to(device)
        if self.predictions is None: return loss
        predicted = self.predictions[index].long().to(device)
        width = learner.net.layers[-1].weight.shape[0]
        output = torch.nn.functional.one_hot(predicted, width).to(next(learner.net.parameters()))
        return loss, output

    def state(self):
        return {'count': self.count, 'task_steps': self.task_steps,
                'losses': self.losses[:self.count].clone(),
                'accuracies': None if self.accuracies is None else self.accuracies[:self.count].clone(),
                'predictions': None if self.predictions is None else self.predictions[:self.count].clone()}

    def restore(self, state):
        self.count = state['count']
        for name in ('losses', 'accuracies', 'predictions'):
            value = state[name]
            if value is not None: getattr(self, name)[:self.count].copy_(value)

    def tasks(self):
        for start in range(0, self.count, self.task_steps):
            end = min(start+self.task_steps, self.count)
            yield {'task': start//self.task_steps+1, 'step': end,
                   'online_error': float(self.losses[start:end].double().mean()),
                   'online_accuracy': None if self.accuracies is None else float(self.accuracies[start:end].double().mean())}


class Prediction(nn.Module):
    """Functional-call bridge to LoP's original predict method."""
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, x):
        return self.native.predict(x)[0]


def sites_for(model):
    native = model.native
    if hasattr(native, 'conv1'):
        return [HookSite(model, 'native.conv1', 'native.conv2', 'conv', between=('native.layers.1', 'native.pool')),
                HookSite(model, 'native.conv2', 'native.conv3', 'conv', between=('native.layers.3', 'native.pool')),
                HookSite(model, 'native.conv3', 'native.fc1', 'conv_linear', between=('native.layers.5', 'native.pool')),
                HookSite(model, 'native.fc1', 'native.fc2', 'linear', between=('native.layers.7',)),
                HookSite(model, 'native.fc2', 'native.fc3', 'linear', between=('native.layers.9',))]
    return [HookSite(model, f'native.layers.{i}', f'native.layers.{i+2}', 'linear',
                     between=(f'native.layers.{i+1}',)) for i in range(0, len(native.layers)-2, 2)]


def gnt_state(gnt):
    if gnt is None:
        return None
    return copy.deepcopy({k: v for k, v in vars(gnt).items()
                          if k not in ('net', 'opt', 'loss_func', 'm')})


def learner_state(learner):
    return detached({'model': learner.net.state_dict(), 'optimizer': learner.opt.state_dict(),
                          'gnt': gnt_state(getattr(learner, 'gnt', None)),
                          'previous_features': learner.previous_features})


def restore_learner(learner, state):
    learner.net.load_state_dict(state['model']); learner.opt.load_state_dict(state['optimizer'])
    if state['gnt'] is not None:
        vars(learner.gnt).update(copy.deepcopy(state['gnt']))
    learner.previous_features = copy.deepcopy(state['previous_features'])


class LearnHook:
    def __init__(self, benchmark, arm, params, measure=None, *, seed=0, checkpoint=None,
                 resume=None, observe=None, time_steps=False, timing_started=None, timing_interval=50, progress=None, status=None, checkpoint_seconds=900):
        self.benchmark, self.arm, self.params = benchmark, arm, params
        self.progress, self.status = progress, status
        self.checkpoint_seconds = checkpoint_seconds
        self.last_checkpoint = self.task_started = self.report_started = time.monotonic()
        self.report_step = 0
        self.last_frame = None
        self.last_status = 0.
        self.last_task_ms = None
        self.buffer = MeasurementBuffer(params['mini_batch_size'] if benchmark == 'imagenet' else 64)
        self.measurement = MeasurementHook(measure, seed=seed) if measure else None
        self.step, self.rows = 0, []
        self.checkpoint, self.resume, self.observe = checkpoint, resume, observe
        self.last_learner, self.shadow, self.selected = None, None, None
        self.shadows, self.cbp_candidates = {}, []
        self.init_row_norms = None
        self.measurement_point_seconds = []
        self.time_steps = time_steps
        self.timing_started = time.monotonic() if timing_started is None else timing_started
        self.timing_interval = timing_interval
        self.startup_seconds = None
        self.step_wall_seconds = []
        self._timing_mark = None
        self._timing_step = 0
        self._timing_measurement = 0.
        self.points = {1, 2, 5, 10, 20, 50, 100, 150, 200}
        if benchmark == 'regression':
            self.task_steps = int(params['flip_after'])
        elif benchmark == 'mnist':
            self.task_steps = int(params.get('change_after', 60000) / params.get('mini_batch_size', 1))
        else:
            self.task_steps = int(params.get('examples_per_epoch', 1200) * params['num_showings'] / params['mini_batch_size'])
        if self.task_steps < 1:
            raise ValueError('task must contain at least one step')
        capacity = int(params['num_data_points']) if benchmark == 'regression' else params['num_tasks']*self.task_steps
        self.results = StepLog(capacity, self.task_steps, params.get('mini_batch_size', 1), benchmark != 'regression')

    def _shadow(self, learner):
        rates = self.params['gir_replacement_rates']
        self.cbp_candidates = []
        for rate in rates:
            from lop.algos.gnt import GnT
            from lop.algos.convGnT import ConvGnT
            kwargs = dict(net=learner.net.layers, opt=learner.opt, hidden_activation=learner.net.act_type,
                          replacement_rate=rate, decay_rate=.99,
                          maturity_threshold=self.params['gir_maturity'], util_type='adaptable_contribution',
                          device=str(next(learner.net.parameters()).device))
            if rate not in self.shadows:
                if self.benchmark == 'imagenet':
                    self.shadows[rate] = ConvGnT(**kwargs, init='default', num_last_filter_outputs=learner.net.last_filter_output)
                else:
                    self.shadows[rate] = GnT(**kwargs, init='kaiming', accumulate=self.benchmark == 'mnist')
            shadow = self.shadows[rate]
            with _global_state():
                selected = shadow.test_features(learner.previous_features)[0]
            label = f'cbp_rr_{rate:g}' if self.benchmark == 'regression' else 'cbp'
            self.cbp_candidates.append((label, shadow, selected))
        self.shadow, self.selected = self.cbp_candidates[0][1:]

    def _checkpoint(self, learner, frame):
        if self.checkpoint is None:
            return
        arrays = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy()
                  for k, v in frame.f_locals.items() if isinstance(v, (torch.Tensor, np.ndarray))
                  and k not in ('x', 'y', 'inputs', 'outputs', 'x_train', 'y_train', 'x_test', 'y_test',
                                'batch_x', 'batch_y', 'pixel_permutation', 'data_permutation', 'example_order')}
        # Preserve the cumulatively permuted MNIST data: skipped CBP RNG draws
        # make the replayed permutation prefix unsuitable for live training.
        if self.benchmark == 'mnist':
            arrays.update({k: frame.f_locals[k].clone() for k in ('x', 'y')})
        state = {'step': self.step, 'learner': learner_state(learner), 'rng': rng_state(),
                 'results': self.results.state(), 'rows': self.rows, 'arrays': arrays,
                 'buffer': list(self.buffer.samples), 'shadows': {r: gnt_state(s) for r, s in self.shadows.items()},
                 'init_row_norms': self.init_row_norms,
                 'measurement_point_seconds': self.measurement_point_seconds,
                 'last_task_ms': self.last_task_ms,
                 'generators': {} if self.measurement is None else
                     {k: g.get_state() for k, g in self.measurement.generators.items()}}
        size = self.checkpoint(state)
        self.last_checkpoint = time.monotonic()
        detail = ''
        if self.step and self.step % self.task_steps == 0:
            row = list(self.results.tasks())[-1]
            detail = (f' task={row["task"]} steps={self.task_steps} '
                      f'ms/step={self.last_task_ms} '
                      f'online_accuracy={row["online_accuracy"]} online_error={row["online_error"]}')
        log(f'{self.benchmark} {self.arm} step={self.step}{detail} checkpoint written ({size} bytes)')

    def _restore_resume(self, learner, frame):
        state = self.resume
        restore_learner(learner, state['learner']); restore_rng(state['rng'])
        self.results.restore(state['results']); self.rows = copy.deepcopy(state['rows'])
        self.buffer.samples.extend(state['buffer'])
        self.init_row_norms = state['init_row_norms']
        self.last_task_ms = state.get('last_task_ms')
        self.measurement_point_seconds = copy.deepcopy(state.get('measurement_point_seconds', []))
        for k, v in state['arrays'].items():
            current = frame.f_locals.get(k)
            if isinstance(current, torch.Tensor): current.copy_(v)
            elif isinstance(current, np.ndarray): np.copyto(current, v)
        if self.measurement is not None:
            from methods.vna import dedicated_generator
            for k, v in state['generators'].items():
                g = dedicated_generator(self.measurement.seed, self.measurement.purpose, torch.device(k))
                g.set_state(v.cpu()); self.measurement.generators[k] = g
        if state['shadows']:
            self._shadow(learner)
            for rate, values in state['shadows'].items(): vars(self.shadows[rate]).update(values)
        self.resume = None
        self.task_started = self.report_started = time.monotonic()
        self.report_step = self.step

    def after_or_replay(self, original, learner, x, target):
        frame = inspect.currentframe().f_back
        while frame and frame.f_code.co_name not in ('expr', 'online_expr', 'repeat_expr'):
            frame = frame.f_back
        if frame is None:
            frame = inspect.currentframe().f_back
        self.last_frame = frame
        self.last_learner = learner
        if self.resume is not None and self.step < self.resume['step']:
            if self.step == 0: self.results.restore(self.resume['results'])
            result = self.results.replay(self.step, learner)
            self.step += 1
            if self.step == len(self.results.losses):
                self._restore_resume(learner, frame)
            return result
        if self.resume is not None:
            self._restore_resume(learner, frame)
        now = time.monotonic()
        if self.status and (self.step % self.task_steps == 0 or now-self.last_status >= 30):
            self.status.update(task=self.step//self.task_steps+1, step=self.step, phase='training')
            self.last_status = now
        if self.progress:
            self.progress.boundary(lambda: self._checkpoint(learner, frame),
                                   point_seconds=max((p['seconds'] for p in self.measurement_point_seconds), default=0))
        if self.step % self.task_steps == 0:
            self.buffer.samples.clear()
            self._checkpoint(learner, frame)
        elif now-self.last_checkpoint >= self.checkpoint_seconds:
            self._checkpoint(learner, frame)
        if self.init_row_norms is None:
            self.init_row_norms = [float(neuron_norms({'weight': layer.weight.detach(),
                                                      'bias': layer.bias.detach() if layer.bias is not None else None}).mean())
                                   for layer in list(learner.net.layers)[::2][:-1]]
        result = original(learner, x=x, target=target)
        previous_buffer_size = len(self.buffer)
        self.buffer.update((x, target))
        first_point = previous_buffer_size < self.buffer.m and len(self.buffer) == self.buffer.m
        self.step += 1
        task = (self.step-1)//self.task_steps + 1
        within = (self.step-1)%self.task_steps + 1
        self.results.append(result, target)
        if self.observe: self.observe(learner, x, target, task, self.step)
        if self.measurement is not None and self.arm == 'bp':
            self._shadow(learner)
            if task in self.points and (first_point or within == max(1, self.task_steps//2)):
                model = Prediction(learner.net)
                spec = MSE(reduction='mean') if self.benchmark == 'regression' else CE()
                ctx = {'targets': self.buffer.batch()[1],
                       'loss_fn': lambda z, c: learner.loss_func(z, c['targets'].to(z) if self.benchmark == 'regression' else c['targets']),
                       'task': task, 'step': self.step, 'point': 'first' if first_point else 'mid',
                       'within_task_step': within, 'buffer_task_purity': 1.0, 'buffer_m': self.buffer.m,
                       'shadow': self.shadow, 'cbp_units': self.selected, 'learner': learner}
                ctx['init_row_norms'] = self.init_row_norms
                ctx['cbp_candidates'] = self.cbp_candidates
                device = next(learner.net.parameters()).device
                if self.time_steps and device.type == 'cuda': torch.cuda.synchronize(device)
                started = time.monotonic()
                rows = self.measurement(model, self.buffer.batch(), spec, ctx)
                if self.time_steps and device.type == 'cuda': torch.cuda.synchronize(device)
                self.measurement_point_seconds.append({'batch_size': len(self.buffer), 'seconds': time.monotonic()-started})
                self.rows.extend(rows or [])
        if self.step % 10000 == 0 or within == self.task_steps:
            elapsed = time.monotonic()-self.report_started
            log(f'{self.benchmark} {self.arm} task={task} step={self.step} '
                f'ms/step={1000*elapsed/max(1, self.step-self.report_step):.3f}')
            self.report_started, self.report_step = time.monotonic(), self.step
        if within == self.task_steps:
            self.last_task_ms = 1000*(time.monotonic()-self.task_started)/within
            row = list(self.results.tasks())[-1]
            log(f'{self.benchmark} {self.arm} task={task} steps={within} '
                f'ms/step={1000*(time.monotonic()-self.task_started)/within:.3f} '
                f'online_accuracy={row["online_accuracy"]} online_error={row["online_error"]}')
            self.task_started = time.monotonic()
        return result

    def timed_learn(self, original, learner, x, target):
        device = next(learner.net.parameters()).device
        def sync():
            if device.type == 'cuda': torch.cuda.synchronize(device)
        if self._timing_mark is None:
            sync()
            self._timing_mark = time.monotonic()
            self.startup_seconds = self._timing_mark - self.timing_started
        result = self.after_or_replay(original, learner, x, target)
        # GPU windows amortize synchronization; CPU steps are timed individually.
        if device.type != 'cuda' or self.step % self.timing_interval == 0 or self.step % self.task_steps == 0:
            sync()
            now = time.monotonic()
            measurement = sum(p['seconds'] for p in self.measurement_point_seconds)
            count = self.step - self._timing_step
            seconds = max(0., now - self._timing_mark - (measurement - self._timing_measurement)) / count
            self.step_wall_seconds.extend([seconds] * count)
            self._timing_mark, self._timing_step = now, self.step
            self._timing_measurement = measurement
        return result

    @contextmanager
    def install(self):
        from lop.algos.bp import Backprop
        from lop.algos.cbp import ContinualBackprop
        from lop.algos.convCBP import ConvCBP
        originals = {cls: cls.learn for cls in (Backprop, ContinualBackprop, ConvCBP)}
        try:
            for cls, original in originals.items():
                def wrapped(learner, x, target, original=original):
                    return (self.timed_learn(original, learner, x, target) if self.time_steps
                            else self.after_or_replay(original, learner, x, target))
                cls.learn = wrapped
            yield self
        finally:
            for cls, original in originals.items(): cls.learn = original
