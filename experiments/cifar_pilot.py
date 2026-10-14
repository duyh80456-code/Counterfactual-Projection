"""Shared bounds and cost reporting for the CIFAR wiring pilot."""
from itertools import islice
import math
import torch

GATE_KEYS = ('cg_converged', 'rank', 'mu_rel', 'damping_norm', 'task', 'epoch',
             'end_of_task', 'rho_joint', 'significant_N1', 'V_tiny_V',
             'V_random_frob', 'train_acc')


class LimitedLoader:
    """Pass through loader attributes without consuming batches past the bound."""
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit

    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit else len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        return islice(self.loader, self.limit) if self.limit else iter(self.loader)


def add_arguments(parser):
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pilot-batches', type=positive_batches, default=20)


def positive_batches(value):
    from argparse import ArgumentTypeError
    value = int(value)
    if value < 1:
        raise ArgumentTypeError('--pilot-batches must be positive')
    return value


def start_memory(device):
    if str(device).startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def pilot_summary(epochs, points, rows, device, *, full_epochs, full_points,
                  full_batches, pilot_batches):
    epoch_seconds = [r['seconds'] for r in epochs]
    point_seconds = [p['seconds'] for p in points]
    epoch_mean = sum(epoch_seconds)/len(epoch_seconds)
    point_mean = sum(point_seconds)/len(point_seconds)
    ratio = full_batches / pilot_batches
    return {'epoch_seconds': epoch_seconds, 'seconds_per_epoch': epoch_mean,
            'seconds_per_measurement_point': point_mean,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(torch.device(device))
                if str(device).startswith('cuda') else 0,
            'full_epochs': full_epochs, 'full_measurement_points': full_points,
            'full_batches_per_epoch': full_batches, 'pilot_batches_per_epoch': pilot_batches,
            'batch_ratio': ratio,
            'projected_full_gpu_hours': (epoch_mean*full_epochs*ratio + point_mean*full_points)/3600,
            'gate_keys_present': bool(rows) and all(all(k in r for k in GATE_KEYS) for r in rows),
            'projection_note': 'Reduced pilot ranks/nulls/CG; full measurement cost may be higher.'}


def batches(samples, batch_size):
    return math.ceil(samples/batch_size)
