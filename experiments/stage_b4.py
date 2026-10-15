"""B4: Phase-0 SGD on imported quarter-width Gromo ResNet-18, offline measures."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from experiments import train_recipe_checkpoints as recipe
from experiments.cifar_pilot import LimitedLoader, add_arguments, start_memory, pilot_summary, batches
from experiments.vna_support import setup_data, augmented_batch
from experiments.vna_provenance import code_commit
from experiments.stage_b5 import write_json, write_csv
from diagnostics.icarl_measure import B3Measurement
from diagnostics.stage_b_smoke import FakeMeasurement
from diagnostics.vna_align import MeasurementHook
from diagnostics.vna_gate import evaluate_gate, setting_config
from methods.vna import _module_path, dedicated_generator
from methods.vna_losses import CE
from methods.vna_sites import HookSite
from experiments.vna_resume import Progress, log

POINTS = set(range(20, 201, 20)) | {98, 148}


def build_model(device='cpu'):
    from dual_growth.adapters import GromoResNet18
    return GromoResNet18(num_classes=100, start_width=.25, device=device,
                         use_preactivation=False).to(device)


def sites_for(model):
    sites = []
    for ref in model.growing_blocks():
        path = _module_path(model, model.block(ref.name))
        sites.append(HookSite(model, path+'.first_layer.layer', path+'.second_layer.layer', 'conv',
                              between=(path+'.first_layer.post_layer_function',)))
    return sites


def context(ctx, batch): return CE(), {'targets': batch[1]}


def run(args):
    os.environ.setdefault('VNA_JOB', 'b4')
    out = Path(args.out or ('outputs/pilot/b4' if args.pilot else 'outputs/stageB/b4')).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.fake_data: torch.set_num_threads(1)
    if args.pilot: start_memory(args.device)
    identity = {'setting': 'B4', 'student': 'resnet18-quarter', 'teacher_hash': None,
                'code_commit': code_commit(), 'upstream_commit': '8d19107b61a9459a9021065a329b699adcb0f25b', 'mu_candidates': sorted({args.mu_rel_default, .01}),
                'seed': 0, 'device': args.device, 'fake_data': args.fake_data,
                'smoke': args.smoke, 'pilot': args.pilot, 'workers': args.workers,
                'pilot_batches': args.pilot_batches}
    if args.mode == 'train':
        params = recipe.parser().parse_args([])
        params.architecture = 'resnet18-quarter'
        params.device, params.fake_data, params.data_root = args.device, args.fake_data, args.data_dir
        params.output, params.workers = str(out/'recipe'), args.workers
        params.resume = args.resume or (str(out/'recipe/latest.pt') if (out/'recipe/latest.pt').exists() else None)
        params.vna_identity = identity
        if args.smoke or args.pilot: params.max_epochs = 2
        if args.fake_data: params.limit_batches = 2; params.batch_size = 4
        if args.pilot:
            params.limit_batches = args.pilot_batches
            params.eval_limit_batches = 5
        latest = recipe.run(params)
        # Report the official test once, using the final model, never a best model.
        final_path = out/'final_test.json'
        saved_final = json.loads(final_path.read_text()) if final_path.exists() else {}
        if not params.resume or saved_final.get('epoch') != (2 if args.smoke or args.pilot else 200):
            from experiments.shared_protocol import evaluate
            from torch.utils.data import DataLoader
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != (2 if args.smoke or args.pilot else 200): raise ValueError('incomplete B4 training')
            final_model = build_model(args.device)
            final_model.load_state_dict(checkpoint['model'])
            _, evaluation, _, _, eval_indices, _ = setup_data(args.data_dir, args.fake_data, 0)
            if args.fake_data:
                from torch.utils.data import TensorDataset, Subset
                evaluation = TensorDataset(torch.nn.functional.interpolate(evaluation.tensors[0], size=32), evaluation.tensors[1])
                evaluation = Subset(evaluation, eval_indices[:4])
            else:
                from torchvision.datasets import CIFAR100
                evaluation = CIFAR100(args.data_dir, train=False, transform=evaluation.transform, download=False)
            test_loader = DataLoader(evaluation, batch_size=64, num_workers=args.workers)
            if args.pilot: test_loader = LimitedLoader(test_loader, 5)
            metrics = evaluate(final_model, test_loader, args.device)
            write_json(final_path, {'final_test_acc': 100*metrics['accuracy'], 'epoch': checkpoint['epoch'],
                                   'pilot': args.pilot, 'smoke': args.smoke, 'fake_data': args.fake_data})
        if not args.smoke and not args.pilot:
            write_json(out/'training_summary.json', {'status': 'COMPLETED', 'identity': identity})
            return
    model = build_model(args.device)
    meta = {'schema_version': 'stage1d-v1', 'setting': 'B4', 'code_commit': code_commit(),
            'seed': 0, 'device': args.device, 'pilot': args.pilot, 'smoke': args.smoke, 'fake_data': args.fake_data,
            'parameter_count': sum(p.numel() for p in model.parameters())}
    train, _, indices, _, _, _ = setup_data(args.data_dir, args.fake_data, 0)
    if args.fake_data:
        from torch.utils.data import TensorDataset
        train = TensorDataset(torch.nn.functional.interpolate(train.tensors[0], size=32), train.tensors[1])
    g = dedicated_generator(0, 'b4-checkpoint-batch')
    rows, epochs, costs = [], [], []
    measure = (FakeMeasurement(meta, sites_for, context) if args.fake_data and not args.pilot else
               B3Measurement(meta, mu_rel_default=args.mu_rel_default, context_fn=context,
                             sites_fn=sites_for, ranks=(4,) if args.pilot else (4, 16, 64), groups=False,
                             null_repeats=2 if args.pilot else 8, cg_iters=200 if args.pilot else 1000))
    isolated = MeasurementHook(measure, seed=0)
    progress = Progress(out, identity, name='measurement_latest.pt')
    if progress.state:
        state = progress.state
        rows, epochs, costs = state['rows'], state['epochs'], state['costs']
        g.set_state(state['batch_generator'].cpu())
        for key, value in state['generators'].items():
            stream = dedicated_generator(isolated.seed, isolated.purpose, torch.device(key))
            stream.set_state(value.cpu()); isolated.generators[key] = stream
        progress.restore_artifacts()
    elif (out/'measurements.csv').exists():
        raise ValueError('B4 measurements lack a resumable identity checkpoint')
    def checkpoint_measure():
        progress.save({'rows': rows, 'epochs': epochs, 'costs': costs,
                       'batch_generator': g.get_state(),
                       'generators': {k: stream.get_state() for k, stream in isolated.generators.items()}})
    def boundary():
        progress.boundary(checkpoint_measure, epochs[-1]['seconds'] if epochs else 0.,
                          costs[-1]['seconds'] if costs else 0.)
    started = time.monotonic()
    paths = sorted((out/'recipe').glob('epoch_*.pt'))
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        epoch = checkpoint['epoch']
        if args.pilot:
            if epoch != 2: continue
        elif epoch not in POINTS and not (args.smoke and epoch == 2): continue
        if checkpoint.get('code_commit') != identity['code_commit']: raise ValueError('B4 checkpoint code identity mismatch')
        if checkpoint.get('identity') != identity: raise ValueError('B4 checkpoint protocol identity mismatch')
        if checkpoint['protocol']['architecture'] != 'resnet18-quarter': raise ValueError('wrong B4 checkpoint')
        if checkpoint['protocol']['fake_data'] != args.fake_data: raise ValueError('B4 data mismatch')
        if epoch in {r['epoch'] for r in epochs}: continue
        boundary()
        log(f'measurement start epoch={epoch}')
        model.load_state_dict(checkpoint['model'])
        history = checkpoint['history'][-1]
        ctx = {'epoch': epoch, 'task': 0, 'step': epoch, 'end_of_task': False, 'known_classes': 0,
               'train_acc': history['train']['accuracy']}
        batch = augmented_batch(train, indices, 4 if args.fake_data else 128, g, args.device)
        t = time.monotonic()
        rows.extend(isolated(model, batch, CE(), ctx))
        costs.append({'epoch': epoch, 'seconds': time.monotonic()-t})
        log(f'measurement end epoch={epoch} seconds={costs[-1]["seconds"]:.3f}')
        if len(costs) == 1:
            training = sum(h['seconds'] for h in checkpoint['history'])/epoch*200
            log(f'projected total seconds={training + costs[0]["seconds"]*(len(POINTS)-1):.1f}')
        epochs.append({**meta, 'epoch': epoch, 'train_acc': ctx['train_acc'],
                       'train_loss': history['train']['loss'], 'eval_acc': history['eval']['accuracy'],
                       'eval_loss': history['eval']['loss'], 'lr': history['lr'], 'seconds': history['seconds']})
        write_csv(out/'measurements.csv', rows); write_csv(out/'epochs.csv', epochs)
        checkpoint_measure()
        boundary()
    if not epochs: raise ValueError('no B4 measurement checkpoints available')
    by_mu = {}
    for mu in (1e-2, 1e-3):
        cfg = setting_config('B4', mu_rel_default=mu)
        cfg['final_train_acc'] = epochs[-1]['train_acc']
        gate = {**meta, **evaluate_gate(rows, cfg)}
        expected = POINTS if not args.smoke else {2}
        if args.smoke or args.fake_data or not expected <= {r['epoch'] for r in epochs}: gate['status'] = 'INVALID'
        if args.pilot: gate['status'] = 'PILOT'
        by_mu[str(mu)] = gate
        write_json(out/f'gate_mu_{mu}.json', gate)
    official_mu = args.official_mu
    gate = {'by_mu': by_mu, 'official_mu': official_mu,
            'status': by_mu[str(official_mu)]['status'] if official_mu is not None else 'PENDING_STAGE_A'}
    write_json(out/'gate.json', gate)
    history = json.loads((out/'recipe/history.json').read_text())
    write_csv(out/'epochs.csv', [{**meta, 'epoch': h['epoch'], 'train_acc': h['train']['accuracy'],
              'train_loss': h['train']['loss'], 'eval_acc': h['eval']['accuracy'],
              'eval_loss': h['eval']['loss'], 'lr': h['lr'], 'seconds': h['seconds']} for h in history])
    write_json(out/'summary.json', {**meta, 'identity': identity, 'mu_rel_default': args.mu_rel_default, 'mu_source': args.mu_source, 'gate': gate, 'seconds': time.monotonic()-started,
               'training_seconds': sum(r['seconds'] for r in history), 'epochs_run': len(history),
               'measurement_point_seconds': costs, 'resume_supported': True, 'status': 'COMPLETED',
               'final_test': json.loads((out/'final_test.json').read_text()) if (out/'final_test.json').exists() else None,
               **(pilot_summary(history, costs, rows, args.device, full_epochs=200, full_points=12,
                   full_batches=batches(len(indices), 4 if args.fake_data else 64),
                   pilot_batches=min(args.pilot_batches, batches(len(indices), 4 if args.fake_data else 64)))
                   if args.pilot else {})})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'measure'))
    add_arguments(p)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mu-rel-default', type=float, default=.01)
    p.add_argument('--mu-source', default='cli/default')
    p.add_argument('--official-mu', type=float, choices=(1e-2, 1e-3))
    p.add_argument('--resume')
    return p


if __name__ == '__main__': run(parser().parse_args())
