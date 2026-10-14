"""B4: Phase-0 SGD on imported quarter-width Gromo ResNet-18, offline measures."""
import argparse
import json
from pathlib import Path
import time
import torch
from experiments import train_recipe_checkpoints as recipe
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
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.fake_data: torch.set_num_threads(1)
    if args.mode == 'train':
        params = recipe.parser().parse_args([])
        params.architecture = 'resnet18-quarter'
        params.device, params.fake_data, params.data_root = args.device, args.fake_data, args.data_dir
        params.output, params.workers, params.resume = str(out/'recipe'), args.workers, args.resume
        if args.smoke: params.max_epochs = 2
        if args.fake_data: params.limit_batches = 2; params.batch_size = 4
        latest = recipe.run(params)
        # Report the official test once, using the final model, never a best model.
        final_path = out/'final_test.json'
        saved_final = json.loads(final_path.read_text()) if final_path.exists() else {}
        if not args.resume or saved_final.get('epoch') != (2 if args.smoke else 200):
            from experiments.shared_protocol import evaluate
            from torch.utils.data import DataLoader
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != (2 if args.smoke else 200): raise ValueError('incomplete B4 training')
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
            metrics = evaluate(final_model, DataLoader(evaluation, batch_size=64, num_workers=args.workers), args.device)
            write_json(final_path, {'final_test_acc': 100*metrics['accuracy'], 'epoch': checkpoint['epoch'],
                                   'smoke': args.smoke, 'fake_data': args.fake_data})
        if not args.smoke: return
    model = build_model(args.device)
    meta = {'schema_version': 'stage1d-v1', 'setting': 'B4', 'code_commit': code_commit(),
            'seed': 0, 'device': args.device, 'smoke': args.smoke, 'fake_data': args.fake_data,
            'parameter_count': sum(p.numel() for p in model.parameters())}
    train, _, indices, _, _, _ = setup_data(args.data_dir, args.fake_data, 0)
    if args.fake_data:
        from torch.utils.data import TensorDataset
        train = TensorDataset(torch.nn.functional.interpolate(train.tensors[0], size=32), train.tensors[1])
    g = dedicated_generator(0, 'b4-checkpoint-batch')
    rows, epochs, costs = [], [], []
    measure = (FakeMeasurement(meta, sites_for, context) if args.fake_data else
               B3Measurement(meta, mu_rel_default=args.mu_rel_default, context_fn=context,
                             sites_fn=sites_for, ranks=(4, 16, 64), groups=False))
    isolated = MeasurementHook(measure, seed=0)
    started = time.monotonic()
    paths = sorted((out/'recipe').glob('epoch_*.pt'))
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        epoch = checkpoint['epoch']
        if epoch not in POINTS and not (args.smoke and epoch == 2): continue
        if checkpoint['protocol']['architecture'] != 'resnet18-quarter': raise ValueError('wrong B4 checkpoint')
        if checkpoint['protocol']['fake_data'] != args.fake_data: raise ValueError('B4 data mismatch')
        model.load_state_dict(checkpoint['model'])
        history = checkpoint['history'][-1]
        ctx = {'epoch': epoch, 'task': 0, 'step': epoch, 'end_of_task': False, 'known_classes': 0,
               'train_acc': history['train']['accuracy']}
        batch = augmented_batch(train, indices, 4 if args.fake_data else 128, g, args.device)
        t = time.monotonic()
        rows.extend(isolated(model, batch, CE(), ctx))
        costs.append({'epoch': epoch, 'seconds': time.monotonic()-t})
        epochs.append({**meta, 'epoch': epoch, 'train_acc': ctx['train_acc'],
                       'train_loss': history['train']['loss'], 'eval_acc': history['eval']['accuracy'],
                       'eval_loss': history['eval']['loss'], 'lr': history['lr'], 'seconds': history['seconds']})
        write_csv(out/'measurements.csv', rows); write_csv(out/'epochs.csv', epochs)
    if not epochs: raise ValueError('no B4 measurement checkpoints available')
    cfg = setting_config('B4', mu_rel_default=args.mu_rel_default)
    cfg['final_train_acc'] = epochs[-1]['train_acc']
    gate = {**meta, **evaluate_gate(rows, cfg)}
    expected = POINTS if not args.smoke else {2}
    if args.smoke or args.fake_data or not expected <= {r['epoch'] for r in epochs}: gate['status'] = 'INVALID'
    write_json(out/'gate.json', gate)
    history = json.loads((out/'recipe/history.json').read_text())
    write_csv(out/'epochs.csv', [{**meta, 'epoch': h['epoch'], 'train_acc': h['train']['accuracy'],
              'train_loss': h['train']['loss'], 'eval_acc': h['eval']['accuracy'],
              'eval_loss': h['eval']['loss'], 'lr': h['lr'], 'seconds': h['seconds']} for h in history])
    write_json(out/'summary.json', {**meta, 'gate': gate, 'seconds': time.monotonic()-started,
               'training_seconds': sum(r['seconds'] for r in history), 'epochs_run': len(history),
               'measurement_point_seconds': costs, 'resume_supported': True,
               'final_test': json.loads((out/'final_test.json').read_text()) if (out/'final_test.json').exists() else None})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'measure'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='outputs/stageB/b4')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mu-rel-default', type=float, default=.01)
    p.add_argument('--resume')
    return p


if __name__ == '__main__': run(parser().parse_args())
