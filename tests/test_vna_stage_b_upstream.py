"""Pinned upstream checks; absent clones are explicit skips, never fake passes."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'loss-of-plasticity': 'a6b79580d85f3025bdb601566d3627c5f489f13b',
    'PyCIL': 'f3509b8ca3f20660ce4aa13f19d5283de81b4b35',
    'RepDistiller': 'b84f547c5db6a35318d4671d7d5c4de74c822403',
}


def pinned(repo):
    path = ROOT/'third_party'/repo
    if not (path/'.git').is_dir():
        pytest.skip(f'{repo} pinned clone absent; fetch failed/needs network: bash scripts/fetch_third_party.sh')
    assert subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() == PINS[repo]
    assert any(path.glob('LICENSE*')), f'{repo}: declared license file absent'
    return path


@pytest.mark.parametrize('repo', PINS)
def test_pinned_sources_patches_apply_cleanly(repo, tmp_path):
    path = pinned(repo)
    clean = tmp_path/repo
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(path), str(clean)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(clean), 'checkout', '--detach', PINS[repo]], check=True, capture_output=True)
    for patch in sorted((ROOT/'third_party'/'patches'/repo).glob('*.patch')):
        subprocess.run(['git', '-C', str(clean), 'apply', '--check', str(patch)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(clean), 'apply', str(patch)], check=True, capture_output=True)


@pytest.mark.parametrize('repo,entry,imports', [
    ('RepDistiller', 'train_student.py', 'from distiller_zoo.KD import DistillKL; from models import model_dict'),
    ('PyCIL', 'main.py', 'from models.icarl import ICarl, _KD_loss; from utils.data_manager import DataManager'),
])
def test_upstream_entrypoint_import_and_help(repo, entry, imports):
    path = pinned(repo)
    subprocess.run([sys.executable, '-c', imports], cwd=path, check=True, capture_output=True, timeout=60)
    result = subprocess.run([sys.executable, entry, '--help'], cwd=path, check=True, capture_output=True, text=True, timeout=60)
    assert 'usage' in result.stdout.lower()


@pytest.mark.parametrize('stem', ['deep_ffnn', 'fix_ltu_net', 'conv_net'])
def test_lop_native_network_entrypoints_one_step(stem):
    path = pinned('loss-of-plasticity')
    matches = list(path.rglob(stem+'.py'))
    assert len(matches) == 1, f'expected unique native LoP {stem} module'
    # Isolate upstream package names; import its definition, instantiate the
    # original class, and perform one synthetic step without rewriting it.
    code = r'''
import importlib.util, inspect, sys, torch
from torch import nn
spec = importlib.util.spec_from_file_location('native_lop_net', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
classes = [c for c in vars(module).values() if inspect.isclass(c) and issubclass(c, nn.Module)
           and c is not nn.Module and c.__module__ == module.__name__]
assert len(classes) == 1, classes
signature = inspect.signature(classes[0])
known = {'num_inputs': 20, 'num_features': 20, 'input_size': 20, 'input_dim': 20,
         'num_outputs': 2, 'output_size': 2, 'output_dim': 2,
         'num_hidden': 5, 'num_hidden_units': 5, 'n_hidden': 5, 'hidden_size': 5,
         'num_hidden_layers': 3, 'act_type': 'relu'}
kwargs = {}
for name, parameter in signature.parameters.items():
    if parameter.default is inspect.Parameter.empty:
        assert name in known, ('unknown mandatory upstream argument', name)
        kwargs[name] = known[name]
model = classes[0](**kwargs)
convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
first = convs[0] if convs else next(m for m in model.modules() if isinstance(m, nn.Linear))
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
inputs = torch.randn(2, first.in_channels, 32, 32) if convs else torch.randn(2, first.in_features)
output = model(inputs)
assert isinstance(output, torch.Tensor)
optimizer.zero_grad(); output.square().mean().backward(); optimizer.step()
'''
    subprocess.run([sys.executable, '-c', code, str(matches[0])], cwd=path, check=True, capture_output=True, timeout=60)
