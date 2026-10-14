"""Pinned upstream checks; absent clones are explicit skips, never fake passes."""
from pathlib import Path
import subprocess
import sys
import os
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
        message = f'{repo} pinned clone absent; run bash scripts/fetch_third_party.sh'
        if os.environ.get('REQUIRE_UPSTREAM') == '1':
            pytest.fail(message)
        pytest.skip(message)
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
    ('PyCIL', 'main.py', 'from models.icarl import iCaRL, _KD_loss; from utils.data_manager import DataManager'),
])
def test_upstream_entrypoint_import_and_help(repo, entry, imports):
    path = pinned(repo)
    env = {**os.environ, 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'}
    subprocess.run([sys.executable, '-c', imports], cwd=path, env=env, check=True, capture_output=True, timeout=60)
    result = subprocess.run([sys.executable, entry, '--help'], cwd=path, env=env, check=True, capture_output=True, text=True, timeout=60)
    assert 'usage' in result.stdout.lower()


@pytest.mark.parametrize('stem', ['deep_ffnn', 'fix_ltu_net', 'conv_net'])
def test_lop_native_network_entrypoints_one_step(stem):
    path = pinned('loss-of-plasticity')
    matches = list(path.rglob(stem+'.py'))
    assert len(matches) == 1, f'expected unique native LoP {stem} module'
    # Isolate upstream package names; import its definition, instantiate the
    # original class, and perform one synthetic step without rewriting it.
    code = r'''
import sys, torch
from torch import nn
# Native symbols and predict signatures verified in the pinned sources.
stem = sys.argv[1]
if stem == 'deep_ffnn':
    from lop.nets.deep_ffnn import DeepFFNN
    model = DeepFFNN(input_size=20, num_features=5, num_outputs=2, num_hidden_layers=3)
    inputs = torch.randn(2, 20)
elif stem == 'fix_ltu_net':
    from lop.nets.fix_ltu_net import FixLTUNet
    model = FixLTUNet(num_inputs=20, num_features=5)
    inputs = torch.randn(2, 20)
else:
    from lop.nets.conv_net import ConvNet
    model = ConvNet(num_classes=2)
    inputs = torch.randn(2, 3, 32, 32)
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
output, features = model.predict(inputs)
assert isinstance(output, torch.Tensor)
optimizer.zero_grad(); output.square().mean().backward(); optimizer.step()
'''
    subprocess.run([sys.executable, '-c', code, stem], cwd=path, check=True, capture_output=True, timeout=60)
