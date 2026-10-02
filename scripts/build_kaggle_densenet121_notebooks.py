"""Build three isolated CIFAR-DenseNet121 block-boundary E-to-O notebooks."""

import copy
import json
from pathlib import Path


TEMPLATE = Path("notebooks/kaggle_vgg16_seed1_end_to_end_t4x2.ipynb")


def cell_text(cell):
    return "".join(cell["source"])


base = json.loads(TEMPLATE.read_text())
replacements = {
    "# CIFAR-VGG16-BN stall experiment — seed 1":
        "# CIFAR-DenseNet121 block-boundary experiment — seed 1",
    "full-width Gromo CIFAR-VGG16-BN": "CIFAR-DenseNet121",
    "all 8 internal VGG16 conv links": "all 4 DenseBlock boundaries",
    "/kaggle/working/vgg16_seed1_stall150_v1":
        "/kaggle/working/densenet121_seed1_stall150_v1",
    "cifar-vgg16-bn-sgd-multistep-200-v1-post200-val-best":
        "cifar-densenet121-sgd-multistep-200-v1-post200-val-best",
    '"architecture") == "CIFAR-VGG16-BN"':
        '"architecture") == "CIFAR-DenseNet121"',
    'f"random_init_vgg16_bn_seed_{SEED}"':
        'f"random_init_densenet121_seed_{SEED}"',
    '"--architecture", "vgg16"': '"--architecture", "densenet121"',
    '"--site", "stages.2.links.0"':
        '"--site", "core.features.denseblock3"',
    '"architecture": "CIFAR-VGG16-BN"':
        '"architecture": "CIFAR-DenseNet121"',
    "did not run CIFAR-VGG16-BN": "did not run CIFAR-DenseNet121",
    '"CIFAR-VGG16-BN"': '"CIFAR-DenseNet121"',
}
for cell in base["cells"]:
    text = cell_text(cell)
    for old, new in replacements.items():
        text = text.replace(old, new)
    cell["source"] = text.splitlines(keepends=True)

for seed in (0, 1, 2):
    notebook = copy.deepcopy(base)
    substitutions = {
        "seed 1": f"seed {seed}",
        "seed1": f"seed{seed}",
        "SEED = 1": f"SEED = {seed}",
        '"--seed", "1"': f'"--seed", "{seed}"',
    }
    for cell in notebook["cells"]:
        text = cell_text(cell)
        for old, new in substitutions.items():
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    destination = Path(
        f"notebooks/kaggle_densenet121_seed{seed}_end_to_end_t4x2.ipynb")
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(destination)
