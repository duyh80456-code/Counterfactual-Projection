"""Build seed 0/1/2 CIFAR-VGG16-BN Vanilla/O-only/E-to-O notebooks."""

import copy
import json
from pathlib import Path


TEMPLATE = Path("notebooks/kaggle_resnet34_seed1_end_to_end_t4x2.ipynb")


def cell_text(cell):
    return "".join(cell["source"])


base = json.loads(TEMPLATE.read_text())
replacements = {
    "# CIFAR-ResNet34 stall experiment — seed 1":
        "# CIFAR-VGG16-BN stall experiment — seed 1",
    "full-width Gromo CIFAR-ResNet34": "full-width Gromo CIFAR-VGG16-BN",
    "all 16 ResNet34 BasicBlocks":
        "all 12 adjacent VGG16 conv interfaces (8 native + 4 operator-aware MaxPool-bridge)",
    "rollback with patience 10": "rollback with patience 15",
    "/kaggle/working/resnet34_seed1_stall150_v1":
        "/kaggle/working/vgg16_seed1_stall150_v3",
    "cifar-resnet34-sgd-multistep-200-v1-post200-val-best":
        "cifar-vgg16-bn-sgd-multistep-200-v1-post200-val-best",
    '"architecture") == "CIFAR-ResNet34"':
        '"architecture") == "CIFAR-VGG16-BN"',
    'f"random_init_resnet34_seed_{SEED}"':
        'f"random_init_vgg16_bn_seed_{SEED}"',
    '"--architecture", "resnet34"': '"--architecture", "vgg16"',
    '"architecture") != "CIFAR-ResNet34"':
        '"architecture") != "CIFAR-VGG16-BN"',
    '"architecture": "CIFAR-ResNet34"':
        '"architecture": "CIFAR-VGG16-BN"',
    "did not run CIFAR-ResNet34": "did not run CIFAR-VGG16-BN",
    '"CIFAR-ResNet34"': '"CIFAR-VGG16-BN"',
}
for cell in base["cells"]:
    text = cell_text(cell)
    for old, new in replacements.items():
        text = text.replace(old, new)
    cell["source"] = text.splitlines(keepends=True)

phase2 = cell_text(base["cells"][8])
needle = '"--seed", "1", "--architecture", "vgg16",\n'
if needle not in phase2:
    raise RuntimeError("VGG notebook template lacks the Phase-2 architecture")
phase2 = phase2.replace(
    needle, needle + '        "--site", "auto",\n'
    '        "--o-only-site", "stages.2.links.0",\n'
    '        "--site-selection-mode", "all_functional_gain",\n')
phase2 = phase2.replace(
    '        "--line-search-scales", "0.0125,0.025,0.05"]',
    '        "--line-search-scales", "0.025,0.05,0.1,0.2",\n'
    '        "--projection-samples", "64"]')
phase2 = phase2.replace(
    '        "--retrigger-patience", "10",\n', "")
commands_needle = '''method_names = ("ours_e_driven_o", "o_projection_only")
commands = {
    name: [sys.executable, "-m", "experiments.run_plateau_fork",
           "--method", name] + base_args(OUTPUT / name)
    for name in method_names}
'''
commands_replacement = '''method_names = ("ours_e_driven_o", "o_projection_only")
retrigger_patience = {
    "ours_e_driven_o": 15,
    "o_projection_only": 10,
}
commands = {
    name: [sys.executable, "-m", "experiments.run_plateau_fork",
           "--method", name] + base_args(OUTPUT / name) + [
               "--retrigger-patience", str(retrigger_patience[name])]
    for name in method_names}
'''
if commands_needle not in phase2:
    raise RuntimeError("VGG notebook template lacks the Phase-2 command map")
phase2 = phase2.replace(commands_needle, commands_replacement)
resume_needle = '''               and (item["payload"].get("protocol", {}).get(
                   "intervention_schedule") or {}).get("mode") ==
                   expected_schedule_mode]
'''
resume_replacement = '''               and (item["payload"].get("protocol", {}).get(
                   "intervention_schedule") or {}).get("mode") ==
                   expected_schedule_mode
               and (name != "ours_e_driven_o" or
                    (item["payload"].get("protocol", {}).get(
                        "intervention_schedule") or {}).get("patience") ==
                    retrigger_patience[name])
               and item["payload"].get("protocol", {}).get(
                   "protocol_version") == 3
               and item["payload"].get("protocol", {}).get(
                   "intervention_config") == {
                       "projection_samples": 64,
                       "line_search_scales": [0.025, 0.05, 0.1, 0.2]}]
'''
if resume_needle not in phase2:
    raise RuntimeError("VGG notebook template lacks the resume protocol filter")
phase2 = phase2.replace(resume_needle, resume_replacement)
base["cells"][8]["source"] = phase2.splitlines(keepends=True)

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
        f"notebooks/kaggle_vgg16_seed{seed}_end_to_end_t4x2.ipynb")
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(destination)
