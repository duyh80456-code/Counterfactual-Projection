"""Import original full Bypass checkpoint bytes and retain phase/optimizer/RNG."""
from pathlib import Path
import shutil
import torch
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.deit_resume import FULL_STATE, most_advanced
from experiments.shared_protocol import sha256_file
from experiments.run_deit_bypass import KIND, identity_for, validate_progress, finish


def prepare_bypass_resume(roots, output, fork_path, fork, config, horizon):
    output=Path(output);identity=identity_for(sha256_file(fork_path),config,horizon);states=[]
    for index,root in enumerate(dict.fromkeys(map(str,[output,*roots]))):
        if not Path(root).exists():continue
        found,_=discover_checkpoints(root,output/'_bypass_resume_cache'/str(index),kind={KIND},
            payload_filter=lambda p:p.get('protocol')==fork['protocol'] and FULL_STATE.issubset(p),
            payload_transform=lambda p:{k:v for k,v in p.items() if k not in FULL_STATE and k!='best_original'})
        states.extend(found)
    if any(item['payload'].get('run_identity')!=identity for item in states):
        raise ValueError('Bypass resume fork/config mismatch; attach one matching trajectory')
    selected=most_advanced(states,'completed_epochs','Bypass latest')
    if not selected:return {'action':'start','completed_epochs':0}
    state=torch.load(selected['path'],map_location='cpu',weights_only=False)
    validate_progress(state,identity,fork)
    target=output/'arms/deit_bypass/checkpoint_latest.pt';target.parent.mkdir(parents=True,exist_ok=True)
    if selected['path'].resolve()!=target.resolve():
        temporary=target.with_suffix('.importing')
        try:shutil.copy2(selected['path'],temporary);temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    completed=state['completed_epochs']==horizon
    if completed:finish(state,target.parent)
    return {'action':'completed' if completed else 'resume','completed_epochs':state['completed_epochs'],'phase':state['phase']}
