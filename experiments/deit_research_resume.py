"""Import hash-bound v2 proposals and full continuation progress from mounted archives."""
from pathlib import Path
import shutil
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.deit_resume import unique,most_advanced
from experiments.shared_protocol import sha256_file


def import_research(roots,output,fork,raw_hash):
    output=Path(output);items=[]
    for i,root in enumerate(dict.fromkeys(map(str,[output,*roots]))):
        if not Path(root).exists():continue
        def compatible(p):
            if p.get('kind')=='deit_v2_generator_reference':
                ident=p.get('run_identity',{})
                return ident.get('protocol')==fork['protocol'] and ident.get('raw_reference_hash')==raw_hash
            return (p.get('protocol')==fork['protocol'] and
                    p.get('run_identity',{}).get('method','').startswith('cp_v2_'))
        found,_=discover_checkpoints(root,output/'_research_resume_cache'/str(i),
            kind={'deit_v2_generator_reference','deit_projection_aware_latest'},payload_filter=compatible,
            payload_transform=lambda p:{k:v for k,v in p.items() if k not in ('model','optimizer','scheduler','rng',
                'train_loader_generator_state','e_controller','previous_direction','candidates')})
        items.extend(found)
    refs=[p for p in items if p['payload']['kind']=='deit_v2_generator_reference']
    latest=[p for p in items if p['payload']['kind']=='deit_projection_aware_latest']
    reference=unique(refs,'v2 generator reference')
    if latest and not reference:raise FileNotFoundError('v2 arm resume needs its original generator_reference.pt')
    if not reference:return
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()==target.resolve():return
        temporary=target.with_suffix('.importing')
        try:shutil.copy2(source,temporary);temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    target=output/'generator_reference.pt';copy(reference['path'],target);h=sha256_file(target)
    methods={p['payload']['run_identity']['method'] for p in latest}
    for method in methods:
        candidates=[p for p in latest if p['payload']['run_identity']['method']==method]
        if any(p['payload']['run_identity']['generator_reference_hash']!=h for p in candidates):
            raise ValueError('v2 progress is bound to another generator reference')
        selected=most_advanced(candidates,'completed_epochs',method)
        copy(selected['path'],output/method/'checkpoint_latest.pt')
