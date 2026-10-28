"""Dataset-only staging and lightweight API gates for the four Kaggle notebooks."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import torch
from experiments.io_utils import atomic_json_save
from diagnostics.vna_align import _global_state


def dependency_report():
    import sysconfig
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    conflicts=[];exceptions=[]
    # Private setuptools vendored dist-info is not an installed environment dependency.
    for package in metadata.distributions(path=[sysconfig.get_paths()['purelib']]):
        for value in package.requires or []:
            requirement=Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            try:version=metadata.version(requirement.name)
            except metadata.PackageNotFoundError:version=None
            if version is not None and (not requirement.specifier or requirement.specifier.contains(version,prereleases=True)):continue
            row=dict(package=package.metadata['Name'],requires=value,installed=version)
            if (canonicalize_name(row['package'])=='torch' and canonicalize_name(requirement.name)=='typing-extensions'
                    and package.version.split('+')[0]=='2.2.2' and version=='4.4.0'):
                exceptions.append(row)
            else:conflicts.append(row)
    if conflicts:raise RuntimeError('Unapproved dependency conflicts: '+json.dumps(conflicts))
    return dict(conflicts=conflicts,tested_compatibility_exceptions=exceptions)


def replay_gate():
    from types import SimpleNamespace
    from torch import nn
    from torch.utils.data import TensorDataset
    from avalanche.benchmarks.utils import make_classification_dataset
    from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
    from experiments.standard_online import with_origin,replay_mask
    from methods.cl_replay import ReplayLoss
    from methods.cl_lpr import official_module
    with _global_state():
        torch.manual_seed(5)
        incoming=make_classification_dataset(TensorDataset(torch.ones(4,3),torch.zeros(4,dtype=torch.long)),task_labels=0)
        memory=make_classification_dataset(TensorDataset(torch.full((4,3),9.),torch.zeros(4,dtype=torch.long)),task_labels=0)
        batch=next(iter(ReplayDataLoader(with_origin(incoming,False),with_origin(memory,True),batch_size=2,batch_size_mem=2,shuffle=False,num_workers=0)))
        mask=replay_mask(SimpleNamespace(mbatch=batch,mb_y=batch[1]))
        assert torch.equal(mask,batch[0][:,0].eq(9)) and mask.sum()==2
        permutation=torch.tensor([3,0,2,1]);permuted=[v[permutation] for v in batch]
        assert torch.equal(replay_mask(SimpleNamespace(mbatch=permuted,mb_y=permuted[1])),permuted[0][:,0].eq(9))
        native_batch=next(iter(ReplayDataLoader(incoming,memory,batch_size=2,batch_size_mem=2,shuffle=False,num_workers=0)))
        assert native_batch[0][:2,0].eq(1).all() and native_batch[0][2:,0].eq(9).all()
        model=nn.Linear(3,2).double();x=batch[0].double();y=batch[1];z=model(x)
        losses=ReplayLoss(y,mask,kind='er')
        gradients=[torch.cat([g.flatten() for g in torch.autograd.grad(c(z),tuple(model.parameters()),retain_graph=True)]) for c in losses.components]
        total=torch.cat([g.flatten() for g in torch.autograd.grad(losses.total(z),tuple(model.parameters()))])
        torch.testing.assert_close(gradients[0]+gradients[1],total,rtol=1e-10,atol=1e-10)
        for c,d in zip(losses.components,ReplayLoss(y[permutation],mask[permutation],kind='er').components):
            torch.testing.assert_close(c(z.detach()),d(z.detach()[permutation]))
        cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text())['lpr']
        plugin=official_module().LPRPlugin({k:SimpleNamespace(**v) for k,v in cfg.items()},SimpleNamespace(buffer=memory))
        model=nn.Linear(3,2)
        strategy=SimpleNamespace(model=model,device='cpu',is_training=True,train_epochs=3,
            clock=SimpleNamespace(train_iterations=0,train_exp_counter=1,train_exp_iterations=0))
        z=model(batch[0]);loss=torch.nn.functional.cross_entropy(z,y)
        plugin.before_backward(strategy);loss.backward();before=model.weight.grad.clone();plugin.after_backward(strategy)
        assert plugin._preconditioner_invs and not torch.equal(before,model.weight.grad)
    return dict(replay_loader='Avalanche0.4 real minibatch; data first, memory second',explicit_origin=True,
        permutation_safe=True,gradient_decomposition=True,lpr_callbacks=True,lpr_inverse_layers=list(plugin._preconditioner_invs))


def geometry_gate():
    from torch import nn
    from methods.cl_h2 import GeometryCorrection
    from methods.cl_replay import CorrectionConfig,ReplayLoss
    from methods.cl_state import State,vector
    class Toy(nn.Module):
        def __init__(self):super().__init__();self.hidden=nn.Linear(3,3);self.head=nn.Linear(3,2)
        def standard_features(self,x):return self.hidden(x)
        def forward(self,x):return self.head(self.standard_features(x))
    with _global_state():
        torch.manual_seed(12);model=Toy().double();opt=torch.optim.SGD(model.parameters(),lr=.001)
        x=torch.randn(4,3,dtype=torch.double);mask=torch.tensor([False,False,True,True]);losses=ReplayLoss(torch.zeros(4,dtype=torch.long),mask,kind='er')
        initial=State(model,opt);results=[];events=[]
        cfg=CorrectionConfig(kernel='gn',damping_relative=1.,trace_probes=1,cg_tolerance=1e-6,cg_iterations=100,max_trials=4)
        for weight in (0.,10.):
            initial.restore();engine=GeometryCorrection(cfg)
            events.append(engine.step(model,opt,x,losses,[],.001,replay_inputs=x[mask],geometry_weight=weight));results.append(vector(model).clone())
        assert events[0]['accepted'] and events[1]['accepted']
        difference=float((results[0]-results[1]).norm());assert difference>1e-10
        initial.restore()
    return dict(actual_projection_changed=True,physical_update_difference=difference,weights=[0.,10.],fixture='CPU toy, not a benchmark')


def h2_environment(require_cuda=False):
    import torchvision,avalanche
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.benchmarks.scenarios import OnlineCLScenario
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.training.storage_policy import ClassBalancedBuffer
    from avalanche.training.supervised import Naive
    from methods.cl_lpr import official_module,PIN,SOURCE_SHA256
    from methods.cl_h2 import GeometryCorrection
    if sys.version_info[:2]!=(3,10):raise RuntimeError('The verified H2 environment requires Python3.10, without Protocol shims')
    expected={'torch':'2.2.2','torchvision':'0.17.2','avalanche-lib':'0.4.0','typing-extensions':'4.4.0',
              'torchmetrics':'0.11.4','pytorchcv':'0.0.67','wandb':'0.15.12','protobuf':'4.25.8',
              'scipy':'1.15.3','pandas':'2.2.3','matplotlib':'3.10.8','scikit-learn':'1.6.1','grpcio':'1.62.3',
              'hydra-core':'1.3.2','jsonlines':'4.0.0','kornia':'0.7.0','dill':'0.3.8','gdown':'5.2.0'}
    versions={name:metadata.version(name) for name in expected}
    for name,version in expected.items():
        if versions[name].split('+')[0]!=version:raise RuntimeError(f'{name}: expected {version}, found {versions[name]}')
    module=official_module()
    assert all(callable(api) for api in (SplitCIFAR100,OnlineCLScenario,ReplayPlugin,ClassBalancedBuffer,Naive,module.LPRPlugin,GeometryCorrection))
    with _global_state():
        model=torch.nn.Linear(3,2);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        strategy=Naive(model,optimizer,train_mb_size=2,train_epochs=3,evaluator=None,eval_every=-1,device='cpu')
        assert strategy.model is model
    if require_cuda and (not torch.cuda.is_available() or torch.cuda.device_count()!=2):raise RuntimeError('Isolated Python must see CUDA and two T4 GPUs')
    if require_cuda:
        value=torch.ones(2,2,device='cuda:0')@torch.ones(2,2,device='cuda:0');torch.cuda.synchronize()
        assert value.eq(2).all()
    return dict(python=sys.version,platform=platform.platform(),versions=versions,cuda_available=torch.cuda.is_available(),
        installed_packages={d.metadata['Name']:d.version for d in metadata.distributions(path=[__import__('sysconfig').get_paths()['purelib']])},
        cuda_version=torch.version.cuda,devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        lpr_commit=PIN,lpr_source_sha256=SOURCE_SHA256,**dependency_report(),replay=replay_gate(),geometry=geometry_gate())


def download_imagenet_classes(stage,needed):
    import shutil,tarfile,zipfile
    import gdown
    target=Path(stage)/'cbp_imagenet32';target.mkdir(parents=True,exist_ok=True)
    missing={name for name in needed if not (target/name).is_file()}
    if not missing:return str(target)
    archive=Path(stage)/'cbp_imagenet32.download'
    if not archive.is_file():
        temporary=archive.with_suffix('.part')
        print('Downloading original CBP ImageNet32 archive to Kaggle disk',flush=True)
        result=gdown.download(id='1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z',output=str(temporary),quiet=False,resume=True)
        if result is None:raise RuntimeError('CBP dataset download failed; attach the original dataset instead')
        temporary.replace(archive)
    def save(name,source):
        name=Path(name).name
        if name not in missing:return
        temporary=target/(name+'.tmp')
        with source,temporary.open('wb') as dest:shutil.copyfileobj(source,dest)
        temporary.replace(target/name);missing.remove(name)
    print('Extracting only required CBP class files',flush=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not member.is_dir() and Path(member.filename).name in missing:save(member.filename,bundle.open(member))
    else:
        with tarfile.open(archive,'r:*') as bundle:
            for member in bundle:
                if member.isfile() and Path(member.name).name in missing:save(member.name,bundle.extractfile(member))
    if missing:raise FileNotFoundError('CBP archive missing required classes: '+', '.join(sorted(missing)))
    archive.unlink()
    return str(target)


def prepare_data(branch,search,stage,config):
    from torchvision.datasets import CIFAR100,MNIST
    search,stage=Path(search),Path(stage);stage.mkdir(parents=True,exist_ok=True)
    candidates=[]
    needed={'train','test','meta'} if branch in ('H1','H2') else {'train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'}
    if branch=='H3':
        from experiments.standard_imagenet import effective_params,required_classes
        from experiments.runners import lop
        lop.upstream_imports();params,_=effective_params(config)
        classes=required_classes(0,params['num_tasks']);needed={f'{int(i)}.npy' for i in classes}
    visited=set()
    print(f'Scanning dataset input recursively: {search}',flush=True)
    def scan_error(error):
        raise OSError(f'Cannot scan dataset input {error.filename}; refusing silent download fallback') from error
    for directory,dirs,files in os.walk(search,followlinks=True,onerror=scan_error):
        resolved=Path(directory).resolve()
        if resolved in visited:dirs[:]=[];continue
        visited.add(resolved)
        dirs[:]=[name for name in dirs if name!='.git']
        folder=Path(directory)
        if all((folder/n).is_file() or (branch=='H4' and (folder/(n+'.gz')).is_file()) for n in needed):
            candidates.append(folder)
            print(f'Found dataset files: {folder}',flush=True)
    candidates=sorted(set(p.resolve() for p in candidates))
    if len(candidates)>1:raise ValueError(f'Set DATA_DIR to a single dataset root: {candidates}')
    print(f'Dataset scan complete: {len(visited)} directories, {len(candidates)} compatible roots',flush=True)
    if branch=='H3':
        if not candidates:return download_imagenet_classes(stage,needed)
        return str(candidates[0])
    if candidates:
        if branch in ('H1','H2'):
            target=stage/'cifar-100-python'
            if not target.exists():target.symlink_to(candidates[0],target_is_directory=True)
            elif target.resolve()!=candidates[0]:raise ValueError('staged CIFAR source mismatch')
        else:
            import gzip,shutil
            raw=stage/'MNIST/raw';raw.mkdir(parents=True,exist_ok=True)
            for name in needed:
                target=raw/name
                if target.exists():continue
                if (candidates[0]/name).exists():target.symlink_to(candidates[0]/name)
                else:
                    temporary=target.with_suffix('.tmp')
                    with gzip.open(candidates[0]/(name+'.gz'),'rb') as source,temporary.open('wb') as dest:shutil.copyfileobj(source,dest)
                    temporary.replace(target)
    dataset=CIFAR100 if branch in ('H1','H2') else MNIST
    # Official torchvision integrity checks and mirrors; never substitute synthetic data.
    train=dataset(str(stage),train=True,download=not candidates);test=dataset(str(stage),train=False,download=not candidates)
    expected=(50000,10000,100) if branch in ('H1','H2') else (60000,10000,10)
    if (len(train),len(test))!=expected[:2] or set(map(int,train.targets))!=set(range(expected[2])) or set(map(int,test.targets))!=set(range(expected[2])):
        raise ValueError('official dataset sizes/label space mismatch')
    return str(stage)


def run(args):
    from experiments.vna_provenance import code_commit
    from diagnostics.kernel_ablation import flags
    config=json.loads(Path(args.config).read_text())
    with _global_state():
        torch.manual_seed(0)
        if args.device!='cpu' and (not torch.cuda.is_available() or torch.cuda.device_count()!=2):raise RuntimeError('Select GPU T4 x2')
        environment=h2_environment(args.device!='cpu') if args.branch=='H2' else None
        if environment:print('H2 ENVIRONMENT VERIFIED',json.dumps(environment),flush=True)
        data=prepare_data(args.branch,args.data_dir or '/kaggle/input',args.stage,config)
        result=dict(branch=args.branch,source_commit=code_commit(),data_dir=data,config=config,runtime=flags(),fixture_tests='separate CPU/API fixtures, not benchmark results')
        if args.branch=='H2':
            result['environment']=environment
            from experiments.standard_online import native_setup
            scenario,model=native_setup(config,data,0)
            order=list(map(int,scenario.classes_order_original_ids));assert len(order)==100 and set(order)==set(range(100))
            result['class_order']=order;result['experience_sizes']=[len(e.dataset) for e in scenario.train_stream]
            batch=scenario.train_stream[0].dataset[0][0].unsqueeze(0).repeat(2,1,1,1)
        elif args.branch=='H3':
            from experiments.standard_imagenet import data_identity,required_classes,effective_params,ImageNetwork
            from lop.nets.conv_net import ConvNet
            params,overrides=effective_params(config);classes=required_classes(0,params['num_tasks'])
            result.update(required_files=data_identity(data,classes),class_pairs=classes.reshape(-1,2).tolist(),upstream_overrides=overrides)
            model=ImageNetwork(ConvNet());batch=torch.from_numpy(__import__('numpy').load(Path(data)/f'{int(classes[0])}.npy')[:2]).float()
        elif args.branch=='H4':
            from methods.cl_h4 import DomainNetwork,DomainMNIST
            from torchvision.datasets import MNIST
            from torchvision.transforms import ToTensor
            model=DomainNetwork(config['normalization']);dataset=MNIST(data,train=True,download=False,transform=ToTensor())
            result['domains']=[DomainMNIST(dataset,d,'rotated',angle_step=config['angle_step']).identity() for d in range(config['domains'])]
            assert len(result['domains'])==20
            batch=torch.stack([dataset[i][0] for i in range(2)])
        else:
            from experiments.runners import pycil
            pycil.upstream_imports()
            from utils.inc_net import IncrementalNet
            from torchvision.datasets import CIFAR100
            from torchvision.transforms import ToTensor
            params=json.loads((pycil.UPSTREAM/'exps/icarl.json').read_text());params['device']=[torch.device('cpu')]
            model=IncrementalNet(params,False);model.update_fc(5)
            order=__import__('numpy').random.RandomState(0).permutation(100).tolist();result['class_order']=order
            assert sorted(sum([order[i:i+5] for i in range(0,100,5)],[]))==list(range(100))
            dataset=CIFAR100(data,train=True,download=False,transform=ToTensor());batch=torch.stack([dataset[i][0] for i in range(2)])
        model=model.to(args.device);model.eval()
        with torch.no_grad():output=model(batch.to(args.device));output=output['logits'] if isinstance(output,dict) else output
        if not bool(torch.isfinite(output).all()):raise RuntimeError('nonfinite native backbone preflight output')
        result.update(status='PASSED',native_forward_shape=list(output.shape),trained_benchmark=False)
    atomic_json_save(result,Path(args.out));print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',choices=('H1','H2','H3','H4'),required=True)
    p.add_argument('--config',required=True);p.add_argument('--data-dir');p.add_argument('--stage',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        atomic_json_save(dict(status='FAILED_PREFLIGHT',error=repr(exc),trained_benchmark=False),Path(args.out));raise
