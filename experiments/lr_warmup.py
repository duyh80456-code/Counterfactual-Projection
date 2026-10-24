"""Opt-in task warmup around a native training update; schedulers retain their LR."""
from contextlib import contextmanager
import inspect
from experiments.vna_resume import log


def nonnegative_epochs(value):
    value=int(value)
    if value<0:raise ValueError('warmup-epochs must be nonnegative')
    return value


@contextmanager
def task_warmup(optimizer,epochs,steps_per_epoch,step,task):
    if not epochs:
        yield
        return
    total=epochs*steps_per_epoch
    scheduled=[group['lr'] for group in optimizer.param_groups]
    factor=min(step/total,1.)
    try:
        if step<total:
            for group,lr in zip(optimizer.param_groups,scheduled):group['lr']=lr*factor
        if step in (1,total,total+1):
            log(f'warmup task={task} step={step} S={total} lr_sched={scheduled} lr={[g["lr"] for g in optimizer.param_groups]}')
        yield
    finally:
        if step<total:
            for group,lr in zip(optimizer.param_groups,scheduled):group['lr']=lr


def optimizer_frame(loader=None):
    frame=inspect.currentframe().f_back
    while frame:
        values=frame.f_locals
        if ('epoch' in values and 'optimizer' in values and hasattr(values.get('self'),'_cur_task') and
                (loader is None or values.get('train_loader') is loader)):
            return frame
        frame=frame.f_back
    if loader is None:raise RuntimeError('warmup requires a native training frame with its optimizer')
    return None
