"""Stream-driven ETT lifecycle: no extra training steps and no pre-training rollback."""
from collections import deque
from copy import deepcopy
from dataclasses import replace as site_copy
import math
import time
import torch
from ett.core.expand import candidates,expand
from ett.core.stats import gram,regularize,measurement
from ett.core.select import select
from ett.core.compensate import compensate,columns,data_residual
from ett.core.rebuild import rebuild,slot_order


def measured_loss(model,batch,loss_fn,training=True):
    with measurement(model,training),torch.no_grad():return float(loss_fn(model(batch[0]),batch[1]))


class ETT:
    def __init__(self,model,optimizer,sites,*,rho=.1,h=10,T=10000,maturity=None,tau=.01,
                 window=4,seed=0,init='gradmax_pool',drop=False,cycle_at_task_start=False):
        if rho not in (0.,.05,.1) or h<0 or T<1 or tau not in (.01,.05):raise ValueError('unregistered ETT controls')
        self.model=model;self.optimizer=optimizer;self.sites=sites
        self.rho=rho;self.h=h;self.T=T;self.maturity=T if maturity is None else maturity
        self.tau=tau;self.window=window;self.init=init;self.drop=drop
        self.cycle_at_task_start=cycle_at_task_start
        self.generator=torch.Generator().manual_seed(seed)
        self.ages={s.incoming:torch.zeros(s.width,dtype=torch.long) for s in sites}
        self.churn={s.incoming:torch.zeros(s.width,dtype=torch.long) for s in sites}
        self.steps=0;self.last_event=0;self.cycle=0;self.active=None;self.recent=deque(maxlen=window);self.events=[]

    def before_batch(self,batch,loss_fn,task_start=False):
        if self.rho==0:return
        if self.active is not None and self.active['trained']>=self.h:self.contract(batch,loss_fn)
        due=self.steps-self.last_event>=self.T or task_start and self.cycle_at_task_start
        if self.active is None and due:
            self.begin(batch,loss_fn)
            if self.h==0:self.contract(batch,loss_fn)

    def begin(self,batch,loss_fn):
        if self.active is not None:raise ValueError('cycle already expanded')
        start=time.perf_counter();original={};energy=0.;pool=0
        for site in self.sites:
            k=max(1,round(self.rho*site.width)) if self.rho else 0
            original[site.incoming]=site.width
            if k:
                proposal=candidates(site,*batch,loss_fn,k,self.generator,self.init)
                energy+=proposal['gradient_energy'];pool+=proposal['pool_size']
                expand(site,self.optimizer,proposal)
                self.ages[site.incoming]=torch.cat((self.ages[site.incoming],torch.zeros(k,dtype=torch.long)))
        self.active=dict(original=original,trained=0,gradient_energy=energy,pool_size=pool,
                         expand_seconds=time.perf_counter()-start,train_seconds=0.)
        self.last_event=self.steps;self.cycle+=1
        if self.h==0:self.recent.append(tuple(t.detach().clone() for t in batch))

    def after_step(self,batch,seconds=0.):
        self.steps+=1
        for age in self.ages.values():age.add_(1)
        if self.active is not None:
            self.active['trained']+=1;self.active['train_seconds']+=seconds
            self.recent.append(tuple(t.detach().clone() for t in batch))

    def contract(self,check_batch,loss_fn,acceptance=True):
        if self.active is None:return []
        if not self.recent:raise ValueError('no trained statistics window')
        records=[]
        for site in self.sites:
            original=self.active['original'][site.incoming];k=site.width-original
            if not k:continue
            row=dict(cycle=self.cycle,step=self.steps,site=site.incoming,k=k,h=self.active['trained'],
                expand_seconds=self.active['expand_seconds'],train_seconds=self.active['train_seconds'])
            start=time.perf_counter();G=gram(site,list(self.recent));W=site.weights()
            energies=G.diag()[:site.width*site.group_size].reshape(site.width,site.group_size).sum(1)
            threshold=1e-12*float(energies.sum())/site.width
            dead=[u for u in range(site.width) if (u>=original or self.ages[site.incoming][u]>=self.maturity)
                  and (float(energies[u])==0 or float(energies[u])<threshold)]
            dead.sort(key=lambda u:(u>=original,u))
            exact_dead=len(dead)>=k and all(float(energies[u])==0 for u in dead[:k])
            if exact_dead:
                # C1 precedes C3: exact-zero removals need no invertible Gram or LS solve.
                Greg=G;attempts=[]
                result=dict(dropped=dead[:k],g_score=0.,g_back=0.,g=0.,dead_units=len(dead),
                            virtual_kept=k-sum(u>=original for u in dead[:k]))
                if site.width<=20 and k<=3 and self.cycle%10==0:result.update(opt=0.,ratio=None)
            else:
                Greg,L,attempts=regularize(G,site.width*site.group_size)
                if Greg is None:
                    row.update(status='FAILED_CHOLESKY',cholesky_attempts=attempts)
                    self.events.append(row)
                    raise ArithmeticError('ETT Gram Cholesky failed; preserve expanded checkpoint')
                result=select(G,Greg,W,site.width,k,original,self.ages[site.incoming],self.maturity,
                              self.cycle,group_size=site.group_size)
            row.update(stats_seconds=time.perf_counter()-start,cholesky_attempts=attempts,
                       exact_dead_path=exact_dead)
            row.update(result,select_seconds=0. if exact_dead else time.perf_counter()-start)
            dropped=list(range(original,site.width)) if self.drop else result['dropped']
            def build(drop):
                keep=slot_order(original,[u for u in range(site.width) if u not in drop])
                cols=columns(keep,site.group_size)
                if site.intercept:cols.append(len(G)-1)
                removed_cols=columns(drop,site.group_size)
                if all(float(G.diag()[c])==0 for c in removed_cols):out=W[cols].clone()
                else:out=compensate(Greg,W,cols)
                return keep,cols,out
            start=time.perf_counter();kept,cols,out=build(dropped)
            row['compensate_seconds']=time.perf_counter()-start
            reference=measured_loss(self.model,check_batch,loss_fn)
            reference_eval=measured_loss(self.model,check_batch,loss_fn,False)
            start=time.perf_counter()
            trial_model,trial_opt=deepcopy((self.model,self.optimizer));trial_site=site_copy(site,model=trial_model)
            rebuild(trial_site,trial_opt,kept,out)
            row['trial_rebuild_seconds']=time.perf_counter()-start
            start=time.perf_counter()
            compact=measured_loss(trial_model,check_batch,loss_fn);compact_eval=measured_loss(trial_model,check_batch,loss_fn,False)
            delta=compact-reference
            accepted=math.isfinite(delta) and delta<=self.tau*reference
            row.update(delta=delta,delta_eval=compact_eval-reference_eval,expanded_loss=reference,
                       accepted=accepted if acceptance else None,fallback=bool(acceptance and not accepted),
                       check_seconds=time.perf_counter()-start)
            if acceptance and not accepted:
                dropped=list(range(original,site.width));kept,cols,out=build(dropped)
            start=time.perf_counter();rebuild(site,self.optimizer,kept,out)
            row['rebuild_seconds']=time.perf_counter()-start
            if row['fallback']:
                row['delta_fallback']=measured_loss(self.model,check_batch,loss_fn)-reference
                row['delta_fallback_eval']=measured_loss(self.model,check_batch,loss_fn,False)-reference_eval
            row.update(virtual_kept=sum(u>=original for u in kept),
                g_selected_metric=result['g'],g_actual_data=data_residual(G,W,cols,out),
                status='FALLBACK' if row['fallback'] else 'CONTRACTED')
            oldage=self.ages[site.incoming];self.ages[site.incoming]=oldage[kept].clone()
            for slot,u in enumerate(kept):
                if u>=original:self.ages[site.incoming][slot]=0;self.churn[site.incoming][slot]+=1
            records.append(row);self.events.append(row)
        self.active=None;self.recent.clear();return records

    def state_dict(self):
        return deepcopy(dict(steps=self.steps,last_event=self.last_event,cycle=self.cycle,active=self.active,
            ages=self.ages,churn=self.churn,recent=list(self.recent),events=self.events,rng=self.generator.get_state()))

    def load_state_dict(self,state):
        for key in ('steps','last_event','cycle','active','ages','churn','events'):setattr(self,key,deepcopy(state[key]))
        self.recent=deque(deepcopy(state['recent']),maxlen=self.window);self.generator.set_state(state['rng'])
