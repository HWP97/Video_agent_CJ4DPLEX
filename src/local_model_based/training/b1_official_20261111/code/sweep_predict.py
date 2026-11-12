# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
# 실행: 평가기가 호출; futureGT인수 없음
"""동결 모델·train memory와 관측 graph만 사용해 모든 세부 조건을 추론한다."""
from functools import lru_cache
import json
import torch
from sweep_data import HERE,context,probabilities
from sweep_models import Head,Denoiser,denoise
from compact_graph import histories,observed_bank
from log_objective import restored_scores

PLAN=json.loads((HERE/'runplan.json').read_text())
MASK_NAMES=tuple(f'{pattern}__{anchor}__{step}' for pattern in PLAN['masked_patterns'] for anchor,step in PLAN['masked_anchor_steps'])
METHODS=('impartail_ode',*PLAN['head_configs'],*PLAN['derived_configs'],*MASK_NAMES)


@lru_cache(maxsize=4)
def load(mode: str,stage: str):
    models={};supports={}
    for name in PLAN['head_configs']+PLAN['masked_patterns']:
        saved=torch.load(HERE/f'assets/{stage}_{mode}_{name}.pt',weights_only=False)
        model=Denoiser() if name in PLAN['masked_patterns'] else Head(name)
        model.load_state_dict(saved['state'],strict=True);model.eval();models[name]=model;supports[name]=saved['support']
    memory=torch.load(HERE/f'assets/{stage}_{mode}_memory.pt',weights_only=False)
    fit_saved=torch.load(HERE/f'assets/development_{mode}_linear_none.pt',weights_only=False)
    fit=Head('linear_none');fit.load_state_dict(fit_saved['state']);fit.eval()
    return models,supports,memory,fit


def predict(graph,future,official,loaded):
    models,supports,memory,fit=loaded
    h,coverage=histories(graph);c=len(h);n=len(future)
    repeated=h.repeat(n,1,1)
    delta=torch.tensor([(f-graph['actual_relation_cutoff'])/30 for f in future]).repeat_interleave(c)
    base=probabilities(official)
    raw_base=torch.cat((official['attention_distribution'].softmax(-1),official['spatial_distribution'],official['contacting_distribution']),-1)
    assert len(base)==n*c
    ctx=context(h)
    similarities=torch.nn.functional.normalize(ctx,dim=-1)@torch.nn.functional.normalize(memory['context'],dim=-1).T
    neighbors=similarities.topk(min(8,len(memory['context'])),dim=1).indices
    remembered=memory['profile'][neighbors].mean(1)
    bins=torch.bucketize(torch.arange(n),torch.linspace(0,n,4).round().long()[1:-1],right=True)
    recall=remembered[:,:,bins].permute(2,0,1).reshape(n*c,26)
    with torch.inference_mode():
        scores={}
        log_multiplier=None
        for name,model in models.items():
            if name in PLAN['masked_patterns']:
                continue
            logits=model(repeated,delta,base)
            scores[name]=restored_scores(logits,base,raw_base) if name=='log_rank_residual' else logits.sigmoid()
            if name=='log_rank_residual':
                log_multiplier=(logits-base.log()).exp()
        scores['fit_only_linear']=fit(repeated,delta,base).sigmoid()
        scores['existing_refit_linear']=scores['linear_none']
        for anchor in ('anchor','direct','linear'):
            scores[f'memory_{anchor}']=.5*scores[f'{anchor}_none']+.5*recall
        scores['baseline_head_half']=.5*base+.5*scores['family_head']
        scores['head_control']=scores['macro_direct']
        guarded=scores['macro_residual'].clone()
        for first,last in ((0,3),(3,9),(9,26)):
            winner=base[:,first:last].argmax(-1)+first
            rare=supports['macro_residual'][winner]<128
            guarded[rare,first:last]=base[rare,first:last]
        scores['rare_guard']=guarded
        scores['order_baseline']=base;scores['diversity_baseline']=base;scores['diversity_linear']=scores['linear_none']
        for pattern in PLAN['masked_patterns']:
            for steps in (1,3):
                profile=denoise(ctx,models[pattern],steps)
                decoded=profile[:,:,bins].permute(2,0,1).reshape(n*c,26)
                for anchor,requested_steps in PLAN['masked_anchor_steps']:
                    if steps!=requested_steps:
                        continue
                    selected={'baseline':base,'linear':scores['fit_only_linear'],'memory':scores['memory_linear'],'direct':decoded}[anchor]
                    scores[f'{pattern}__{anchor}__{steps}']=decoded if anchor=='direct' else .5*decoded+.5*selected
    bank=observed_bank(graph,future)
    outputs={'impartail_ode':dict(bank,**{k:official[k].clone() for k in ('attention_distribution','spatial_distribution','contacting_distribution')})}
    for name,p in scores.items():
        assert torch.isfinite(p).all()
        value={k:v.clone() for k,v in bank.items()}
        value.update(attention_distribution=p[:,:3].clamp_min(1e-7).log(),spatial_distribution=p[:,3:9],contacting_distribution=p[:,9:])
        outputs[name]=value
        if name=='log_rank_residual':
            outputs[name]['post_softmax_scores']=p
            outputs[name]['post_softmax_multiplier']=log_multiplier
    outputs['order_baseline']={k:v.clone() for k,v in outputs['impartail_ode'].items()}
    outputs['diversity_baseline']={k:v.clone() for k,v in outputs['impartail_ode'].items()}
    assert set(outputs)==set(METHODS)
    return outputs
