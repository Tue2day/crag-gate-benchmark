import asyncio
from collections import Counter
from dataclasses import replace
import json

import httpx
import numpy as np
import pytest

from jev_gate.baselines import baselines
from jev_gate.budget import Budget,BudgetExceeded,estimate_attempt
from jev_gate.evaluate_top1 import evaluate
from jev_gate.io import read_json,read_jsonl,digest,write_jsonl
from jev_gate.jev import request_body,cache_key,all_async,call_api,RateLimiter
from jev_gate.gpt import request_body as gpt_body
from jev_gate.metrics import paired_ap_interval
from jev_gate.tasks import label
from jev_gate.top1 import prepare,rerank,partition,select_development,challenge_variant


@pytest.fixture
def topcfg(cfg):
    return replace(cfg,task='top1_exact',seed=20261025,development_size=20,
        development_pool_size=80,development_negative_target=8,development_min_negatives=4,
        validation_size=20,test_size=40,challenge_pool_size=80,challenge_size=8,
        challenge_s=3,challenge_c=3,challenge_i=2,target_risk=0.02,secondary_risk=0.05,
        min_accepted=3,concurrency=2,requests_per_second=100000,bootstrap_samples=30)


def group(labels=('S','E','C','I','S')):
    return {'query_id':'example','query':'camera', 'source_split':'train','split':'test',
            'candidates':[{'id':f'p{i}','title':f'camera {l} {i}','label':l,'brand':None,
                           'color':None,'bullet_points':'','ce_score':float(5-i)} for i,l in enumerate(labels)]}


def test_task_truth_and_same_single_candidate_input(topcfg):
    g=group()
    assert label(g,'top1_exact')==0 and label(g,'top5_any_exact')==1
    a=request_body(g,topcfg);b=gpt_body(g,topcfg)
    assert a['state']==json.loads(b['messages'][1]['content'])
    assert 'candidates' not in a['state']
    assert a['state']['target_candidate']['id']=='p0'
    assert len(a['state']['rerank_features'])==11
    assert '"label"' not in json.dumps(a) and '"y"' not in json.dumps(a)
    old=request_body(g,replace(topcfg,task='top5_any_exact'))
    assert cache_key(a)!=cache_key(old)


def test_inverse_weights_restore_screening_prevalence(topcfg):
    pool=[{'query_id':str(i),'y':int(i<80)} for i in range(100)]
    selected=select_development(pool,topcfg)
    assert len(selected)==20 and sum(g['y']==0 for g in selected)==8
    weighted=np.average([g['y']==0 for g in selected],weights=[g['sampling_weight'] for g in selected])
    assert weighted==pytest.approx(0.2)
    assert sum(g['sampling_weight'] for g in selected)==pytest.approx(20)
    assert all(0<g['sampling_probability']<=1 for g in selected)
    with pytest.raises(ValueError,match='natural negatives'):
        select_development([{'y':1}]*80+[{'y':0}],topcfg)


def test_exclusion_and_disjoint_partitions(topcfg):
    groups=[]
    for i in range(300):
        groups.append({**group(),'query_id':str(i),'query':f'query {i}',
                       'source_split':'train' if i<150 else 'test'})
    rows=partition(groups,topcfg,{'query 0','query 160'})
    assert not {'0','160'} & {r['query_id'] for r in rows}
    assert len({r['query'] for r in rows})==len(rows)
    assert partition(groups,topcfg,{'query 0','query 160'})==rows


def test_challenge_filters_without_promoting_a_low_rank_item(topcfg):
    counts=Counter()
    g=group(('E','S','C','I','I'))
    g['candidates'] += [{'id':f'x{i}','title':'other','label':'I','ce_score':float(-i-1)} for i in range(5)]
    r=challenge_variant(g,counts,topcfg)
    assert r['challenge_type']=='S' and all(c['label']!='E' for c in r['candidates'])
    assert r['candidates'][0]['id']=='p1'
    counts['S']=topcfg.challenge_s
    r=challenge_variant(g,counts,topcfg)
    assert r['challenge_type']=='C' and r['candidates'][0]['id']=='p2'
    counts['C']=topcfg.challenge_c
    r=challenge_variant(g,counts,topcfg)
    assert r['challenge_type']=='I' and r['candidates'][0]['id']=='p3'


def test_budget_shared_pending_unknown_and_resume(topcfg,tmp_path):
    cfg=replace(topcfg,api_budget_usd=0.10,budget_safety_usd=0.01)
    async def scenario():
        budget=Budget(tmp_path/'budget.json',cfg)
        first=await budget.reserve(0.06)
        with pytest.raises(BudgetExceeded): await budget.reserve(0.04)
        await budget.settle(first,0.01)
        second=await budget.reserve(0.05)
        await budget.settle(second,None)
        await budget.reserve(0.02)
    asyncio.run(scenario())
    resumed=Budget(tmp_path/'budget.json',cfg)
    assert resumed.state['reported_usd']==pytest.approx(0.01)
    assert resumed.state['unknown_reserved_usd']==pytest.approx(0.07)
    assert not resumed.state['pending']


def test_budget_blocks_before_network_and_counts_reported_cost(topcfg,tmp_path):
    cfg=replace(topcfg,api_budget_usd=0.0001,budget_safety_usd=0.00001)
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(200,json={})
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(BudgetExceeded):
                await call_api(client,request_body(group(),cfg),cfg,RateLimiter(1e5),
                               budget=Budget(tmp_path/'budget.json',cfg))
    asyncio.run(scenario());assert not calls
    assert estimate_attempt(gpt_body(group(),cfg),'gpt',cfg)>0


def test_paired_ap_orientation_and_reproducibility():
    y=[1,1,0,0];a=[0.9,0.8,0.2,0.1];b=[0.1,0.2,0.8,0.9]
    x=paired_ap_interval(y,a,b,30,42)
    assert x['delta']>0 and x==paired_ap_interval(y,a,b,30,42)
    assert paired_ap_interval([1,1],[0,1],[1,0],30) is None


def test_full_top1_mock_and_frozen_workpoints(topcfg):
    topcfg.validate()
    prepare(topcfg,True);rerank(topcfg,True);baselines(topcfg,True)
    rows=read_jsonl(topcfg.root(True)/'ranked.jsonl')
    assert all(g['y']==int(g['candidates'][0]['label']=='E') for g in rows)
    assert sum(g['split']=='development' for g in rows)==20
    assert sum(g['split']=='test' for g in rows)==40
    assert len({g['query_id'] for g in rows})==len(rows)
    asyncio.run(all_async(topcfg,True))
    predictions=read_jsonl(topcfg.root(True)/'jev.jsonl')
    assert not any(r['split']=='development' for r in predictions)
    assert all(r['task']=='top1_exact' for r in predictions)
    result=evaluate(topcfg,True)
    assert set(result['workpoints'])=={'0.02','0.05'}
    assert result['completeness']['natural_test_n']==40
    assert set(result['metrics'])=={'all_pass','ce_threshold','logistic','jev','gpt'}
    assert result['budget'] is None
    frozen=read_json(topcfg.root(True)/'operating_points.json')
    evaluate(topcfg,True)
    assert read_json(topcfg.root(True)/'operating_points.json')==frozen
    topcfg.target_risk=0.01
    with pytest.raises(ValueError,match='Frozen'): evaluate(topcfg,True)


def test_degenerate_validation_stops_before_paid_stage(topcfg):
    prepare(topcfg,True)
    rows=read_jsonl(topcfg.root(True)/'prepared.jsonl')
    for g in rows:
        if g['split']=='validation':
            for c in g['candidates']: c['label']='E'
    write_jsonl(topcfg.root(True)/'prepared.jsonl',rows)
    with pytest.raises(ValueError,match='already meets'):
        rerank(topcfg,True)
    assert not read_json(topcfg.root(True)/'eligibility.json')['ready']
    assert not (topcfg.root(True)/'budget.json').exists()


def test_duplicate_api_process_lock_is_not_stolen(topcfg):
    root=topcfg.root(True);root.mkdir(parents=True)
    lock=root/'api-all.lock';lock.write_text('prior-process',encoding='utf-8')
    with pytest.raises(ValueError,match='already running'):
        asyncio.run(all_async(topcfg,True))
    assert lock.read_text(encoding='utf-8')=='prior-process'
