import asyncio
from dataclasses import replace
import json
import math
import numpy as np
import pytest
from jev_gate import fusion
from jev_gate.config import Config
from jev_gate.io import read_json,read_jsonl,write_json,write_jsonl
from jev_gate.jev import all_async

@pytest.fixture
def fusioncfg(cfg):
    return replace(cfg,task='top1_exact',protocol='fusion_v3',seed=2026102503,
        fit_size=80,select_size=40,certification_size=40,test_size=80,
        fit_min_class=4,min_accepted=8,certification_min_accepted=8,
        challenge_pool_size=80,challenge_size=8,challenge_s=3,challenge_c=3,challenge_i=2,
        target_risk=.05,secondary_risk=.02,concurrency=2,requests_per_second=100000,
        bootstrap_samples=20,api_budget_usd=3,budget_safety_usd=.1,preflight_max_estimate_usd=2.7)


def test_new_config_and_prior_paths():
    c=Config.load('config.top1-fusion-v3.toml')
    assert c.target_risk==.05 and c.secondary_risk==.02
    assert len(c.prior_prepared_paths)==2
    assert all(__import__('pathlib').Path(p).is_absolute() for p in c.prior_prepared_paths)


def test_partition_is_natural_deterministic_and_disjoint(fusioncfg):
    groups=[{'query_id':str(i),'query':f'Query {i}','source_split':'train' if i<200 else 'test',
             'candidates':[{'id':str(p)} for p in range(5)]} for i in range(600)]
    rows=fusion.partition(groups,fusioncfg,{'query 0','query 201'})
    assert {r['query_id'] for r in rows}.isdisjoint({'0','201'})
    assert len(rows)==80+40+40+80+80
    assert len({r['query'] for r in rows})==len(rows)
    assert fusion.partition(groups,fusioncfg,{'query 0','query 201'})==rows
    assert all(r['source_split']=='test' for r in rows if r['split'] in ('certification','test'))


def test_cp_upper_known_edges_and_family_correction():
    assert fusion.cp_upper(0,0,.05)==1 and fusion.cp_upper(10,10,.05)==1
    assert fusion.cp_upper(0,100,.05)==pytest.approx(1-.05**.01)
    assert fusion.cp_upper(0,100,.05/12)>fusion.cp_upper(0,100,.05)
    assert fusion.cp_upper(0,300,.05/12)<.02
    with pytest.raises(ValueError):fusion.cp_upper(2,1,.05)


def test_fusion_order_clip_and_model_roundtrip(fusioncfg):
    g={'candidates':[{'ce_score':float(5-i)} for i in range(5)]}
    x=fusion.fusion_features(g,.8)
    assert len(x)==12 and x[-1]==pytest.approx(math.log(4))
    assert np.isfinite(fusion.fusion_features(g,0)).all()
    assert np.isfinite(fusion.fusion_features(g,1)).all()
    with pytest.raises(ValueError): fusion.fusion_features(g,1.1)
    model=fusion.fitted_model(np.array([x,x+1,x+2,x+3]),[0,0,1,1],fusioncfg)
    assert fusion.model_probability(model,x)<fusion.model_probability(model,x+3)


def test_guard_no_paid_phases_out_of_order(fusioncfg):
    fusion.prepare(fusioncfg,True);fusion.rerank(fusioncfg,True);fusion.preflight(fusioncfg,True)
    with pytest.raises(ValueError):fusion.api_guard(fusioncfg,True,['fit'])
    with pytest.raises(ValueError):fusion.api_guard(fusioncfg,True,['certification'])
    with pytest.raises(ValueError):fusion.api_guard(fusioncfg,True,['test','challenge'])
    with pytest.raises(ValueError):fusion.prepare(fusioncfg,True)
    with pytest.raises(ValueError):fusion.rerank(fusioncfg,True)
    changed=replace(fusioncfg,min_accepted=7)
    with pytest.raises(ValueError,match='Frozen'):fusion.protocol_check(changed,True)


def test_full_seven_gate_mock_protocol_and_no_test_fit(fusioncfg):
    fusion.prepare(fusioncfg,True);fusion.rerank(fusioncfg,True);fusion.preflight(fusioncfg,True)
    asyncio.run(all_async(fusioncfg,True,splits=['fit','select']))
    root=fusioncfg.root(True)
    assert len(read_jsonl(root/'jev.jsonl'))==120
    fusion.fit_gates(fusioncfg,True);model=read_json(root/'gate_models.json')
    groups=read_jsonl(root/'ranked.jsonl')
    assert set(model['fit_query_ids'])=={g['query_id'] for g in groups if g['split']=='fit'}
    assert len(model['models'])==3
    fusion.fit_gates(fusioncfg,True)
    fusion.freeze_thresholds(fusioncfg,True);frozen=read_json(root/'operating_points.json')
    assert len(frozen['workpoints']['0.05'])==6
    asyncio.run(all_async(fusioncfg,True,splits=['certification']))
    fusion.certify(fusioncfg,True);audit=read_json(root/'certification.json')
    assert audit['family_tests']==12
    assert all(v['alpha']==pytest.approx(.05/12) for w in audit['workpoints'].values() for v in w.values())
    assert any(not v['passed'] for w in audit['workpoints'].values() for v in w.values())
    asyncio.run(all_async(fusioncfg,True,splits=['test','challenge']))
    assert len(read_jsonl(root/'jev.jsonl'))==len(groups)
    report=fusion.evaluate(fusioncfg,True)
    assert len(report['metrics'])==7 and report['mock']
    assert report['completeness']['common_test_n']==80
    assert report['budget'] is None
    assert read_json(root/'operating_points.json')==frozen
    fusion.certify(fusioncfg,True)
    with pytest.raises(ValueError,match='already frozen'):fusion.api_guard(fusioncfg,True,['fit','select'])
    with pytest.raises(ValueError,match='already frozen'):fusion.api_guard(fusioncfg,True,['certification'])
    # Certification bytes cannot be silently edited, even when the model/data stay unchanged.
    audit['family_alpha']=.2;write_json(root/'certification.json',audit)
    with pytest.raises(ValueError,match='fingerprint'):fusion.evaluate(fusioncfg,True)


def test_fit_minimum_blocks_all_paid_work(fusioncfg):
    fusion.prepare(fusioncfg,True)
    root=fusioncfg.root(True);rows=read_jsonl(root/'prepared.jsonl')
    for g in rows:
        if g['split']=='fit':
            for c in g['candidates']:c['label']='E'
    write_jsonl(root/'prepared.jsonl',rows)
    with pytest.raises(ValueError,match='paid phase blocked'):fusion.rerank(fusioncfg,True)
    assert not read_json(root/'eligibility.json')['ready']
    assert not (root/'protocol_frozen.json').exists()


def test_known_retry_costs_survive_with_shared_budget(fusioncfg,tmp_path):
    import httpx
    from jev_gate.jev import call_api,RateLimiter,request_body
    from jev_gate.budget import Budget
    cfg=replace(fusioncfg,jev_model='typesafe/jev-1.13-20260917')
    def handler(request):
        if not hasattr(handler,'called'):
            handler.called=True
            return httpx.Response(503,json={'usage':{'cost':.00001}})
        return httpx.Response(200,json={'model':cfg.jev_model,
            'answers':{'has_exact_match':{'type':'noul','noul':.8}},
            'usage':{'input_tokens':100,'output_tokens':22,'cost':.00002}})
    async def no_sleep(seconds):pass
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            b=Budget(tmp_path/'ledger.json',cfg)
            r=await call_api(client,{'model':cfg.jev_model},cfg,RateLimiter(1e5),sleep=no_sleep,budget=b)
            assert r['status']=='success' and len(r['attempts'])==2
            assert r['known_attempt_cost_usd']==pytest.approx(.00003)
            assert not r['cost_may_exclude_failed_attempts']
            assert b.state['reported_usd']==pytest.approx(.00003)
            assert b.state['unknown_reserved_usd']==0
    asyncio.run(run())


def test_decision_price_uses_exact_dated_provider_endpoint():
    import httpx
    model='typesafe/jev-1.13-20260917'
    def handler(request):
        assert str(request.url).endswith(model+'/endpoints')
        return httpx.Response(200,json={'data':{'id':'typesafe/jev-1.13','endpoints':[
            {'provider_name':'TypeSafe','name':'TypeSafe | '+model,'status':0,
             'pricing':{'prompt':'0.000000042','completion':'0'}}]}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        result=fusion.catalog_quote(c,{},'jev',model)
        assert result['endpoint_name'].endswith(model)
        assert float(result['pricing']['prompt'])*1e6==pytest.approx(.042)
        with pytest.raises(ValueError):fusion.catalog_quote(c,{},'gpt','openai/unknown')
    def wrong_build(request):
        return httpx.Response(200,json={'data':{'id':'typesafe/jev-1.13','endpoints':[
            {'provider_name':'TypeSafe','name':'TypeSafe | typesafe/jev-latest','status':0,
             'pricing':{'prompt':'0.000000042','completion':'0'}}]}})
    with httpx.Client(transport=httpx.MockTransport(wrong_build)) as c:
        with pytest.raises(ValueError,match='exact dated'):fusion.catalog_quote(c,{},'jev',model)
