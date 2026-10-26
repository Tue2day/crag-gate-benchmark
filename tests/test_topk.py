import asyncio
from dataclasses import replace
import copy,json
import httpx
import numpy as np
import pytest
from jev_gate import topk
from jev_gate.config import Config
from jev_gate.io import read_json,read_jsonl,write_json,write_jsonl

@pytest.fixture
def kc(cfg):
 return replace(cfg,protocol='topk_v4',task='topk_filter',seed=2026102504,
  development_size=5,test_size=30,challenge_pool_size=80,challenge_size=8,challenge_s=3,challenge_c=3,challenge_i=2,
  gpt_max_tokens=768,api_budget_usd=1.8,budget_safety_usd=.1,preflight_max_estimate_usd=1.7,
  concurrency=2,requests_per_second=100000,bootstrap_samples=30)

def group():
 return {'query_id':'x','query':'requested phone','split':'test','task':'topk_filter','experiment_protocol':'topk_v4','mock':True,
  'candidates':[{'id':str(i),'label':l,'title':'product','brand':None,'color':None,'bullet_points':'','ce_score':float(5-i)} for i,l in enumerate(['C','E','E','S','I'])]}

def pred(ids):return {'selected_ids':ids}

def test_schema_inputs_labels_and_15_questions(kc):
 g=group();a=topk.jev_body(g,kc);b=topk.gpt_body(g,kc)
 assert a['state']==json.loads(b['messages'][1]['content'])
 assert len(a['questions'])==15 and len(a['state']['candidates'])==5
 assert set(a['state']['candidates'][0])=={'rank','id','title','brand','color','bullet_points','ce_score'}
 changed=copy.deepcopy(g)
 for c in changed['candidates']:c['label']='I'
 assert topk.jev_body(changed,kc)==a
 assert topk.gpt_body(changed,kc)==b
 assert 'probability_exact' not in json.dumps(b)
 Config.load('config.topk-v4.toml')

def test_shared_filter_and_route_unknown_is_not_violation():
 rows=[{'id':str(i),'is_core_intent_matched':True,'is_accessory_trap':False,'primary_constraint':s} for i,s in enumerate(['NONE','UNKNOWN','BRAND'])]
 p=topk.adapt(rows,['0','1','2'],'log','test')
 assert p['selected_ids']==['0'] and p['routing_action']=='PASS_FILTERED'
 assert p['candidate_decisions'][1]['violated_constraints']==[]
 assert p['candidate_decisions'][1]['constraint_evidence_unknown']
 assert p['candidate_decisions'][2]['violated_constraints']==['BRAND']
 rows[0]['is_accessory_trap']=True
 assert topk.adapt(rows,['0','1','2'],'','test')['routing_action']=='FALLBACK'
 with pytest.raises(ValueError):topk.adapt(rows,['0','0','2'],'','test')

def test_multifield_response_parsing(kc):
 g=group();answers={}
 for i in range(5):
  for field in ('core','accessory'):answers[f'c{i}_{field}']={'type':'noul','noul':.9 if field=='core' else .1}
  answers[f'c{i}_constraint']={'type':'choice','choice':'NONE','probabilities':{k:float(k=='NONE') for k in topk.CONSTRAINTS}}
 raw={'model':kc.jev_model,'answers':answers,'usage':{'cost':.0001}}
 p,u=topk.decoder(g,'jev')(raw,kc.jev_model)
 assert p['selected_ids']==[str(i) for i in range(5)] and p['routing_action']=='PASS_ALL'
 assert u['reported_cost_usd']==.0001
 bad=copy.deepcopy(raw);bad['answers']['c0_constraint']['choice']='bad'
 with pytest.raises(ValueError):topk.decoder(g,'jev')(bad,kc.jev_model)
 rows=[{'id':str(i),'is_core_intent_matched':True,'is_accessory_trap':i==0,'primary_constraint':'NONE'} for i in range(5)]
 raw={'model':kc.gpt_model,'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'candidate_decisions':rows,'reasoning':'brief'})}}]}
 p,_=topk.decoder(g,'gpt')(raw,kc.gpt_model)
 assert p['selected_ids']==['1','2','3','4']
 raw['choices'][0]['finish_reason']='length'
 with pytest.raises(ValueError):topk.decoder(g,'gpt')(raw,kc.gpt_model)

def test_known_quality_counts_rescue_and_empty_risk():
 g=group();c=topk.query_counts(g,pred(['1','2']))
 assert (c['tp'],c['fp'],c['fn'],c['tn'])==(2,0,0,3)
 assert c['rescue_success']==1 and c['safe_route_success']==1 and c['route_correct']==1
 r=topk.aggregate([c]);assert r['kept_precision']==1 and r['exact_retention']==1 and r['query_contamination_risk']==0
 r=topk.aggregate([topk.query_counts(g,pred(['0','1']))])
 assert r['kept_precision']==.5 and r['exact_retention']==.5 and r['query_contamination_risk']==1
 r=topk.aggregate([topk.query_counts(g,pred([]))]);assert r['query_contamination_risk'] is None and r['unnecessary_fallback_rate']==1
 with pytest.raises(ValueError):topk.query_counts(g,pred(['not-an-id']))

def test_paired_bootstrap_clusters():
 a=[topk.query_counts(group(),pred(['1','2']))]*5;b=[topk.query_counts(group(),pred(['0','1']))]*5
 x=topk.bootstrap(a,b,20,42)
 assert x['delta_ci95']['query_contamination_risk']==[-1.,-1.]
 assert x['delta_ci95']['exact_retention']==[.5,.5]
 assert x==topk.bootstrap(a,b,20,42)

def test_full_mock_and_freezes(kc):
 kc.validate();topk.prepare(kc,True);topk.rerank(kc,True);topk.preflight(kc,True)
 with pytest.raises((ValueError,FileNotFoundError)):topk.run(kc,True,['test','challenge'])
 topk.run(kc,True,['development']);topk.check_probe(kc,True)
 with pytest.raises(ValueError,match='already frozen'):topk.run(kc,True,['development'])
 topk.run(kc,True,['test','challenge']);r=topk.evaluate(kc,True)
 assert r['completeness']['natural_n']==30 and r['completeness']['common_fraction']==1
 assert set(r['metrics'])=={'jev','gpt'} and r['mock'] and r['budget'] is None
 with pytest.raises(ValueError):topk.prepare(kc,True)
 with pytest.raises(ValueError):topk.rerank(kc,True)
 changed=replace(kc,title_chars=50)
 with pytest.raises(ValueError,match='Frozen'):topk.checked(changed,True)
 assert len(read_jsonl(kc.root(True)/'jev.jsonl'))==len(read_jsonl(kc.root(True)/'ranked.jsonl'))

def test_invalid_contract_preserves_raw_reply_and_known_cost(kc):
 from jev_gate.jev import call_api,RateLimiter
 from jev_gate.budget import Budget
 raw={'model':kc.jev_model,'answers':{},'usage':{'cost':.00001}}
 async def scenario():
  async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=raw))) as client:
   b=Budget(kc.root(True)/'budget.json',kc)
   r=await call_api(client,topk.jev_body(group(),kc),kc,RateLimiter(1e5),budget=b,parse_fn=topk.decoder(group(),'jev'))
   assert r['status']=='failed' and r['error']=='invalid_response' and r['response']==raw
   assert r['known_attempt_cost_usd']==pytest.approx(.00001)
 asyncio.run(scenario())

def test_missing_pilot_billing_blocks_full_batch(kc,monkeypatch):
 g=group();g['split']='development'
 monkeypatch.setattr(topk,'checked',lambda cfg,mock:[g])
 monkeypatch.setattr(topk,'prediction_index',lambda *a:{m:{'x':{'status':'success','usage':{'reported_cost_usd':None}}} for m in ('jev','gpt')})
 with pytest.raises(ValueError,match='lacks billed usage'):topk.check_probe(kc,False)
