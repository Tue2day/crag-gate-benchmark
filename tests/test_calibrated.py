from dataclasses import replace
import copy
import ssl
import asyncio

import httpx
import numpy as np
import pytest
from jev_gate import calibrated as c, topk
from jev_gate.config import Config
from jev_gate.io import read_json, read_jsonl, write_json
from test_topk import group

@pytest.fixture
def vc(cfg):
 return replace(cfg, protocol=c.PROTOCOL, task='topk_filter', calibration_size=8, test_size=10,
  challenge_size=0, challenge_pool_size=0, challenge_s=0, challenge_c=0, challenge_i=0,
  core_threshold_grid=[.1,.5,.9], calibration_retention_slack=.01, retention_noninferiority_margin=.02,
  gpt_max_tokens=768, api_budget_usd=4.2, budget_safety_usd=.1, preflight_max_estimate_usd=4.05,
  bootstrap_samples=30)

def prediction(g, probs, accessory=None, constraints=None):
 rows=[{'id': item['id'], 'is_core_intent_matched':p>=.5, 'core_probability':p,
  'is_accessory_trap': (accessory or [False]*5)[i], 'primary_constraint':(constraints or ['NONE']*5)[i]}
  for i,(item,p) in enumerate(zip(g['candidates'],probs))]
 return topk.adapt(rows,[r['id'] for r in rows],'log','test')

def test_partition_normalized_exclusion_and_sources(vc):
 gs=[]
 for i in range(50):
  g=group();gs.append({**g,'query_id':str(i),'query':f'q {i}','source_split':'train' if i<20 else 'test'})
 gs.append({**gs[-1],'query_id':'dupe','query':' Q   49 '})
 rows=c.partition(gs,vc,{'q 0','q 49'})
 assert len(rows)==18 and len({x['query'] for x in rows})==18
 assert not {'q 0','q 49'}&{x['query'] for x in rows}
 assert all(x['source_split']==('train' if x['split']=='calibration' else 'test') for x in rows)
 assert rows==c.partition(gs,vc,{'q 0','q 49'})
 with pytest.raises(ValueError):c.partition(gs,replace(vc,test_size=100))

def test_adjusts_core_only_without_mutating_raw():
 g=group();p=prediction(g,[.2,.3,.8,.9,.9],accessory=[False,False,True,False,False],constraints=['NONE','NONE','NONE','UNKNOWN','BRAND'])
 old=copy.deepcopy(p);r=c.apply_core_threshold(p,.1)
 assert r['selected_ids']==['0','1'] and p==old
 assert c.apply_core_threshold(p,.5)['selected_ids']==[]
 p['candidate_decisions'][0]['core_probability']=float('nan')
 with pytest.raises(ValueError):c.apply_core_threshold(p,.1)

def test_selection_rule_tie_break_and_infeasible(vc):
 g=group();g['split']='calibration'
 # E items are 1/2. At .1 both E retained, at .5 one, at .9 none.
 j=prediction(g,[.05,.3,.8,.05,.05]);gpt=prediction(g,[0,1,1,0,0])
 idx={'jev':{'x':{'status':'success','prediction':j}},'gpt':{'x':{'status':'success','prediction':gpt}}}
 r=c.select_threshold([g],idx,vc);assert r['threshold']==.1 and r['usable']
 idx['jev']['x']['prediction']=prediction(g,[0,.01,.01,0,0])
 assert not c.select_threshold([g],idx,vc)['usable']
 idx['jev']['x']['prediction']=prediction(g,[0,.99,.99,0,0])
 assert c.select_threshold([g],idx,vc)['threshold']==.9
 g['split']='test'
 with pytest.raises(ValueError,match='calibration only'):c.select_threshold([g],idx,vc)

def test_bootstrap_matches_query_cluster_reference(vc):
 gs=group();a=[topk.query_counts(gs,{'selected_ids':['1','2']})]*4+[topk.query_counts(gs,{'selected_ids':['1']})]*3
 b=[topk.query_counts(gs,{'selected_ids':['0','1']})]*7
 fast=c.bootstrap(a,b,100,42);slow=topk.bootstrap(a,b,100,42)
 for k in topk.BOOT_METRICS:
  assert np.allclose(fast['delta_ci95'][k],slow['delta_ci95'][k],equal_nan=True)
 assert fast['retention_delta_one_sided95_lower']>=0

@pytest.mark.parametrize('lower,precision,expected',[(-.019,[.001,.02],True),(-.021,[.01,.02],False),(-.02,[.01,.02],False),(-.01,[-.001,.02],False)])
def test_noninferiority_is_one_sided_not_zero_crossing(vc,lower,precision,expected):
 b={'retention_delta_one_sided95_lower':lower,'delta_ci95':{'kept_precision':precision}}
 assert c.decisions(b,vc)['joint_quality_evidence']==expected
 assert not c.decisions(b,vc,False)['joint_quality_evidence']

def test_full_mock_order_and_tamper_guards(vc):
 vc.validate();Config.load('config.topk-calibrated-v5.toml')
 c.prepare(vc,True);c.rerank(vc,True);c.preflight(vc,True)
 with pytest.raises(ValueError):c.run(vc,True,['test'])
 with pytest.raises(ValueError):c.run(vc,True,['calibration','test'])
 c.run(vc,True,['calibration']);f=c.freeze_thresholds(vc,True)
 assert f['usable'] and f['threshold']==.5
 assert c.freeze_thresholds(vc,True)==f
 with pytest.raises(ValueError):c.run(vc,True,['calibration'])
 c.run(vc,True,['test']);r=c.evaluate(vc,True)
 assert r['mock'] and r['completeness']['natural_n']==10 and r['completeness']['common_fraction']==1
 p=vc.root(True)/'threshold_frozen.json';bad=read_json(p);bad['threshold']=.1;write_json(p,bad)
 with pytest.raises(ValueError,match='inconsistent'):c.frozen_threshold(vc,True)
 with pytest.raises(ValueError):c.prepare(vc,True)
 with pytest.raises(ValueError):c.checked(replace(vc,test_size=9),True)

def test_no_feasible_point_stops_test_without_retuning(vc):
 c.prepare(vc,True);c.rerank(vc,True);c.preflight(vc,True);c.run(vc,True,['calibration'])
 p=vc.root(True)/'jev.jsonl';rows=read_jsonl(p)
 from jev_gate.io import write_jsonl
 for row in rows:
  for d in row['prediction']['candidate_decisions']:d['core_probability']=0
 write_jsonl(p,rows);f=c.freeze_thresholds(vc,True);assert not f['usable']
 with pytest.raises(ValueError,match='No calibration threshold'):c.run(vc,True,['test'])
 assert all(r['split']=='calibration' for r in read_jsonl(p))

def test_native_ssl_exception_is_recorded_retried_and_reserved(vc):
 from jev_gate.jev import call_api, RateLimiter
 from jev_gate.budget import Budget
 calls=[]
 def handler(request):
  calls.append(1)
  if len(calls)==1:raise ssl.SSLError('synthetic SSL interruption')
  return httpx.Response(200,json={'model':vc.jev_model,'usage':{'cost':.0001}})
 async def scenario():
  async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
   b=Budget(vc.root(True)/'ssl_budget.json',vc)
   r=await call_api(client,{'model':vc.jev_model},vc,RateLimiter(1e5),budget=b,
    sleep=lambda _:asyncio.sleep(0),parse_fn=lambda raw,model:({'selected_ids':[]},{'reported_cost_usd':.0001}))
   assert r['status']=='success' and len(r['attempts'])==2 and r['attempts'][0]['error']=='transport_error'
   assert r['cost_may_exclude_failed_attempts'] and b.state['unknown_reserved_usd']>0
 asyncio.run(scenario())
