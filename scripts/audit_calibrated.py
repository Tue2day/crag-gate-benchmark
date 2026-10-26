import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_gate.io import read_json,read_jsonl,digest,write_json
from jev_gate.config import Config
from jev_gate import topk
from jev_gate.jev import cache_key,ENDPOINT
from jev_gate.gpt import ENDPOINT as GPT_ENDPOINT
from jev_gate.data import normalize_query
root=Path(sys.argv[1]);f=read_json(root/'protocol_frozen.json');cfg=Config(**f['config'])
ranked=read_jsonl(root/'ranked.jsonl');prepared=read_jsonl(root/'prepared.jsonl');th=read_json(root/'threshold_frozen.json')
assert digest(ranked)==f['ranked_fingerprint'] and digest(prepared)==f['prepared_fingerprint']
for name,hash in f['source_hashes'].items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==hash
assert len({normalize_query(g['query']) for g in prepared})==len(prepared)
assert len([g for g in ranked if g['split']=='calibration'])==800
assert len([g for g in ranked if g['split']=='test'])==2000
index={m:{r['query_id']:r for r in read_jsonl(root/f'{m}.jsonl')} for m in ('jev','gpt')}
for g in ranked:
 for m,builder,endpoint in [('jev',topk.jev_body,ENDPOINT),('gpt',topk.gpt_body,GPT_ENDPOINT)]:
  r=index[m][g['query_id']];assert r['experiment_protocol']=='topk_calibrated_v5' and not r['mock']
  assert r['cache_key']==cache_key(builder(g,cfg),False,endpoint)
  if r['status']=='success':
   p,u=topk.decoder(g,m)(r['response'],cfg.jev_model if m=='jev' else cfg.gpt_model)
   assert p==r['prediction'] and u==r['usage']
cal=[g for g in ranked if g['split']=='calibration' and all(index[m][g['query_id']]['status']=='success' for m in index)]
assert th['calibration_receipt_fingerprint']==digest({m:[index[m][g['query_id']] for g in ranked if g['split']=='calibration'] for m in index})

def selected(g,m,t):
 return {d['id'] for d in index[m][g['query_id']]['prediction']['candidate_decisions'] if (d['core_probability']>=t if m=='jev' else d['is_core_intent_matched']) and not d['is_accessory_trap'] and d['primary_constraint']=='NONE'}

def vectors(groups,m,t):
 rows=[]
 for g in groups:
  s=selected(g,m,t);good={c['id'] for c in g['candidates'] if c['label']=='E'};bad={c['id'] for c in g['candidates']}-good
  tp=len(s&good);fp=len(s&bad);fn=len(good-s);tn=len(bad-s);rescue=g['candidates'][0]['label']!='E' and bool(good)
  rows.append([tp,fp,fn,tn,int(bool(s)),int(bool(s&bad)),int(bool(s) and not s&bad) if good else int(not s),int(rescue),int(rescue and bool(s) and not s&bad and g['candidates'][0]['id'] not in s)])
 return np.asarray(rows)

def metrics(x,n):
 tp,fp,fn,tn,passed,polluted,safe,rescue,success=x
 def ratio(a,b):return a/b if b else np.nan
 return dict(kept_precision=ratio(tp,tp+fp),exact_retention=ratio(tp,tp+fn),query_coverage=passed/n,query_contamination_risk=ratio(polluted,passed),safe_route_success=safe/n,rescue_success_rate=ratio(success,rescue),candidate_f1=ratio(2*tp,2*tp+fp+fn))
target=metrics(vectors(cal,'gpt',.5).sum(0),len(cal))['exact_retention'];grid=[]
for t in cfg.core_threshold_grid:
 v=metrics(vectors(cal,'jev',t).sum(0),len(cal));grid.append((t,v))
feasible=[(t,v) for t,v in grid if v['exact_retention']>=target-cfg.calibration_retention_slack and np.isfinite(v['kept_precision'])]
assert bool(feasible)==th['usable']
chosen=max(feasible,key=lambda z:(z[1]['kept_precision'],z[0]))[0] if feasible else None
assert chosen==th['threshold']
print('calibration independently verified',chosen)
if chosen is None:
 print('No feasible threshold, no test evaluation.');sys.exit(0)
report=read_json(root/'report/metrics.json');natural=[g for g in ranked if g['split']=='test' and all(index[m][g['query_id']]['status']=='success' for m in index)]
assert len(natural)==report['completeness']['common_n'];a=vectors(natural,'jev',chosen);b=vectors(natural,'gpt',.5)
for m,x in [('jev',a),('gpt',b)]:
 for k,v in metrics(x.sum(0),len(x)).items():assert np.isclose(v,report['metrics'][m][k],equal_nan=True)
for row in csv.DictReader((root/'report/query_predictions.csv').open(encoding='utf-8-sig',newline='')):
 g=next(g for g in natural if g['query_id']==row['query_id']);m=row['method']
 assert json.loads(row['selected_ids'])==[c['id'] for c in g['candidates'] if c['id'] in selected(g,m,chosen)]
for row in csv.DictReader((root/'report/candidate_predictions.csv').open(encoding='utf-8-sig',newline='')):
 g=next(g for g in natural if g['query_id']==row['query_id']);s=selected(g,row['method'],chosen)
 assert (row['keep']=='True')==(row['product_id'] in s)
seed=cfg.seed;rng=np.random.default_rng(seed);delta={k:[] for k in topk.BOOT_METRICS}
for _ in range(cfg.bootstrap_samples):
 ii=rng.integers(0,len(a),len(a));ma=metrics(a[ii].sum(0),len(a));mb=metrics(b[ii].sum(0),len(b))
 for k in delta:delta[k].append(ma[k]-mb[k])
for k,v in delta.items():
 v=np.asarray(v);v=v[np.isfinite(v)];ci=np.quantile(v,[.025,.975]).tolist() if len(v) else None
 assert np.allclose(ci,report['bootstrap']['delta_ci95'][k],equal_nan=True)
assert np.isclose(np.quantile(delta['exact_retention'],.05),report['bootstrap']['retention_delta_one_sided95_lower'])
result=dict(passed=True,raw_records_checked_per_model={m:len(idx) for m,idx in index.items()},source_and_data_fingerprints_verified=True,
 calibration_threshold_independently_selected=chosen,test_query_n=len(a),candidate_csv_rows_checked=10*len(a),query_csv_rows_checked=2*len(a),
 bootstrap_samples=cfg.bootstrap_samples,all_seven_delta_intervals_and_one_sided_retention_bound_match=True)
write_json(root/'report/independent_metric_check.json',result);print(json.dumps(result,indent=2))
