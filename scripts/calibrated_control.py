"""Run/monitor/export the pre-registered independent calibration experiment."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_gate import calibrated as c, topk
from jev_gate.config import Config
from jev_gate.data import normalize_query
from jev_gate.io import digest, load_env, read_json, read_jsonl, write_json

p=argparse.ArgumentParser();p.add_argument('command',choices=['run','monitor','export'])
p.add_argument('--config',default='config.topk-calibrated-v5.toml');a=p.parse_args()
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
project=Path(a.config).resolve().parent;cfg=Config.load(a.config);root=cfg.root();load_env(project/'.env')
if a.command=='run':
 for step in [['preflight'],['api-all','--splits','calibration'],['freeze-thresholds'],['api-all','--splits','test'],['evaluate']]:
  print('STAGE: '+' '.join(step),flush=True)
  subprocess.run([sys.executable,'-u','-m','jev_gate',*step,'--config',str(Path(a.config).resolve())],cwd=project,check=True)
 raise SystemExit(0)
if a.command=='monitor':
 result={}
 for m in c.METHODS:
  path=root/f'{m}_journal.jsonl';rows={}
  if path.exists():
   for r in read_jsonl(path):rows[r['query_id']]=r
  result[m]={'n':len(rows),'splits':dict(Counter(r['split'] for r in rows.values())),
   'status':dict(Counter(r['status'] for r in rows.values())),
   'known_cost_usd':sum(r.get('known_attempt_cost_usd',0) for r in rows.values())}
 if (root/'threshold_frozen.json').exists():result['frozen_threshold']=read_json(root/'threshold_frozen.json')['threshold']
 print(json.dumps(result,indent=2));raise SystemExit(0)
# Evaluate the frozen strategy first; this also verifies raw API parsing and calibrated-file integrity.
report=c.evaluate(cfg,False);groups=c.checked(cfg,False);f=c.frozen_threshold(cfg,False)
idx=c.prediction_index(cfg,False,groups,{'calibration','test'})
prepared=read_jsonl(root/'prepared.jsonl')
if cfg.exclusion_manifest_path:
 from jev_gate.publication import exclusions
 hashes,_=exclusions(cfg,read_json(root/'prepare_metadata.json')['sources'],4)
 assert not hashes & {digest(normalize_query(g['query'])) for g in prepared}
else:
 old={normalize_query(g['query']) for path in cfg.prior_prepared_paths for g in read_jsonl(path)}
 assert not old & {normalize_query(g['query']) for g in prepared}
audit={'passed':True,'prior_overlap':0,'natural_n':cfg.test_size,'calibration_n':cfg.calibration_size,
 'threshold':f['threshold'],'raw_records_checked_per_model':{},'counts':{},'account_before':read_json(root/'protocol_frozen.json')['account_before']}
# Independent item and query counts using selected IDs, with no aggregate/query_counts helper.
for m in c.METHODS:
 tp=fp=fn=tn=passed=polluted=safe=0
 for g in groups:
  if g['split']!='test' or any(idx[k][g['query_id']]['status']!='success' for k in c.METHODS):continue
  raw=idx[m][g['query_id']]['prediction'];selected={d['id'] for d in raw['candidate_decisions'] if
   (d['core_probability']>=f['threshold'] if m=='jev' else d['is_core_intent_matched']) and
   not d['is_accessory_trap'] and d['primary_constraint']=='NONE'}
  good={i['id'] for i in g['candidates'] if i['label']=='E'};bad={i['id'] for i in g['candidates']}-good
  tp+=len(selected&good);fp+=len(selected&bad);fn+=len(good-selected);tn+=len(bad-selected)
  passed+=bool(selected);polluted+=bool(selected&bad);safe+=bool(selected and not selected&bad) if good else not selected
 expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,passed=passed,polluted=polluted,safe_route_success=safe)
 assert all(report['metrics'][m]['counts'][k]==v for k,v in expected.items())
 audit['counts'][m]=expected;audit['raw_records_checked_per_model'][m]=len(idx[m])
import httpx
with httpx.Client(timeout=60) as client:
 response=client.get('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']});response.raise_for_status();d=response.json()['data']
audit['account_after']={k:d[k] for k in ('total_credits','total_usage')};audit['account_delta']=d['total_usage']-audit['account_before']['total_usage']
audit['remaining']=d['total_credits']-d['total_usage'];audit['known_experiment_cost']=sum(v['known_cost_usd'] for v in report['costs'].values())
audit['unattributed_account_delta']=audit['account_delta']-audit['known_experiment_cost']
audit['threshold_fingerprint']=digest(f)
write_json(root/'report/integrity_audit.json',audit)
files=list((root/'report').glob('*'))+list(root.glob('*_metadata.json'))+list((root/'source').glob('*.py'))
files+=[root/n for n in ('protocol_frozen.json','threshold_frozen.json','calibration_grid.csv','budget.json','ranked.jsonl','prepared.jsonl','jev.jsonl','gpt.jsonl','jev_journal.jsonl','gpt_journal.jsonl')]
files+=[project/Path(a.config).name,project/'docs/EXPERIMENT_REPORT.md']
bundle=project.parent/'jev-gate-topk-calibrated-v5-results.zip'
with ZipFile(bundle,'w',ZIP_DEFLATED) as z:
 for path in files:
  assert path.name!='.env'
  name=str(path.relative_to(project)).replace('\\','/') if path.is_relative_to(project) and not path.is_relative_to(root) else str(path.relative_to(root)).replace('\\','/')
  z.write(path,'jev-gate-topk-calibrated-v5/'+name)
print(json.dumps({'audit':audit,'metrics':report['metrics'],'conclusions':report['conclusions'],'costs':report['costs'],'latency':report['latency'],'bundle':str(bundle)},indent=2))
