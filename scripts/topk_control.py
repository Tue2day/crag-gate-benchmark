"""Frozen public Top-K proxy runner, monitor and raw-record audit/export."""
import argparse
from collections import Counter
from pathlib import Path
import csv,json,math,os,subprocess,sys
from zipfile import ZipFile,ZIP_DEFLATED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_gate.config import Config
from jev_gate.io import read_json,read_jsonl,write_json,digest,load_env
from jev_gate.data import normalize_query
from jev_gate import topk
p=argparse.ArgumentParser();p.add_argument('command',choices=['run','monitor','export']);p.add_argument('--config',default='config.topk-v4.toml')
a=p.parse_args();cfg=Config.load(a.config);root=cfg.root();project=Path(a.config).resolve().parent
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
load_env(project/'.env')
if a.command=='run':
 for step in [['preflight'],['probe'],['check-probe'],['api-all','--splits','test','challenge'],['evaluate']]:
  print('STAGE: '+' '.join(step),flush=True)
  subprocess.run([sys.executable,'-u','-m','jev_gate',*step,'--config',str(Path(a.config).resolve())],cwd=project,check=True)
 raise SystemExit(0)
if a.command=='monitor':
 out={}
 for m in ('jev','gpt'):
  rows={};path=root/f'{m}_journal.jsonl'
  if path.exists():
   with path.open(encoding='utf-8') as f:
    for line in f:
     try:r=json.loads(line)
     except ValueError:continue
     rows[r['query_id']]=r
  vals=list(rows.values());out[m]={'n':len(vals),'split':dict(Counter(r['split'] for r in vals)),
   'status':dict(Counter(r['status'] for r in vals)),'errors':dict(Counter(r.get('error') for r in vals if r['status']!='success')),
   'cost':sum(r.get('known_attempt_cost_usd',0) for r in vals)}
 print(json.dumps(out,indent=2));raise SystemExit(0)
import httpx
f=read_json(root/'protocol_frozen.json');groups=topk.checked(cfg,False);idx={g['query_id']:g for g in groups}
prepared=read_jsonl(root/'prepared.jsonl');assert digest(prepared)==f['prepared_fingerprint']
assert len({normalize_query(g['query']) for g in prepared})==len(prepared)
if cfg.exclusion_manifest_path:
 from jev_gate.publication import exclusions
 hashes,_=exclusions(cfg,read_json(root/'prepare_metadata.json')['sources'],3)
 assert not hashes & {digest(normalize_query(g['query'])) for g in prepared}
else:
 old={normalize_query(g['query']) for path in cfg.prior_prepared_paths for g in read_jsonl(path)}
 assert not old & {normalize_query(g['query']) for g in prepared}
audit={'passed':True,'prior_overlap':0,'api':{},'checks':{}}
records={}
for m in ('jev','gpt'):
 rows=read_jsonl(root/f'{m}.jsonl');records[m]={r['query_id']:r for r in rows};assert len(rows)==len(groups)
 for r in rows:
  g=idx[r['query_id']];assert not r['mock'] and r['task']==topk.TASK and r['experiment_protocol']==topk.PROTOCOL
  if r['status']=='success':
   pred,usage=topk.decoder(g,m)(r['response'],cfg.jev_model if m=='jev' else cfg.gpt_model)
   assert pred==r['prediction'] and usage==r['usage']
 audit['api'][m]={'n':len(rows),'success':sum(r['status']=='success' for r in rows),'attempts':sum(len(r['attempts']) for r in rows),
  'known_cost':sum(r.get('known_attempt_cost_usd',0) for r in rows),'cache_hits':sum(r['cache_hit'] for r in rows)}
report=read_json(root/'report/metrics.json')
# Reconstruct actual network measurements across resumes. Cache-read time is never used.
from jev_gate.evaluate import percentile_summary
history={};history_costs={};recorded_attempts={}
for method in ('jev','gpt'):
 history[method]={};known=0.;attempts=0
 for row in read_jsonl(root/f'{method}_journal.jsonl'):
  if row['cache_hit']:continue
  known+=row.get('known_attempt_cost_usd',0);attempts+=len(row['attempts'])
  if row['status']=='success':history[method].setdefault(row['query_id'],row)
 history_costs[method]=known;recorded_attempts[method]=attempts
latency_ids=[g['query_id'] for g in groups if g['split']=='test' and all(g['query_id'] in history[m] for m in history)]
full_latency={}
for method in history:
 samples=[history[method][qid] for qid in latency_ids]
 full_latency[method]={'http':percentile_summary([x['elapsed_ms'] for row in samples for x in row['attempts']]),
  'gate':percentile_summary([row['total_elapsed_ms'] for row in samples if row['total_elapsed_ms'] is not None])}
ledger=read_json(root/'budget.json')
events=read_json(root/'run_events.json') if (root/'run_events.json').exists() else []
execution={'history_known_costs':history_costs,'recorded_noncache_attempts':recorded_attempts,
 'paired_natural_latency_queries':len(latency_ids),'latency_uncached_history':full_latency,
 'reserved_attempts_in_ledger':ledger['sequence'],'events':events,
 'note':'First successful actual network invocation per query across resumes. No cache-read latency; process-level interruption/restart interval not included. Failed orphan attempts may lack timing/usage.'}
write_json(root/'report/execution_audit.json',execution)
report['latency_uncached_history']=full_latency
report['latency_history_definition']=execution['note']
report['orphan_usage_unknown']=ledger['unknown_reserved_usd']>0
for method in history:
 report['costs'][method]['known_cost_last_success_records']=report['costs'][method]['known_cost_usd']
 report['costs'][method]['known_cost_usd']=history_costs[method]
 report['costs'][method]['http_attempts']=recorded_attempts[method]
 audit['api'][method]['known_cost']=history_costs[method]
 audit['api'][method]['attempts']=recorded_attempts[method]
write_json(root/'report/metrics.json',report)
audit['execution_audit_fingerprint']=digest(execution)
# Independent counts directly from IDs and original judgments; no classifier helper is used.
common=[g for g in groups if g['split']=='test' and all(records[m][g['query_id']]['status']=='success' for m in records)]
for m in records:
 tp=fp=fn=tn=passed=polluted=safe=0
 for g in common:
  selected=set(records[m][g['query_id']]['prediction']['selected_ids']);good={c['id'] for c in g['candidates'] if c['label']=='E'}
  bad={c['id'] for c in g['candidates']}-good
  tp+=len(selected&good);fp+=len(selected&bad);fn+=len(good-selected);tn+=len(bad-selected)
  passed+=bool(selected);polluted+=bool(selected&bad);safe+=bool(selected and not selected&bad) if good else not selected
 actual=report['metrics'][m];expected={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'passed':passed,'polluted':polluted,'safe_route_success':safe}
 assert all(actual['counts'][k]==v for k,v in expected.items())
 audit['checks'][m]=expected
r=httpx.get('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},timeout=30)
if r.status_code!=200:raise ValueError('Account snapshot unavailable')
d=r.json()['data'];audit['account_before']=f['account_before'];audit['account_after']={k:d[k] for k in ('total_credits','total_usage')}
audit['known_experiment_cost']=sum(v['known_cost'] for v in audit['api'].values());audit['account_delta']=d['total_usage']-f['account_before']['total_usage'];audit['remaining']=d['total_credits']-d['total_usage']
write_json(root/'report/integrity_audit.json',audit)
files=list((root/'report').glob('*'))+list(root.glob('*_metadata.json'))
files+=[root/n for n in ('protocol_frozen.json','probe_checked.json','budget.json','ranked.jsonl','jev.jsonl','gpt.jsonl','jev_journal.jsonl','gpt_journal.jsonl')]
if (root/'run_events.json').exists():files.append(root/'run_events.json')
files+=[project/'config.topk-v4.toml',project/'docs/EXPERIMENT_REPORT.md']+sorted((project/'jev_gate').glob('*.py'))
bundle=project.parent/'jev-gate-topk-v4-results.zip'
with ZipFile(bundle,'w',ZIP_DEFLATED) as z:
 for path in files:
  name='source/'+path.name if path.parent.name=='jev_gate' else (str(path.relative_to(project)).replace('\\','/') if path.is_relative_to(project) and not path.is_relative_to(root) else str(path.relative_to(root)).replace('\\','/'))
  assert path.name!='.env';z.write(path,'jev-gate-topk-v4/'+name)
print(json.dumps({'audit':audit,'metrics':report['metrics'],'challenge':report['challenge'],'bootstrap':report['bootstrap'],
 'latency':report['latency'],'models':report['models'],'budget':report['budget'],'bundle':str(bundle)},indent=2))
