"""Run/monitor/export the frozen v3 protocol. No secrets are printed or archived."""
import argparse
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from zipfile import ZipFile,ZIP_DEFLATED

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_gate.config import Config
from jev_gate.io import read_json,read_jsonl,write_json,digest,load_env
from jev_gate.data import normalize_query
from jev_gate import fusion

parser=argparse.ArgumentParser()
parser.add_argument('command',choices=['run','monitor','export'])
parser.add_argument('--config',default='config.top1-fusion-v3.toml')
a=parser.parse_args()
cfg=Config.load(a.config);root=cfg.root();project=Path(a.config).resolve().parent
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
load_env(project/'.env')

if a.command=='run':
    steps=[['preflight'],['api-all','--splits','fit','select'],['fit-gates'],['freeze-thresholds'],
           ['api-all','--splits','certification'],['certify'],['api-all','--splits','test','challenge'],['evaluate']]
    for step in steps:
        print('STAGE: '+' '.join(step),flush=True)
        subprocess.run([sys.executable,'-u','-m','jev_gate',*step,'--config',str(Path(a.config).resolve())],cwd=project,check=True)
    print('All real stages complete; ready for audit/export',flush=True)
    raise SystemExit(0)

if a.command=='monitor':
    result={'stages':{n:(root/n).exists() for n in ('prepared.jsonl','ranked.jsonl','protocol_frozen.json','gate_models.json','operating_points.json','certification.json','report/metrics.json')}}
    for method in ('jev','gpt'):
        rows={};path=root/f'{method}_journal.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as f:
                for line in f:
                    try:r=json.loads(line)
                    except ValueError:continue
                    rows[r['query_id']]=r
        values=list(rows.values())
        result[method]={'completed':len(values),'splits':dict(Counter(r['split'] for r in values)),
            'statuses':dict(Counter(r['status'] for r in values)),
            'errors':dict(Counter(r.get('error') for r in values if r['status']!='success')),
            'reported_cost_usd':sum(r.get('known_attempt_cost_usd',r['usage'].get('reported_cost_usd') or 0) for r in values)}
    print(json.dumps(result,ensure_ascii=True,indent=2));raise SystemExit(0)

from jev_gate.jev import request_body,parse_response as parse_jev
from jev_gate.gpt import parse_response as parse_gpt
from sklearn.metrics import average_precision_score,brier_score_loss
import httpx
import numpy as np

groups=fusion.protocol_check(cfg,False);prepared=read_jsonl(root/'prepared.jsonl')
assert digest(prepared)==read_json(root/'protocol_frozen.json')['prepared_fingerprint']
assert len({normalize_query(g['query']) for g in prepared})==len(prepared)
old={normalize_query(g['query']) for p in cfg.prior_prepared_paths for g in read_jsonl(p)}
assert not old & {normalize_query(g['query']) for g in prepared}
report=read_json(root/'report/metrics.json');models=fusion.checked_artifact(root,'gate_models.json')
ops=fusion.checked_artifact(root,'operating_points.json');cert=fusion.checked_artifact(root,'certification.json')
assert set(models['fit_query_ids'])<={g['query_id'] for g in groups if g['split']=='fit'}
assert len(models['fit_query_ids'])>=.99*cfg.fit_size
assert set(cert['query_ids'])<={g['query_id'] for g in groups if g['split']=='certification'}
audit={'passed':True,'prior_overlap':0,'prepared_unique_queries':len(prepared),'api':{},'metric_checks':[]}
for method,parser,model in [('jev',parse_jev,cfg.jev_model),('gpt',parse_gpt,cfg.gpt_model)]:
    rows=read_jsonl(root/f'{method}.jsonl')
    assert len(rows)==len(groups)
    for r in rows:
        assert not r['mock'] and r['task']==cfg.task and r['experiment_protocol']==cfg.protocol
        if r['status']=='success':
            prob,usage=parser(r['response'],model)
            assert prob==r['probability'] and usage==r['usage']
    audit['api'][method]={'n':len(rows),'success':sum(r['status']=='success' for r in rows),
        'cache_hits':sum(r['cache_hit'] for r in rows),'attempts':sum(len(r['attempts']) for r in rows),
        'reported_cost_usd':sum(r.get('known_attempt_cost_usd',r['usage'].get('reported_cost_usd') or 0) for r in rows)}
rows=list(csv.DictReader((root/'report/query_predictions.csv').open(encoding='utf-8-sig')))
rows=[r for r in rows if r['split']=='test'];y=np.array([int(r['y']) for r in rows])
for method in fusion.METHODS:
    scores=np.array([float(r[method]) for r in rows]);metric=report['metrics'][method]
    assert math.isclose(average_precision_score(1-y,-scores),metric['bad_average_precision'])
    if metric['brier'] is not None:assert math.isclose(brier_score_loss(y,scores),metric['brier'])
    for q,w in report['workpoints'].items():
        s=w['methods'][method];mask=scores>=s['threshold'] if s['candidate_available'] else np.zeros(len(y),bool)
        assert int(mask.sum())==s['accepted'] and int(((y==0)&mask).sum())==s['bad_accepted']
        audit['metric_checks'].append({'target':q,'method':method,'accepted':s['accepted'],'bad_accepted':s['bad_accepted']})
key=os.environ.get('OPENROUTER_API_KEY','').strip()
with httpx.Client(timeout=30) as client:
    response=client.get('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+key})
    if response.status_code!=200:raise ValueError('Account snapshot unavailable')
    data=response.json()['data']
frozen=read_json(root/'protocol_frozen.json');before=frozen['account_before']
audit['account_before']=before;audit['account_after']={k:data[k] for k in ('total_credits','total_usage')}
audit['account_delta_usd']=data['total_usage']-before['total_usage']
audit['remaining_usd']=data['total_credits']-data['total_usage']
audit['recorded_experiment_cost_usd']=sum(v['reported_cost_usd'] for v in audit['api'].values())
write_json(root/'report/integrity_audit.json',audit)
files=list((root/'report').glob('*'))+list(root.glob('*_metadata.json'))
files+=[root/n for n in ('protocol_frozen.json','operating_points.json','gate_models.json','certification.json','eligibility.json','budget.json','development_review.csv','ranked.jsonl','jev.jsonl','gpt.jsonl')]
files+=[project/'config.top1-fusion-v3.toml',project/'docs/EXPERIMENT_REPORT.md']
files+=sorted((project/'jev_gate').glob('*.py'))
bundle=project.parent/'jev-gate-fusion-v3-results.zip'
with ZipFile(bundle,'w',ZIP_DEFLATED) as z:
    for p in files:
        name=('source/'+p.name) if p.parent.name=='jev_gate' else (p.name if p.parent==project else str(p.relative_to(root)).replace('\\','/'))
        assert p.name!='.env'
        z.write(p,'jev-gate-fusion-v3/'+name)
print(json.dumps({'audit':audit,'completeness':report['completeness'],'workpoints':report['workpoints'],
    'metrics':{k:{kk:vv for kk,vv in v.items() if kk!='reliability'} for k,v in report['metrics'].items()},
    'ap_comparisons':report['ap_comparisons'],'bundle':str(bundle)},indent=2))
