"""Top-1 evaluation: natural-test metrics and constructed challenges never pool together."""
from collections import Counter
import csv
import json
import os
import random

import numpy as np

from .evaluate import write_csv, percentile_summary
from .io import digest, metadata, read_json, read_jsonl, write_json
from .jev import request_body, cache_key, ENDPOINT as JEV_ENDPOINT
from .gpt import request_body as gpt_body, ENDPOINT as GPT_ENDPOINT
from .metrics import (choose_threshold,accepted_mask,gate_metrics,bad_average_precision,
                      paired_coverage_interval,paired_ap_interval,reliability,risk_coverage_curve)

METHODS=('all_pass','ce_threshold','logistic','jev','gpt')


def evaluate(cfg,mock=False):
    root=cfg.root(mock)
    groups=read_jsonl(root/'ranked.jsonl')
    fingerprint=digest(groups)
    artifacts={m:read_jsonl(root/f'{m}.jsonl') for m in ('baselines','jev','gpt')}
    for rows in [groups,*artifacts.values()]:
        if len({g['query_id'] for g in rows})!=len(rows): raise ValueError('Duplicate query IDs')
        if any(g.get('mock')!=mock or g.get('task')!='top1_exact' for g in rows):
            raise ValueError('Mixed task/mode rejected')
    identities={}
    for m in artifacts:
        meta=read_json(root/f'{m}_metadata.json')
        if meta['ranked_fingerprint']!=fingerprint or meta.get('task')!='top1_exact':
            raise ValueError(f'Stale or wrong-task {m} artifact')
        if m!='baselines':
            if meta['completed']!=meta['expected']:
                raise ValueError('API run incomplete; do not freeze official workpoints from a partial run')
            identities[m]={'requested':cfg.jev_model if m=='jev' else cfg.gpt_model,
                           'resolved':meta['resolved_models'],'providers':meta.get('providers',[])}
            if len(identities[m]['resolved'])>1: raise ValueError('Mixed model builds')
    baseline_meta=read_json(root/'baselines_metadata.json')
    if baseline_meta['logistic_status']!='available':
        raise ValueError('Top-1 requires a trained two-class logistic baseline')
    index={m:{g['query_id']:g for g in rows} for m,rows in artifacts.items()}
    for g in groups:
        if g['y']!=int(g['candidates'][0]['label']=='E'): raise ValueError('Top-1 truth mismatch')
        if g['query_id'] not in index['baselines']: raise ValueError('Missing baseline')
        for m,builder,endpoint in [('jev',request_body,JEV_ENDPOINT),('gpt',gpt_body,GPT_ENDPOINT)]:
            r=index[m].get(g['query_id'])
            if r and r['cache_key']!=cache_key(builder(g,cfg),mock,endpoint):
                raise ValueError('Current input/prompt differs from recorded prediction')
    common=[g for g in groups if all(index[m].get(g['query_id'],{}).get('status')=='success' for m in ('jev','gpt'))]
    validation=[g for g in common if g['split']=='validation']
    test=[g for g in common if g['split']=='test']
    challenge=[g for g in common if g['split']=='challenge']
    natural_test=[g for g in groups if g['split']=='test']
    def scores(rows,m):
        return np.asarray([index[m][g['query_id']]['probability'] if m in ('jev','gpt')
                           else index['baselines'][g['query_id']][m] for g in rows],dtype=float)
    vy=np.asarray([g['y'] for g in validation],dtype=int)
    y=np.asarray([g['y'] for g in test],dtype=int)
    risks=[cfg.target_risk,cfg.secondary_risk]
    val_data={'task':cfg.task,'risks':risks,'min_accepted':cfg.min_accepted,'models':identities,
              'rows':[{'query_id':g['query_id'],'y':g['y'],
                       **{m:float(scores([g],m)[0]) for m in METHODS}} for g in validation]}
    val_hash=digest(val_data)
    path=root/'operating_points.json'
    if path.exists():
        frozen=read_json(path)
        if frozen['task']!=cfg.task or frozen['validation_hash']!=val_hash:
            raise ValueError('Frozen protocol changed; use a new run directory, never retune on test')
    else:
        frozen={'task':cfg.task,'validation_hash':val_hash,'workpoints':{}}
        for risk in risks:
            ops={m:choose_threshold(vy,scores(validation,m),risk,cfg.min_accepted) for m in METHODS if m!='all_pass'}
            ops['all_pass']={'available':True,'threshold':0.0,'accepted':len(vy),
                             'risk':float((1-vy).mean()) if len(vy) else None}
            frozen['workpoints'][str(risk)]=ops
        write_json(path,frozen)
    warnings=[]
    if sum(g['y']==0 for g in natural_test)<50: warnings.append('Fewer than 50 natural test negatives; results exploratory; no resampling')
    if len(validation)<cfg.validation_size: warnings.append('API failures reduced the common validation set')
    if len(test)<len(natural_test): warnings.append('Quality uses common successful test queries; failures separately reported')
    if len(vy) and (1-vy).mean()<=cfg.target_risk: warnings.append('All-pass already meets primary validation target; no coverage-gain claim')
    base={};costs={}
    for m in METHODS:
        p=scores(test,m)
        base[m]={'n':len(test),'bad_average_precision':bad_average_precision(y,p),
                 'brier':float(np.mean((p-y)**2)) if len(y) and m in ('logistic','jev','gpt') else None,
                 'probability_source':{'logistic':'weighted_supervised_logistic','jev':'jev_native_noul','gpt':'gpt_elicited_json'}.get(m),
                 'reliability':reliability(y,p) if m in ('logistic','jev','gpt') else None}
        if m in ('jev','gpt'):
            records=list(index[m].values())
            uncached=[index[m][g['query_id']] for g in natural_test if g['query_id'] in index[m] and not index[m][g['query_id']]['cache_hit']]
            base[m]['latency_gate']=percentile_summary([r['total_elapsed_ms'] for r in uncached])
            base[m]['latency_http']=percentile_summary([a['elapsed_ms'] for r in uncached for a in r['attempts']])
            charges=[r['usage'].get('reported_cost_usd') for r in records]
            costs[m]={'requests':len(records),'known_charge_records':sum(c is not None for c in charges),
                      'reported_cost_usd':sum(c for c in charges if c is not None),
                      'input_tokens':sum(r['usage'].get('input_tokens') or 0 for r in records),
                      'output_tokens':sum(r['usage'].get('output_tokens') or 0 for r in records),
                      'attempts':sum(len(r['attempts']) for r in records),
                      'unknown_failed_attempt_cost':any(r['cost_may_exclude_failed_attempts'] for r in records)}
        else:
            latency=percentile_summary([index['baselines'][g['query_id']]['latency_ms'][m] for g in natural_test])
            base[m]['latency_gate']=latency;base[m]['latency_http']=None
    workpoints={};rows=[];predictions=[]
    for risk in risks:
        ops=frozen['workpoints'][str(risk)]
        masks={m:accepted_mask(scores(test,m),ops[m]) for m in METHODS}
        result={}
        for m in METHODS:
            s={**gate_metrics(y,masks[m]),'operating_point_available':ops[m]['available'],'threshold':ops[m]['threshold']}
            subgroups={}
            for subtype in ('S','C','I'):
                subset=[g for g in challenge if g['challenge_type']==subtype]
                mask=accepted_mask(scores(subset,m),ops[m])
                subgroups[subtype]={'n':len(subset),'intercept_rate':float((~mask).mean()) if len(mask) and ops[m]['available'] else None}
            s['challenge_by_type']=subgroups
            cm=accepted_mask(scores(challenge,m),ops[m])
            s['challenge_intercept_rate']=float((~cm).mean()) if len(cm) and ops[m]['available'] else None
            result[m]=s
            rows.append({'mode':'SIMULATED' if mock else 'REAL','task':cfg.task,'target_risk':risk,'method':m,
                         **s,'bad_average_precision':base[m]['bad_average_precision'],'brier':base[m]['brier'],
                         'http_p50_ms':(base[m]['latency_http'] or {}).get('p50_ms'),
                         'http_p95_ms':(base[m]['latency_http'] or {}).get('p95_ms'),
                         'gate_p95_ms':base[m]['latency_gate']['p95_ms']})
        comparisons={}
        for reference in ('ce_threshold','logistic','gpt'):
            delta=float((masks['jev'].astype(float)-masks[reference].astype(float)).mean()) if len(y) else None
            ci=paired_coverage_interval(masks['jev'],masks[reference],cfg.bootstrap_samples,cfg.seed)
            jr,rr=result['jev']['risk'],result[reference]['risk']
            claim=(risk==cfg.target_risk and sum(y==0)>=50 and ops['jev']['available'] and ops[reference]['available']
                   and jr is not None and rr is not None and jr<=risk and rr<=risk and ci is not None and ci[0]>0)
            comparisons[f'jev_vs_{reference}']={'coverage_delta':delta,'coverage_ci95':ci,
                                               'primary_positive_evidence':bool(claim)}
        workpoints[str(risk)]={'methods':result,'comparisons':comparisons}
    ap_comparisons={f'jev_vs_{ref}':paired_ap_interval(y,scores(test,'jev'),scores(test,ref),cfg.bootstrap_samples,cfg.seed)
                    for ref in ('ce_threshold','logistic','gpt')}
    operational={}
    for risk in risks:
        operational[str(risk)]={}
        for m in METHODS:
            decisions=[]
            for g in natural_test:
                success=m not in ('jev','gpt') or index[m].get(g['query_id'],{}).get('status')=='success'
                decisions.append(bool(accepted_mask(scores([g],m),frozen['workpoints'][str(risk)][m])[0]) if success else False)
            operational[str(risk)][m]=gate_metrics([g['y'] for g in natural_test],decisions)
    completeness={'natural_test_n':len(natural_test),'natural_test_negatives':sum(g['y']==0 for g in natural_test),
                  'common_test_n':len(test),'common_validation_n':len(validation),'common_challenge_n':len(challenge),
                  'challenge_counts':dict(Counter(g['challenge_type'] for g in groups if g['split']=='challenge')),
                  'failed_or_missing':{m:sum(index[m].get(g['query_id'],{}).get('status')!='success' for g in natural_test) for m in ('jev','gpt')}}
    report={'task':cfg.task,'mock':mock,'warnings':warnings,'models':identities,'metrics':base,
            'workpoints':workpoints,'ap_comparisons':ap_comparisons,'operational_all_test':operational,
            'completeness':completeness,'costs':costs,'budget':read_json(root/'budget.json') if not mock else None,
            'validation_hash':val_hash,'ranked_fingerprint':fingerprint,
            'primary_comparison':'Jev vs CE at 2% empirical validation target; other comparisons exploratory',
            'notes':['Fixed thresholds; bootstrap excludes fitting/threshold uncertainty.',
                     'GPT probability is elicited, not token logprobs; Brier differences do not isolate training effects.',
                     'Natural test unweighted. Constructed challenge never pooled with natural traffic.',
                     'Gate latency includes pacing/retries; HTTP includes network/OpenRouter, not pure inference.',
                     'Two model processes share one local budget via api-all; unknown attempt charges conservatively reserved.']}
    dest=root/'report';write_json(dest/'metrics.json',report)
    write_csv(dest/'summary.csv',rows,['mode','task','target_risk','method','n','accepted','bad_accepted','coverage','risk',
        'operating_point_available','threshold','bad_average_precision','brier','challenge_intercept_rate','http_p50_ms','http_p95_ms','gate_p95_ms'])
    errors=[]
    primary=frozen['workpoints'][str(cfg.target_risk)]
    for g in groups:
        if g['split'] not in ('test','challenge'): continue
        p={'query_id':g['query_id'],'split':g['split'],'challenge_type':g.get('challenge_type',''),
           'y':g['y'],'top1_label':g['candidates'][0]['label']}
        for m in METHODS:
            success=m not in ('jev','gpt') or index[m].get(g['query_id'],{}).get('status')=='success'
            value=float(scores([g],m)[0]) if success else None
            p[m]=value
            if success:
                accepted=bool(accepted_mask([value],primary[m])[0])
                if int(accepted)!=g['y'] and primary[m]['available']:
                    errors.append({'method':m,'query_id':g['query_id'],'split':g['split'],'query':g['query'],
                        'title':g['candidates'][0]['title'],'label':g['candidates'][0]['label'],'score':value,
                        'kind':'bad_pass' if accepted else 'unnecessary_fallback'})
            else:
                errors.append({'method':m,'query_id':g['query_id'],'split':g['split'],'kind':'api_failure'})
        predictions.append(p)
    write_csv(dest/'query_predictions.csv',predictions,['query_id','split','challenge_type','y','top1_label',*METHODS])
    write_csv(dest/'errors.csv',errors,['method','query_id','split','query','title','label','score','kind'])
    audit=random.Random(cfg.seed).sample(natural_test,min(100,len(natural_test)))
    write_csv(dest/'manual_audit.csv',[{'query_id':g['query_id'],'query':g['query'],'title':g['candidates'][0]['title'],
        'label':g['candidates'][0]['label'],'reviewer_label':''} for g in audit],['query_id','query','title','label','reviewer_label'])
    plots(dest,y,{m:scores(test,m) for m in METHODS},mock)
    (dest/'REPORT.md').write_text('# '+('SIMULATED — NOT BENCHMARK RESULTS' if mock else 'REAL Top-1 experiment')+'\n\n'+
        '\n\n'.join(report['notes'])+'\n\n'+ '\n'.join('- '+w for w in warnings)+'\n',encoding='utf-8')
    metadata(cfg,mock,'evaluate',{'task':cfg.task,'ranked_fingerprint':fingerprint,'validation_hash':val_hash})
    print(f'Top-1 report: {dest}',flush=True)
    for w in warnings: print('WARNING: '+w)
    return report


def plots(dest,y,scores,mock):
    cache=dest.parent/'.plot_cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'mpl'));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label='SIMULATED — NOT BENCHMARK RESULTS' if mock else 'REAL Top-1 experiment'
    fig,ax=plt.subplots(figsize=(8,5))
    for m,p in scores.items():
        c,r=risk_coverage_curve(y,p);ax.plot(c,r,label=m,marker='o' if m=='all_pass' else None)
    ax.set(xlabel='Coverage',ylabel='Wrong / accepted',title=label+'\nRisk–coverage, frozen-policy evaluation')
    ax.legend();fig.tight_layout();fig.savefig(dest/'risk_coverage.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.plot([0,1],[0,1],'--',color='gray')
    for m in ('logistic','jev','gpt'):
        points=reliability(y,scores[m]);ax.plot([p['predicted'] for p in points],[p['observed'] for p in points],'o-',label=m)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted P(Top-1 Exact)',ylabel='Observed Exact fraction',title=label+'\nReliability')
    ax.legend();fig.tight_layout();fig.savefig(dest/'reliability.png',dpi=160);plt.close(fig)
