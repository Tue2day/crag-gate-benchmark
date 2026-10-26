"""Pre-registered fusion protocol: separate fit/select/certification/final-test sets."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os
import random
import time

import numpy as np
from .config import Config
from .data import normalize_query, synthetic_groups
from .io import digest, metadata, read_json, read_jsonl, write_json, write_jsonl
from .baselines import features, FEATURE_NAMES
from .metrics import (choose_threshold, accepted_mask, gate_metrics, bad_average_precision,
                      paired_coverage_interval, paired_ap_interval, reliability)
from .evaluate import write_csv, percentile_summary
from .top1 import load_groups, challenge_variant
from .rerank import TransformerReranker, MockReranker, candidate_text

METHODS=('all_pass','ce_threshold','logistic','jev','gpt','jev_fusion','gpt_fusion')
GATES=METHODS[1:]
PHASES=({'fit','select'},{'certification'},{'test','challenge'})


def partition(groups,cfg,excluded):
    seen=set(excluded); pools={'train':[],'test':[]}
    for g in sorted(groups,key=lambda g:(g['source_split']!='train',g['query_id'])):
        q=normalize_query(g['query'])
        if q and q not in seen and len(g['candidates'])>=5:
            seen.add(q); pools[g['source_split']].append(g)
    rng=random.Random(cfg.seed)
    for pool in pools.values(): rng.shuffle(pool)
    need_train=cfg.fit_size+cfg.select_size
    need_test=cfg.certification_size+cfg.test_size+cfg.challenge_pool_size
    if len(pools['train'])<need_train or len(pools['test'])<need_test:
        raise ValueError('Insufficient unused eligible queries; do not change sample size or seed automatically')
    sections=[('fit',pools['train'][:cfg.fit_size]),('select',pools['train'][cfg.fit_size:need_train]),
              ('certification',pools['test'][:cfg.certification_size]),
              ('test',pools['test'][cfg.certification_size:cfg.certification_size+cfg.test_size]),
              ('challenge_pool',pools['test'][cfg.certification_size+cfg.test_size:need_test])]
    return [{**g,'split':s,'task':'top1_exact','experiment_protocol':'fusion_v3','constructed':False}
            for s,rows in sections for g in rows]


def prepare(cfg,mock=False):
    root=cfg.root(mock)
    if (root/'protocol_frozen.json').exists(): raise ValueError('Protocol frozen; never prepare again in this directory')
    excluded=set(); previous=[]; sources={}
    if mock:
        prototypes=synthetic_groups(cfg.seed)
        ntrain=cfg.fit_size+cfg.select_size+10
        total=ntrain+cfg.certification_size+cfg.test_size+cfg.challenge_pool_size+10
        groups=[]
        for i in range(total):
            g=prototypes[i%len(prototypes)]
            groups.append({**g,'query_id':f'v3m{i}','query':f'Fusion synthetic query {i}: '+g['query'],
                'source_split':'train' if i<ntrain else 'test',
                'candidates':[{**c,'id':f'{i}-{c["id"]}'} for c in g['candidates']]})
        products=None
    else:
        if len(cfg.prior_prepared_paths or [])!=2: raise ValueError('Exactly two prior prepared artifacts required')
        groups,products,sources=load_groups(cfg)
        for path in cfg.prior_prepared_paths:
            path=Path(path)
            rows=read_jsonl(path); meta=read_json(path.parent/'prepare_metadata.json')
            rank=read_json(path.parent/'rerank_metadata.json')
            if meta.get('mock') or digest(rows)!=meta['prepared_fingerprint'] or sources!=meta['sources']:
                raise ValueError('Prior data artifact/version mismatch')
            if rank['identity'].get('revision')!=cfg.reranker_revision:
                raise ValueError('BGE revision differs from prior experiments')
            if root==path.parent: raise ValueError('Cannot overwrite prior run')
            excluded.update(normalize_query(g['query']) for g in rows)
            previous.append({'path':str(path),'fingerprint':digest(rows),'n':len(rows)})
    rows=partition(groups,cfg,excluded)
    if products is not None:
        ids={c['id'] for g in rows for c in g['candidates']}
        lookup=products[products.product_id.isin(ids)].set_index('product_id').fillna('').to_dict('index')
        for g in rows:
            for c in g['candidates']:
                p=lookup[c['id']]
                c.update(title=str(p['product_title']),brand=str(p['product_brand']) or None,
                         color=str(p['product_color']) or None,bullet_points=str(p['product_bullet_point']))
    for g in rows: g['mock']=mock
    write_jsonl(root/'prepared.jsonl',rows)
    metadata(cfg,mock,'prepare',{'task':cfg.task,'experiment_protocol':cfg.protocol,'sources':sources,
        'prior_artifacts':previous,'excluded_normalized_queries':len(excluded),
        'prepared_fingerprint':digest(rows),'counts':dict(Counter(g['split'] for g in rows))})
    print('Fusion prepared: '+str(dict(Counter(g['split'] for g in rows))),flush=True)


def rerank(cfg,mock=False):
    root=cfg.root(mock)
    if (root/'protocol_frozen.json').exists(): raise ValueError('Protocol frozen; cannot rerank')
    rows=read_jsonl(root/'prepared.jsonl')
    ranker=MockReranker() if mock else TransformerReranker(cfg)
    def score(g):
        if g.get('mock')!=mock or g.get('experiment_protocol')!='fusion_v3': raise ValueError('Wrong task/mode')
        pairs=[(g['query'],candidate_text(c,cfg)) for c in g['candidates']]
        key=digest({'pairs':pairs,'identity':ranker.identity}); path=root/'rerank_cache'/f'{key}.json'
        hit=path.exists(); start=time.perf_counter()
        values=read_json(path)['scores'] if hit else ranker.score(pairs)
        if not hit: write_json(path,{'scores':values,'identity':ranker.identity})
        if len(values)!=len(g['candidates']): raise ValueError('Wrong score length')
        ordered=sorted(({**c,'ce_score':float(v)} for c,v in zip(g['candidates'],values)),key=lambda c:(-c['ce_score'],c['id']))
        return {**g,'candidates':ordered,'y':int(ordered[0]['label']=='E'),
                'rerank_cache_hit':hit,'rerank_elapsed_ms':(time.perf_counter()-start)*1000}
    ranked=[]
    for split in ('fit','select','certification','test'):
        current=[]
        for i,g in enumerate(g for g in rows if g['split']==split):
            r=score(g);r['candidates']=r['candidates'][:5];current.append(r)
            if (i+1)%100==0: print(f'{split} rerank {i+1}',flush=True)
        if split=='fit':
            counts=Counter(g['y'] for g in current)
            if min(counts.get(0,0),counts.get(1,0))<cfg.fit_min_class:
                write_json(root/'eligibility.json',{'ready':False,'reason':'Fewer than minimum fit examples in one class','counts':dict(counts)})
                raise ValueError('Natural fit set has fewer than required examples in one class; paid phase blocked')
        ranked.extend(current)
    challenges=[];counts=Counter();checked=0
    for g in (g for g in rows if g['split']=='challenge_pool'):
        if len(challenges)>=cfg.challenge_size: break
        checked+=1; variant=challenge_variant(score(g),counts,cfg)
        if variant: counts[variant['challenge_type']]+=1;challenges.append(variant)
        if checked%100==0: print(f'Challenge screened {checked}: {dict(counts)}',flush=True)
    ranked+=challenges
    write_jsonl(root/'ranked.jsonl',ranked)
    write_json(root/'eligibility.json',{'ready':True,'task':cfg.task,'experiment_protocol':cfg.protocol,
        'ranked_fingerprint':digest(ranked),'fit_counts':dict(Counter(g['y'] for g in ranked if g['split']=='fit'))})
    metadata(cfg,mock,'rerank',{'task':cfg.task,'experiment_protocol':cfg.protocol,'identity':ranker.identity,
        'effective_batch_size':ranker.batch_size,'prepared_fingerprint':digest(rows),
        'ranked_fingerprint':digest(ranked),'challenge_counts':dict(counts),'challenge_screened':checked,
        'counts':dict(Counter(g['split'] for g in ranked))})
    fit=[g for g in ranked if g['split']=='fit']; review=[]
    for y in (0,1): review.extend([g for g in fit if g['y']==y][:40])
    write_csv(root/'development_review.csv',[{'query_id':g['query_id'],'query':g['query'],
        'title':g['candidates'][0]['title'],'official_label':g['candidates'][0]['label'],'reviewer_label':''} for g in review],
        ['query_id','query','title','official_label','reviewer_label'])
    print('Fusion ranked '+str(dict(Counter(g['split'] for g in ranked))),flush=True)


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def load_ranked(cfg,mock):
    groups=read_jsonl(cfg.root(mock)/'ranked.jsonl')
    if len({g['query_id'] for g in groups})!=len(groups) or len({normalize_query(g['query']) for g in groups})!=len(groups):
        raise ValueError('Duplicate query')
    if any(g.get('mock')!=mock or g.get('experiment_protocol')!='fusion_v3' or g['y']!=int(g['candidates'][0]['label']=='E') for g in groups):
        raise ValueError('Wrong protocol/mode/truth')
    return groups


def protocol_check(cfg,mock):
    root=cfg.root(mock); frozen=read_json(root/'protocol_frozen.json')
    if frozen['source_hashes']!=source_hashes() or frozen['config']!=cfg.dict(): raise ValueError('Frozen source/config changed')
    groups=load_ranked(cfg,mock)
    if frozen['ranked_fingerprint']!=digest(groups): raise ValueError('Frozen data changed')
    return groups


def catalog_quote(client,models,method,model):
    entry=models.get(model)
    if entry:
        return {'catalog_id':entry['id'],'pricing':entry['pricing'],'source':'/api/v1/models'}
    if method!='jev': raise ValueError('Pinned model price not available in catalog: '+model)
    # Decision models are omitted from the chat catalog. The documented provider
    # endpoints catalog exposes the dated build and price without a paid call.
    url='https://openrouter.ai/api/v1/models/'+model+'/endpoints'
    response=client.get(url)
    if response.status_code!=200: raise ValueError('Decision-model endpoint catalog unavailable')
    data=response.json().get('data',{})
    endpoints=[e for e in data.get('endpoints',[]) if e.get('provider_name')=='TypeSafe'
               and e.get('name','').split(' | ')[-1]==model and e.get('status')==0]
    if len(endpoints)!=1: raise ValueError('Cannot verify exact dated Jev provider/build price')
    e=endpoints[0]
    return {'catalog_id':data['id'],'pricing':e['pricing'],'source':url,'endpoint_name':e['name']}


def preflight(cfg,mock=False):
    import httpx
    from .jev import request_body
    from .gpt import request_body as gpt_body
    root=cfg.root(mock);groups=load_ranked(cfg,mock)
    if (root/'protocol_frozen.json').exists(): protocol_check(cfg,mock);return
    eligible=read_json(root/'eligibility.json')
    if not eligible.get('ready') or eligible['ranked_fingerprint']!=digest(groups): raise ValueError('Free checks not ready')
    for g in groups:
        a=request_body(g,cfg)['state'];b=json.loads(gpt_body(g,cfg)['messages'][1]['content'])
        if a!=b or a['target_candidate']['id']!=g['candidates'][0]['id']: raise ValueError('API input mismatch')
    estimate=0.; prices={};account=None
    if not mock:
        key=os.environ.get('OPENROUTER_API_KEY','').strip()
        if not key: raise ValueError('Missing OPENROUTER_API_KEY')
        with httpx.Client(timeout=60) as client:
            response=client.get('https://openrouter.ai/api/v1/models');response.raise_for_status()
            models={m['id']:m for m in response.json()['data']}
            for method,model in [('jev',cfg.jev_model),('gpt',cfg.gpt_model)]:
                entry=catalog_quote(client,models,method,model)
                price=entry['pricing'];inp=float(price['prompt'])*1e6;out=float(price['completion'])*1e6
                expected_in=cfg.input_price_per_million if method=='jev' else cfg.gpt_input_price_per_million
                expected_out=0 if method=='jev' else cfg.gpt_output_price_per_million
                if not math.isclose(inp,expected_in,abs_tol=1e-9) or not math.isclose(out,expected_out,abs_tol=1e-9):
                    raise ValueError('Model price changed; review protocol before any paid calls')
                prices[method]={k:v for k,v in entry.items() if k!='pricing'}
                prices[method].update(input_per_million=inp,output_per_million=out)
            response=client.get('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+key})
            if response.status_code!=200: raise ValueError('Cannot verify credits before paid phase')
            data=response.json()['data'];account={k:data[k] for k in ('total_credits','total_usage')}
        # Estimate from previous REAL token receipts per serialized UTF-8 request byte; add 20% contingency.
        oldcfg=Config(**read_json(Path(cfg.prior_prepared_paths[1]).parent/'protocol_frozen.json')['config'])
        oldroot=Path(cfg.prior_prepared_paths[1]).parent
        oldgroups={g['query_id']:g for g in read_jsonl(oldroot/'ranked.jsonl')}
        for method,builder in [('jev',request_body),('gpt',gpt_body)]:
            records=[r for r in read_jsonl(oldroot/f'{method}.jsonl') if r['status']=='success' and not r['mock']]
            nbytes=sum(len(json.dumps(builder(oldgroups[r['query_id']],oldcfg),ensure_ascii=False).encode()) for r in records)
            ratio=sum(r['usage']['input_tokens'] for r in records)/nbytes
            mean_out=sum(r['usage']['output_tokens'] or 0 for r in records)/len(records)
            newbytes=sum(len(json.dumps(builder(g,cfg),ensure_ascii=False).encode()) for g in groups)
            prices[method]['estimated_input_tokens_per_utf8_byte']=ratio
            prices[method]['estimated_mean_output_tokens']=mean_out
            estimate+=(newbytes*ratio*prices[method]['input_per_million']+len(groups)*mean_out*prices[method]['output_per_million'])/1e6
        estimate*=1.20
        if estimate>cfg.preflight_max_estimate_usd: raise ValueError(f'Estimated spend with 20% contingency ${estimate:.4f} exceeds preflight cap; no paid calls')
        if account['total_credits']-account['total_usage']<cfg.api_budget_usd: raise ValueError('Available balance below full experiment budget')
    metadata(cfg,mock,'preflight',{'estimated_total_usd':estimate,'prices':prices,'account_before':account})
    write_json(root/'protocol_frozen.json',{'task':cfg.task,'experiment_protocol':cfg.protocol,'mock':mock,
        'config':cfg.dict(),'source_hashes':source_hashes(),'ranked_fingerprint':digest(groups),
        'prepared_fingerprint':digest(read_jsonl(root/'prepared.jsonl')),
        'created_utc':read_json(root/'preflight_metadata.json')['created_utc'],
        'prompt_hashes':{'jev':digest(request_body(groups[0],cfg)['questions']),
                        'gpt':digest(gpt_body(groups[0],cfg)['messages'][0])},
        'estimated_total_usd':estimate,'prices':prices,'account_before':account})
    print(f'Protocol frozen; forecast including 20% contingency ${estimate:.4f}',flush=True)


def checked_artifact(root,name):
    if not (root/name).exists(): raise ValueError('Missing frozen stage: '+name)
    value=read_json(root/name)
    body={k:v for k,v in value.items() if k!='fingerprint'}
    if digest(body)!=value.get('fingerprint'): raise ValueError('Artifact fingerprint mismatch: '+name)
    if name=='gate_models.json' and value['protocol_fingerprint']!=digest(read_json(root/'protocol_frozen.json')):
        raise ValueError('Fitted model belongs to a different frozen protocol')
    return value


def save_frozen(root,name,value):
    value={**value,'fingerprint':digest(value)}
    path=root/name
    if path.exists() and read_json(path)!=value: raise ValueError('Frozen stage changed; new run directory required: '+name)
    if not path.exists(): write_json(path,value)
    return value


def api_guard(cfg,mock,splits):
    protocol_check(cfg,mock)
    selected=set(splits or [])
    if selected not in PHASES or len(splits or [])!=len(selected): raise ValueError('Use --splits fit select, certification, or test challenge')
    root=cfg.root(mock)
    if selected=={'certification'}:
        frozen=checked_artifact(root,'operating_points.json');fit=checked_artifact(root,'gate_models.json')
        if frozen['model_fingerprint']!=fit['fingerprint']: raise ValueError('Stale frozen model')
    if selected=={'test','challenge'}:
        cert=checked_artifact(root,'certification.json');frozen=checked_artifact(root,'operating_points.json')
        if cert['operating_points_fingerprint']!=frozen['fingerprint']: raise ValueError('Stale audit')
    # Once a later stage is frozen, earlier API stages cannot be rewritten.
    if selected=={'fit','select'} and (root/'gate_models.json').exists(): raise ValueError('Fit/select phase already frozen')
    if selected=={'certification'} and (root/'certification.json').exists(): raise ValueError('Certification already frozen')


def prediction_index(cfg,mock,groups,splits):
    from .jev import request_body,cache_key,ENDPOINT
    from .gpt import request_body as gpt_body,ENDPOINT as GPT_ENDPOINT
    root=cfg.root(mock);index={}
    for method,builder,endpoint in [('jev',request_body,ENDPOINT),('gpt',gpt_body,GPT_ENDPOINT)]:
        rows=read_jsonl(root/f'{method}.jsonl')
        if len({r['query_id'] for r in rows})!=len(rows): raise ValueError('Duplicate API results')
        indexed={r['query_id']:r for r in rows}
        for g in (g for g in groups if g['split'] in splits):
            r=indexed.get(g['query_id'])
            if r is None: raise ValueError('Incomplete API phase; do not freeze results')
            if r.get('task')!=cfg.task or r.get('experiment_protocol')!=cfg.protocol or r.get('mock')!=mock:
                raise ValueError('Mixed task/protocol/mode predictions')
            if r['cache_key']!=cache_key(builder(g,cfg),mock,endpoint): raise ValueError('Stale API prompt or input')
        builds={r['model'] for r in rows if r['status']=='success'}
        if len(builds)>1: raise ValueError('Mixed model builds')
        index[method]=indexed
    return index


def common_rows(groups,index,splits):
    return [g for g in groups if g['split'] in splits and all(index[m].get(g['query_id'],{}).get('status')=='success' for m in ('jev','gpt'))]


def fusion_features(g,probability=None):
    x=features(g)
    if probability is None:return x
    if not np.isfinite(probability) or not 0<=probability<=1: raise ValueError('Invalid fusion probability')
    p=np.clip(probability,1e-4,1-1e-4)
    return np.r_[x,np.log(p/(1-p))]


def fitted_model(x,y,cfg):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1.0,max_iter=1000,random_state=cfg.seed).fit(scaler.transform(x),y)
    return {'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
            'coefficients':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
            'C':1.0,'max_iter':1000,'seed':cfg.seed}


def model_probability(model,x):
    from scipy.special import expit
    return float(expit(((x-np.asarray(model['scaler_mean']))/np.asarray(model['scaler_scale']))@np.asarray(model['coefficients'])+model['intercept']))


def fit_gates(cfg,mock=False):
    groups=protocol_check(cfg,mock);root=cfg.root(mock)
    index=prediction_index(cfg,mock,groups,{'fit','select'})
    fit=common_rows(groups,index,{'fit'}); expected=sum(g['split']=='fit' for g in groups)
    y=np.asarray([g['y'] for g in fit])
    if len(fit)/expected<0.99 or min(sum(y==0),sum(y==1))<cfg.fit_min_class: raise ValueError('Insufficient complete fit coverage/classes')
    models={}
    for method in ('logistic','jev_fusion','gpt_fusion'):
        api=method.split('_')[0]
        x=np.asarray([fusion_features(g,index[api][g['query_id']]['probability'] if method!='logistic' else None) for g in fit])
        models[method]={**fitted_model(x,y,cfg),'features':FEATURE_NAMES+([api+'_probability_logit'] if method!='logistic' else [])}
    save_frozen(root,'gate_models.json',{'task':cfg.task,'experiment_protocol':cfg.protocol,'mock':mock,
        'protocol_fingerprint':digest(read_json(root/'protocol_frozen.json')),
        'fit_query_ids':[g['query_id'] for g in fit],'fit_label_counts':dict(Counter(str(int(v)) for v in y)),
        'fit_input_fingerprint':digest([{'g':g,**{m:index[m][g['query_id']]['probability'] for m in index}} for g in fit]),'models':models})
    print(f'Three models fitted on {len(fit)} natural fit queries only',flush=True)


def scores_for(rows,index,models):
    result={m:[] for m in METHODS};latency={m:[] for m in METHODS}
    for g in rows:
        start=time.perf_counter();x=fusion_features(g);feature_ms=(time.perf_counter()-start)*1000
        result['all_pass'].append(1.);result['ce_threshold'].append(float(x[0]))
        latency['all_pass'].append(0.);latency['ce_threshold'].append(0.)
        for api in ('jev','gpt'):
            r=index[api][g['query_id']];result[api].append(r['probability'])
            latency[api].append(r['total_elapsed_ms'])
        for method in ('logistic','jev_fusion','gpt_fusion'):
            api=method.split('_')[0];start=time.perf_counter()
            xx=fusion_features(g,index[api][g['query_id']]['probability']) if method!='logistic' else x
            result[method].append(model_probability(models[method],xx))
            local_ms=(time.perf_counter()-start)*1000+feature_ms
            latency[method].append(local_ms if method=='logistic' else
                (None if index[api][g['query_id']]['cache_hit'] else (index[api][g['query_id']]['total_elapsed_ms'] or 0)+local_ms))
    return {m:np.asarray(v,dtype=float) for m,v in result.items()},latency


def freeze_thresholds(cfg,mock=False):
    groups=protocol_check(cfg,mock);root=cfg.root(mock);fit=checked_artifact(root,'gate_models.json')
    index=prediction_index(cfg,mock,groups,{'fit','select'});rows=common_rows(groups,index,{'select'})
    if len(rows)/cfg.select_size<0.99: raise ValueError('Selection common coverage below 99%')
    scores,_=scores_for(rows,index,fit['models']);y=[g['y'] for g in rows]
    ops={str(q):{m:choose_threshold(y,scores[m],q*cfg.selection_risk_fraction,cfg.min_accepted) for m in GATES}
         for q in (cfg.target_risk,cfg.secondary_risk)}
    save_frozen(root,'operating_points.json',{'task':cfg.task,'experiment_protocol':cfg.protocol,
        'model_fingerprint':fit['fingerprint'],'selection_n':len(rows),
        'selection_fingerprint':digest({'ids':[g['query_id'] for g in rows],'y':y,'scores':{m:s.tolist() for m,s in scores.items()}}),
        'workpoints':ops})
    print('Candidate thresholds frozen; certification may begin',flush=True)


def cp_upper(errors,n,alpha):
    from scipy.stats import beta
    if type(errors) is not int or type(n) is not int or not 0<=errors<=n or not 0<alpha<1: raise ValueError('Invalid binomial inputs')
    if n==0 or errors==n:return 1.0
    return float(beta.ppf(1-alpha,errors+1,n-errors))


def certify(cfg,mock=False):
    groups=protocol_check(cfg,mock);root=cfg.root(mock);frozen=checked_artifact(root,'operating_points.json');fit=checked_artifact(root,'gate_models.json')
    if frozen['model_fingerprint']!=fit['fingerprint']: raise ValueError('Stale model')
    index=prediction_index(cfg,mock,groups,{'fit','select','certification'});rows=common_rows(groups,index,{'certification'})
    completeness=len(rows)/cfg.certification_size;scores,_=scores_for(rows,index,fit['models']);y=[g['y'] for g in rows]
    alpha=cfg.family_alpha/(len(GATES)*2);out={}
    for q,ops in frozen['workpoints'].items():
        out[q]={}
        for method,op in ops.items():
            s=gate_metrics(y,accepted_mask(scores[method],op));upper=cp_upper(s['bad_accepted'],s['accepted'],alpha)
            passed=op['available'] and completeness>=0.99 and s['accepted']>=cfg.certification_min_accepted and upper<=float(q)
            out[q][method]={**s,'candidate_available':op['available'],'cp_upper':upper,'alpha':alpha,'passed':bool(passed)}
    save_frozen(root,'certification.json',{'task':cfg.task,'experiment_protocol':cfg.protocol,
        'operating_points_fingerprint':frozen['fingerprint'],'model_fingerprint':fit['fingerprint'],
        'n':len(rows),'common_coverage':completeness,'family_tests':len(GATES)*2,'family_alpha':cfg.family_alpha,
        'query_ids':[g['query_id'] for g in rows],'certification_fingerprint':digest({'ids':[g['query_id'] for g in rows],'y':y,'scores':{m:s.tolist() for m,s in scores.items()}}),
        'workpoints':out})
    print('Independent certification frozen: '+str({q:[m for m,s in v.items() if s['passed']] for q,v in out.items()}),flush=True)


def evaluate(cfg,mock=False):
    from .evaluate_top1 import plots
    root=cfg.root(mock);groups=protocol_check(cfg,mock)
    fit=checked_artifact(root,'gate_models.json');frozen=checked_artifact(root,'operating_points.json');cert=checked_artifact(root,'certification.json')
    if frozen['model_fingerprint']!=fit['fingerprint'] or cert['operating_points_fingerprint']!=frozen['fingerprint']: raise ValueError('Stale frozen artifacts')
    index=prediction_index(cfg,mock,groups,{'fit','select','certification','test','challenge'})
    test=common_rows(groups,index,{'test'});challenge=common_rows(groups,index,{'challenge'})
    y=np.array([g['y'] for g in test]);scores,latency=scores_for(test,index,fit['models']);cscores,_=scores_for(challenge,index,fit['models'])
    base={};costs={};warnings=[]
    alltest=[g for g in groups if g['split']=='test'];coverage=len(test)/len(alltest)
    if coverage<0.99:warnings.append('Common successful natural-test coverage below 99%; exploratory only')
    if sum(y==0)<50:warnings.append('Fewer than 50 natural-test negatives; exploratory only')
    if cert['common_coverage']<0.99:warnings.append('Risk certification common coverage below 99%; not certified')
    for method in METHODS:
        brier=float(np.mean((scores[method]-y)**2)) if method not in ('all_pass','ce_threshold') else None
        base[method]={'n':len(test),'bad_average_precision':bad_average_precision(y,scores[method]),'brier':brier,
            'reliability':reliability(y,scores[method]) if brier is not None else None,
            'latency_gate':percentile_summary([v for v in latency[method] if v is not None]),
            'latency_http':None}
    for method in ('jev','gpt'):
        rows=list(index[method].values());uncached=[index[method][g['query_id']] for g in alltest if not index[method][g['query_id']]['cache_hit']]
        base[method]['latency_http']=percentile_summary([a['elapsed_ms'] for r in uncached for a in r['attempts']])
        base[method]['latency_gate']=percentile_summary([r['total_elapsed_ms'] for r in uncached if r['total_elapsed_ms'] is not None])
        costs[method]={'requests':len(rows),'successful':sum(r['status']=='success' for r in rows),
            'attempts':sum(len(r['attempts']) for r in rows),'reported_cost_usd':sum(r.get('known_attempt_cost_usd',r['usage'].get('reported_cost_usd') or 0) for r in rows),
            'input_tokens':sum(r['usage'].get('input_tokens') or 0 for r in rows),'output_tokens':sum(r['usage'].get('output_tokens') or 0 for r in rows),
            'unknown_failed_attempt_cost':any(r['cost_may_exclude_failed_attempts'] for r in rows)}
    workpoints={};summary=[]
    for q,ops in frozen['workpoints'].items():
        ops={**ops,'all_pass':{'available':True,'threshold':0.}}
        masks={m:accepted_mask(scores[m],ops[m]) for m in METHODS};entries={}
        for method in METHODS:
            s=gate_metrics(y,masks[method]);op=ops[method];passed=method!='all_pass' and cert['workpoints'][q][method]['passed']
            bytype={}
            for subtype in ('S','C','I'):
                mask=np.array([g['challenge_type']==subtype for g in challenge],dtype=bool)
                decisions=accepted_mask(cscores[method][mask],op)
                bytype[subtype]={'n':int(mask.sum()),'intercept_rate':float((~decisions).mean()) if len(decisions) and op['available'] else None}
            c=accepted_mask(cscores[method],op)
            entries[method]={**s,'candidate_available':op['available'],'certified':passed,
                'operationally_available':method=='all_pass' or passed,'threshold':op['threshold'],
                'challenge_by_type':bytype,'challenge_intercept_rate':float((~c).mean()) if len(c) and op['available'] else None}
            summary.append({'mode':'SIMULATED' if mock else 'REAL','target_risk':q,'method':method,
                **{k:v for k,v in entries[method].items() if k not in ('challenge_by_type','risk_wilson_95')},
                'bad_average_precision':base[method]['bad_average_precision'],'brier':base[method]['brier']})
        comp={}
        for ref in ('logistic','ce_threshold','jev','gpt','gpt_fusion'):
            valid=ops['jev_fusion']['available'] and ops[ref]['available']
            ci=paired_coverage_interval(masks['jev_fusion'],masks[ref],cfg.bootstrap_samples,cfg.seed) if valid else None
            delta=float((masks['jev_fusion'].astype(float)-masks[ref].astype(float)).mean()) if valid and len(y) else None
            jr=entries['jev_fusion'];rr=entries[ref]
            evidence=(valid and coverage>=0.99 and sum(y==0)>=50 and jr['certified'] and rr['certified'] and jr['risk_wilson_95'] is not None and rr['risk_wilson_95'] is not None
                      and jr['risk_wilson_95'][1]<=float(q) and rr['risk_wilson_95'][1]<=float(q) and ci is not None and ci[0]>0)
            comp['jev_fusion_vs_'+ref]={'coverage_delta':delta,'coverage_ci95':ci,'positive_evidence':bool(evidence),
                                       'primary':ref=='logistic' and float(q)==cfg.target_risk}
        workpoints[q]={'methods':entries,'comparisons':comp}
    ap={f'jev_fusion_vs_{ref}':paired_ap_interval(y,scores['jev_fusion'],scores[ref],cfg.bootstrap_samples,cfg.seed) for ref in ('logistic','ce_threshold','jev','gpt','gpt_fusion')}
    report={'task':cfg.task,'experiment_protocol':cfg.protocol,'mock':mock,'metrics':base,'workpoints':workpoints,'ap_comparisons':ap,
        'costs':costs,'budget':read_json(root/'budget.json') if not mock else None,'warnings':warnings,
        'primary_comparison':'Jev fusion vs logistic at 5%; CP audit passed and both test Wilson upper bounds <=5%; positive paired coverage CI',
        'models':{m:{'requested':cfg.jev_model if m=='jev' else cfg.gpt_model,'resolved':sorted({r['model'] for r in index[m].values() if r['status']=='success'}),
                      'providers':sorted({r.get('provider') for r in index[m].values() if r.get('provider')})} for m in ('jev','gpt')},
        'completeness':{'natural_test_n':len(alltest),'common_test_n':len(test),'common_test_fraction':coverage,'common_test_negatives':int(sum(y==0)),
            'common_challenge_n':len(challenge),'failed_or_missing':{m:sum(index[m][g['query_id']]['status']!='success' for g in alltest) for m in index}},
        'certification':cert,'operating_points_fingerprint':frozen['fingerprint'],'ranked_fingerprint':digest(groups),
        'notes':['Candidate-policy test results are diagnostic when certification failed; not deployable risk-compliant policies.',
                 'Intervals condition on fixed models/thresholds and exclude fitting/selection uncertainty.',
                 'CP audit assumes independent same-distribution queries; not an arbitrary production guarantee.',
                 'Auxiliary AP comparisons exploratory; GPT probability elicited, not token logprobs.',
                 'Natural test unweighted; constructed challenge reported separately; latency includes network/pacing.',
                 'Fusion reuses one API request per query; no extra remote calls.']}
    dest=root/'report';write_json(dest/'metrics.json',report)
    write_csv(dest/'summary.csv',summary,['mode','target_risk','method','candidate_available','certified','operationally_available','accepted','bad_accepted','coverage','risk','threshold','bad_average_precision','brier','challenge_intercept_rate'])
    predictions=[];errors=[]
    for subset,ss in ((test,scores),(challenge,cscores)):
        for i,g in enumerate(subset):
            r={'query_id':g['query_id'],'split':g['split'],'challenge_type':g.get('challenge_type',''),'y':g['y'],'top1_label':g['candidates'][0]['label']}
            for method in METHODS:r[method]=float(ss[method][i])
            predictions.append(r)
            for q,w in workpoints.items():
                for method,s in w['methods'].items():
                    if not s['candidate_available']: continue
                    accepted=r[method]>=s['threshold']
                    if int(accepted)!=g['y']:errors.append({'target_risk':q,'method':method,'certified':s['certified'],'query_id':g['query_id'],'split':g['split'],
                        'query':g['query'],'title':g['candidates'][0]['title'],'label':g['candidates'][0]['label'],'score':r[method],
                        'kind':'bad_pass' if accepted else 'unnecessary_fallback'})
    write_csv(dest/'query_predictions.csv',predictions,['query_id','split','challenge_type','y','top1_label',*METHODS])
    write_csv(dest/'errors.csv',errors,['target_risk','method','certified','query_id','split','query','title','label','score','kind'])
    manual=random.Random(cfg.seed).sample(alltest,min(100,len(alltest)))
    write_csv(dest/'manual_audit.csv',[{'query_id':g['query_id'],'query':g['query'],'title':g['candidates'][0]['title'],'official_label':g['candidates'][0]['label'],'reviewer_label':''} for g in manual],
        ['query_id','query','title','official_label','reviewer_label'])
    plots(dest,y,scores,mock)
    # The generic plot helper includes the original three probabilities; add the fusion calibration plot.
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5));ax.plot([0,1],[0,1],'--',color='gray')
    for method in ('logistic','jev','gpt','jev_fusion','gpt_fusion'):
        pts=base[method]['reliability'];ax.plot([p['predicted'] for p in pts],[p['observed'] for p in pts],'o-',label=method)
    ax.set(xlabel='Predicted P(Top-1 Exact)',ylabel='Observed Exact fraction',xlim=(0,1),ylim=(0,1),title=('SIMULATED' if mock else 'REAL')+' fusion reliability')
    ax.legend();fig.tight_layout();fig.savefig(dest/'reliability_fusion.png',dpi=160);plt.close(fig)
    metadata(cfg,mock,'evaluate',{'task':cfg.task,'experiment_protocol':cfg.protocol,'ranked_fingerprint':digest(groups)})
    print('Fusion final report: '+str(dest),flush=True)
    return report
