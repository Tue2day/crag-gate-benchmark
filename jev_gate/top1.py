"""Independent Top-1 protocol. Enrichment uses only genuine natural ranking errors."""
from collections import Counter
from pathlib import Path
import random
import time

from .data import FILES, validate_parquet, file_digest, normalize_query, synthetic_groups
from .io import digest, metadata, read_json, read_jsonl, write_json, write_jsonl
from .rerank import TransformerReranker, MockReranker, candidate_text


def load_groups(cfg):
    import pandas as pd
    sources = {}
    for name in FILES:
        p = Path(cfg.data_dir)/name
        validate_parquet(p)
        sources[name] = {"sha256": file_digest(p), "bytes": p.stat().st_size}
    ex = pd.read_parquet(Path(cfg.data_dir)/FILES[0], columns=[
        "query_id", "query", "product_id", "product_locale", "esci_label", "split"])
    products = pd.read_parquet(Path(cfg.data_dir)/FILES[1], columns=[
        "product_id", "product_locale", "product_title", "product_brand", "product_color", "product_bullet_point"])
    ex = ex[(ex.product_locale=="us") & ex.split.isin(["train","test"])].dropna().copy()
    products = products[products.product_locale=="us"].dropna(subset=["product_id","product_title"]).copy()
    products = products[products.product_title.str.strip().ne("")]
    if products.duplicated(["product_id","product_locale"]).any():
        raise ValueError("Duplicate product metadata")
    ex["esci_label"] = ex.esci_label.str.upper()
    if not ex.esci_label.isin(["E","S","C","I"]).all():
        raise ValueError("Unknown judgment")
    if (ex.groupby(["query_id","product_id"]).esci_label.nunique()>1).any():
        raise ValueError("Conflicting judgments")
    ex = ex.drop_duplicates(["query_id","product_id"])
    ex = ex[ex.product_id.isin(products.product_id)]
    groups=[]
    for (split,qid),f in ex.groupby(["split","query_id"],sort=True):
        if f["query"].nunique()!=1:
            raise ValueError("Query ID maps to multiple texts")
        if len(f)>=5:
            groups.append({"query_id":str(qid),"query":str(f["query"].iloc[0]),"source_split":split,
                           "candidates":[{"id":str(p),"label":l} for p,l in zip(f.product_id,f.esci_label)]})
    products["product_id"] = products.product_id.astype(str)
    return groups, products, sources


def partition(groups, cfg, excluded=()):
    seen = set(excluded)
    pools={"train":[],"test":[]}
    for g in sorted(groups,key=lambda g:(g["source_split"]!="train",g["query_id"])):
        text=normalize_query(g["query"])
        if text and text not in seen and len(g["candidates"])>=5:
            seen.add(text)
            pools[g["source_split"]].append(g)
    rng=random.Random(cfg.seed)
    for pool in pools.values(): rng.shuffle(pool)
    if len(pools['train'])<cfg.validation_size+cfg.development_pool_size or len(pools['test'])<cfg.test_size:
        raise ValueError("Insufficient disjoint eligible queries for fixed protocol")
    sections=[('validation',pools['train'][:cfg.validation_size]),
              ('development_pool',pools['train'][cfg.validation_size:cfg.validation_size+cfg.development_pool_size]),
              ('test',pools['test'][:cfg.test_size]),
              ('challenge_pool',pools['test'][cfg.test_size:cfg.test_size+cfg.challenge_pool_size])]
    return [{**g,'split':s,'task':'top1_exact','constructed':False} for s,selected in sections for g in selected]


def prepare(cfg,mock=False):
    if mock:
        prototypes=synthetic_groups(cfg.seed)
        train_n=cfg.development_pool_size+cfg.validation_size+10
        total=train_n+cfg.test_size+cfg.challenge_pool_size+10
        groups=[]
        for i in range(total):
            g=prototypes[i%len(prototypes)]
            groups.append({**g,'query_id':f'm{i}','query':f'Mock query {i}: '+g['query'],
                'source_split':'train' if i<train_n else 'test',
                'candidates':[{**c,'id':f'{i}-{c["id"]}'} for c in g['candidates']]})
        products=None; sources={}; excluded=set(); prior=[]
    else:
        if not cfg.prior_prepared_path:
            raise ValueError("Top-1 requires the first-run prepared artifact for exclusion")
        prior=read_jsonl(cfg.prior_prepared_path)
        previous_meta=read_json(Path(cfg.prior_prepared_path).parent/'prepare_metadata.json')
        previous_ranker=read_json(Path(cfg.prior_prepared_path).parent/'rerank_metadata.json')['identity']
        if previous_meta.get('mock') or digest(prior)!=previous_meta['prepared_fingerprint']:
            raise ValueError('Prior prepared artifact does not match its original real metadata')
        if cfg.reranker_model!=previous_ranker['model'] or cfg.reranker_revision!=previous_ranker['revision']:
            raise ValueError('Top-1 must use the same pinned BGE model commit as the prior run')
        if cfg.root()==Path(cfg.prior_prepared_path).parent:
            raise ValueError("Never overwrite the prior experiment directory")
        excluded={normalize_query(g['query']) for g in prior}
        groups,products,sources=load_groups(cfg)
        if sources!=previous_meta['sources']:
            raise ValueError('ESCI files differ from the first experiment; do not silently change dataset version')
    rows=partition(groups,cfg,excluded)
    if products is not None:
        wanted={c['id'] for g in rows for c in g['candidates']}
        lookup=products[products.product_id.isin(wanted)].set_index('product_id').fillna('').to_dict('index')
        for g in rows:
            for c in g['candidates']:
                p=lookup[c['id']]
                c.update(title=str(p['product_title']),brand=str(p['product_brand']) or None,
                         color=str(p['product_color']) or None,bullet_points=str(p['product_bullet_point']))
    for g in rows: g['mock']=mock
    write_jsonl(cfg.root(mock)/'prepared.jsonl',rows)
    metadata(cfg,mock,'prepare',{'task':cfg.task,'sources':sources,'prior_excluded_queries':len(excluded),
        'prior_artifact_fingerprint':digest(prior),'prepared_fingerprint':digest(rows),
        'counts':dict(Counter(g['split'] for g in rows))})
    print(f"Top-1 prepared: {dict(Counter(g['split'] for g in rows))}; excluded prior queries: {len(excluded)}")


def select_development(pool,cfg):
    classes={y:[g for g in pool if g['y']==y] for y in (0,1)}
    n0=min(cfg.development_negative_target,len(classes[0]))
    if n0<cfg.development_min_negatives:
        raise ValueError(f"Only {n0} genuine natural negatives; need {cfg.development_min_negatives}; paid stage blocked")
    sizes={0:n0,1:cfg.development_size-n0}
    if len(classes[1])<sizes[1]:
        raise ValueError("Insufficient natural positives for fixed development size")
    rng=random.Random(cfg.seed+1)
    selected=[]
    for y in (0,1):
        rng.shuffle(classes[y])
        probability=sizes[y]/len(classes[y])
        selected.extend({**g,'split':'development','sampling_probability':probability,
                         'sampling_weight':1/probability} for g in classes[y][:sizes[y]])
    mean_weight=sum(g['sampling_weight'] for g in selected)/len(selected)
    for g in selected: g['sampling_weight']/=mean_weight
    rng.shuffle(selected)
    return selected


def challenge_variant(scored,counts,cfg):
    quotas={'S':cfg.challenge_s,'C':cfg.challenge_c,'I':cfg.challenge_i}
    for subtype,allowed in [('S',{'S','C','I'}),('C',{'C','I'}),('I',{'I'})]:
        if counts[subtype]>=quotas[subtype]: continue
        candidates=[c for c in scored['candidates'] if c['label'] in allowed][:5]
        if len(candidates)==5 and candidates[0]['label']==subtype:
            return {**scored,'candidates':candidates,'split':'challenge','constructed':True,
                    'challenge_type':subtype,'y':0}
    return None


def rerank(cfg,mock=False):
    root=cfg.root(mock)
    rows=read_jsonl(root/'prepared.jsonl')
    if any(g.get('mock')!=mock or g.get('task')!='top1_exact' for g in rows):
        raise ValueError('Wrong task/mode in prepared artifact')
    ranker=MockReranker() if mock else TransformerReranker(cfg)
    def score(g):
        pairs=[(g['query'],candidate_text(c,cfg)) for c in g['candidates']]
        key=digest({'pairs':pairs,'identity':ranker.identity})
        cache=root/'rerank_cache'/f'{key}.json'
        hit=cache.exists(); start=time.perf_counter()
        values=read_json(cache)['scores'] if hit else ranker.score(pairs)
        if not hit: write_json(cache,{'scores':values,'identity':ranker.identity})
        if len(values)!=len(g['candidates']): raise ValueError('Score length mismatch')
        ordered=sorted(({**c,'ce_score':float(v)} for c,v in zip(g['candidates'],values)),key=lambda c:(-c['ce_score'],c['id']))
        return {**g,'candidates':ordered,'y':int(ordered[0]['label']=='E'),
                'rerank_cache_hit':hit,'rerank_elapsed_ms':(time.perf_counter()-start)*1000}
    screened=[]
    pool=[g for g in rows if g['split']=='development_pool']
    for i,g in enumerate(pool):
        r=score(g); r['candidates']=r['candidates'][:5]; screened.append(r)
        if (i+1)%100==0: print(f'Development screening {i+1}/{len(pool)}',flush=True)
    write_jsonl(root/'screened_development.jsonl',screened)
    try:
        dev=select_development(screened,cfg)
    except ValueError as exc:
        write_json(root/'eligibility.json',{'ready':False,'reason':str(exc),'task':cfg.task})
        raise
    validations=[]
    for g in (g for g in rows if g['split']=='validation'):
        r=score(g);r['candidates']=r['candidates'][:5]; validations.append(r)
    val_risk=sum(g['y']==0 for g in validations)/len(validations)
    if val_risk<=cfg.target_risk:
        reason=f'Natural validation all-pass risk {val_risk:.4f} already meets primary target; no seed changes; paid stage blocked'
        write_json(root/'eligibility.json',{'ready':False,'reason':reason,'task':cfg.task})
        write_jsonl(root/'ranked_validation_diagnostic.jsonl',validations)
        raise ValueError(reason)
    tests=[]
    for i,g in enumerate(g for g in rows if g['split']=='test'):
        r=score(g);r['candidates']=r['candidates'][:5];tests.append(r)
        if (i+1)%100==0: print(f'Natural test reranking {i+1}/{cfg.test_size}',flush=True)
    challenges=[];counts=Counter();checked=0
    quota=cfg.challenge_s+cfg.challenge_c+cfg.challenge_i
    for g in (g for g in rows if g['split']=='challenge_pool'):
        if len(challenges)>=quota: break
        checked+=1
        variant=challenge_variant(score(g),counts,cfg)
        if variant:
            counts[variant['challenge_type']]+=1;challenges.append(variant)
        if checked%100==0: print(f'Challenge screen {checked}: {dict(counts)}',flush=True)
    result=dev+validations+tests+challenges
    write_jsonl(root/'ranked.jsonl',result)
    write_json(root/'eligibility.json',{'ready':True,'task':cfg.task,'validation_all_pass_risk':val_risk,
        'development_n':len(dev),'natural_negative_pool':sum(g['y']==0 for g in screened),
        'ranked_fingerprint':digest(result)})
    metadata(cfg,mock,'rerank',{'task':cfg.task,'identity':ranker.identity,
        'effective_batch_size':ranker.batch_size,'prepared_fingerprint':digest(rows),
        'ranked_fingerprint':digest(result),'screened_development_fingerprint':digest(screened),
        'screening_class_counts':dict(Counter(g['y'] for g in screened)),
        'development_class_counts':dict(Counter(g['y'] for g in dev)),
        'challenge_counts':dict(counts),'challenge_screened':checked,
        'development_no_exact_fraction':sum(g['y']==0 for g in dev)/len(dev)})
    # A reproducible evidence review sheet; blank review fields are not human annotations.
    import csv
    review=[]
    for y in (0,1): review.extend([g for g in dev if g['y']==y][:40])
    with (root/'development_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['query_id','query','title','official_label','reviewer_label','notes']);w.writeheader()
        for g in review: w.writerow({'query_id':g['query_id'],'query':g['query'],
                                    'title':g['candidates'][0]['title'],'official_label':g['candidates'][0]['label']})
    print(f'Top-1 ready: {len(result)} selected queries, development {dict(Counter(g["y"] for g in dev))}, challenge {dict(counts)}',flush=True)
