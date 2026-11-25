"""Question-level aggregation and deterministic resampling, no fabricated CIs."""
import math
import random
from collections import defaultdict
from statistics import mean, pstdev


def bootstrap(values, seed=42, iterations=2000):
    if iterations<1:raise ValueError('Positive bootstrap iterations required')
    if not values:return None
    rng=random.Random(seed);n=len(values)
    samples=sorted(mean(rng.choices(values,k=n)) for _ in range(iterations))
    return dict(mean=mean(values),ci95=[samples[int(.025*(iterations-1))],samples[int(.975*(iterations-1))]],n=n,seed=seed,iterations=iterations)


def aggregate_runs(rows, metric, fixed_context=True):
    grouped=defaultdict(list)
    seen=set()
    for row in rows:
        key=(row['query_id'],row['run_id'])
        if key in seen:raise ValueError('Duplicate query/run')
        seen.add(key);grouped[row['query_id']].append(row)
    per=[];within=[]
    for qid,rs in grouped.items():
        if fixed_context and len({r['context_snapshot_hash'] for r in rs})!=1:
            raise ValueError('Context changes confound generator stability')
        values=[r[metric] for r in rs if r.get(metric) is not None]
        if values:
            per.append(dict(query_id=qid,mean=mean(values),runs=len(values)))
            if len(values)>1:within.append(pstdev(values))
    return dict(per_question=per,bootstrap=bootstrap([x['mean'] for x in per]),
                between_question_std=pstdev([x['mean'] for x in per]) if per else None,
                mean_within_question_std=mean(within) if within else None)


def paired_bootstrap(a,b,seed=42,iterations=2000):
    if set(a)!=set(b):raise ValueError('Paired comparison requires identical query IDs')
    return bootstrap([b[k]-a[k] for k in sorted(a)],seed,iterations)


def cluster_bootstrap(rows,metric,seed=42,iterations=2000):
    """Average repeated runs per question, resample complete topic families."""
    questions=defaultdict(list);families={}
    for row in rows:
        if row.get(metric) is not None:questions[row['query_id']].append(row[metric])
        families[row['query_id']]=row['topic_group']
    groups=defaultdict(list)
    for qid,values in questions.items():groups[families[qid]].append(mean(values))
    if not groups:return None
    units=list(groups.values());rng=random.Random(seed)
    means=[]
    for _ in range(iterations):
        sample=[value for unit in rng.choices(units,k=len(units)) for value in unit]
        means.append(mean(sample))
    means.sort()
    return dict(mean=mean([x for unit in units for x in unit]),ci95=[means[int(.025*(iterations-1))],means[int(.975*(iterations-1))]],
                questions=len(questions),topic_groups=len(units),seed=seed,iterations=iterations)


def mcnemar(a,b):
    if set(a)!=set(b):raise ValueError('Paired labels required')
    if any(x not in (0,1) for x in [*a.values(),*b.values()]):raise ValueError('Binary outcomes required')
    n01=sum(a[k]==0 and b[k]==1 for k in a);n10=sum(a[k]==1 and b[k]==0 for k in a)
    n=n01+n10
    p=min(1,2*sum(math.comb(n,i) for i in range(min(n01,n10)+1))/2**n) if n else 1
    return dict(n01=n01,n10=n10,exact_two_sided_p=p)


def calibration(human,judge):
    if set(human)!=set(judge) or not human:raise ValueError('Matched nonempty calibration IDs required')
    keys=sorted(human);a=[human[k] for k in keys];b=[judge[k] for k in keys]
    labels=sorted(set(a+b));n=len(a)
    observed=sum(x==y for x,y in zip(a,b))/n
    expected=sum(a.count(x)/n*b.count(x)/n for x in labels)
    span=max(labels)-min(labels)
    weighted_observed=sum((x-y)**2 for x,y in zip(a,b))/n
    weighted_expected=sum((x-y)**2*a.count(x)/n*b.count(y)/n for x in labels for y in labels)
    def ranks(xs):
        ordered=sorted(xs)
        return [(ordered.index(x)+1 + len(ordered)-ordered[::-1].index(x))/2 for x in xs]
    x,y=ranks(a),ranks(b);mx,my=mean(x),mean(y)
    denom=math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
    return dict(n=n,agreement=observed,kappa=(observed-expected)/(1-expected) if expected<1 else None,
        quadratic_weighted_kappa=1-weighted_observed/weighted_expected if weighted_expected else None,
        spearman=sum((v-mx)*(w-my) for v,w in zip(x,y))/denom if denom else None,
        mean_absolute_error=mean(abs(v-w) for v,w in zip(a,b)),
        disagreements=[k for k in keys if human[k]!=judge[k]])
