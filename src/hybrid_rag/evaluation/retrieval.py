"""Section ranking and strict evidence-span coverage are distinct metrics."""
import math


def covered(target, ranges):
    cursor=target[0]
    for a,b in sorted(ranges):
        if a>cursor:break
        if b>cursor:cursor=b
        if cursor>=target[1]:return True
    return False


def resolve_results(items, sections, chunks):
    resolved=[]
    for item in items:
        if 'chunk_id' in item:
            c=chunks[item['chunk_id']];s=sections[c.section_id]
            ranges=[(p.section_start,p.section_end) for p in c.content_parts] or [(c.section_start,c.section_end)]
            resolved.append(dict(section_id=s.section_id,ranges=ranges))
        else:
            s=sections[item['section_id']]
            # A section ID alone must not pretend a partial Chunk retrieved a whole Section.
            if item.get('scope')=='section':ranges=[(0,len(s.content))]
            elif 'ranges' in item:ranges=item['ranges']
            else:raise ValueError('Section results require scope=section or explicit Section-relative ranges')
            if any(not 0<=a<b<=len(s.content) for a,b in ranges):raise ValueError('Invalid retrieved range')
            resolved.append(dict(section_id=s.section_id,ranges=ranges))
    return resolved


def retrieval_metrics(question, ranked, k):
    if k<1:raise ValueError('K must be positive')
    if not question.answerable:return None
    gold=question.section_relevance
    seen=set();gains=[];ranks=[];ranges={}
    for rank,item in enumerate(ranked[:k],1):
        sid=item['section_id'];ranges.setdefault(sid,[]).extend(item['ranges'])
        grade=gold.get(sid,0) if sid not in seen else 0
        gains.append(2**grade-1)
        if grade:ranks.append(rank)
        seen.add(sid)
    relevant={sid for sid,grade in gold.items() if grade>0}
    matched={sid for sid in seen if gold.get(sid,0)>0}
    hit=[e for e in question.gold_evidence if covered(e.section_span,ranges.get(e.section_id,[]))]
    supported={f for e in hit for f in e.supports}
    required={f.fact_id for f in question.reference_facts if f.required}
    ideal=sorted([2**g-1 for g in gold.values() if g>0],reverse=True)[:k]
    dcg=lambda xs:sum(g/math.log2(i+2) for i,g in enumerate(xs))
    return dict(hit_section=float(bool(matched)),recall_section=len(matched)/len(relevant),
        mrr=1/ranks[0] if ranks else 0,ndcg=dcg(gains)/dcg(ideal) if ideal else 0,
        evidence_hit=float(bool(hit)),required_fact_coverage=len(required & supported)/len(required),
        all_required_facts=float(required<=supported),unjudged_sections=len(seen-relevant))
