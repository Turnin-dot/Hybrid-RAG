"""Small deterministic context selection over a frozen ordered candidate list."""
import re
from hybrid_rag.retrieval.generation import build_prompt


def select_contexts(question,candidates,strategy,count_tokens,max_contexts=10,prompt_budget=6000):
    if strategy=='top5':return candidates[:5]
    if strategy=='top10':return candidates[:10]
    if strategy!='dynamic':raise ValueError('Unknown context strategy')
    # Prefer diverse Sections within the same ranked pool; never use gold labels.
    selected=[];sections=set();seen=set()
    for unique_only in [True,False]:
        for c in candidates:
            if c['chunk_id'] in seen or unique_only and c['section_id'] in sections:continue
            tokens=set(re.findall(r'\w+',c['text'].lower()))
            if any(len(tokens & t)/max(1,len(tokens | t))>.85 for _,t in selected):continue
            proposal=[r for r,_ in selected]+[c]
            if count_tokens(build_prompt(question,proposal))>prompt_budget:continue
            selected.append((c,tokens));seen.add(c['chunk_id']);sections.add(c['section_id'])
            if len(selected)==max_contexts:break
        if len(selected)==max_contexts:break
    # Return the selection in its original rank order for predictable input.
    return sorted([r for r,_ in selected],key=lambda r:r['rank'])
