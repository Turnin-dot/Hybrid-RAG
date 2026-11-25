"""Recompute evidence ranks from saved original candidate order; no models."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hybrid_rag.evaluation.retrieval import covered


def evidence_ranks(question, candidates):
    ranges = {}
    first_section = None
    found = [None]*len(question['gold_evidence'])
    for rank, item in enumerate(candidates, 1):
        sid = item['section_id']
        ranges.setdefault(sid, []).extend(item['ranges'])
        if first_section is None and question['section_relevance'].get(sid, 0) > 0:
            first_section = rank
        for i, evidence in enumerate(question['gold_evidence']):
            if found[i] is None and covered(evidence['section_span'], ranges.get(evidence['section_id'], [])):
                found[i] = rank
    hits = [rank for rank in found if rank is not None]
    return dict(first_section_rank=first_section, first_complete_evidence_rank=min(hits) if hits else None,
        evidence_completion_ranks=found)


def distribution(ranks):
    hits = [r for r in ranks if r is not None]
    return dict(total=len(ranks), found=len(hits), missing=len(ranks)-len(hits),
        conditional_mean=statistics.mean(hits) if hits else None,
        conditional_median=statistics.median(hits) if hits else None,
        top1=sum(r <= 1 for r in hits), top5=sum(r <= 5 for r in hits), top10=sum(r <= 10 for r in hits))


def analyze(predictions, questions):
    groups = {}
    details = []
    for row in predictions:
        q = questions[row['query_id']]
        if not q['answerable']:
            continue
        ranks = evidence_ranks(q, row['results'])
        detail = dict(query_id=q['query_id'], strategy=row['strategy'], **ranks)
        details.append(detail)
        groups.setdefault(row['strategy'], []).append(detail)
    summaries = {}
    for strategy, rows in groups.items():
        summaries[strategy] = dict(
            first_evidence=distribution([r['first_complete_evidence_rank'] for r in rows]),
            first_section=distribution([r['first_section_rank'] for r in rows]),
            all_gold_evidence=distribution([rank for r in rows for rank in r['evidence_completion_ranks']]))
    pairs = []
    by = {(r['query_id'], r['strategy']): r for r in details}
    for qid, strategy in by:
        if strategy != 'hybrid' or (qid, 'hybrid_rerank') not in by:
            continue
        before = by[qid, strategy]['first_complete_evidence_rank']
        after = by[qid, 'hybrid_rerank']['first_complete_evidence_rank']
        pairs.append(dict(query_id=qid, before=before, after=after,
            outcome='both_missing' if before is None and after is None else 'new_hit' if before is None
                else 'lost_hit' if after is None else 'improved' if after < before
                else 'worsened' if after > before else 'unchanged'))
    return dict(summary=summaries, per_query=details, paired_hybrid_rerank=pairs,
        paired_counts={key: sum(r['outcome'] == key for r in pairs)
            for key in ['improved', 'worsened', 'unchanged', 'both_missing', 'new_hit', 'lost_hit']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError(args.output)
    dataset = ROOT/'data/eval/hybridrag-en-v2/questions.jsonl'
    questions = {q['query_id']: q for q in map(json.loads, dataset.read_text(encoding='utf-8').splitlines())}
    inputs = [dataset]
    result = dict(definition='Original Chunk positions, no Section compression. Evidence rank is the first prefix fully covering a gold quote via real source ranges. Missing ranks stay null. Means are conditional on found evidence, not 1/MRR.', runs={})
    for split in ['dev', 'test']:
        path = ROOT/f'data/eval/runs/repair-v2/retrieval_ablation_{split}_abcd.json'
        inputs.append(path)
        source = json.loads(path.read_text(encoding='utf-8'))
        result['runs'][split] = analyze(source['predictions'], questions)
    path = ROOT/'data/eval/runs/test80-recovery-20261116-v1/retrieval_diagnostics.jsonl'
    inputs.append(path)
    current = [dict(query_id=r['query_id'], strategy='hybrid_rerank', results=r['candidates'])
        for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())]
    result['runs']['actual_test80'] = analyze(current, questions)
    result['input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    md = ['# 保存候选的证据排名复算\n\n',
        '当前问答使用 Hybrid+Rerank，先重排40个融合候选，再取Top5；另有Top10上下文实验。四策略均已有真实本地模型消融。此处仅离线复算，无新检索或API调用。\n\n',
        '“首条完整证据排名”指按原始Chunk顺序取前缀，首次完整覆盖任一Gold引文的位置；不是所有必需事实均已覆盖。均值仅对命中样本计算，未命中另列，不用1/MRR代替平均排名。\n\n',
        '| 数据 | 策略 | 可答题 | 命中/未命中@40 | 首条证据均值/中位数 | Top5 | Top10 |\n| --- | --- | ---: | --- | --- | --- | --- |\n']
    for split, run in result['runs'].items():
        for strategy, summary in run['summary'].items():
            s = summary['first_evidence']
            md.append(f"| {split} | {strategy} | {s['total']} | {s['found']}/{s['missing']} | {s['conditional_mean']:.4f}/{s['conditional_median']} | {s['top5']}/{s['total']} | {s['top10']}/{s['total']} |\n")
    md.append('\n全部Gold引文逐条的完成排名分布、配对改善/退化题号、首个相关Section排名和输入哈希见同名JSON。多段联合覆盖时记录完成前缀位置，不假装存在单个完整命中Chunk。\n')
    for split in ['dev', 'test']:
        md.append(f"\n{split} 的 C→D 配对计数：{json.dumps(result['runs'][split]['paired_counts'], ensure_ascii=False)}。\n")
    args.output.with_suffix('.md').write_text(''.join(md), encoding='utf-8')
    print(json.dumps({split: run['summary'] for split, run in result['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
