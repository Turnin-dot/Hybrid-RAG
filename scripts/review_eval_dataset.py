"""Import explicit human review decisions into a NEW versioned dataset copy."""
import argparse,hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.dataset import load_questions


def apply_reviews(source,decisions,output,version):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Source/output overlap')
    questions=load_questions(source,allow_unreviewed=True)
    rows=[json.loads(s) for s in Path(decisions).read_text(encoding='utf-8').splitlines()]
    if len({r['query_id'] for r in rows})!=len(rows):raise ValueError('Duplicate decision IDs')
    by={q.query_id:q for q in questions}
    for row in rows:
        if row['query_id'] not in by:raise ValueError('Unknown decision query')
        decision=row['decision']
        if decision not in {'pending_review','approved','rejected'}:raise ValueError('Invalid decision')
        if decision!='pending_review':
            if not row.get('reviewer') or not row['reviewer'].strip():raise ValueError('Human reviewer name required')
            checks=row.get('checks',{})
            keys=['clarity','atomic_facts','evidence_sufficiency','answerability','split_integrity']
            if any(type(checks.get(k)) is not bool for k in keys):raise ValueError('Five explicit boolean checks required for reviewed decisions')
            if decision=='approved' and not all(checks[k] for k in keys):raise ValueError('All human checks must be explicitly confirmed')
            if decision=='rejected' and not row.get('notes','').strip():raise ValueError('Rejected decisions require explanatory notes')
            q=by[row['query_id']];q.annotation_status=decision;q.reviewed_by=row['reviewer'];q.reviewed_at=datetime.now(timezone.utc).isoformat()
    old=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    if not version or version==old['dataset_version']:raise ValueError('A new dataset version is required')
    shutil.copytree(source,output)
    for q in questions:q.dataset_version=version
    def write(name,qs):(output/name).write_text(''.join(q.model_dump_json()+'\n' for q in qs),encoding='utf-8')
    write('questions.jsonl',questions);write('dev.jsonl',[q for q in questions if q.split=='dev']);write('test.jsonl',[q for q in questions if q.split=='test'])
    write('pilot30.jsonl',[q for q in questions if q.query_id in set(old['pilot_ids'])])
    old.update(dataset_version=version,parent_version=old['dataset_version'],review_decisions_sha256=hashlib.sha256(Path(decisions).read_bytes()).hexdigest(),
               status='human_review_complete' if all(q.annotation_status=='approved' for q in questions) else 'partial_human_review')
    # Update review banners without changing the recorded physical line numbers.
    for filename in ['review_all.md','review_pilot30.md']:
        text=(output/filename).read_text(encoding='utf-8').splitlines(keepends=True)
        text[1]=f"Dataset version: {version}. Review status: {old['status']}; per-question decisions are in questions.jsonl.\n"
        (output/filename).write_text(''.join(text),encoding='utf-8')
    old['artifacts_sha256']={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['questions.jsonl','dev.jsonl','test.jsonl','pilot30.jsonl','review_all.md','review_pilot30.md']}
    (output/'metadata.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
    (output/'review_decisions.jsonl').write_bytes(Path(decisions).read_bytes())
    template=output/'experiment_manifest.template.json'
    if template.exists():
        manifest=json.loads(template.read_text(encoding='utf-8'))
        manifest.update(dataset_version=version,dataset_sha256=hashlib.sha256((output/'questions.jsonl').read_bytes()).hexdigest())
        template.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from hybrid_rag.evaluation.dataset import validate_dataset
    report=validate_dataset(output,ROOT)
    (output/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['errors']:raise ValueError('Reviewed dataset failed structural validation')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--version',required=True);a=p.parse_args()
    apply_reviews(a.source,a.decisions,a.output,a.version)
