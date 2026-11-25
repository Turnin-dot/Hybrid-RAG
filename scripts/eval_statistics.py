import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.statistics import calibration,paired_bootstrap,mcnemar
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--mode',choices=['calibration','paired'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def load(path):
        records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        if len({r['query_id'] for r in records})!=len(records):raise ValueError('Aggregate repeated runs before paired/calibration analysis')
        return {r['query_id']:r['score'] for r in records}
    x,y=load(a.a),load(a.b)
    if a.mode=='calibration':
        human=[json.loads(s) for s in a.a.read_text(encoding='utf-8').splitlines()]
        if any(not str(r.get('reviewer') or '').strip() or type(r.get('score')) not in (int,float) or r.get('score') not in (0,.5,1) for r in human):raise ValueError('Explicit human reviewer and 0/0.5/1 scores required; pending templates cannot be scored')
    result=calibration(x,y) if a.mode=='calibration' else dict(paired_bootstrap=paired_bootstrap(x,y),mcnemar=mcnemar(x,y) if all(v in (0,1) for v in [*x.values(),*y.values()]) else None)
    if a.mode=='calibration':result.update(status='preliminary_small_sample' if len(x)<40 else 'calibrated_sample',note='Human labels are required; constant-score kappa/correlation is undefined, not perfect.',human_reviewers=sorted({r['reviewer'] for r in human}))
    result['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.a,a.b]}
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
