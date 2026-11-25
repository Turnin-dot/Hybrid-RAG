import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.dataset import validate_dataset
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'data/eval/hybridrag-en-v2');a=p.parse_args()
    report=validate_dataset(a.dataset,ROOT)
    (a.dataset/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(int(bool(report['errors'])))
