"""Run all 40 Dev questions, including unanswerable cases, for human calibration.

Uses the existing generation evaluator, fixed retrieval settings, and a fresh
output directory. Real API configuration is required unless --dry-run is given.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from run_generation_eval import main as run_evaluation

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--index',type=Path,default=ROOT/'data/retrieval/index-v2')
    parser.add_argument('--dry-run',action='store_true',help='Run retrieval only; no DeepSeek calls or quality scores')
    args=parser.parse_args(argv)
    output=args.output or ROOT/'data/eval/runs'/('dev40-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ'))
    return run_evaluation(['--dataset',str(ROOT/'data/eval/hybridrag-en-v2'),
        '--index',str(args.index),'--subset','dev','--query-ids','all','--expected-count','40',
        '--allow-unreviewed','--provider','deepseek','--model','deepseek-chat',
        '--judge-model','deepseek-chat','--strategy','hybrid_rerank','--candidate-k','40','--top-k','5',
        '--dry-run' if args.dry_run else '--require-api','--output',str(output)])


if __name__=='__main__':raise SystemExit(main())
