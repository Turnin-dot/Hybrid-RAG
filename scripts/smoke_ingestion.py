"""Build the four-file export from checked-in synthetic fixtures, without downloads."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hybrid_rag.ingestion.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/synthetic-smoke')
    args = parser.parse_args()
    source = args.output.parent/(args.output.name+'-source')
    if args.output.exists() or source.exists():
        raise FileExistsError('Use a new output directory for each smoke run')
    for product in ('langchain','langgraph','deepagents'):
        (source/'oss'/product).mkdir(parents=True, exist_ok=True)
    fixture = ROOT/'tests/fixtures/cleaner'
    shutil.copyfile(fixture/'page.mdx', source/'oss/langgraph/page.mdx')
    shutil.copytree(fixture/'snippets', source/'snippets')
    report = run_pipeline(source, args.output, paths=['oss/langgraph/page.mdx'])
    print(json.dumps({key:report[key] for key in ('status','documents_completed','sections_generated','chunks_generated','quality_errors')},indent=2))
    return int(report['status']=='failed')


if __name__ == '__main__':
    raise SystemExit(main())
