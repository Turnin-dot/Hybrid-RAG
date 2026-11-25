"""Run the four-file ingestion pipeline. Existing outputs are never overwritten."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybrid_rag.ingestion.pipeline import run_pipeline, refine_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=ROOT / "docs/src")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--include-md", action="store_true")
    parser.add_argument("--paths", nargs="+")
    parser.add_argument("--strategy", choices=("baseline", "improved"), default="baseline")
    parser.add_argument("--from-export", type=Path, help="Refine existing four-file export; skip Cleaner and Parser")
    parser.add_argument("--input-chunks", type=Path, help="Optional existing improved.chunks.jsonl for --from-export")
    parser.add_argument("--tokenizer", type=Path, help="Local selected embedding tokenizer; no model weights loaded")
    parser.add_argument("--target-tokens", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--token-overlap", type=int, default=32)
    parser.add_argument("--input-prefix", default='', help="Actual document input prefix; included in embedding_text and Token budget")
    args = parser.parse_args()
    if args.from_export:
        if args.paths or args.include_md:parser.error('--from-export uses the existing complete export scope')
        report=refine_export(args.from_export,args.output,args.tokenizer or ROOT/'data/models/bge-base-en-v1.5-tokenizer',
            input_chunks=args.input_chunks,target_tokens=args.target_tokens,max_tokens=args.max_tokens,token_overlap=args.token_overlap,input_prefix=args.input_prefix)
    else:
        if args.input_chunks or args.tokenizer:parser.error('--input-chunks/--tokenizer require --from-export')
        report = run_pipeline(args.docs_root, args.output, chunk_size=args.chunk_size,
                              chunk_overlap=args.chunk_overlap, include_md=args.include_md, paths=args.paths, strategy=args.strategy)
    summary={k:report[k] for k in ("status", "documents_completed", "documents_skipped", "sections_generated", "chunks_generated", "oversized_chunks", "quality_errors", "failures")}
    if 'token_refinement' in report:
        t=report['token_refinement']
        summary['tokens']=dict(input_chunks=t['input_chunks'],unchanged_small=t['unchanged_small_chunks'],
            over_max_before=t['before']['over_max'],over_max_after=t['after']['over_max'],maximum=t['after']['p100'],repeatable=t['repeatable'])
    print(json.dumps(summary,indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
