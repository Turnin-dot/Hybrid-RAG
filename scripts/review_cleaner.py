"""Export before/after evidence for the three agreed real-document regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.ingestion.cleaner import MDXCleaner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/cleaner_review")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    docs = ROOT / "docs"
    commit = subprocess.run(["git", "-C", str(docs), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    cleaner = MDXCleaner(docs / "src")
    summaries = []
    for relative in ("oss/langgraph/persistence.mdx", "oss/langgraph/overview.mdx", "oss/deepagents/subagents.mdx"):
        data = (docs / "src" / relative).read_bytes()
        raw = cleaner.from_text(data.decode("utf-8-sig"), source_path=relative,
                                repository_commit=commit, product=relative.split("/")[1])
        result = cleaner.clean(raw)
        assert result.model_dump_json() == cleaner.clean(raw).model_dump_json()
        assert not any(d.source_path.endswith("-js.mdx") for d in result.dependencies)
        name = Path(relative).stem
        (output / f"{name}.before.mdx").write_bytes(data)
        (output / f"{name}.after.md").write_text(result.content, encoding="utf-8", newline="\n")
        (output / f"{name}.cleaned.json").write_text(result.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        summaries.append({"source_path": relative, "source_sha256": hashlib.sha256(data).hexdigest(),
                          "body_chars_before": len(raw.content), "chars_after": len(result.content),
                          "dependencies": len(result.dependencies), "warnings": [w.model_dump() for w in result.warnings],
                          "revision_id": result.revision_id, "repeatable": True})
    report = {"repository_commit": commit, "cleaner_version": result.cleaner_version,
              "target_language": "python", "documents": summaries}
    (output / "review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
