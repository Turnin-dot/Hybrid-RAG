"""Run the existing Cleaner over a corpus and save evidence for human review.

This is a batch test harness, not a Section Parser or a quality acceptance
decision. Default scope: every MDX under the three OSS product directories.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from markdown_it import MarkdownIt
from hybrid_rag.ingestion.cleaner import CLEANER_VERSION, MDXCleaner

from hybrid_rag.ingestion.validator import digest, trusted_path, source_fences, audit_document

PRODUCTS = ("langchain", "langgraph", "deepagents")




def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_line(handle, value: object) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def git_info(root: Path) -> dict:
    results = {}
    for name, args in (("commit", ["rev-parse", "HEAD"]),
                       ("status", ["status", "--porcelain", "--untracked-files=no"])):
        try:
            process = subprocess.run(["git", "-C", str(root), *args],
                                     capture_output=True, text=True, encoding="utf-8", timeout=30)
            results[name] = process.stdout.strip() if process.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            results[name] = None
    return results




def discover(root: Path, products: list[str], include_md: bool) -> list[tuple[str, Path]]:
    files = []
    for product in products:
        directory = root / "oss" / product
        if not directory.is_dir():
            raise ValueError(f"Missing product directory: {directory}")
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in ({".mdx", ".md"} if include_md else {".mdx"}):
                files.append((product, path))
    return sorted(files, key=lambda item: item[1].relative_to(root).as_posix())






def run_batch(docs_root: Path, output: Path, products: list[str], *, include_md: bool = False,
              strict_warnings: bool = False, quiet: bool = False) -> int:
    root = docs_root.resolve(strict=True)
    output = output.resolve()
    protected = root.parent if root.name == "src" else root
    if output.is_relative_to(protected) or protected.is_relative_to(output):
        raise ValueError("Output must not overlap the original docs repository")
    products = list(dict.fromkeys(products))
    files = discover(root, products, include_md)
    if not files:
        raise ValueError("No documents found; refusing to report an empty run as success")
    output.mkdir(parents=True, exist_ok=False)
    cleaner = MDXCleaner(root)
    git_before = git_info(root)
    rows = []
    warning_codes = Counter()
    finding_counts = Counter()
    document_ids = set()
    hard_errors = 0
    failed = 0
    per_product = {name: Counter() for name in products}
    log_names = ("documents.jsonl", "warnings.jsonl", "quality_findings.jsonl", "failures.jsonl", "redirects.jsonl")
    handles = [None] * len(log_names)
    try:
        for i, name in enumerate(log_names):
            handles[i] = (output / name).open("w", encoding="utf-8", newline="\n")
        documents, warnings_log, findings_log, failures_log, redirects_log = handles
        for number, (product, original) in enumerate(files, 1):
            relative = original.relative_to(root).as_posix()
            row = {"source_path": relative, "product": product, "status": "failed"}
            per_product[product]["discovered"] += 1
            try:
                source = trusted_path(root, relative)
                data = source.read_bytes()
                row["source_sha256"] = digest(data)
                before_path = Path("before") / relative
                after_path = Path("after") / (relative + ".cleaned.md")
                before_artifact = output / before_path
                before_artifact.parent.mkdir(parents=True, exist_ok=True)
                before_artifact.write_bytes(data)
                row["before_file"] = before_path.as_posix()
                raw = cleaner.from_text(data.decode("utf-8-sig"), source_path=relative,
                                        repository_commit=git_before["commit"], product=product)
                result = cleaner.clean(raw)
                repeated = cleaner.clean(raw)
                findings, metrics = audit_document(raw, result, root)
                for dependency in result.dependencies:
                    try:
                        dependency_source = trusted_path(root, dependency.source_path)
                        dependency_bytes = dependency_source.read_bytes()
                        snapshot = output / "before" / dependency_source.relative_to(root)
                        if snapshot.exists() and snapshot.read_bytes() != dependency_bytes:
                            findings.append({"check": "dependency_changed_during_run", "severity": "error",
                                             "detail": "Dependency differs from its earlier snapshot", "dependency": dependency.source_path})
                        elif not snapshot.exists():
                            snapshot.parent.mkdir(parents=True, exist_ok=True)
                            snapshot.write_bytes(dependency_bytes)
                        if digest(dependency_bytes) != dependency.content_hash:
                            findings.append({"check": "dependency_snapshot_mismatch", "severity": "error",
                                             "detail": "Snapshot bytes differ from cleaner dependency hash", "dependency": dependency.source_path})
                    except (OSError, ValueError) as exc:
                        findings.append({"check": "dependency_snapshot_failed", "severity": "error", "detail": str(exc)})
                repeatable = result.model_dump_json() == repeated.model_dump_json()
                if not repeatable:
                    findings.append({"check": "not_repeatable", "severity": "error", "detail": "Two clean calls returned different JSON"})
                if source.read_bytes() != data:
                    findings.append({"check": "source_changed", "severity": "error", "detail": "Original page changed during the run"})
                if result.doc_id in document_ids:
                    findings.append({"check": "duplicate_doc_id", "severity": "error", "detail": "Multiple selected pages share a doc_id"})
                document_ids.add(result.doc_id)
                artifact = output / after_path
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(result.content, encoding="utf-8", newline="\n")
                redirect = result.document_kind == "redirect"
                write_line(redirects_log if redirect else documents, result.to_document().model_dump(mode="json"))
                for warning in result.warnings:
                    write_line(warnings_log, {"document_source_path": relative, **warning.model_dump(mode="json")})
                for finding in findings:
                    write_line(findings_log, {"source_path": relative, **finding})
                counts = Counter(w.code for w in result.warnings)
                checks = Counter(f["check"] for f in findings)
                errors = sum(f["severity"] == "error" for f in findings)
                warning_codes.update(counts)
                finding_counts.update(checks)
                hard_errors += errors
                needs_review = bool(counts or findings)
                row.update(status="skipped" if redirect else "completed", document_kind=result.document_kind,
                           skip_reason="redirect_only" if redirect else None, redirect_url=result.redirect_url,
                           doc_id=result.doc_id, revision_id=result.revision_id,
                           after_file=after_path.as_posix(), cleaned_sha256=digest(result.content.encode("utf-8")),
                           repeatable=repeatable, warning_counts=dict(counts), quality_checks=dict(checks),
                           quality_errors=errors, review_required=needs_review, **metrics)
                per_product[product]["skipped" if redirect else "completed"] += 1
                per_product[product]["with_warnings"] += bool(counts)
                per_product[product]["review_required"] += needs_review
            except Exception as exc:
                failed += 1
                per_product[product]["failed"] += 1
                row.update(error_type=type(exc).__name__, error=str(exc), review_required=True)
                write_line(failures_log, {**row, "traceback": traceback.format_exc()})
            rows.append(row)
            if not quiet:
                print(f"[{number}/{len(files)}] {relative} {row['status']}", flush=True)
    finally:
        for handle in handles:
            if handle is not None:
                handle.close()

    # Include every flagged page, plus a deterministic control sample of up to 30.
    flagged = sorted((row for row in rows if row.get("review_required")),
                     key=lambda row: (row["status"] != "failed", -row.get("quality_errors", 0),
                                      -sum(row.get("warning_counts", {}).values()), row["source_path"]))
    controls = sorted((row for row in rows if row["status"] == "completed" and not row.get("review_required")),
                      key=lambda row: digest(row["source_path"].encode("utf-8")))[:30]
    with (output / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            write_line(handle, row)
    with (output / "review_queue.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in [*flagged, *controls]:
            write_line(handle, {**row, "review_reason": "flagged" if row.get("review_required") else "control_sample"})
    git_after = git_info(root)
    if git_before != git_after:
        hard_errors += 1
    status = "failed" if failed or hard_errors else "review_required" if flagged else "automated_checks_passed"
    report = {
        "report_version": 2,
        "status": status, "manual_review_status": "pending",
        "note": "Automated checks and heuristics do not establish semantic fidelity.",
        "scope": {"products": products, "extensions": [".mdx", ".md"] if include_md else [".mdx"], "target_language": "python"},
        "docs_root": str(root), "output": str(output),
        "repository_before": git_before, "repository_after": git_after,
        "cleaner_version": CLEANER_VERSION,
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("pydantic", "PyYAML", "markdown-it-py")}},
        "implementation_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in (
            Path(__file__).resolve(), ROOT / "src/hybrid_rag/ingestion/cleaner.py", ROOT / "src/hybrid_rag/schemas/document.py")},
        "documents_discovered": len(files), "documents_processed": len(files) - failed,
        "documents_completed": sum(row["status"] == "completed" for row in rows),
        "documents_skipped": sum(row["status"] == "skipped" for row in rows),
        "documents_failed": failed, "quality_errors": hard_errors,
        "documents_with_warnings": sum(bool(row.get("warning_counts")) for row in rows),
        "documents_without_flags": sum(row["status"] == "completed" and not row.get("review_required") for row in rows),
        "warning_counts": dict(sorted(warning_codes.items())),
        "quality_check_counts": dict(sorted(finding_counts.items())),
        "products": {name: dict(counts) for name, counts in per_product.items()},
        "manual_review_queue": {"flagged": len(flagged), "control_samples": len(controls)},
    }
    write_json(output / "ingestion_report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "output", "documents_discovered", "documents_completed", "documents_skipped", "documents_failed", "quality_errors")}, indent=2))
    return 1 if status == "failed" else 2 if strict_warnings and flagged else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=ROOT / "docs/src")
    parser.add_argument("--output", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--products", nargs="+", choices=PRODUCTS, default=list(PRODUCTS))
    parser.add_argument("--include-md", action="store_true", help="Include .md in addition to .mdx")
    parser.add_argument("--strict-warnings", action="store_true", help="Exit 2 if human review is required")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-%fZ")
    output = args.output or ROOT / "data/processed/full_cleaning" / stamp
    try:
        return run_batch(args.docs_root, output, args.products, include_md=args.include_md,
                         strict_warnings=args.strict_warnings, quiet=args.quiet)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
