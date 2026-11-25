"""Deterministic four-file ingestion export, with explicit quality status."""
import hashlib
import json
from pathlib import Path
import subprocess
from collections import Counter
from importlib.metadata import version
from .cleaner import MDXCleaner, CLEANER_VERSION
from .loader import discover
from .parser import SectionParser, PARSER_VERSION
from .chunker import Chunker, CHUNKER_VERSION
from .validator import audit_document, validate_records


def run_pipeline(docs_root, output, *, products=("langchain", "langgraph", "deepagents"),
                 include_md=False, chunk_size=800, chunk_overlap=120, paths=None, strategy="baseline"):
    root, output = Path(docs_root).resolve(strict=True), Path(output).resolve()
    protected = root.parent if root.name == "src" else root
    if output.is_relative_to(protected) or protected.is_relative_to(output):
        raise ValueError("Output overlaps original source repository")
    files = list(discover(root, products, include_md))
    if paths is not None:
        wanted = set(paths)
        files = [(p, f) for p, f in files if f.relative_to(root).as_posix() in wanted]
        if len(files) != len(wanted):
            raise ValueError("Requested input files were not all discovered")
    if not files:
        raise ValueError("Empty input scope")
    if strategy not in {"baseline", "improved"}:
        raise ValueError("Unknown chunking strategy")
    from .improved_chunker import ImprovedChunker, IMPROVED_VERSION
    cleaner, parser = MDXCleaner(root), SectionParser()
    chunker = (ImprovedChunker if strategy == "improved" else Chunker)(chunk_size, chunk_overlap)
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    report = dict(status="pending", manual_review_status="pending", cleaner_version=CLEANER_VERSION,
                  parser_version=PARSER_VERSION, chunker_version=IMPROVED_VERSION if strategy == "improved" else CHUNKER_VERSION, repository_commit=commit,
                  config=dict(products=sorted(products), include_md=include_md, chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy),
                  documents_discovered=len(files), documents_completed=0, documents_skipped=0,
                  sections_generated=0, chunks_generated=0, oversized_chunks=0,
                  redirects=[], failures=[], validation_errors=[], cleaning_findings=[], documents=[], input_sha256={})
    warnings = Counter()
    lengths = []
    ids = {name: set() for name in ("documents", "sections", "chunks")}
    handles = {}
    try:
        for name in ids:
            handles[name] = (output / (name + ".jsonl")).open("w", encoding="utf-8", newline="\n")
        for product, path in files:
            relative = path.relative_to(root).as_posix()
            try:
                data = path.read_bytes()
                report["input_sha256"][relative] = hashlib.sha256(data).hexdigest()
                raw = cleaner.from_text(data.decode("utf-8-sig"), source_path=relative, product=product, repository_commit=commit)
                doc = cleaner.clean(raw)
                if doc.document_kind == "redirect":
                    report["documents_skipped"] += 1
                    report["redirects"].append(dict(source_path=relative, target=doc.redirect_url, reason="redirect_only"))
                    continue
                findings, _ = audit_document(raw, doc, root)
                report["cleaning_findings"].extend(dict(source_path=relative, **finding) for finding in findings)
                sections = parser.parse(doc)
                chunks = [chunk for section in sections for chunk in chunker.split(section)]
                errors = validate_records(doc, sections, chunks, chunk_size)
                report["validation_errors"].extend(dict(source_path=relative, check=e) for e in errors)
                for name, records, key in (("documents", [doc], "doc_id"), ("sections", sections, "section_id"), ("chunks", chunks, "chunk_id")):
                    for record in records:
                        identity = getattr(record, key)
                        if identity in ids[name]:
                            report["validation_errors"].append(dict(source_path=relative, check="duplicate_" + key))
                        ids[name].add(identity)
                        handles[name].write(record.model_dump_json() + "\n")
                warnings.update(w.code for w in doc.warnings)
                lengths.extend(len(c.content) for c in chunks)
                report["documents_completed"] += 1
                report["sections_generated"] += len(sections)
                report["chunks_generated"] += len(chunks)
                report["oversized_chunks"] += sum(c.oversized for c in chunks)
                report["documents"].append(dict(source_path=relative, sections=len(sections), chunks=len(chunks),
                    oversized=sum(c.oversized for c in chunks), warnings=[w.model_dump() for w in doc.warnings]))
                if path.read_bytes() != data:
                    report["validation_errors"].append(dict(source_path=relative, check="source_changed"))
            except Exception as exc:
                report["failures"].append(dict(source_path=relative, error=f"{type(exc).__name__}: {exc}"))
    finally:
        for handle in handles.values():
            handle.close()
    report["warning_counts"] = dict(sorted(warnings.items()))
    report["documents_failed"] = len(report["failures"])
    report["runtime_versions"] = {name: version(name) for name in ("pydantic", "markdown-it-py", "langchain-text-splitters")}
    report["implementation_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))}
    ordered = sorted(lengths)
    report["chunk_lengths"] = {"p50": ordered[int((len(ordered)-1)*.5)] if ordered else 0,
                               "p95": ordered[int((len(ordered)-1)*.95)] if ordered else 0,
                               "max": max(ordered, default=0)}
    report["quality_errors"] = len(report["validation_errors"]) + sum(f["severity"] == "error" for f in report["cleaning_findings"])
    report["status"] = "failed" if report["failures"] or report["quality_errors"] else "review_required" if warnings or report["cleaning_findings"] else "automated_checks_passed"
    (output / "ingestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def refine_export(source, output, tokenizer_path, *, input_chunks=None,
                  target_tokens=384, max_tokens=512, token_overlap=32,input_prefix=''):
    """Refine a frozen four-file export without rerunning Cleaner or Parser.

    Short Chunk records are copied verbatim. New fragment indices use spare
    Section-local indices so existing records remain entirely unchanged; the
    JSONL order and source offsets retain reading order.
    """
    from collections import defaultdict
    from transformers import AutoTokenizer
    from hybrid_rag.schemas.document import Document, Section, Chunk
    from .token_chunking import TokenRefiner, TOKEN_CHUNKER_VERSION

    source,output=Path(source).resolve(strict=True),Path(output).resolve()
    chunks_path=Path(input_chunks).resolve(strict=True) if input_chunks else source/'chunks.jsonl'
    if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Source/output overlap')
    if output.is_relative_to(Path(tokenizer_path).resolve()):raise ValueError('Output overlaps tokenizer')
    protected=Path(__file__).resolve().parents[3]/'docs'
    if output.is_relative_to(protected) or protected.is_relative_to(output):raise ValueError('Output overlaps original docs')
    if output.exists():raise FileExistsError(output)
    def load(path,model):return [model.model_validate_json(s) for s in path.read_text(encoding='utf-8').splitlines()]
    documents=load(source/'documents.jsonl',Document)
    sections=load(source/'sections.jsonl',Section)
    originals=load(chunks_path,Chunk)
    if len({c.chunk_id for c in originals})!=len(originals):raise ValueError('Duplicate input Chunk ID')
    by_section=defaultdict(list)
    for c in originals:by_section[c.section_id].append(c)
    if set(by_section)-{s.section_id for s in sections}:raise ValueError('Unknown input Section')
    report=json.loads((source/'ingestion_report.json').read_text(encoding='utf-8'))
    size=report['config']['chunk_size']
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
    options=dict(target_tokens=target_tokens,max_tokens=max_tokens,overlap=token_overlap,chunk_size=size,input_prefix=input_prefix)
    refiner=TokenRefiner(tokenizer,**options)
    baseline=Chunker(size,report['config']['chunk_overlap'])
    def build(worker):return [c for s in sections for c in baseline.refine_tokens(s,by_section[s.section_id],worker)]
    chunks=build(refiner)
    repeated=build(TokenRefiner(tokenizer,**options))
    old={c.chunk_id:c for c in originals}
    raw_lines={json.loads(s)['chunk_id']:s for s in chunks_path.read_text(encoding='utf-8').splitlines()}
    def encode(records):
        return ''.join((raw_lines[c.chunk_id] if c.chunk_id in old and c==old[c.chunk_id] else c.model_dump_json())+'\n' for c in records)
    encoded=encode(chunks)
    repeatable=encoded==encode(repeated)
    errors=[];section_groups=defaultdict(list);chunk_groups=defaultdict(list)
    for s in sections:section_groups[s.doc_id].append(s)
    for c in chunks:chunk_groups[c.doc_id].append(c)
    for d in documents:
        errors.extend(dict(source_path=d.source_path,check=e) for e in validate_records(d,section_groups[d.doc_id],chunk_groups[d.doc_id],size))
    if len({c.chunk_id for c in chunks})!=len(chunks):errors.append(dict(check='duplicate_chunk_id'))
    if not repeatable:errors.append(dict(check='token_refinement_not_repeatable'))
    after_by_id={c.chunk_id:c for c in chunks}
    small=[c for c in originals if refiner.count(input_prefix+c.embedding_text)<=max_tokens]
    unchanged=all(after_by_id.get(c.chunk_id)==(c.model_copy(update={'embedding_text':input_prefix+c.embedding_text}) if input_prefix else c) for c in small)
    if not unchanged:errors.append(dict(check='small_chunk_modified'))
    # Exact parent coverage, including whitespace, is enforced in TokenRefiner;
    # Validator also checks full Section coverage and every rendered structure.
    def distribution(values):
        ordered=sorted(values)
        return {**{f'p{p}':ordered[int((len(ordered)-1)*p/100)] if ordered else 0 for p in (0,50,90,95,99,100)},
                'mean':sum(values)/len(values) if values else 0,'over_max':sum(n>max_tokens for n in values)}
    before=[refiner.count(input_prefix+c.embedding_text) for c in originals]
    after=[refiner.count(c.embedding_text) for c in chunks]
    unresolved_ids={f['chunk_id'] for f in refiner.findings}
    if any(n>max_tokens for n in after) or refiner.findings:errors.append(dict(check='hard_token_limit_failed',over_max=sum(n>max_tokens for n in after)))
    if {c.chunk_id for c,n in zip(chunks,after) if n>max_tokens}!=unresolved_ids:errors.append(dict(check='unreported_token_overflow'))
    report['token_refinement']=dict(version=TOKEN_CHUNKER_VERSION,config=refiner.config,
        tokenizer=dict(path=str(Path(tokenizer_path).resolve()),fingerprint=refiner.fingerprint,
            files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(tokenizer_path).glob('*')) if p.is_file()},
            versions={p:version(p) for p in ('transformers','tokenizers')},add_special_tokens=True,truncation=False,field='embedding_text'),
        source_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'documents.jsonl',source/'sections.jsonl',chunks_path,source/'ingestion_report.json']},
        input_chunks=len(originals),output_chunks=len(chunks),unchanged_small_chunks=len(small),small_chunks_unchanged=unchanged,
        all_original_chunk_characters_preserved=True,
        original_body_order_verified=True,
        before=distribution(before),after=distribution(after),repeatable=repeatable,
        chunks_sha256=hashlib.sha256(encoded.encode('utf-8')).hexdigest(),changes=refiner.changes,unresolved=refiner.findings,
        parent_reassembly=refiner.reconstructions,
        hard_limit_acceptance='passed' if not errors and not any(n>max_tokens for n in after) else 'failed',
        continuation_policy='token_source_continuation marks exact raw source excerpts, including crossing fences, rows or statements; these excerpts are not standalone executable examples. Parent-local ranges record repeated context and allow exact reassembly.',
        per_chunk_tokens=[dict(chunk_id=c.chunk_id,embedding_tokens=n) for c,n in zip(chunks,after)],
        index_policy='Untouched records keep chunk_index. New siblings use spare Section-local indices; JSONL and source offsets retain reading order.',
        overlap_policy='Up to configured Tokens of whole structural units, or recursive prose overlap; repeated fences and headers are additional explicit context. Existing inter-parent overlap is preserved.')
    report['implementation_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    report['chunker_version']=report['chunker_version']+'+'+TOKEN_CHUNKER_VERSION
    report['config']['token_refinement']=refiner.config
    if input_chunks:report['config']['input_chunks_override']=str(chunks_path)
    report['chunks_generated']=len(chunks);report['oversized_chunks']=sum(c.oversized for c in chunks)
    report['validation_errors']=errors
    report['quality_errors']=len(errors)+sum(f['severity']=='error' for f in report.get('cleaning_findings',[]))
    for row in report.get('documents',[]):
        local=[c for c in chunks if c.source_path==row['source_path']]
        row.update(chunks=len(local),oversized=sum(c.oversized for c in local))
    ordered=sorted(len(c.content) for c in chunks)
    report['chunk_lengths']={f'p{p}':ordered[int((len(ordered)-1)*p/100)] if ordered else 0 for p in (50,95)}
    report['chunk_lengths']['max']=max(ordered,default=0)
    report['status']='failed' if report['quality_errors'] or report.get('failures') else 'review_required' if refiner.findings or report.get('warning_counts') or report.get('cleaning_findings') else 'automated_checks_passed'
    report['manual_review_status']='pending'
    output.mkdir(parents=True,exist_ok=False)
    for name in ['documents.jsonl','sections.jsonl']:(output/name).write_bytes((source/name).read_bytes())
    (output/'chunks.jsonl').write_text(encoded,encoding='utf-8',newline='\n')
    (output/'ingestion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
