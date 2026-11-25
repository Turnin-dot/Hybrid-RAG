# Citation validation retries

`generate()` and `answer_question()` default to `citation_retries=2`: one initial
request plus at most two correction requests (three total). Set 0 to reproduce
the earlier single-request policy, or 3 for at most four requests. Values outside
0..3, non-integers and booleans are rejected before model invocation.

The evaluator exposes `--citation-retries 2` and records this policy in the run
manifest/report. Correction requests reuse the same original question, excerpts
and source numbering, append the last rejected JSON and structured validator
feedback, and request a full corrected answer. The exact-quote validator is not
relaxed; Markdown, whitespace and source numbering remain significant.

Only citation matching/missing-marker/out-of-range-source errors trigger this
policy. Network errors use the existing SDK retry setting; invalid JSON/schema
does not trigger citation retries. SDK retries are separate, so three logical
generation requests can still cause more transport attempts.

Rejected responses and their usage remain in `generation_attempts` and evaluator
`generation_traces.jsonl`, with `attempt`, `output_validation` and error details.
Successful results aggregate reported usage across attempts once; evaluator
statistics sum individual invocation traces, not the aggregate answer again.
Missing usage is explicitly counted. Exhaustion raises ValueError with attached
`generation_attempts`; no rejected result is marked valid and no retrieval or
Judge request is repeated by this mechanism.

This changes recovery policy for NEW runs. Historical Test80 files remain
immutable, and 77/80 first-attempt protocol validity cannot be overwritten by a
later recovered score. Label recovery runs separately and include all calls and
costs. Do not tune retrieval, prompts or validation from these holdout failures.
The original frozen Test80 launcher explicitly keeps --citation-retries 0;
use the general evaluation entry with an explicit retry policy for a new,
separately labelled recovery experiment.
The existing --retry-judge-from does not recover failed generation questions.

Tests are offline mocks and saved-response diagnostics, not evidence that the
real provider will successfully repair all three failed Test80 answers.

## Recover the three historical generation failures

```powershell
python scripts/recover_generation_failures.py --source data/eval/runs/test80-20261116T141248-384266Z --query-ids q058,q077,q090 --citation-retries 2 --output data/eval/runs/test80-recovery-20261116-v1
```

Use an unused output directory for each experiment. This entry reads the saved
selected contexts and rejected responses; it does not load or query the index.
Each historical rejected response counts as attempt 1, so `2` permits at most
two NEW generation calls (attempts 2 and 3), not a new initial call plus retries.
The original question, base prompt, source order, Judge rules and dataset hashes
are checked before calls. Passing generations receive the unchanged grounded-v2
Judge; accepted historical judgments and score objects are retained verbatim.

`generation_report.json` separates `first_attempt` and `after_recovery` protocol
validity rates. These rates do not measure factual correctness. All new calls,
including failed ones, are saved in `recovery_generation_traces.jsonl` and
`recovery_judge_traces.jsonl`; combined traces include the historical calls once.
The merged answer/Judge/automatic-score JSONL files retain original lines as a
byte-for-byte prefix. `verification.json` checks the historical directory stayed
unchanged. Source failures remain in `parent_failures.jsonl`, and any unresolved
new failures keep the new report failed. Partial checkpoints are saved after each
question; an interrupted run without a final report is incomplete.

Read the new `human_review.md` and fill `human_scores.md` independently before
opening `judge_review.md`. All 80 questions are included, with source and answer
locations in `review_locations.json`. Recovery is a different protocol/version
from the original first-attempt experiment; it does not replace the old result.
