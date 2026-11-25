"""Explicit asset gates for integration tests; synthetic tests remain executable."""
import inspect
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_TESTS = {
    'test_real_document_regression', 'test_real_three_document_pipeline',
    'test_real_failed_layout_regression', 'test_real_official_conditional_variants',
    'test_real_redirect_pages_are_not_quality_failures',
}
HISTORY_TESTS = {
    'test_actual_dev_selection_has_36_answerable_and_4_unanswerable',
    'test_actual_test80_preflight_and_plan_do_not_call_models',
    'test_freeze_rejects_settings_and_file_drift_before_execution',
    'test_test80_entry_has_one_frozen_strategy_and_checks_after_run',
    'test_midrun_freeze_drift_marks_result_failed',
    'test_merge_preserves_77_rows_and_original_contexts_without_retrieval',
    'test_preflight_rejects_accepted_id_without_creating_output',
    'test_offline_replay_preserves_history_and_only_changes_semantic_conflict',
    'test_runner_passes_one_explicit_retry_policy_to_all_three_groups',
}

def pytest_collection_modifyitems(items):
    for item in items:
        name = item.originalname or item.name
        reason = None
        if name in DOC_TESTS and not (ROOT/'docs/src/oss').is_dir():
            reason = 'External upstream docs are intentionally not distributed; see README.'
        elif name in HISTORY_TESTS and not (ROOT/'data/eval/hybridrag-en-v2/questions.jsonl').is_file():
            reason = 'Integration test requires local-only frozen evaluation assets/history.'
        elif name == 'test_actual_hard512_chunks_are_one_to_one' and not (ROOT/'data/processed/hard512-improved-v2/chunks.jsonl').is_file():
            reason = 'Generated corpus is intentionally excluded from the public repository.'
        elif item.path.name == 'test_evaluation.py' and 'DATASET' in inspect.getsource(item.function) and not (ROOT/'data/eval/hybridrag-en-v1/questions.jsonl').is_file():
            reason = 'Frozen evaluation dataset not distributed; synthetic metric tests still run.'
        elif name == 'test_reranker_windows_are_explicit_and_pair_bounded' and not (ROOT/'data/models/bge-base-en-v1.5-tokenizer/tokenizer.json').is_file():
            reason = 'Local tokenizer not downloaded; see README model preparation.'
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
