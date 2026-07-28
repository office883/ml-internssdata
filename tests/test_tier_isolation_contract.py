from __future__ import annotations

import pytest

from heocr_unified.verifier import VerificationError, enforce_config_tier, enforce_acceptance


def _row(tier: str, weight: float) -> dict:
    return {
        "data_tier": tier,
        "label_trust": tier,
        "recommended_sampling_weight": weight,
    }


def test_config_suffix_and_row_tier_must_match_exactly() -> None:
    enforce_config_tier("modern_print_lines", _row("gold", 1.0))
    enforce_config_tier("modern_print_lines_extended", _row("extended", 0.4))
    enforce_config_tier("modern_print_lines_quarantine", _row("quarantine", 0.0))
    with pytest.raises(VerificationError, match="config/tier"):
        enforce_config_tier("modern_print_lines", _row("extended", 0.4))
    with pytest.raises(VerificationError, match="config/tier"):
        enforce_config_tier("modern_print_lines_extended", _row("gold", 1.0))


def test_quarantine_is_never_train_weighted() -> None:
    with pytest.raises(VerificationError, match="quarantine sampling weight"):
        enforce_config_tier("modern_print_lines_quarantine", _row("quarantine", 0.1))
    with pytest.raises(VerificationError, match="positive sampling weight"):
        enforce_config_tier("modern_print_lines", _row("gold", 0.0))


def test_acceptance_uses_gold_counts_not_all_rows() -> None:
    config = {
        "acceptance": {
            "minimum_total_rows": 10,
            "minimum_train_rows": 8,
            "minimum_recognition_lines": 8,
            "minimum_unique_texts": 6,
            "minimum_human_train": 1,
            "minimum_human_validation": 1,
            "minimum_human_test": 1,
            "minimum_architecture_natural_lines": 1,
            "minimum_architecture_structured_lines": 1,
            "minimum_pages": 1,
            "minimum_mixed_bidi": 1,
            "minimum_with_digits": 1,
            "minimum_with_combining_marks": 1,
        }
    }
    summary = {
        "all_rows": 1000,
        "gold_rows": 9,
        "gold_train_rows": 9,
        "gold_recognition_lines": 9,
        "gold_unique_texts": 9,
        "human_train": 1,
        "human_validation": 1,
        "human_test": 1,
        "architecture_natural_lines": 1,
        "architecture_structured_lines": 1,
        "pages": 1,
        "mixed_bidi": 1,
        "with_digits": 1,
        "with_combining_marks": 1,
        "architecture_gold_total": 1,
        "architecture_gold_accounted": 1,
        "leakage_errors": 0,
        "integrity_errors": 0,
    }
    with pytest.raises(VerificationError, match="gold_rows"):
        enforce_acceptance(summary, config=config, mini=False)
