from __future__ import annotations

import pytest

from heocr_unified.config import load_config
from heocr_unified.verifier import VerificationError, enforce_acceptance


def _passing(config: dict) -> dict:
    a=config["acceptance"]
    return {
        "gold_rows":a["minimum_total_rows"], "gold_train_rows":a["minimum_train_rows"],
        "gold_recognition_lines":a["minimum_recognition_lines"], "gold_unique_texts":a["minimum_unique_texts"],
        "human_train":a["minimum_human_train"], "human_validation":a["minimum_human_validation"],
        "human_test":a["minimum_human_test"], "architecture_primary_lines":a["minimum_architecture_natural_lines"],
        "architecture_extra_variants":a["minimum_architecture_extra_variants"],
        "architecture_structured_lines":a["minimum_architecture_structured_lines"],
        "pages":a["minimum_pages"], "mixed_bidi":a["minimum_mixed_bidi"],
        "with_digits":a["minimum_with_digits"], "with_combining_marks":a["minimum_with_combining_marks"],
        "verified_pointed_rerender":a["minimum_verified_pointed_rerender"],
        "architecture_gold_total":100, "architecture_gold_accounted":100,
        "leakage_errors":0, "integrity_errors":0,
    }


def test_full_acceptance_is_fail_closed() -> None:
    config=load_config(None,overrides={"upload":False})
    summary=_passing(config)
    enforce_acceptance(summary,config=config,mini=False)
    summary["pages"]-=1
    with pytest.raises(VerificationError,match="pages"):
        enforce_acceptance(summary,config=config,mini=False)


def test_mini_acceptance_requires_all_contract_categories_not_full_volume() -> None:
    config=load_config(None,overrides={"upload":False})
    summary={
        "gold_rows":100,"gold_train_rows":50,"gold_recognition_lines":70,"gold_unique_texts":50,
        "human_train":1,"human_validation":1,"human_test":1,
        "architecture_primary_lines":1,"architecture_extra_variants":1,"architecture_structured_lines":1,"pages":1,
        "mixed_bidi":1,"with_digits":1,"with_combining_marks":1,"verified_pointed_rerender":1,
        "architecture_gold_total":1,"architecture_gold_accounted":1,
        "leakage_errors":0,"integrity_errors":0,
        "required_configs_present":True,"required_source_families_present":True,
    }
    enforce_acceptance(summary,config=config,mini=True)
    summary["human_test"]=0
    with pytest.raises(VerificationError,match="human_test"):
        enforce_acceptance(summary,config=config,mini=True)


def test_required_config_presence_accepts_extended_or_quarantine_variants() -> None:
    from heocr_unified.verifier import required_config_families_present

    required = {"historical_print_lines", "modern_print_words", "biblical_pointed_lines"}
    present = {
        "historical_print_lines_extended",
        "modern_print_words_quarantine",
        "biblical_pointed_lines",
    }
    assert required_config_families_present(required, present)
