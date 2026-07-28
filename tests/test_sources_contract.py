from __future__ import annotations

from heocr_unified.sources import SourceTask, classify_source_task, sort_evaluation_first


def test_human_eval_is_processed_before_every_train_source() -> None:
    tasks = [
        SourceTask("htr", "r", "stage1_synthetic_pretrain/train-000.parquet", "train"),
        SourceTask("htr", "r", "stage3_human_finetune/test-000.parquet", "test"),
        SourceTask("foundation", "f", "shards/train-modern-natural-000.tar", "train"),
        SourceTask("htr", "r", "stage3_human_finetune/validation-000.parquet", "validation"),
    ]
    ordered = sort_evaluation_first(tasks)
    assert [x.split for x in ordered[:2]] == ["test", "validation"]


def test_classifies_ocr_character_config_without_promoting_to_lines() -> None:
    mapped = classify_source_task(
        SourceTask("ocr", "r", "webdataset/real_handwriting_characters/train/a.tar", "train")
    )
    assert mapped.output_config == "handwriting_real_characters"
    assert mapped.task == "character_recognition"
    assert mapped.granularity == "character"


def test_diffusion_is_extended_not_gold() -> None:
    mapped = classify_source_task(
        SourceTask("htr", "r", "stage2_diffusion_augmentation/train-000.parquet", "train")
    )
    assert mapped.data_tier == "extended"
    assert mapped.output_config == "handwriting_diffusion_lines"


def test_mini_selection_keeps_all_ocr_configs_and_human_splits() -> None:
    from heocr_unified.sources import select_mini_tasks
    tasks = []
    for config in [
        "modern_print_lines", "modern_print_words", "synthetic_handwriting_lines",
        "real_handwriting_characters", "historical_handwriting_lines", "biblical_pointed_lines",
        "historical_print_lines", "lexicographic_print_lines", "rabbinic_print_lines",
    ]:
        tasks.append(SourceTask("ocr", "o", f"webdataset/{config}/train/{config}-train-00000.tar", "train", size=100))
    tasks.extend([
        SourceTask("htr", "h", "stage3_human_finetune/train-00000.parquet", "train", size=100),
        SourceTask("htr", "h", "stage3_human_finetune/validation-00000.parquet", "validation", size=100),
        SourceTask("htr", "h", "stage3_human_finetune/test-00000.parquet", "test", size=100),
    ])
    selected = select_mini_tasks(tasks)
    assert len([task for task in selected if task.family == "ocr"]) == 9
    assert {task.split for task in selected if "stage3_human" in task.path} == {"train", "validation", "test"}


def test_mini_selection_keeps_one_ocr_shard_per_config_and_split() -> None:
    from heocr_unified.sources import select_mini_tasks
    tasks = []
    configs = [
        "modern_print_lines", "modern_print_words", "synthetic_handwriting_lines",
        "real_handwriting_characters", "historical_handwriting_lines", "biblical_pointed_lines",
        "historical_print_lines", "lexicographic_print_lines", "rabbinic_print_lines",
    ]
    for config in configs:
        for split in ("train", "validation", "test"):
            tasks.append(SourceTask(
                "ocr", "o", f"webdataset/{config}/{split}/{config}-{split}-00000.tar",
                split, size=100,
            ))
            tasks.append(SourceTask(
                "ocr", "o", f"webdataset/{config}/{split}/{config}-{split}-00001.tar",
                split, size=200,
            ))
    selected = select_mini_tasks(tasks)
    assert len(selected) == len(configs) * 3
    assert {(task.split, task.path.split("/")[1]) for task in selected} == {
        (split, config) for config in configs for split in ("train", "validation", "test")
    }
