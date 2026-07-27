from __future__ import annotations

from pathlib import Path

from heocr_unified.cli import _parser


def test_cli_has_one_command_run_and_no_deprecated_uploader() -> None:
    args=_parser().parse_args(["run","--no-upload"])
    assert args.command=="run" and args.no_upload
    launcher=Path("RUN_ME.command").read_text(encoding="utf-8")
    assert "upload-large-folder" not in launcher
    assert "heocr_unified run" in launcher
    assert "python3.12 python3.11 python3.13" in launcher
    assert "Python 3.11–3.13" in launcher
