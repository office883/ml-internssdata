from __future__ import annotations

from pathlib import Path

from heocr_unified.cli import _parser


def test_cli_has_one_command_run_and_no_deprecated_uploader() -> None:
    args=_parser().parse_args(["run","--no-upload"])
    assert args.command=="run" and args.no_upload
    assert _parser().parse_args(["probe-upload"]).command == "probe-upload"
    launcher=Path("RUN_ME.command").read_text(encoding="utf-8")
    assert "upload-large-folder" not in launcher
    assert "heocr_unified run" in launcher
    assert "python3.12 python3" in launcher
    assert "Python 3.12.x is required" in launcher
    assert "python3.11" not in launcher
    assert "python3.13" not in launcher



def test_run_probes_private_write_access_before_any_build(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    import heocr_unified.cli as cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"builder_version":"15.0.0","output_repo":"ssdataanalysis/hebrew-ocr-unified-sota-v1",'
        '"work_dir":"%s","upload":true,"private":true,"deep_remote_verify":true,'
        '"minimum_free_gib":0}' % str(tmp_path / "work"),
        encoding="utf-8",
    )
    events: list[str] = []
    paths = SimpleNamespace(output=tmp_path / "out", state=tmp_path / "state")

    monkeypatch.setattr(cli, "get_token", lambda: "write-token")
    monkeypatch.setattr(
        cli,
        "probe_private_write_access",
        lambda **kwargs: events.append("probe") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        cli,
        "run_local_build",
        lambda config, mini: events.append("mini-build" if mini else "full-build") or (paths, {}),
    )
    monkeypatch.setattr(
        cli,
        "finalize_local_release",
        lambda *args, mini, **kwargs: events.append("mini-finalize" if mini else "full-finalize") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        cli,
        "upload_private_release",
        lambda *args, **kwargs: events.append("upload") or {"status": "PASS"},
    )

    assert cli.main(["run", "--config", str(config_path)]) == 0
    assert events == [
        "probe",
        "mini-build",
        "mini-finalize",
        "full-build",
        "full-finalize",
        "upload",
    ]
