from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import get_token

from .config import load_config
from .finalize import finalize_local_release
from .orchestrator import BuildPaths, prepare_environment, run_local_build
from .upload import upload_private_release
from .verifier import verify_output_dataset


def _parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="heocr-unified",description="Fail-closed unified Hebrew OCR dataset builder")
    parser.add_argument("command",choices=["preflight","mini","build","verify","upload","run"])
    parser.add_argument("--config",default="config.json")
    parser.add_argument("--work-dir")
    parser.add_argument("--output-repo")
    parser.add_argument("--minimum-free-gib",type=float)
    parser.add_argument("--mini",action="store_true",help="verify mini output for the verify command")
    parser.add_argument("--no-upload",action="store_true")
    parser.add_argument("--shallow-remote-verify",action="store_true")
    return parser


def _load(args) -> dict:
    path=Path(args.config)
    overrides={}
    if args.work_dir: overrides["work_dir"]=args.work_dir
    if args.output_repo: overrides["output_repo"]=args.output_repo
    if args.minimum_free_gib is not None: overrides["minimum_free_gib"]=args.minimum_free_gib
    if args.no_upload: overrides["upload"]=False
    if args.shallow_remote_verify: overrides["deep_remote_verify"]=False
    return load_config(path if path.exists() else None,overrides=overrides)


def _print(payload) -> None:
    print(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),flush=True)


def main(argv: Sequence[str]|None=None) -> int:
    args=_parser().parse_args(argv)
    config=_load(args)
    token=get_token() or os.environ.get("HF_TOKEN")
    if args.command=="preflight":
        if not token: raise RuntimeError("Hugging Face authentication is required")
        paths=BuildPaths.create(config["work_dir"],mini=True)
        _,renderer,tasks,inventory=prepare_environment(config,paths,token=token)
        _print({"status":"PASS","tasks":len(tasks),"fonts":len(renderer.fonts),"inventory":{k:v.get("revision") for k,v in inventory.items()}})
        return 0
    if args.command=="mini":
        paths,build=run_local_build(config,mini=True)
        ready=finalize_local_release(paths.output,registry_path=paths.state/"registry.sqlite",config=config,mini=True)
        _print({"build":build,"ready":ready,"output":str(paths.output)})
        return 0
    if args.command=="build":
        paths,build=run_local_build(config,mini=False)
        ready=finalize_local_release(paths.output,registry_path=paths.state/"registry.sqlite",config=config,mini=False)
        _print({"build":build,"ready":ready,"output":str(paths.output)})
        return 0
    if args.command=="verify":
        paths=BuildPaths.create(config["work_dir"],mini=bool(args.mini))
        summary=verify_output_dataset(paths.output,registry_path=paths.state/"registry.sqlite",config=config,mini=bool(args.mini))
        _print(summary); return 0
    if args.command=="upload":
        if not token: raise RuntimeError("Hugging Face authentication is required")
        paths=BuildPaths.create(config["work_dir"],mini=False)
        result=upload_private_release(paths.output,repo_id=config["output_repo"],token=token,deep_verify=bool(config["deep_remote_verify"]))
        _print(result); return 0
    if args.command=="run":
        mini_paths,mini_build=run_local_build(config,mini=True)
        mini_ready=finalize_local_release(mini_paths.output,registry_path=mini_paths.state/"registry.sqlite",config=config,mini=True)
        full_paths,full_build=run_local_build(config,mini=False)
        full_ready=finalize_local_release(full_paths.output,registry_path=full_paths.state/"registry.sqlite",config=config,mini=False)
        result={"mini":{"build":mini_build,"ready":mini_ready},"full":{"build":full_build,"ready":full_ready},"output":str(full_paths.output)}
        if config.get("upload"):
            if not token: raise RuntimeError("Hugging Face authentication is required")
            result["remote"]=upload_private_release(full_paths.output,repo_id=config["output_repo"],token=token,deep_verify=bool(config["deep_remote_verify"]))
        _print(result); return 0
    raise AssertionError(args.command)
