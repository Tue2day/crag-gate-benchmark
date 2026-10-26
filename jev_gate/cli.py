import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from .config import Config
from .io import load_env


def doctor(cfg):
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "pandas", "pyarrow", "sklearn", "httpx", "matplotlib", "torch", "transformers")}
    cuda = mps = False
    gpu_name = None
    if packages["torch"]:
        import torch
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
    from .data import FILES
    result = {"python": sys.version.split()[0], "packages": packages,
              "cuda": cuda, "gpu": gpu_name, "mps": mps,
              "api_key_configured": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
              "api_provider": "OpenRouter", "api_key_variable": "OPENROUTER_API_KEY",
              "jev_fixed_model": cfg.jev_model,
              "gpt_model": cfg.gpt_model,
              "data_files_exist": {name: (Path(cfg.data_dir)/name).is_file() for name in FILES},
              "run_dir": cfg.run_dir,
              "note": "Local checks only; no API/network requests. Existence is not Parquet validation."}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ESCI Jev Gate experiment; real and mock outputs are isolated")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "download-data", "prepare", "rerank", "baselines", "jev", "gpt", "api-all", "evaluate", "preflight", "fit-gates", "freeze-thresholds", "certify", "probe", "check-probe", "verify-results"):
        p = commands.add_parser(name)
        p.add_argument("--config", help="TOML config (default: config.example.toml)")
        if name == 'api-all': p.add_argument('--splits',nargs='+')
        if name == 'verify-results':
            p.add_argument('--experiment',choices=['v4','v5'],default='v5')
            p.add_argument('--artifacts',help='Optional exported benchmark directory')
            p.add_argument('--raw-artifacts',help='Optional detailed release ZIP for native-response verification')
        if name not in {"doctor", "download-data"}:
            p.add_argument("--mock", action="store_true", help="Use only the isolated synthetic run")
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify-results':
            from .published_results import verify
            result = verify(args.experiment, args.artifacts, args.raw_artifacts)
            print(json.dumps(result, indent=2))
            return
        cfg = Config.load(args.config)
        load_env()
        if args.config:
            load_env(Path(args.config).resolve().parent / ".env")
        if args.command == "doctor":
            doctor(cfg)
        elif args.command == 'download-data':
            from .data import download
            download(cfg)
        elif cfg.protocol == 'topk_calibrated_v5':
            from . import calibrated
            if args.command=='api-all':calibrated.run(cfg,args.mock,args.splits)
            elif args.command in {'prepare','rerank','preflight','freeze-thresholds','evaluate'}:
                getattr(calibrated,args.command.replace('-','_'))(cfg,args.mock)
            else:raise ValueError('Use prepare, rerank, preflight, api-all --splits calibration, freeze-thresholds, api-all --splits test, evaluate')
        elif cfg.protocol == 'topk_v4' and args.command in {'prepare','rerank','preflight','probe','check-probe','api-all','evaluate'}:
            from . import topk
            if args.command=='api-all':topk.run(cfg,args.mock,args.splits)
            elif args.command=='probe':topk.run(cfg,args.mock,['development'])
            else:getattr(topk,args.command.replace('-','_'))(cfg,args.mock)
        elif args.command == "download-data":
            from .data import download
            download(cfg)
        elif args.command == "prepare":
            if cfg.protocol == 'fusion_v3':
                from .fusion import prepare
            elif cfg.task == "top1_exact":
                from .top1 import prepare
            else:
                from .data import prepare
            prepare(cfg, args.mock)
        elif args.command == "rerank":
            if cfg.protocol == 'fusion_v3':
                from .fusion import rerank
            elif cfg.task == "top1_exact":
                from .top1 import rerank
            else:
                from .rerank import rerank
            rerank(cfg, args.mock)
        elif args.command == "baselines":
            if cfg.protocol in {'fusion_v3','topk_v4'}: raise ValueError('No legacy baselines command for this protocol')
            from .baselines import baselines
            baselines(cfg, args.mock)
        elif args.command in {"jev", "gpt"}:
            if cfg.task in {'top1_exact','topk_filter'}:
                raise ValueError('Use api-all for Top-1 so both models share one budget')
            from .jev import run
            run(cfg, args.mock, method=args.command)
        elif args.command == 'api-all':
            from .jev import run_all
            run_all(cfg,args.mock,splits=args.splits)
        elif args.command in {'preflight','fit-gates','freeze-thresholds','certify'}:
            if cfg.protocol != 'fusion_v3': raise ValueError('Command requires fusion_v3 protocol')
            from . import fusion
            getattr(fusion,args.command.replace('-','_'))(cfg,args.mock)
        elif args.command == "evaluate":
            if cfg.protocol == 'fusion_v3':
                from .fusion import evaluate
            elif cfg.task == 'top1_exact':
                from .evaluate_top1 import evaluate
            else:
                from .evaluate import evaluate
            evaluate(cfg, args.mock)
    except KeyboardInterrupt:
        print("Interrupted. Successful caches can be resumed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        from .jev import AuthenticationError
        if isinstance(exc, (ValueError, FileNotFoundError, ModuleNotFoundError, AuthenticationError)):
            print(f"ERROR: {exc}", file=sys.stderr)
        else:
            # Avoid accidentally echoing authentication headers from third-party exception strings.
            print(f"ERROR: {type(exc).__name__}; inspect configuration, network and data. No mock fallback was used.", file=sys.stderr)
        raise SystemExit(1)
