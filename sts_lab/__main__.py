import argparse
import json
from pathlib import Path

from .evaluation import atomic_json, run_suite


def main():
    parser = argparse.ArgumentParser(prog="sts-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="run all five variants against all five families")
    evaluate.add_argument("--split", choices=("development", "held_out"), default="held_out")
    evaluate.add_argument("--output", default="reports")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument("--network", action="store_true")
    doctor_parser.add_argument("--output")
    audio = commands.add_parser("inspect-audio")
    audio.add_argument("path")
    args = parser.parse_args()
    if args.command == "evaluate":
        report = run_suite(args.split, output=args.output)
        print(json.dumps({"selection": report["selection"], "runs": report["run_count"],
                          "results": [{k:s[k] for k in ("variant", "passed", "total")} for s in report["summaries"]]}, indent=2))
    elif args.command == "serve":
        from .server import serve
        serve(args.port)
    elif args.command == "doctor":
        from .runtime import doctor
        result = doctor(args.network)
        if args.output:
            atomic_json(Path(args.output), result)
        print(json.dumps(result, indent=2))
    elif args.command == "inspect-audio":
        from .audio import inspect_wav
        print(json.dumps(inspect_wav(args.path), indent=2))


if __name__ == "__main__":
    main()
