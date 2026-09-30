from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.evaluation_runner import EvaluationRunner, load_questions
from app.services.verification_service import DEFAULT_FALLBACK_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable benchmark executions across all three query modes.")
    parser.add_argument("--question-id", action="append", help="Run one question ID; repeat to select several.")
    parser.add_argument("--ids", help="Comma-separated question IDs to run.")
    parser.add_argument("--category", help="Run questions in one category.")
    parser.add_argument("--limit", type=int, help="Run at most this many selected questions.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--pace-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Checkpoint/results JSON path.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing checkpoint at --output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1 or args.pace_seconds < 0 or (args.limit is not None and args.limit < 1):
        print("Invalid top-k, pacing interval, or limit.", file=sys.stderr)
        return 2
    question_ids = set(args.question_id or [])
    if args.ids:
        question_ids.update(item.strip() for item in args.ids.split(",") if item.strip())
    output_path = args.output
    if output_path is None:
        if args.resume:
            print("--resume requires an existing --output path.", file=sys.stderr)
            return 2
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = ROOT / "data" / "evaluation_runs" / f"evaluation-{timestamp}.json"
    elif not output_path.is_absolute():
        output_path = ROOT / output_path

    dataset_version, questions = load_questions()
    fallback_version = DEFAULT_FALLBACK_VERSION
    if args.resume and output_path.is_file():
        try:
            checkpoint = json.loads(output_path.read_text(encoding="utf-8"))
            fallback_version = checkpoint.get("configuration", {}).get(
                "faithfulness_fallback_version",
                DEFAULT_FALLBACK_VERSION,
            )
        except (OSError, json.JSONDecodeError) as error:
            print(f"Unable to read checkpoint ({type(error).__name__}): {output_path}.", file=sys.stderr)
            return 1
    runner = EvaluationRunner(
        top_k=args.top_k,
        max_depth=args.max_depth,
        pacing_seconds=args.pace_seconds,
        fallback_version=fallback_version,
    )
    try:
        run = runner.run(
            questions=questions,
            dataset_version=dataset_version,
            output_path=output_path,
            question_ids=question_ids or None,
            category=args.category,
            limit=args.limit,
            resume=args.resume,
        )
    except Exception as error:
        print(f"Evaluation stopped ({type(error).__name__}); checkpoint retained at {output_path}.", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "result_path": str(output_path),
                "summary": run["summary"],
                "configuration": run["configuration"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())