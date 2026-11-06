#!/usr/bin/env python3
"""Validate one checkpoint exactly as training validates, without training.

Usage:
    python scripts/validate_only.py KEY --initial_clean_r1 C0 [train.py flags ...]

The train.py flags must name the model, the checkpoint (--resume, with --resume_model_only),
the run directory (--run_dir) and an adversarial --attack: the attack is never run, it only
makes validation measure the rank-PGD budgets of --val_rank_epsilons as an adversarial run
does. C0 is the pretrained model's clean R@1, against which the clean drop is measured.

The run directory receives the log files, a validation_recalls.jsonl holding one record with
"epoch": 12 in the trainer's format, and run_status.json "completed", so
scripts/screen_summary.py lists the checkpoint next to 12-epoch training screens. The
checkpoint itself is only read.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.checkpoints import write_run_status
from src.train_loop import build_validation_metrics_record

# The epoch label of the record: screens are compared at their 12th epoch.
SCREEN_EPOCH = 12


def split_arguments(argv: Sequence[str]) -> Tuple[str, float, List[str]]:
    """KEY, the --initial_clean_r1 value and the remaining train.py flags."""
    if not argv or argv[0].startswith("-"):
        raise SystemExit("usage: validate_only.py KEY --initial_clean_r1 C0 [train.py flags ...]")
    key, rest = argv[0], list(argv[1:])
    initial: Optional[str] = None
    remaining: List[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--initial_clean_r1":
            if index + 1 >= len(rest):
                raise SystemExit("--initial_clean_r1 needs a value")
            initial = rest[index + 1]
            index += 2
            continue
        if token.startswith("--initial_clean_r1="):
            initial = token.split("=", 1)[1]
        else:
            remaining.append(token)
        index += 1
    if initial is None:
        raise SystemExit("--initial_clean_r1 is required (the pretrained model's clean R@1)")
    try:
        return key, float(initial), remaining
    except ValueError:
        raise SystemExit(f"--initial_clean_r1 is not a number: {initial}") from None


def write_screen_record(
    run_dir: Path, metrics: Dict[str, object], scores: Dict[str, object], initial_clean_r1: float
) -> Dict[str, object]:
    """Write the record and the completed status; a rerun replaces the earlier record."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = build_validation_metrics_record(SCREEN_EPOCH, metrics, scores, initial_clean_r1=initial_clean_r1)
    (run_dir / "validation_recalls.jsonl").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    write_run_status(run_dir, "completed", final_epoch=SCREEN_EPOCH)
    return record


def result_row(key: str, resume: str, queries: int, metrics: Dict[str, object], scores: Dict[str, object], seconds: float):
    return {
        "label": key,
        "resume": resume,
        "queries": int(queries),
        "clean_R@1": float(metrics["NoAttack"]["recalls"]["R@1"]),
        **{name: float(value["recalls"]["R@1"]) for name, value in metrics.items() if name != "NoAttack"},
        "robust": float(scores["robust_score"]),
        "seconds": round(seconds),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return
    key, initial_clean_r1, train_argv = split_arguments(argv)

    import commons
    from src import supervlad_compat
    from src.cli import parse_arguments
    from src.config import validate_cuda_runtime
    from src.data import setup_datasets
    from src.faiss_utils import validate_faiss_runtime
    from src.models import get_model_adapter
    from src.rank_validation import evaluate_rank_validation, sample_validation_queries, select_checkpoint

    args = parse_arguments(train_argv)
    if args.resume is None or args.run_dir is None:
        raise SystemExit("--resume (the checkpoint) and --run_dir (where the record goes) are required")
    run_dir = Path(args.run_dir)
    supervlad_compat.setup_logging(str(run_dir), allow_existing=True)
    validate_cuda_runtime(args)
    validate_faiss_runtime(args.device)
    commons.make_deterministic(args.seed)

    val_ds, _ = setup_datasets(args)
    adapter = get_model_adapter(args.model)
    bundle = adapter.build(args)
    model = bundle.model.to(args.device)
    args.features_dim = bundle.descriptor_dim
    adapter.load_weights(model, args.resume, args)
    model.eval()

    queries = sample_validation_queries(val_ds.get_positives(), args.val_queries, args.val_query_seed)
    start = time.time()
    metrics = evaluate_rank_validation(args, model, val_ds, queries)
    scores = select_checkpoint(metrics, initial_clean_r1, args.selection_clean_budgets, args.is_clean_only)
    write_screen_record(run_dir, metrics, scores, initial_clean_r1)
    row = result_row(key, args.resume, len(queries), metrics, scores, time.time() - start)
    print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
