#!/usr/bin/env python3
"""Unified evaluation entry point.

Usage:
    python eval.py [rank] <rank_eval arguments>          # rank attacks (default)
    python eval.py perceptual <perceptual_eval arguments>  # perceptual attacks

The first argument selects the evaluator; every remaining argument is
forwarded unchanged. Run `python eval.py --help` or
`python eval.py perceptual --help` for the per-mode options.
"""
import sys

MODES = ("rank", "perceptual")


def resolve_mode(argv):
    if argv and argv[0] in MODES:
        return argv[0], list(argv[1:])
    return "rank", list(argv)


def main() -> None:
    argv = sys.argv[1:]
    mode, forwarded = resolve_mode(argv)
    token_consumed = len(forwarded) != len(argv)
    prog = f"{sys.argv[0]} {mode}" if token_consumed else sys.argv[0]
    sys.argv = [prog, *forwarded]
    if mode == "perceptual":
        from src.perceptual_eval import main as run_eval
    else:
        from src.rank_eval import main as run_eval
    run_eval()


if __name__ == "__main__":
    main()
