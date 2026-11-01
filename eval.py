#!/usr/bin/env python3
"""Unified evaluation entry point.

Usage:
    python eval.py [rank] <rank_eval arguments>   # rank-PGD / rank-APGD attacks

Every argument after the optional `rank` token is forwarded unchanged to
src/rank_eval.py. Run `python eval.py --help` for the options.
"""
import sys

MODES = ("rank",)


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
    from src.rank_eval import main as run_eval

    run_eval()


if __name__ == "__main__":
    main()
