#!/usr/bin/env python3
"""CLI for the final T4 signal-alignment report.

CPU-only, read-only analysis of an already-finalized trust/centrality
report. Runs no model, no CUDA, and no dataset inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.t4_signal_alignment_report import (
    T4SignalAlignmentError,
    build_report,
    render_markdown,
    write_markdown_report,
    write_report_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the final T4 signal-alignment report")
    parser.add_argument("--trust-report", type=Path, required=True, help="Path to the finalized trust/centrality report JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the derived report JSON")
    parser.add_argument(
        "--canonical-stats", type=Path, default=None,
        help="Optional path to a canonical per-class pixel-union artifact (enables the union-weighted proxy)",
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=None,
        help="Optional path to write a deterministic Markdown rendering of the report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args.trust_report, canonical_stats_path=args.canonical_stats)
        write_report_json(report, args.output)
        if args.markdown_output is not None:
            write_markdown_report(report, args.markdown_output)
        print(
            "T4_SIGNAL_ALIGNMENT_REPORT_OK "
            f"decision={report['decision']['decision']} "
            f"direct_anchor_net={report['strict_t4']['direct_anchor_net']} "
            f"output={args.output}"
        )
    except T4SignalAlignmentError as error:
        print(f"T4 SIGNAL ALIGNMENT REPORT FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
