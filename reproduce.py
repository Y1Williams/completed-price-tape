#!/usr/bin/env python3
"""Run the deterministic price tape example. See --help for commands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parser():
    result = argparse.ArgumentParser(
        description="Reproduce the fixed deterministic price tape example locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  verify   Reevaluate saved controls and check finer quadrature. No optimization.
  compute  Rerun all original optimization nodes, verify and compare the results.
  plot     Plot the data in --data-dir.
  demo     Build the offline demonstration from the data in --data-dir.
  all      Verify reference, compute, verify/compare, then plot and demo fresh data.

Examples:
  python reproduce.py verify
  python reproduce.py compute --output-dir output
  python reproduce.py plot --data-dir output --output-dir output
  python reproduce.py all

Output defaults to output/. Reference JSON files are never overwritten.
Values printed to the terminal use 3 decimals. JSON retains full precision.
""",
    )
    result.add_argument("command", choices=("verify", "compute", "plot", "demo", "all"))
    result.add_argument("--data-dir", type=Path, default=ROOT / "data" / "reference",
                        help="input/reference JSON directory (default: data/reference)")
    result.add_argument("--output-dir", type=Path, default=ROOT / "output",
                        help="generated results directory (default: output)")
    result.add_argument("--font", default="STIXGeneral", help="installed plot font (default: STIXGeneral)")
    result.add_argument("--threads", type=int, default=1, help="numerical-library threads (default: 1)")
    result.add_argument("--beta-order", type=int, default=64, help="verification base beta order (default: 64)")
    result.add_argument("--execution-order", type=int, default=12, help="verification base cell order (default: 12)")
    result.add_argument("--recovery-order", type=int, default=80, help="verification base recovery order (default: 80)")
    result.add_argument("--check-beta-order", type=int, default=96, help="verification refined beta order (default: 96)")
    result.add_argument("--check-execution-order", type=int, default=24, help="verification refined cell order (default: 24)")
    result.add_argument("--check-recovery-order", type=int, default=120, help="verification refined recovery order (default: 120)")
    return result


def portable_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "<external-directory>/" + path.name


def main(argv=None):
    args = parser().parse_args(argv)
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    reference_dir = (ROOT / "data" / "reference").resolve()
    if output_dir == reference_dir or reference_dir in output_dir.parents:
        parser().error("--output-dir must be outside data/reference.")
    if args.command in ("compute", "all") and data_dir == output_dir:
        parser().error("Use different --data-dir and --output-dir for computation.")
    if args.threads < 1:
        parser().error("--threads must be positive.")
    # Set before importing NumPy/SciPy, including when invoked outside a shell.
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from tape_example.model import Quadrature
    from tape_example.numerics import compare_rerun, compute, verify

    base = Quadrature(args.beta_order, args.execution_order, args.recovery_order)
    refined = Quadrature(args.check_beta_order, args.check_execution_order, args.check_recovery_order)
    if any(fine <= coarse for fine, coarse in zip(refined.as_dict().values(), base.as_dict().values())):
        parser().error("Every refined quadrature order must exceed its base order.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = "failed"
    error = None
    try:
        if args.command in ("verify", "all"):
            verify(data_dir, output_dir, quadrature=base, refined=refined)
        if args.command in ("compute", "all"):
            compute(output_dir)
            verify(output_dir, output_dir, quadrature=base, refined=refined,
                   report_name="computed_verification.json")
            compare_rerun(data_dir, output_dir)
        input_dir = output_dir if args.command == "all" else data_dir
        if args.command == "plot":
            from tape_example.presentation import plot
            plot(input_dir, output_dir, font=args.font)
        if args.command in ("demo", "all"):
            from tape_example.presentation import demo
            demo(input_dir, output_dir, font=args.font)
        status = "passed"
    except Exception as exc:
        error = type(exc).__name__
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": args.command,
                  "data_dir": portable_path(data_dir), "output_dir": portable_path(output_dir),
                  "font": args.font, "threads": args.threads,
                  "verification_base_quadrature": base.as_dict(),
                  "verification_refined_quadrature": refined.as_dict(),
                  "status": status, "error_type": error,
                  "elapsed_seconds": time.perf_counter() - started}
        with (output_dir / "command_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"{args.command}: completed in {time.perf_counter() - started:.3f} s. "
          f"Results: {portable_path(output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
