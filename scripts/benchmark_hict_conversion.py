#!/usr/bin/env python3
"""Benchmark HiCT conversion/scaffolding commands and optionally plot results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


def run_step(step: Step, env: dict[str, str], log_dir: Path) -> dict[str, object]:
    stdout_path = log_dir / f"{step.name}.stdout.log"
    stderr_path = log_dir / f"{step.name}.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("wt", encoding="utf-8") as stdout, stderr_path.open("wt", encoding="utf-8") as stderr:
        result = subprocess.run(step.command, stdout=stdout, stderr=stderr, env=env, check=False)
    elapsed = time.perf_counter() - started
    return {
        "step": step.name,
        "seconds": elapsed,
        "exit_code": result.returncode,
        "command": " ".join(step.command),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "seconds", "exit_code", "command", "stdout", "stderr"])
        writer.writeheader()
        writer.writerows(rows)


def write_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; wrote CSV only.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    by_step: dict[str, list[float]] = {}
    for row in rows:
        by_step.setdefault(str(row["step"]), []).append(float(row["seconds"]))

    labels = list(by_step)
    values = [by_step[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_ylabel("Seconds")
    ax.set_title("HiCT conversion/scaffolding step timing")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "conversion_step_boxplot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for step, step_rows in by_step.items():
        ax.scatter(range(1, len(step_rows) + 1), step_rows, label=step)
    ax.set_xlabel("Run")
    ax.set_ylabel("Seconds")
    ax.set_title("HiCT conversion/scaffolding timing by run")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "conversion_step_scatter.png", dpi=160)
    plt.close(fig)


def build_steps(args: argparse.Namespace, work_dir: Path, run_idx: int) -> list[Step]:
    hict_path = work_dir / f"run{run_idx}.hict.hdf5"
    exported_path = work_dir / f"run{run_idx}.scaffolded.mcool"
    base = [
        str(args.hict),
        "--verbose=false",
        "convert",
    ]
    import_command = base + [
        "mcool-to-hict",
        f"--input={args.source_mcool}",
        f"--output={hict_path}",
        f"--resolutions={args.resolution}",
        "--build-resolution-pyramid=false",
        "--balance-input-coolers=false",
        f"--parallelism={args.threads}",
        f"--compression={args.compression}",
    ]
    export_command = base + [
        "hict-to-mcool",
        f"--input={hict_path}",
        f"--output={exported_path}",
        f"--agp={args.agp}",
        "--apply-agp=true",
        f"--resolutions={args.resolution}",
        f"--parallelism={args.threads}",
        f"--compression={args.compression}",
        "--balance-exported-coolers=false",
    ]
    return [
        Step("mcool_to_hict_single_resolution", import_command),
        Step("hict_to_mcool_with_agp", export_command),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hict", type=Path, required=True, help="HiCT runfile or CLI launcher.")
    parser.add_argument("--source-mcool", type=Path, required=True)
    parser.add_argument("--agp", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=100000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--compression", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "hict-benchmark")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    env = os.environ.copy()
    env.setdefault("HICT_CONVERSION_MAX_MEMORY", "16g")
    env.setdefault("HICT_EXPORT_MAX_MEMORY", "16g")
    env.setdefault("HICT_EXPORT_MODE", "hictk")
    env.setdefault("HICT_NATIVE_PROCESSING", "true")

    for run_idx in range(1, args.runs + 1):
        work_dir = args.output_dir / f"work-run-{run_idx}"
        log_dir = args.output_dir / f"logs-run-{run_idx}"
        work_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for step in build_steps(args, work_dir, run_idx):
                row = run_step(step, env, log_dir)
                row["run"] = run_idx
                rows.append(row)
                if row["exit_code"] != 0:
                    write_csv(rows, args.output_dir / "benchmark.csv")
                    print(json.dumps(row, indent=2), file=sys.stderr)
                    return int(row["exit_code"])
        finally:
            if not args.keep_work:
                shutil.rmtree(work_dir, ignore_errors=True)

    write_csv(rows, args.output_dir / "benchmark.csv")
    write_plots(rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
