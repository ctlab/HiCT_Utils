#!/usr/bin/env python3
"""Validate a scaffolded Cooler/MCool export against an AGP layout.

This script intentionally reads inputs only. Put generated reports under /tmp or
scratch when validating large laboratory datasets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import cooler
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing guard
    print(
        "Missing Python dependency. Install HiCT_Utils requirements first, e.g.:\n"
        "  python3 -m venv HiCT_Utils/.venv\n"
        "  HiCT_Utils/.venv/bin/pip install -r HiCT_Utils/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


@dataclass(frozen=True)
class AgpComponent:
    scaffold: str
    scaffold_start0: int
    scaffold_end: int
    component: str
    component_start0: int
    component_end: int
    orientation: str

    @property
    def length(self) -> int:
        return self.scaffold_end - self.scaffold_start0


def parse_agp(path: Path) -> list[AgpComponent]:
    components: list[AgpComponent] = []
    with path.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) == 8 and fields[4].upper() in {"W", "A", "D", "F", "G", "O", "P"}:
                fields.append("+")
            if len(fields) < 9:
                raise ValueError(f"{path}:{line_no}: expected at least 9 AGP columns")
            if fields[4].upper() in {"N", "U"}:
                continue
            scaffold = fields[0]
            scaffold_start0 = int(fields[1]) - 1
            scaffold_end = int(fields[2])
            component = fields[5]
            component_start0 = int(fields[6]) - 1
            component_end = int(fields[7])
            orientation = fields[8]
            if scaffold_end <= scaffold_start0 or component_end <= component_start0:
                raise ValueError(f"{path}:{line_no}: non-positive AGP interval")
            components.append(
                AgpComponent(
                    scaffold,
                    scaffold_start0,
                    scaffold_end,
                    component,
                    component_start0,
                    component_end,
                    orientation,
                )
            )
    if not components:
        raise ValueError(f"{path}: no AGP sequence components found")
    return components


def cooler_uri(path: Path, resolution: int | None) -> str:
    if path.suffix == ".mcool":
        if resolution is None:
            raise ValueError("--resolution is required for .mcool files")
        return f"{path}::/resolutions/{resolution}"
    return str(path)


def chrom_sizes(c: cooler.Cooler) -> dict[str, int]:
    chroms = c.chroms()[:]
    return dict(zip(chroms["name"].astype(str), chroms["length"].astype(int), strict=True))


def expected_scaffold_sizes(components: Iterable[AgpComponent]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for component in components:
        sizes[component.scaffold] = max(sizes.get(component.scaffold, 0), component.scaffold_end)
    return sizes


def validate_chrom_order(exported: cooler.Cooler, components: list[AgpComponent]) -> list[str]:
    expected_order: list[str] = []
    for component in components:
        if component.scaffold not in expected_order:
            expected_order.append(component.scaffold)
    exported_order = list(exported.chromnames)
    errors: list[str] = []
    if exported_order != expected_order:
        errors.append(
            "Exported chromosome/scaffold order does not match AGP. "
            f"expected first={expected_order[:8]}, observed first={exported_order[:8]}"
        )
    expected_sizes = expected_scaffold_sizes(components)
    observed_sizes = chrom_sizes(exported)
    for name, expected_size in expected_sizes.items():
        observed_size = observed_sizes.get(name)
        if observed_size != expected_size:
            errors.append(f"Scaffold {name!r} size mismatch: expected {expected_size}, observed {observed_size}")
    extra = sorted(set(observed_sizes) - set(expected_sizes))
    if extra:
        errors.append(f"Exported cooler contains {len(extra)} scaffold(s) absent from AGP, first={extra[:8]}")
    return errors


def fetch_dense(c: cooler.Cooler, chrom: str, start: int, end: int) -> np.ndarray:
    matrix = c.matrix(balance=False, sparse=False).fetch((chrom, start, end), (chrom, start, end))
    return np.asarray(matrix)


def fetch_dense_bins(c: cooler.Cooler, start_bin: int, end_bin: int) -> np.ndarray:
    matrix = c.matrix(balance=False, sparse=False)[start_bin:end_bin, start_bin:end_bin]
    return np.asarray(matrix)


def validate_component_windows(
    source: cooler.Cooler,
    exported: cooler.Cooler,
    components: list[AgpComponent],
    queries: int,
    max_bins: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    source_sizes = chrom_sizes(source)
    max_requested_bins = max(1, max_bins)
    candidate_components = [
        component
        for component in components
        if component.orientation in {"+", "1"}
        and component.component in source_sizes
        and component.length >= exported.binsize * 3
    ]
    if not candidate_components:
        return [
            "No positive-orientation AGP components with enough interior source Cooler bins were available "
            "for window validation"
        ]

    errors: list[str] = []
    for query_idx in range(queries):
        component = rng.choice(candidate_components)
        target_extent = exported.extent(component.scaffold)
        source_extent = source.extent(component.component)
        target_component_start_bin = target_extent[0] + component.scaffold_start0 // exported.binsize
        source_component_start_bin = source_extent[0] + component.component_start0 // source.binsize
        component_bins = max(1, (component.length + exported.binsize - 1) // exported.binsize)
        component_bins = min(
            component_bins,
            target_extent[1] - target_component_start_bin,
            source_extent[1] - source_component_start_bin,
        )
        # AGP starts/gaps are often not aligned to the Cooler resolution. The
        # first and last target bins of a component can therefore legitimately
        # aggregate contacts from neighboring components. Strict equality to a
        # single source chromosome bin is only expected for interior bins.
        interior_bins = component_bins - 2
        if interior_bins <= 0:
            continue
        bins = min(max_requested_bins, interior_bins)
        offset_bins = 0 if interior_bins == bins else rng.randrange(0, interior_bins - bins + 1)
        target_start_bin = target_component_start_bin + 1 + offset_bins
        source_start_bin = source_component_start_bin + 1 + offset_bins
        target_end_bin = target_start_bin + bins
        source_end_bin = source_start_bin + bins
        try:
            observed = fetch_dense_bins(exported, target_start_bin, target_end_bin)
            expected = fetch_dense_bins(source, source_start_bin, source_end_bin)
        except Exception as exc:  # noqa: BLE001 - report exact user-facing region
            errors.append(
                f"Query {query_idx}: failed to fetch {component.scaffold} bins {target_start_bin}:{target_end_bin} "
                f"or {component.component} bins {source_start_bin}:{source_end_bin}: {exc}"
            )
            continue
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            errors.append(
                f"Query {query_idx}: raw matrix mismatch for {component.scaffold} bins {target_start_bin}:{target_end_bin} "
                f"vs {component.component} bins {source_start_bin}:{source_end_bin}; "
                f"observed_shape={observed.shape}, expected_shape={expected.shape}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mcool", type=Path, required=True)
    parser.add_argument("--exported-mcool", type=Path, required=True)
    parser.add_argument("--agp", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--max-bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)

    components = parse_agp(args.agp)
    source = cooler.Cooler(cooler_uri(args.source_mcool, args.resolution))
    exported = cooler.Cooler(cooler_uri(args.exported_mcool, args.resolution))

    errors = validate_chrom_order(exported, components)
    errors.extend(validate_component_windows(source, exported, components, args.queries, args.max_bins, args.seed))
    report = {
        "source": str(args.source_mcool),
        "exported": str(args.exported_mcool),
        "agp": str(args.agp),
        "resolution": args.resolution,
        "queries": args.queries,
        "passed": not errors,
        "errors": errors,
    }
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
