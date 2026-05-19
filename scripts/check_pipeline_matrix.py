#!/usr/bin/env python
"""Exhaustively check PipelineManager policy combinations.

This script uses small fake Dataset4D/DetectorGeometry objects plus simulated
RAM/VRAM budgets.  It does not allocate real 4D data or CUDA memory; it tests
the policy resolver itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass

from fast_acbf.recon.pipeline import PipelineManager


PIPELINES = ('speed', 'balanced', 'memory')
STORAGES = ('auto', 'device', 'host', 'none')
FILLS = ('auto', 'precompute', 'lazy', 'on_the_fly')
EXTRACTORS = (
    'auto',
    'device_mask',
    'host_mask',
    'disk_per_pixel',
    'disk_slab',
    'disk_scan_row',
)

SCAN_SHAPE = (64, 64)
RAW_BYTES = 256 * 2**20
NB = 512
VBF_BYTES = NB * SCAN_SHAPE[0] * SCAN_SHAPE[1] * 4
IMAGEFFT_BYTES = VBF_BYTES * 2


@dataclass(frozen=True)
class FakeDataset:
    name: str
    is_lazy: bool
    backend_chunks: tuple[int, int, int, int] | None
    nbytes_float32: int = RAW_BYTES
    scan_shape: tuple[int, int] = SCAN_SHAPE


@dataclass(frozen=True)
class FakeGeometry:
    n_bf_pixels: int = NB


@dataclass(frozen=True)
class ResourceCase:
    name: str
    device: str
    free_vram: int | None
    available_ram: int | None


class SimulatedPipelineManager(PipelineManager):
    def __init__(self, *args, free_vram: int | None, available_ram: int | None, **kwargs):
        self._sim_free_vram = free_vram
        self._sim_available_ram = available_ram
        super().__init__(*args, **kwargs)

    def _free_vram(self) -> int | None:
        return self._sim_free_vram

    def _available_ram(self) -> int | None:
        return self._sim_available_ram


DATASETS = (
    FakeDataset("materialized", is_lazy=False, backend_chunks=None),
    FakeDataset("lazy_contiguous", is_lazy=True, backend_chunks=None),
    FakeDataset("lazy_detector_chunks", is_lazy=True, backend_chunks=(64, 64, 1, 1)),
    FakeDataset("lazy_scan_chunks", is_lazy=True, backend_chunks=(1, 1, 128, 128)),
    FakeDataset("lazy_other_chunks", is_lazy=True, backend_chunks=(4, 4, 16, 16)),
)

RESOURCES = (
    ResourceCase("cpu_ram_raw_fft_fit", "cpu", None, 512 * 2**20),
    ResourceCase("cpu_ram_fft_only", "cpu", None, 64 * 2**20),
    ResourceCase("cpu_ram_tight", "cpu", None, 8 * 2**20),
    ResourceCase("cuda_full_fit", "cuda", 1024 * 2**20, 512 * 2**20),
    ResourceCase("cuda_fft_only_host_raw_fit", "cuda", 32 * 2**20, 512 * 2**20),
    ResourceCase("cuda_fft_only_host_tight", "cuda", 32 * 2**20, 64 * 2**20),
    ResourceCase("cuda_no_fft_host_fit", "cuda", 8 * 2**20, 512 * 2**20),
    ResourceCase("cuda_no_fft_host_tight", "cuda", 8 * 2**20, 8 * 2**20),
)


TREE = f"""
PipelineManager behavior tree

1. Resolve ImageFFT storage
   - explicit device|host|none: respect it, but reject impossible device/host cache sizes
   - auto + pipeline=memory: none
   - auto + speed/balanced:
     - device if ImageFFT fits compute device memory
     - else host if ImageFFT fits host RAM
     - else none

2. Resolve ImageFFT fill
   - storage=none: on_the_fly only
   - persistent storage + explicit precompute|lazy: respect it
   - persistent storage + auto: precompute
   - persistent storage + on_the_fly: invalid

3. Resolve BF extraction strategy
   - explicit strategy: respect it, then validate route feasibility
   - auto:
     - device_mask only when storage is persistent, fill=precompute, CUDA is available,
       pipeline is not memory, and raw+vBF+ImageFFT fit VRAM
     - host_mask if raw is already materialized
     - host_mask for lazy raw only when storage is persistent, pipeline is not memory,
       and raw host materialization plus any host ImageFFT cache fits RAM
     - otherwise disk strategy:
       - contiguous lazy + precompute: disk_scan_row
       - contiguous lazy + on_the_fly/lazy: disk_slab
       - detector-pixel chunks: disk_per_pixel
       - scan-row chunks: disk_scan_row
       - other chunks: disk_per_pixel

4. Whole-pass coercions
   - device_mask means whole-pass precompute:
     raw 4D -> device, full vBF -> device, full ImageFFT -> cache, release raw/vBF
   - device_mask + lazy/on_the_fly or storage=none is invalid
   - host_mask on a lazy source means host materialization first, so RAM must fit

Nominal sizes in this matrix:
   raw={RAW_BYTES / 2**20:.1f} MiB, vBF={VBF_BYTES / 2**20:.1f} MiB,
   ImageFFT={IMAGEFFT_BYTES / 2**20:.1f} MiB
"""


def resolve_case(dataset, resources, pipeline, storage, fill, extractor):
    try:
        manager = SimulatedPipelineManager(
            dataset,
            FakeGeometry(),
            device=resources.device,
            pipeline=pipeline,
            imagefft_storage=storage,
            imagefft_fill=fill,
            extractor_strategy=extractor,
            free_vram=resources.free_vram,
            available_ram=resources.available_ram,
        )
    except Exception as exc:  # noqa: BLE001 - script reports policy errors by class/message.
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "dataset": dataset,
            "resources": resources,
            "pipeline": pipeline,
            "storage_request": storage,
            "fill_request": fill,
            "extractor_request": extractor,
        }
    res = manager.resolution
    return {
        "ok": True,
        "dataset": dataset,
        "resources": resources,
        "pipeline": pipeline,
        "storage_request": storage,
        "fill_request": fill,
        "extractor_request": extractor,
        "storage": res.imagefft_storage,
        "fill": res.imagefft_fill,
        "extractor": res.extractor_strategy,
    }


def assert_invariants(row):
    if not row["ok"]:
        if (
            row["storage_request"] == 'auto'
            and row["fill_request"] == 'auto'
            and row["extractor_request"] == 'auto'
        ):
            raise AssertionError(f"All-auto route should not fail: {format_row(row)}")
        return

    storage = row["storage"]
    fill = row["fill"]
    extractor = row["extractor"]
    dataset = row["dataset"]
    pipeline = row["pipeline"]

    if storage == 'none' and fill != 'on_the_fly':
        raise AssertionError(f"storage=none must resolve to on_the_fly: {format_row(row)}")
    if fill == 'on_the_fly' and storage != 'none':
        raise AssertionError(f"on_the_fly must not keep ImageFFT storage: {format_row(row)}")
    if extractor == 'device_mask' and (storage == 'none' or fill != 'precompute'):
        raise AssertionError(f"device_mask must be whole-pass precompute: {format_row(row)}")
    if row["extractor_request"] == 'auto' and fill != 'precompute' and extractor == 'device_mask':
        raise AssertionError(f"auto extractor chose device_mask for non-precompute: {format_row(row)}")
    if row["extractor_request"] == 'auto' and dataset.is_lazy and storage == 'none':
        if not extractor.startswith('disk_'):
            raise AssertionError(f"lazy storage=none should stream from disk: {format_row(row)}")
    if row["extractor_request"] == 'auto' and pipeline == 'memory' and dataset.is_lazy:
        if not extractor.startswith('disk_'):
            raise AssertionError(f"memory preset should not materialize lazy raw: {format_row(row)}")


def classify(row):
    if not row["ok"]:
        return f"invalid:{row['error_type']}"
    extractor = row["extractor"]
    storage = row["storage"]
    fill = row["fill"]
    if extractor == 'device_mask':
        return f"whole_device_precompute->{storage}"
    if extractor == 'host_mask':
        return f"host_mask->{storage}/{fill}"
    if storage == 'none':
        return f"disk_or_host_on_the_fly:{extractor}"
    return f"{extractor}->{storage}/{fill}"


def format_row(row):
    base = (
        f"dataset={row['dataset'].name}, resources={row['resources'].name}, "
        f"pipeline={row['pipeline']}, storage={row['storage_request']}, "
        f"fill={row['fill_request']}, extractor={row['extractor_request']}"
    )
    if row["ok"]:
        return (
            f"{base} => {row['storage']}/{row['fill']}/{row['extractor']}"
        )
    return f"{base} => {row['error_type']}: {row['error']}"


def run_matrix(verbose: bool = False):
    rows = []
    counts = Counter()
    by_auto = Counter()
    invalid_examples = defaultdict(list)
    for dataset in DATASETS:
        for resources in RESOURCES:
            for pipeline in PIPELINES:
                for storage in STORAGES:
                    for fill in FILLS:
                        for extractor in EXTRACTORS:
                            row = resolve_case(dataset, resources, pipeline, storage, fill, extractor)
                            assert_invariants(row)
                            rows.append(row)
                            label = classify(row)
                            counts[label] += 1
                            if extractor == 'auto' and row["ok"]:
                                by_auto[label] += 1
                            if not row["ok"] and len(invalid_examples[label]) < 3:
                                invalid_examples[label].append(row)
                            if verbose:
                                print(format_row(row))
    return rows, counts, by_auto, invalid_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print every resolved combination")
    parser.add_argument("--no-tree", action="store_true", help="hide the behavior tree")
    args = parser.parse_args()

    if not args.no_tree:
        print(TREE.strip())
        print()

    rows, counts, by_auto, invalid_examples = run_matrix(verbose=args.verbose)
    print(f"Checked {len(rows)} combinations.")
    print()
    print("All routes:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print()
    print("Auto extractor successful routes:")
    for label, count in sorted(by_auto.items()):
        print(f"  {label}: {count}")
    print()
    print("Representative invalid routes:")
    for label, examples in sorted(invalid_examples.items()):
        print(f"  {label}:")
        for row in examples:
            print(f"    - {format_row(row)}")
    print()
    print("Policy invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
