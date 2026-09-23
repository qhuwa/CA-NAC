#!/usr/bin/env python3
"""Stream per-frame graph/Hamiltonian inputs into the CA-NAC WFC calculator."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


import hamgnn_wfc as upstream


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate WFCs while keeping only one frame resident per worker."
    )
    parser.add_argument("--nao-max", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--index-width", type=int, default=4)
    parser.add_argument("--graph-data-path", required=True)
    parser.add_argument("--hamiltonian-path", required=True)
    parser.add_argument("--model-source", required=True, help="Checkpoint/artifact identity for the supplied HamGNN prediction.")
    parser.add_argument("--save-pattern", required=True)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--maxtasksperchild", type=int, default=25)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--software", choices=("abacus",), default="abacus")
    parser.add_argument("--fix-edge-index", action="store_true")
    parser.add_argument(
        "--expected-orbitals",
        type=int,
        help="Require eigen/wfc outputs to have this orbital dimension.",
    )
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def output_is_valid(save_dir, expected_orbitals=None, dtype=None, source=None):
    eigen_path = Path(save_dir) / "eigen.npy"
    wfc_path = Path(save_dir) / "wfc.npy"
    if not eigen_path.is_file() or not wfc_path.is_file():
        return False
    if eigen_path.stat().st_size == 0 or wfc_path.stat().st_size == 0:
        return False
    try:
        if source is not None:
            marker = Path(save_dir) / "wfc-source.json"
            if not marker.is_file() or json.loads(marker.read_text()) != source:
                return False
        eigen = np.load(eigen_path, mmap_mode="r")
        wfc = np.load(wfc_path, mmap_mode="r")
        if dtype is None:
            dtype_valid = True
        else:
            real_dtype = np.dtype(dtype)
            complex_dtype = np.dtype(
                "complex128" if real_dtype == np.dtype("float64") else "complex64"
            )
            dtype_valid = eigen.dtype == real_dtype and wfc.dtype in {
                real_dtype,
                complex_dtype,
            }
        dimension_valid = expected_orbitals is None or (
            eigen.ndim == 1 and eigen.shape[0] == expected_orbitals
        )
        return (
            eigen.ndim == 1
            and wfc.ndim == 2
            and wfc.shape[0] == wfc.shape[1] == eigen.shape[0]
            and dimension_valid
            and dtype_valid
            and np.isfinite(eigen).all()
            and np.isfinite(wfc).all()
        )
    except (OSError, TypeError, ValueError):
        return False


def process_frame(task):
    (
        label,
        graph_path,
        hamiltonian_path,
        save_dir,
        nao_max,
        device,
        dtype,
        software,
        fix_edge_index,
        expected_orbitals,
        source,
    ) = task

    save_dir = Path(save_dir)
    if output_is_valid(save_dir, expected_orbitals, dtype, source):
        return "skipped", label
    if save_dir.exists():
        raise FileExistsError(f"Use a new output leaf; existing output is not reusable: {save_dir}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".wfc-", dir=save_dir.parent))
    graph_dataset = None
    hon_all = None
    hoff_all = None
    try:
        graph_dataset = upstream.load_graph_dataset(graph_path)
        if len(graph_dataset) != 1:
            raise IndexError(
                f"Frame {label} contains {len(graph_dataset)} graphs; expected exactly 1: {graph_path}"
            )
        hon_all, hoff_all = upstream.split_predicted_hamiltonian(
            graph_dataset, hamiltonian_path, nao_max
        )
        if len(hon_all) != 1 or len(hoff_all) != 1:
            raise IndexError(f"Frame {label} did not resolve to one Hon/Hoff pair")

        upstream.ARGS = SimpleNamespace(
            nao_max=nao_max,
            device=device,
            dtype=dtype,
            software=software,
            fix_edge_index=fix_edge_index,
        )
        upstream.HON_ALL = [hon_all[0]]
        upstream.HOFF_ALL = [hoff_all[0]]
        upstream.SAVE_DIRS = [str(tmp_dir)]
        upstream.FRAME_INDICES = [label]
        upstream.calc_wfc(0, graph_dataset[0])

        if not output_is_valid(tmp_dir, expected_orbitals, dtype):
            raise RuntimeError(f"Frame {label} produced invalid WFC output")
        (tmp_dir / "wfc-source.json").write_text(json.dumps(source, indent=2) + "\n")
        tmp_dir.rename(save_dir)
        return "written", label
    finally:
        upstream.HON_ALL = None
        upstream.HOFF_ALL = None
        upstream.SAVE_DIRS = None
        upstream.FRAME_INDICES = None
        graph_dataset = None
        hon_all = None
        hoff_all = None
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()


def main():
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be greater than or equal to --start")
    if args.nproc < 1:
        raise ValueError("--nproc must be at least 1")
    if args.maxtasksperchild < 1:
        raise ValueError("--maxtasksperchild must be at least 1")

    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    if args.nproc > available_cpus:
        raise ValueError(
            f"--nproc={args.nproc} exceeds the {available_cpus} CPUs in this process affinity"
        )

    tasks = []
    missing = []
    valid_existing = 0
    for index in range(args.start, args.end + 1):
        label = f"{index:0{args.index_width}d}" if args.index_width else str(index)
        graph_path = upstream.format_index_pattern(args.graph_data_path, label, index)
        hamiltonian_path = upstream.format_index_pattern(
            args.hamiltonian_path, label, index
        )
        save_dir = upstream.format_index_pattern(args.save_pattern, label, index)
        for kind, path in (("graph", graph_path), ("hamiltonian", hamiltonian_path)):
            if not Path(path).is_file() or Path(path).stat().st_size == 0:
                missing.append(f"{kind} frame {label}: {path}")
        if missing:
            continue
        source = {
            "generator": "wfc_hamgnn", "frame": label,
            "model_source": args.model_source,
            "physical_spin_channel": "SPIN0", "internal_ca_nac_spin_index": 1,
            "sources": [{
                "path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
                "mtime_ns": Path(p).stat().st_mtime_ns,
            } for p in (graph_path, hamiltonian_path)],
            "nao_max": args.nao_max, "dtype": args.dtype,
            "add_h0": args.fix_edge_index, "energy_unit": "eV", "wfc_layout": "bands,ao",
            "expected_orbitals": args.expected_orbitals,
        }
        if output_is_valid(
            save_dir, args.expected_orbitals, args.dtype, source
        ):
            valid_existing += 1
        tasks.append(
            (
                label,
                graph_path,
                hamiltonian_path,
                save_dir,
                args.nao_max,
                args.device,
                args.dtype,
                args.software,
                args.fix_edge_index,
                args.expected_orbitals,
                source,
            )
        )

    if missing:
        shown = "\n".join(missing[:20])
        suffix = "\n..." if len(missing) > 20 else ""
        raise FileNotFoundError(f"Missing required inputs:\n{shown}{suffix}")

    for field in (1, 2, 3):
        paths = [str(Path(task[field]).resolve()) for task in tasks]
        if len(set(paths)) != len(paths):
            raise ValueError("Graph, Hamiltonian and output patterns must resolve to distinct paths per frame.")

    config = {
        "start": args.start,
        "end": args.end,
        "frame_count": len(tasks),
        "nproc": args.nproc,
        "maxtasksperchild": args.maxtasksperchild,
        "available_cpus": available_cpus,
        "dtype": args.dtype,
        "expected_orbitals": args.expected_orbitals,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "valid_existing": valid_existing,
    }
    print(json.dumps(config, indent=2), flush=True)

    started = time.time()
    counts = {"written": 0, "skipped": 0}
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=args.nproc, maxtasksperchild=args.maxtasksperchild
    ) as pool:
        for status, label in pool.imap_unordered(process_frame, tasks, chunksize=1):
            counts[status] += 1
            completed = counts["written"] + counts["skipped"]
            print(
                f"[{completed}/{len(tasks)}] {label}: {status}",
                flush=True,
            )

    invalid_outputs = []
    for task in tasks:
        label, _, _, save_dir = task[:4]
        if not output_is_valid(save_dir, args.expected_orbitals, args.dtype, task[-1]):
            invalid_outputs.append(f"{label}: {save_dir}")
    if invalid_outputs:
        shown = "\n".join(invalid_outputs[:20])
        suffix = "\n..." if len(invalid_outputs) > 20 else ""
        raise RuntimeError(f"Invalid or missing WFC outputs:\n{shown}{suffix}")

    summary = {
        **config,
        **counts,
        "elapsed_seconds": time.time() - started,
        "invalid_outputs": invalid_outputs,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary.with_name(
            f".{args.summary.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.summary)


if __name__ == "__main__":
    main()
