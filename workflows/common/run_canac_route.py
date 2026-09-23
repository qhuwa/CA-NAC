#!/usr/bin/env python3
"""Run CA-NAC on an ordered route of prepared NumPy eigenstate data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ca-nac-root", type=Path, default=os.environ.get("CA_NAC_ROOT", str(Path(__file__).resolve().parents[2])))
    result.add_argument("--run-dir-pattern", required=True)
    result.add_argument("--source", choices=("abacus", "hamgnn"), required=True)
    selection = result.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices", nargs="+")
    selection.add_argument("--start", type=int)
    result.add_argument("--end", type=int)
    result.add_argument("--index-width", type=int, default=4)
    result.add_argument("--band-min", type=int, required=True)
    result.add_argument("--band-max", type=int, required=True)
    result.add_argument("--potim", type=float, required=True, help="Effective frame spacing in fs")
    result.add_argument("--nproc", type=int, default=1)
    result.add_argument("--state-tracking", action="store_true")
    result.add_argument("--summary", required=True, type=Path)
    return result


def labels(args: argparse.Namespace) -> list[tuple[str, int | None]]:
    if args.indices:
        return [(raw, int(raw) if raw.lstrip("+-").isdigit() else None) for raw in args.indices]
    if args.end is None or args.end < args.start:
        raise ValueError("--end >= --start is required with --start")
    return [(f"{value:0{args.index_width}d}" if args.index_width else str(value), value) for value in range(args.start, args.end + 1)]


def expand(pattern: str, label: str, number: int | None) -> Path:
    values = {"index": label, "label": label, "num": number if number is not None else label}
    try:
        return Path(pattern.format(**values)).expanduser().resolve()
    except (KeyError, ValueError) as exc:
        raise ValueError("--run-dir-pattern supports {index}, {label}, and {num}") from exc


def main() -> None:
    args = parser().parse_args()
    if args.ca_nac_root is None:
        raise ValueError("provide --ca-nac-root, set CA_NAC_ROOT, or resolve it from the machine profile")
    root = args.ca_nac_root.expanduser().resolve()
    entry = root / "CAnac.py"
    if not entry.is_file():
        raise FileNotFoundError(f"missing CA-NAC entry point: {entry}")
    if args.band_min < 1 or args.band_max < args.band_min or args.nproc < 1 or not np.isfinite(args.potim) or args.potim <= 0:
        raise ValueError("invalid band window, process count, or timestep")

    selected = labels(args)
    run_paths = [expand(args.run_dir_pattern, label, number) for label, number in selected]
    if len(run_paths) < 2 or len(set(run_paths)) != len(run_paths):
        raise ValueError('At least two distinct frame directories are required')
    numbers = [number for _, number in selected]
    if any(number is None for number in numbers) or any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
        raise ValueError('Frames must have contiguous numeric labels in trajectory order')
    missing = []
    for position, run_path in enumerate(run_paths):
        for name in ("wfc.npy", "eigen.npy"):
            if not (run_path / name).is_file():
                missing.append(str(run_path / name))
        if position < len(run_paths) - 1:
            for name in ('tdoverlap.npy', 'tdoverlap.npy.meta.json'):
                if not (run_path / name).is_file():
                    missing.append(str(run_path / name))
    if missing:
        raise FileNotFoundError("missing CA-NAC inputs:\n" + "\n".join(missing[:20]))

    ao_count = None
    provenance = []
    for position, (run_path, (label, _)) in enumerate(zip(run_paths, selected)):
        wfc = np.load(run_path / 'wfc.npy', mmap_mode='r', allow_pickle=False)
        eigen = np.load(run_path / 'eigen.npy', mmap_mode='r', allow_pickle=False)
        if (eigen.ndim != 1 or wfc.ndim != 2 or wfc.shape != (eigen.size, eigen.size)
                or args.band_max > eigen.size or np.iscomplexobj(wfc)
                or not np.isfinite(wfc).all() or not np.isfinite(eigen).all()):
            raise ValueError(f'Expected finite, full-rank real Gamma eigenstates with the requested bands: {run_path}')
        if ao_count is None:
            ao_count = eigen.size
        if eigen.size != ao_count:
            raise ValueError(f'AO dimension changed: {run_path}')
        record = {'frame': label, 'eigen': str((run_path / 'eigen.npy').resolve()),
                  'wfc': str((run_path / 'wfc.npy').resolve())}
        source_path = run_path / 'wfc-source.json'
        if source_path.is_file():
            source = json.loads(source_path.read_text())
            expected_generator = 'wfc_dft' if args.source == 'abacus' else 'wfc_hamgnn'
            if source.get('generator') != expected_generator or source.get('frame') != label:
                raise ValueError(f'Eigenstate source/frame mismatch: {source_path}')
            record['eigenstate_source'] = source
        if position < len(run_paths) - 1:
            overlap = np.load(run_path / 'tdoverlap.npy', mmap_mode='r', allow_pickle=False)
            metadata = json.loads((run_path / 'tdoverlap.npy.meta.json').read_text())
            signature = metadata.get('input_signature', {})
            if (overlap.shape != (ao_count, ao_count) or not np.isfinite(overlap).all()
                    or metadata.get('status') != 'complete'
                    or signature.get('auto_contract', {}).get('status') != 'validated'):
                raise ValueError(f'Invalid or uncalibrated adjacent-frame overlap: {run_path}')
            for side, expected_label in (('left', label), ('right', selected[position + 1][0])):
                stru = Path(signature.get(f'{side}_stru', {}).get('path', ''))
                if stru.name != 'STRU' or stru.parent.name != expected_label:
                    raise ValueError(f'Overlap {side} frame mismatch: {run_path}')
            record['overlap'] = str((run_path / 'tdoverlap.npy').resolve())
            record['overlap_sidecar'] = str((run_path / 'tdoverlap.npy.meta.json').resolve())
        provenance.append(record)

    sys.path.insert(0, str(root))
    from CAnac import nac_calc

    stored_min, stored_max = args.band_min, args.band_max
    run_dirs = [str(path) + "/" for path in run_paths]
    checking = {
        # Inputs were checked above. Recompute projections so old cached
        # tdolap files cannot silently survive a change of eigenstates.
        "skip_file_verification": True,
        "skip_TDolap_calc": False,
        "skip_NAC_calc": False,
        "onthefly_verification": True,
    }
    nac_calc(
        run_dirs,
        checking,
        nproc=args.nproc,
        is_gamma=True,
        is_reorder=args.state_tracking,
        is_alle=False,
        is_real=True,
        is_combine=False,
        iformat="HFNAMD",
        ibmin=args.band_min,
        ibmax=args.band_max,
        bmin_s=stored_min,
        bmax_s=stored_max,
        omin=args.band_min,
        omax=args.band_max,
        ikpt=1,
        ispin=1,
        icor=1,
        potim=args.potim,
        software="HAMNET",
        wavecar="",
    )
    output_name = 'nac_psrd.npy' if args.state_tracking else 'nac_ps.npy'
    expected_shape = (args.band_max - args.band_min + 1,) * 2
    for run_path in run_paths[:-1]:
        output = np.load(run_path / output_name, allow_pickle=False)
        if output.shape != expected_shape or not np.isfinite(output).all():
            raise ValueError(f'Invalid NAC output: {run_path / output_name}')
    payload = {
        'status': 'complete', 'source': args.source, 'frames': len(run_paths),
        'pairs': len(run_paths) - 1, 'output_name': output_name,
        'output_unit': 'dimensionless', 'rate_fs_inv_formula': 'nac / (2 * potim_fs)',
        'physical_spin_channel': 'SPIN0', 'internal_ca_nac_spin_index': 1,
        'inputs': provenance,
        "ca_nac_root": str(root),
        "run_dirs": run_dirs,
        "indices": [label for label, _ in selected],
        "band_window": [args.band_min, args.band_max],
        "stored_band_window": [stored_min, stored_max],
        "potim_fs": args.potim,
        "state_tracking": args.state_tracking,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_name(f'.{args.summary.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.summary)


if __name__ == "__main__":
    main()
