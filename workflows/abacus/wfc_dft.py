'''
Author: Changwei Zhang
Date: 2023-05-23 09:45:24
Last Modified by:   Changwei Zhang
Last Modified time: 2023-05-23 09:45:24
'''

import argparse
import multiprocessing
import os
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from abacus_csr import ABACUSHS

au2ev = 27.211324570273

DEFAULT_INDEX_WIDTH = 4
DEFAULT_SOC = False
DEFAULT_NPROC = 1
DEFAULT_H_FILE = 'data-HR-sparse_SPIN0.csr'
DEFAULT_S_FILE = 'data-SR-sparse_SPIN0.csr'
ALLOWED_H_FILES = (DEFAULT_H_FILE,)
ALLOWED_S_FILES = (DEFAULT_S_FILE,)


def format_index_pattern(pattern, index_label, index_num=None):
    formatters = [
        lambda: pattern % index_label,
        lambda: pattern.format(index=index_label, label=index_label, raw=index_label),
        lambda: pattern.format(index_label),
    ]
    if index_num is not None:
        formatters = [
            lambda: pattern % index_num,
            lambda: pattern % index_label,
            lambda: pattern.format(index=index_label, label=index_label, raw=index_label, num=index_num, index_num=index_num),
            lambda: pattern.format(index=index_num, label=index_label, raw=index_label, num=index_num, index_num=index_num),
            lambda: pattern.format(index_label),
            lambda: pattern.format(index_num),
        ]
    for formatter in formatters:
        try:
            return formatter()
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    raise ValueError(
        f"Unsupported index pattern: {pattern}. Use '{{index}}' for formatted labels, '{{num}}' for numeric values, '{{:04d}}', or '%04d'."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate wfc.npy and eigen.npy from ABACUS Hamiltonian/overlap CSR files.'
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--start', type=int, help='First frame index.')
    selection.add_argument('--indices', nargs='+', help='Explicit frame labels; preserves zero padding.')
    parser.add_argument('--end', type=int, help='Last frame index (inclusive).')
    parser.add_argument('--index-width', type=int, default=DEFAULT_INDEX_WIDTH, help='Zero-pad indices generated from --start/--end to this width.')
    parser.add_argument(
        '--folder-pattern',
        required=True,
        help="Frame folder pattern. Supports '{index}', '{num}', '{}', '{:04d}', or '%%04d'.",
    )
    parser.add_argument(
        '--save-pattern',
        required=True,
        help="Output folder pattern. Supports '{index}', '{num}', '{}', '{:04d}', or '%%04d'.",
    )
    parser.add_argument(
        '--hamiltonian-file',
        default=DEFAULT_H_FILE,
        help=f'Exact Hamiltonian CSR filename to use inside each frame folder. Only {DEFAULT_H_FILE} is supported.',
    )
    parser.add_argument(
        '--overlap-file',
        default=DEFAULT_S_FILE,
        help=f'Exact overlap CSR filename to use inside each frame folder. Only {DEFAULT_S_FILE} is supported.',
    )
    parser.add_argument('--nproc', type=int, default=DEFAULT_NPROC, help='Worker process count.')
    return parser.parse_args()


def resolve_indices(args):
    if args.indices:
        result = []
        for raw in args.indices:
            try:
                index_num = int(raw)
            except ValueError:
                index_num = None
            result.append((raw, index_num))
        return result
    if args.end is None or args.end < args.start:
        raise ValueError(f'--end ({args.end}) must be >= --start ({args.start}).')
    return [
        (f'{index:0{args.index_width}d}' if args.index_width > 0 else str(index), index)
        for index in range(args.start, args.end + 1)
    ]


def resolve_csr_path(folder, candidates, explicit_filename=None, label='CSR'):
    if explicit_filename:
        allowed = ALLOWED_H_FILES if label == 'Hamiltonian' else ALLOWED_S_FILES
        if explicit_filename not in allowed:
            allowed_list = ', '.join(allowed)
            raise ValueError(
                f'{label} file must be one of [{allowed_list}], got {explicit_filename}'
            )
        explicit_path = os.path.join(folder, explicit_filename)
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f'{label} file {explicit_filename} not found under {folder}'
            )
        return explicit_path

    path = os.path.join(folder, candidates[0])
    if not os.path.isfile(path):
        joined = ', '.join(candidates)
        raise FileNotFoundError(f'Cannot find any of [{joined}] under {folder}')
    return path


def calc_wfc(frame_index, folder, save_dir, soc, hamiltonian_file=None, overlap_file=None):
    h_path = resolve_csr_path(
        folder,
        ALLOWED_H_FILES,
        explicit_filename=hamiltonian_file,
        label='Hamiltonian',
    )
    s_path = resolve_csr_path(
        folder,
        ALLOWED_S_FILES,
        explicit_filename=overlap_file,
        label='Overlap',
    )

    source = {
        'generator': 'wfc_dft', 'frame': frame_index,
        'physical_spin_channel': 'SPIN0', 'internal_ca_nac_spin_index': 1,
        'sources': [{
            'path': str(Path(p).resolve()), 'size': Path(p).stat().st_size,
            'mtime_ns': Path(p).stat().st_mtime_ns,
        } for p in (h_path, s_path)],
        'energy_unit': 'eV', 'wfc_layout': 'bands,ao', 'dtype': 'float32',
    }
    destination = Path(save_dir)
    if destination.exists():
        marker = destination / 'wfc-source.json'
        if marker.is_file() and json.loads(marker.read_text()) == source:
            eigen = np.load(destination / 'eigen.npy', mmap_mode='r', allow_pickle=False)
            wfc = np.load(destination / 'wfc.npy', mmap_mode='r', allow_pickle=False)
            if (eigen.ndim == 1 and wfc.shape == (eigen.size, eigen.size)
                    and eigen.dtype == wfc.dtype == np.dtype('float32')
                    and np.isfinite(eigen).all() and np.isfinite(wfc).all()):
                print(f'Skip validated frame {frame_index}', flush=True)
                return
        raise FileExistsError(f'Use a new output leaf; existing output is not reusable: {destination}')

    fH = ABACUSHS(h_path)
    HK = fH.getHK(stru=None, isH=True, isSOC=soc)
    fS = ABACUSHS(s_path)
    SK = fS.getHK(stru=None, isSOC=soc)
    fH.close()
    fS.close()

    eigen, eigen_vecs = eigh(a=HK, b=SK)

    eigen *= au2ev
    eigen_vecs = np.swapaxes(eigen_vecs, 0, 1)

    lamda = np.einsum('ai, ij, aj -> a', np.conj(eigen_vecs), SK, eigen_vecs).real
    lamda = 1 / np.sqrt(lamda)
    eigen_vecs = eigen_vecs * lamda[:, None]

    if not np.isfinite(eigen).all() or not np.isfinite(eigen_vecs).all():
        raise ValueError(f'Non-finite eigenstates for frame {frame_index}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.wfc-', dir=destination.parent))
    try:
        np.save(temporary / 'eigen.npy', eigen, allow_pickle=False)
        np.save(temporary / 'wfc.npy', eigen_vecs, allow_pickle=False)
        (temporary / 'wfc-source.json').write_text(json.dumps(source, indent=2) + '\n')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        f'Finish frame {frame_index} using {os.path.basename(h_path)} and {os.path.basename(s_path)}',
        flush=True,
    )


def main():
    args = parse_args()
    indices = resolve_indices(args)
    if args.nproc < 1:
        raise ValueError('--nproc must be >= 1.')

    tasks = []
    for index_label, index_num in indices:
        folder = format_index_pattern(args.folder_pattern, index_label, index_num)
        save_dir = format_index_pattern(args.save_pattern, index_label, index_num)
        tasks.append((index_label, folder, save_dir, DEFAULT_SOC, args.hamiltonian_file, args.overlap_file))

    destinations = [str(Path(task[2]).resolve()) for task in tasks]
    if len(set(destinations)) != len(destinations):
        raise ValueError('--save-pattern must resolve to a distinct directory per frame.')

    multiprocessing.freeze_support()
    nproc = min(multiprocessing.cpu_count(), args.nproc)
    with multiprocessing.Pool(processes=nproc) as pool:
        pool.starmap(calc_wfc, tasks)


if __name__ == '__main__':
    main()
