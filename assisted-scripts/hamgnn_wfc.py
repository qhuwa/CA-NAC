"""Internal Gamma-point graph/Hamiltonian assembly for wfc_hamgnn.py.

CPU solver extracted from the existing wfc_cal.py implementation.
"""

import math
import os
import numpy as np
from scipy.linalg import eigh
from abacus_basis import (au2ev, basis_def_13_abacus,
    basis_def_15_abacus, basis_def_27_abacus, basis_def_40_abacus)


ABACUS_BASIS_DEFS = {
    13: basis_def_13_abacus,
    15: basis_def_15_abacus,
    27: basis_def_27_abacus,
    40: basis_def_40_abacus,
}

ARGS = None

HON_ALL = None

HOFF_ALL = None

SAVE_DIRS = None

FRAME_INDICES = None

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

def get_basis_def(software, nao_max):
    if software != 'abacus':
        raise NotImplementedError(
            'This converter only supports ABACUS basis definitions.'
        )

    if nao_max in ABACUS_BASIS_DEFS:
        return ABACUS_BASIS_DEFS[nao_max]

    raise ValueError(f'Unsupported nao_max={nao_max}; supported masks: {sorted(ABACUS_BASIS_DEFS)}')

def load_graph_dataset(graph_data_path):
    with np.load(graph_data_path, allow_pickle=True) as graph_data:
        return list(graph_data['graph'].item().values())

def split_predicted_hamiltonian(graph_dataset, hamiltonian_path, nao_max):
    len_H = []
    for data in graph_dataset:
        len_H.append(len(data.Son))
        len_H.append(len(data.Soff))

    H = np.load(hamiltonian_path, allow_pickle=False).reshape(-1, nao_max, nao_max)
    if len(H) != sum(len_H) or not np.isfinite(H).all() or np.iscomplexobj(H):
        raise ValueError(f'Expected finite real onsite/offsite blocks matching the graph: {hamiltonian_path}')
    Hon_all, Hoff_all = [], []
    idx = 0
    for i in range(0, len(len_H), 2):
        Hon_all.append(H[idx : idx + len_H[i]])
        idx += len_H[i]
        Hoff_all.append(H[idx : idx + len_H[i + 1]])
        idx += len_H[i + 1]
    return Hon_all, Hoff_all

def calc_wfc(idx, data):
    nao_max = ARGS.nao_max
    numerical_dtype = np.dtype(ARGS.dtype)
    Son = np.asarray(data.Son.numpy(), dtype=numerical_dtype).reshape(-1, nao_max, nao_max)
    Soff = np.asarray(data.Soff.numpy(), dtype=numerical_dtype).reshape(-1, nao_max, nao_max)
    Hon = np.asarray(HON_ALL[idx], dtype=numerical_dtype).reshape(-1, nao_max, nao_max)
    Hoff = np.asarray(HOFF_ALL[idx], dtype=numerical_dtype).reshape(-1, nao_max, nao_max)
    edge_index = data.edge_index.numpy()
    species = data.z.numpy()

    if ARGS.fix_edge_index:
        Hon0 = np.asarray(data.Hon0.numpy(), dtype=numerical_dtype).reshape(-1, nao_max, nao_max)
        Hoff0 = np.asarray(data.Hoff0.numpy(), dtype=numerical_dtype).reshape(-1, nao_max, nao_max)

    basis_definition = np.zeros((99, nao_max), dtype=int)
    basis_def = get_basis_def(ARGS.software, nao_max)
    for atomic_number in species:
        if int(atomic_number) not in basis_def:
            raise ValueError(f'No ABACUS orbital mask for Z={atomic_number}, nao_max={nao_max}')
    for atomic_number, orbital_indices in basis_def.items():
        basis_definition[atomic_number][orbital_indices] = 1

    orb_mask = basis_definition[species].reshape(-1)
    orb_mask = orb_mask[:, None] * orb_mask[None, :]

    natoms = len(species)
    HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=numerical_dtype)
    SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=numerical_dtype)

    na = np.arange(natoms)
    HK[na, na, :, :] += Hon[na, :, :]
    SK[na, na, :, :] += Son[na, :, :]

    if not ARGS.fix_edge_index:
        for iedge in range(len(Hoff)):
            HK[edge_index[0, iedge], edge_index[1, iedge]] += Hoff[iedge, :, :]
            SK[edge_index[0, iedge], edge_index[1, iedge]] += Soff[iedge, :, :]
    else:
        HK0 = np.zeros((natoms, natoms, nao_max, nao_max), dtype=numerical_dtype)
        HK0[na, na, :, :] += Hon0[na, :, :]
        for iedge in range(len(Hoff)):
            HK[edge_index[0, iedge], edge_index[1, iedge]] += Hoff[iedge, :, :]
        for iedge in range(len(Hoff0)):
            HK0[edge_index[0, iedge], edge_index[1, iedge]] += Hoff0[iedge, :, :]
            SK[edge_index[0, iedge], edge_index[1, iedge]] += Soff[iedge, :, :]
        HK = HK + HK0

    HK = np.swapaxes(HK, 1, 2).reshape(natoms * nao_max, natoms * nao_max)
    SK = np.swapaxes(SK, 1, 2).reshape(natoms * nao_max, natoms * nao_max)

    HK = HK[orb_mask > 0]
    norbs = int(math.sqrt(HK.size))
    HK = HK.reshape(norbs, norbs)

    SK = SK[orb_mask > 0]
    norbs = int(math.sqrt(SK.size))
    SK = SK.reshape(norbs, norbs)

    if ARGS.device == 'cpu':
        w, v = eigh(a=HK, b=SK, subset_by_index=None)
        eigen = w * au2ev
        eigen_vecs = np.swapaxes(v, -1, -2)

        lamda = np.einsum('ai, ij, aj -> a', np.conj(eigen_vecs), SK, eigen_vecs).real
        lamda = 1 / np.sqrt(lamda)
        eigen_vecs = eigen_vecs * lamda[:, None]
    else:
        raise NotImplementedError

    eigen = np.asarray(eigen, dtype=numerical_dtype)
    eigen_vec_dtype = (
        np.dtype('complex128' if ARGS.dtype == 'float64' else 'complex64')
        if np.iscomplexobj(eigen_vecs)
        else numerical_dtype
    )
    eigen_vecs = np.asarray(eigen_vecs, dtype=eigen_vec_dtype)

    save_dir = SAVE_DIRS[idx]
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'eigen.npy'), eigen[:])
    np.save(os.path.join(save_dir, 'wfc.npy'), eigen_vecs[:, :])
    print(f'Finish frame {FRAME_INDICES[idx]}', flush=True)
