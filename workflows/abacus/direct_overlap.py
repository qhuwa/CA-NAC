#!/usr/bin/env python3
"""Generate direct Gamma-point ABACUS LCAO cross-frame overlaps.

One command supports both a contiguous SCF-frame range and the original
``--left/--right`` single-pair interface.  Batch mode validates completed SCF
outputs, calibrates the native two-center grid against ordinary same-frame
``data-SR`` files, manages the audit contract automatically, and invokes this
same file for every adjacent pair.  No second worker script or staging tree is
required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ALGORITHM_VERSION = "direct-pyabacus-gamma-v4-abacus-native-grid"
SCHEMA_VERSION = 2
BATCH_SCHEMA_VERSION = 2
CONTRACT_SCHEMA_VERSION = 1
DEFAULT_INDEX_WIDTH = 4
DEFAULT_NPROC = 4
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "PYTHONUNBUFFERED": "1",
}
EXPECTED_NAO: int | None = None
TABLE_CUTOFF_BOHR: float | None = None
TABLE_NR: int | None = None
EXPECTED_ORBITAL_RCUT_BOHR: float | None = None
TABLE_MODE = "uniform_fft"
LEGACY_PARAMETERS: dict[str, float] | None = None
ACTIVE_CONTRACT: dict[str, Any] | None = None
ACTIVE_CONTRACT_PATH: Path | None = None
ACTIVE_CONTRACT_SHA256: str | None = None
SPECIES_ELEMENT_MAP: dict[str, str] = {}
ZERO_IDENTITY_ATOL = 1.0e-7
SPARSE_THRESHOLD = 1.0e-10
ANGSTROM_TO_BOHR = 1.8897261254578281

SECTION_NAMES = {
    "ATOMIC_SPECIES",
    "NUMERICAL_ORBITAL",
    "LATTICE_CONSTANT",
    "LATTICE_VECTORS",
    "ATOMIC_POSITIONS",
}
ANGULAR_LABELS = "spdfghiklmnoqrtuvwxyz"


class DirectOverlapError(RuntimeError):
    """Raised for invalid inputs or an unsafe/incomplete output state."""


@dataclass(frozen=True)
class OrbitalSpec:
    species: str
    element: str
    path: Path
    sha256: str
    rcut_bohr: float
    lmax: int
    nzeta: tuple[int, ...]

    @property
    def nao_per_atom(self) -> int:
        return sum((2 * angular_momentum + 1) * count for angular_momentum, count in enumerate(self.nzeta))


@dataclass(frozen=True)
class AODescriptor:
    angular_momentum: int
    zeta: int
    magnetic_m: int


@dataclass
class Structure:
    path: Path
    sha256: str
    species: tuple[str, ...]
    orbital_names: tuple[str, ...]
    lattice_constant_bohr: float
    lattice_vectors: np.ndarray
    cell_bohr: np.ndarray
    coordinate_mode: str
    positions_by_type_bohr: tuple[np.ndarray, ...]

    @property
    def atoms_per_type(self) -> tuple[int, ...]:
        return tuple(int(positions.shape[0]) for positions in self.positions_by_type_bohr)


@dataclass(frozen=True)
class AtomRecord:
    type_index: int
    atom_index_within_type: int
    position_bohr: tuple[float, float, float]
    ao_start: int
    nao: int


@dataclass(frozen=True)
class NeighborImage:
    left_atom: int
    right_atom: int
    displacement_bohr: tuple[float, float, float]


@dataclass
class PreparedInputs:
    left: Structure
    right: Structure
    orbitals: tuple[OrbitalSpec, ...]
    basis_by_type: tuple[tuple[AODescriptor, ...], ...]
    left_atoms: tuple[AtomRecord, ...]
    right_atoms: tuple[AtomRecord, ...]
    neighbors_by_translation: dict[tuple[int, int, int], tuple[NeighborImage, ...]]
    nao: int


@dataclass
class IntegratorContext:
    module_nao: Any
    module_base: Any
    radial_collection: Any
    transformer: Any
    integrator: Any
    pyabacus_version: str
    contract_checks: dict[str, Any]


@dataclass(frozen=True)
class PairTask:
    order: int
    left_label: str
    right_label: str
    left_stru: Path
    right_stru: Path
    output: Path

    @property
    def metadata(self) -> Path:
        return metadata_path_for(self.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate direct Gamma-point cross overlaps either for an inclusive "
            "SCF frame range or for one explicit --left/--right STRU pair."
        )
    )
    pair = parser.add_argument_group("single-pair compatibility mode")
    pair.add_argument("--left", type=Path, help="Left/time-t single-frame ABACUS STRU.")
    pair.add_argument("--right", type=Path, help="Right/time-(t+dt) single-frame ABACUS STRU.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Single-pair output tdoverlap.npy. Required in --left/--right mode unless --parse-only.",
    )
    parser.add_argument(
        "--orbital-dir",
        type=Path,
        help=(
            "Directory containing NUMERICAL_ORBITAL files. Relative orbital names are "
            "otherwise searched beside the left and right STRU files."
        ),
    )
    parser.add_argument(
        "--species-element-map",
        nargs="+",
        metavar="SPECIES=ELEMENT",
        help=(
            "Explicit ABACUS species-label to orbital Element mapping, for example "
            "H1=H H2=H. Labels not listed must match the orbital Element exactly."
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        help="Single-pair compatibility input: validated contract generated by this script's batch mode.",
    )
    parser.add_argument(
        "--expected-nao",
        type=int,
        help="Optional required AO dimension; normally read from --contract.",
    )
    parser.add_argument(
        "--expected-orbital-rcut-bohr",
        type=float,
        help="Optional required maximum numerical-orbital cutoff; normally read from --contract.",
    )
    parser.add_argument(
        "--table-cutoff-bohr",
        type=float,
        help="Explicit table cutoff for manual mode; prefer --contract.",
    )
    parser.add_argument(
        "--table-nr",
        type=int,
        help="Explicit table size for manual mode; prefer --contract.",
    )
    parser.add_argument(
        "--table-mode",
        choices=("uniform_fft", "legacy_kgrid"),
        default="uniform_fft",
        help="Manual table construction mode; ignored when --contract is supplied.",
    )
    parser.add_argument("--lcao-ecut", type=float)
    parser.add_argument("--lcao-dk", type=float)
    parser.add_argument("--lcao-dr", type=float)
    parser.add_argument("--lcao-rmax", type=float)
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Validate STRU/orbital parsing, AO layout, and complete PBC enumeration without importing pyabacus.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when output and completion metadata validate against the current inputs.",
    )
    batch = parser.add_argument_group("contiguous SCF-frame batch mode")
    batch.add_argument(
        "--scf-root",
        type=Path,
        help=(
            "Root containing FRAME/STRU and FRAME/OUT.ABACUS/"
            "{INPUT,data-SR-sparse_SPIN0.csr,running_scf.log}."
        ),
    )
    batch.add_argument("--output-root", type=Path, help="Batch output root.")
    batch.add_argument("--start", type=int, help="First frame, inclusive.")
    batch.add_argument("--end", type=int, help="Last frame, inclusive; N frames produce N-1 overlaps.")
    batch.add_argument("--index-width", type=int, default=DEFAULT_INDEX_WIDTH)
    batch.add_argument(
        "--calibration-indices",
        nargs="+",
        help="Representative padded frame labels; default is first/middle/last left frame.",
    )
    batch.add_argument(
        "--reference-pattern",
        default="{index}/OUT.ABACUS/data-SR-sparse_SPIN0.csr",
        help="Same-frame S(R) calibration path relative to --scf-root.",
    )
    batch.add_argument(
        "--contract-json",
        type=Path,
        help="Automatic audit artifact; default OUTPUT_ROOT/tdoverlap-contract.json.",
    )
    batch.add_argument(
        "--report-json",
        type=Path,
        help="Batch report; default OUTPUT_ROOT/reports/batch-<UTC>-<PID>.json.",
    )
    batch.add_argument("--nproc", type=int, default=DEFAULT_NPROC)
    batch.add_argument("--rtol", type=float, default=1.0e-6)
    batch.add_argument("--atol", type=float, default=1.0e-7)
    batch.add_argument("--max-abs-tol", type=float, default=1.0e-6)
    batch.add_argument("--relative-frobenius-tol", type=float, default=1.0e-7)
    batch.add_argument("--zero-identity-atol", type=float, default=1.0e-7)
    batch.add_argument("--max-table-cutoff-bohr", type=float, default=64.0)
    batch.add_argument("--recalibrate", action="store_true")
    batch.add_argument("--calibrate-only", action="store_true")
    batch.add_argument("--preflight-only", action="store_true")
    batch.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages; errors are still printed.")
    return parser


def configure_contract(args: argparse.Namespace) -> None:
    """Install a validated auto contract or a fully explicit manual contract."""

    global EXPECTED_NAO, EXPECTED_ORBITAL_RCUT_BOHR, TABLE_CUTOFF_BOHR, TABLE_NR
    global TABLE_MODE, LEGACY_PARAMETERS, ACTIVE_CONTRACT, ACTIVE_CONTRACT_PATH
    global ACTIVE_CONTRACT_SHA256, SPECIES_ELEMENT_MAP, ZERO_IDENTITY_ATOL

    # This module is also imported by the calibrator.  Reset every mutable
    # setting so repeated candidate trials cannot inherit the previous one.
    EXPECTED_NAO = None
    EXPECTED_ORBITAL_RCUT_BOHR = None
    TABLE_CUTOFF_BOHR = None
    TABLE_NR = None
    TABLE_MODE = "uniform_fft"
    LEGACY_PARAMETERS = None
    ACTIVE_CONTRACT = None
    ACTIVE_CONTRACT_PATH = None
    ACTIVE_CONTRACT_SHA256 = None
    SPECIES_ELEMENT_MAP = parse_species_element_map(args.species_element_map)
    ZERO_IDENTITY_ATOL = 1.0e-7

    contract: dict[str, Any] | None = None
    if args.contract is not None:
        path = resolved_input(args.contract, "auto-contract JSON")
        try:
            contract = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DirectOverlapError(f"Cannot read auto contract {path}: {exc}") from exc
        if contract.get("schema_version") != 1 or contract.get("status") not in {
            "validated",
            "self_validated",
        }:
            raise DirectOverlapError(f"Auto contract is not complete and validated: {path}")
        declared_id = contract.get("contract_id")
        calculated_id = contract_identifier(contract)
        if declared_id != calculated_id:
            raise DirectOverlapError(
                f"Auto contract ID/content mismatch in {path}; recalibrate rather than editing it"
            )
        declared_worker_sha = contract.get("provenance", {}).get("worker_sha256")
        current_worker_sha = sha256_file(Path(__file__).resolve())
        if declared_worker_sha != current_worker_sha:
            raise DirectOverlapError(
                "The direct-overlap worker differs from the implementation covered by "
                f"{path}; recalibrate the contract"
            )
        basis = contract.get("basis", {})
        contract_mapping = {
            str(species): str(element)
            for species, element in basis.get("species_element_map", {}).items()
        }
        if SPECIES_ELEMENT_MAP and SPECIES_ELEMENT_MAP != contract_mapping:
            raise DirectOverlapError(
                "--species-element-map differs from the mapping in the validated auto contract"
            )
        SPECIES_ELEMENT_MAP = contract_mapping
        table = contract.get("table", {})
        args.expected_nao = basis.get("nao")
        args.expected_orbital_rcut_bohr = basis.get("max_rcut_bohr")
        args.table_cutoff_bohr = table.get("cutoff_bohr")
        args.table_nr = table.get("nr")
        args.table_mode = table.get("mode")
        ZERO_IDENTITY_ATOL = float(table.get("zero_identity_atol", ZERO_IDENTITY_ATOL))
        legacy = table.get("legacy_parameters")
        if legacy is not None:
            LEGACY_PARAMETERS = {key: float(value) for key, value in legacy.items()}
        ACTIVE_CONTRACT = contract
        ACTIVE_CONTRACT_PATH = path
        ACTIVE_CONTRACT_SHA256 = sha256_file(path)

    if args.table_cutoff_bohr is None or args.table_nr is None:
        raise DirectOverlapError("Use --contract or provide both --table-cutoff-bohr and --table-nr")
    if args.table_mode not in {"uniform_fft", "legacy_kgrid"}:
        raise DirectOverlapError(f"Unsupported table mode: {args.table_mode!r}")

    expected_nao = None if args.expected_nao is None else int(args.expected_nao)
    if expected_nao is not None and expected_nao < 1:
        raise DirectOverlapError("--expected-nao must be at least 1")
    rcut = None if args.expected_orbital_rcut_bohr is None else float(args.expected_orbital_rcut_bohr)
    if rcut is not None and (not math.isfinite(rcut) or rcut <= 0.0):
        raise DirectOverlapError("--expected-orbital-rcut-bohr must be finite and positive")

    table_cutoff = float(args.table_cutoff_bohr)
    table_nr = int(args.table_nr)
    if not math.isfinite(table_cutoff) or table_cutoff <= 0.0 or table_nr < 2:
        raise DirectOverlapError("The table cutoff must be positive and --table-nr must be at least 2")
    if (
        args.table_mode == "uniform_fft"
        and rcut is not None
        and table_cutoff + 1.0e-12 < 2.0 * rcut
    ):
        raise DirectOverlapError(
            f"Table cutoff {table_cutoff} is smaller than 2*max_rcut={2.0 * rcut}"
        )

    if args.table_mode == "legacy_kgrid" and LEGACY_PARAMETERS is None:
        values = (args.lcao_ecut, args.lcao_dk, args.lcao_dr, args.lcao_rmax)
        if any(value is None for value in values):
            raise DirectOverlapError("legacy_kgrid requires all four --lcao-* values")
        LEGACY_PARAMETERS = {
            "lcao_ecut": float(args.lcao_ecut),
            "lcao_dk": float(args.lcao_dk),
            "lcao_dr": float(args.lcao_dr),
            "lcao_rmax": float(args.lcao_rmax),
        }

    EXPECTED_NAO = expected_nao
    EXPECTED_ORBITAL_RCUT_BOHR = rcut
    TABLE_CUTOFF_BOHR = table_cutoff
    TABLE_NR = table_nr
    TABLE_MODE = args.table_mode


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def contract_identifier(contract: dict[str, Any]) -> str:
    """Return the stable content ID used by automatic table contracts."""
    payload = dict(contract)
    payload.pop("contract_id", None)
    payload.pop("created_at_utc", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_input(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    # Keep the logical path rather than resolving the final symlink.  Frame STRU
    # files are commonly symlinked while their .orb siblings live beside the
    # link, not beside its target in OUT.ABACUS/STRU.
    absolute = Path(os.path.abspath(absolute))
    if not absolute.is_file():
        raise DirectOverlapError(f"Missing required {label}: {absolute}")
    return absolute


def strip_comment(line: str) -> str:
    return line.split("//", 1)[0].split("#", 1)[0].strip()


def read_clean_lines(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            clean = strip_comment(raw_line)
            if clean:
                result.append((line_number, clean))
    return result


def split_sections(path: Path) -> dict[str, list[tuple[int, str]]]:
    lines = read_clean_lines(path)
    starts: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index, (_, line) in enumerate(lines):
        key = line.upper()
        if key in SECTION_NAMES:
            if key in seen:
                raise DirectOverlapError(f"Duplicate {key} section in {path}")
            starts.append((index, key))
            seen.add(key)

    missing = sorted(SECTION_NAMES - seen)
    if missing:
        raise DirectOverlapError(f"Missing section(s) in {path}: {', '.join(missing)}")

    sections: dict[str, list[tuple[int, str]]] = {}
    starts.sort()
    for position, (start_index, key) in enumerate(starts):
        end_index = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        sections[key] = lines[start_index + 1 : end_index]
    return sections


def parse_three_floats(path: Path, line_number: int, text: str, context: str) -> list[float]:
    fields = text.split()
    if len(fields) < 3:
        raise DirectOverlapError(f"Expected three coordinates for {context} at {path}:{line_number}")
    try:
        values = [float(fields[0]), float(fields[1]), float(fields[2])]
    except ValueError as exc:
        raise DirectOverlapError(f"Invalid coordinates for {context} at {path}:{line_number}: {text}") from exc
    if not all(math.isfinite(value) for value in values):
        raise DirectOverlapError(f"Non-finite coordinates for {context} at {path}:{line_number}")
    return values


def parse_structure(path: Path) -> Structure:
    path = resolved_input(path, "STRU")
    sections = split_sections(path)

    species_lines = sections["ATOMIC_SPECIES"]
    if not species_lines:
        raise DirectOverlapError(f"ATOMIC_SPECIES is empty in {path}")
    species: list[str] = []
    for line_number, text in species_lines:
        fields = text.split()
        if len(fields) < 3:
            raise DirectOverlapError(f"Invalid ATOMIC_SPECIES record at {path}:{line_number}: {text}")
        symbol = fields[0]
        if symbol in species:
            raise DirectOverlapError(f"Duplicate atomic species {symbol!r} in {path}")
        species.append(symbol)

    orbital_lines = sections["NUMERICAL_ORBITAL"]
    if len(orbital_lines) != len(species):
        raise DirectOverlapError(
            f"NUMERICAL_ORBITAL count ({len(orbital_lines)}) does not match ATOMIC_SPECIES "
            f"count ({len(species)}) in {path}"
        )
    orbital_names = tuple(text.split()[0] for _, text in orbital_lines)

    lattice_constant_lines = sections["LATTICE_CONSTANT"]
    if len(lattice_constant_lines) != 1:
        raise DirectOverlapError(f"LATTICE_CONSTANT must contain exactly one value in {path}")
    try:
        lattice_constant = float(lattice_constant_lines[0][1].split()[0])
    except ValueError as exc:
        raise DirectOverlapError(f"Invalid LATTICE_CONSTANT in {path}") from exc
    if not math.isfinite(lattice_constant) or lattice_constant <= 0.0:
        raise DirectOverlapError(f"LATTICE_CONSTANT must be finite and positive in {path}")

    lattice_lines = sections["LATTICE_VECTORS"]
    if len(lattice_lines) != 3:
        raise DirectOverlapError(f"LATTICE_VECTORS must contain exactly three vectors in {path}")
    lattice_vectors = np.asarray(
        [parse_three_floats(path, number, text, "lattice vector") for number, text in lattice_lines],
        dtype=np.float64,
    )
    cell_bohr = lattice_vectors * lattice_constant
    determinant = float(np.linalg.det(cell_bohr))
    if not math.isfinite(determinant) or abs(determinant) < 1.0e-12:
        raise DirectOverlapError(f"Lattice is singular or non-finite in {path}")

    position_lines = sections["ATOMIC_POSITIONS"]
    if not position_lines:
        raise DirectOverlapError(f"ATOMIC_POSITIONS is empty in {path}")
    coordinate_mode_raw = position_lines[0][1].split()[0].lower()
    if coordinate_mode_raw.startswith("direct"):
        coordinate_mode = "direct"
    elif coordinate_mode_raw == "cartesian":
        coordinate_mode = "cartesian"
    elif coordinate_mode_raw in {"cartesian_angstrom", "cartesian_ang"}:
        coordinate_mode = "cartesian_angstrom"
    else:
        raise DirectOverlapError(
            f"Unsupported ATOMIC_POSITIONS mode {position_lines[0][1]!r} in {path}; "
            "supported modes are Direct, Cartesian, and Cartesian_angstrom"
        )

    cursor = 1
    parsed_order: list[str] = []
    positions_raw_by_symbol: dict[str, np.ndarray] = {}
    while cursor < len(position_lines):
        symbol_line_number, symbol_text = position_lines[cursor]
        symbol = symbol_text.split()[0]
        cursor += 1
        if cursor + 1 >= len(position_lines):
            raise DirectOverlapError(f"Incomplete ATOMIC_POSITIONS block for {symbol!r} at {path}:{symbol_line_number}")

        cursor += 1  # The magnetization line does not affect the overlap basis.
        count_line_number, count_text = position_lines[cursor]
        cursor += 1
        try:
            atom_count = int(count_text.split()[0])
        except ValueError as exc:
            raise DirectOverlapError(f"Invalid atom count at {path}:{count_line_number}: {count_text}") from exc
        if atom_count < 0:
            raise DirectOverlapError(f"Negative atom count at {path}:{count_line_number}")
        if cursor + atom_count > len(position_lines):
            raise DirectOverlapError(f"Not enough coordinate records for {symbol!r} in {path}")
        if symbol in positions_raw_by_symbol:
            raise DirectOverlapError(f"Duplicate ATOMIC_POSITIONS block for {symbol!r} in {path}")

        coordinates = [
            parse_three_floats(path, number, text, f"{symbol} atom")
            for number, text in position_lines[cursor : cursor + atom_count]
        ]
        cursor += atom_count
        parsed_order.append(symbol)
        positions_raw_by_symbol[symbol] = np.asarray(coordinates, dtype=np.float64).reshape(atom_count, 3)

    if tuple(parsed_order) != tuple(species):
        raise DirectOverlapError(
            f"ATOMIC_POSITIONS species order {parsed_order} does not match ATOMIC_SPECIES order {species} in {path}"
        )

    positions_by_type_bohr: list[np.ndarray] = []
    for symbol in species:
        raw_positions = positions_raw_by_symbol[symbol]
        if coordinate_mode == "direct":
            positions_bohr = raw_positions @ cell_bohr
        elif coordinate_mode == "cartesian":
            positions_bohr = raw_positions * lattice_constant
        else:
            positions_bohr = raw_positions * ANGSTROM_TO_BOHR
        positions_by_type_bohr.append(np.ascontiguousarray(positions_bohr, dtype=np.float64))

    return Structure(
        path=path,
        sha256=sha256_file(path),
        species=tuple(species),
        orbital_names=orbital_names,
        lattice_constant_bohr=lattice_constant,
        lattice_vectors=lattice_vectors,
        cell_bohr=np.ascontiguousarray(cell_bohr, dtype=np.float64),
        coordinate_mode=coordinate_mode,
        positions_by_type_bohr=tuple(positions_by_type_bohr),
    )


def locate_orbital(
    orbital_name: str,
    orbital_dir: Path | None,
    left_parent: Path,
    right_parent: Path,
) -> Path:
    orbital_path = Path(orbital_name).expanduser()
    candidates: list[Path] = []
    if orbital_path.is_absolute():
        candidates.append(orbital_path)
    else:
        if orbital_dir is not None:
            candidates.append(orbital_dir.expanduser() / orbital_path)
        candidates.append(left_parent / orbital_path)
        candidates.append(right_parent / orbital_path)

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved.resolve()
    raise DirectOverlapError(
        f"Cannot find required orbital file {orbital_name!r}; checked: " + ", ".join(checked)
    )


def parse_species_element_map(values: Sequence[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in values or ():
        if raw.count("=") != 1:
            raise DirectOverlapError(
                f"Invalid species/element mapping {raw!r}; expected SPECIES=ELEMENT"
            )
        species, element = (field.strip() for field in raw.split("=", 1))
        if not species or not element or any(character.isspace() for character in species + element):
            raise DirectOverlapError(
                f"Invalid species/element mapping {raw!r}; labels must be non-empty tokens"
            )
        previous = mapping.get(species)
        if previous is not None and previous != element:
            raise DirectOverlapError(
                f"Conflicting element mappings for species {species!r}: {previous!r} and {element!r}"
            )
        mapping[species] = element
    return dict(sorted(mapping.items()))


def parse_orbital_spec(species: str, path: Path, expected_element: str | None = None) -> OrbitalSpec:
    element: str | None = None
    rcut: float | None = None
    lmax: int | None = None
    nzeta_by_l: dict[int, int] = {}

    for _, line in read_clean_lines(path):
        fields = line.split()
        if fields and fields[0].lower() == "element" and len(fields) >= 2:
            element = fields[-1]
        elif line.lower().startswith("radius cutoff") and fields:
            try:
                rcut = float(fields[-1])
            except ValueError as exc:
                raise DirectOverlapError(f"Invalid Radius Cutoff in orbital file {path}") from exc
        elif fields and fields[0].lower() == "lmax" and len(fields) >= 2:
            try:
                lmax = int(fields[-1])
            except ValueError as exc:
                raise DirectOverlapError(f"Invalid Lmax in orbital file {path}") from exc
        elif len(fields) >= 3 and fields[0].lower() == "number" and fields[1].lower() == "of":
            compact = "".join(fields[2:]).lower()
            if "orbital" not in compact or ">" not in compact:
                continue
            angular_label = compact[0]
            if angular_label not in ANGULAR_LABELS:
                continue
            try:
                count = int(fields[-1])
            except ValueError as exc:
                raise DirectOverlapError(f"Invalid orbital multiplicity in {path}: {line}") from exc
            nzeta_by_l[ANGULAR_LABELS.index(angular_label)] = count

        if line.upper() == "SUMMARY END":
            break

    if element is None or rcut is None or lmax is None:
        raise DirectOverlapError(f"Could not read Element, Radius Cutoff, and Lmax from orbital file {path}")
    expected = species if expected_element is None else expected_element
    if element != expected:
        raise DirectOverlapError(
            f"Orbital file {path} declares element {element!r}, expected {expected!r} "
            f"for ABACUS species {species!r}"
        )
    if not math.isfinite(rcut) or rcut <= 0.0 or lmax < 0:
        raise DirectOverlapError(f"Invalid rcut/Lmax metadata in orbital file {path}")

    missing_l = [angular_momentum for angular_momentum in range(lmax + 1) if angular_momentum not in nzeta_by_l]
    if missing_l:
        raise DirectOverlapError(f"Orbital file {path} is missing multiplicities for l={missing_l}")
    nzeta = tuple(nzeta_by_l[angular_momentum] for angular_momentum in range(lmax + 1))
    if any(count < 0 for count in nzeta) or not any(count > 0 for count in nzeta):
        raise DirectOverlapError(f"Invalid orbital multiplicities in {path}: {nzeta}")

    return OrbitalSpec(
        species=species,
        element=element,
        path=path,
        sha256=sha256_file(path),
        rcut_bohr=rcut,
        lmax=lmax,
        nzeta=nzeta,
    )


def resolve_orbitals(
    left: Structure,
    right: Structure,
    orbital_dir: Path | None,
    species_element_map: dict[str, str] | None = None,
) -> tuple[OrbitalSpec, ...]:
    if left.orbital_names != right.orbital_names:
        raise DirectOverlapError(
            "Left and right STRU files must declare the same NUMERICAL_ORBITAL files in the same order"
        )
    mapping = species_element_map or {}
    unknown = sorted(set(mapping) - set(left.species))
    if unknown:
        raise DirectOverlapError(
            f"--species-element-map contains species absent from STRU: {unknown}"
        )
    orbitals: list[OrbitalSpec] = []
    for species, orbital_name in zip(left.species, left.orbital_names):
        orbital_path = locate_orbital(orbital_name, orbital_dir, left.path.parent, right.path.parent)
        orbitals.append(parse_orbital_spec(species, orbital_path, mapping.get(species)))

    maximum_rcut = max(orbital.rcut_bohr for orbital in orbitals)
    if EXPECTED_ORBITAL_RCUT_BOHR is not None and not math.isclose(
        maximum_rcut, EXPECTED_ORBITAL_RCUT_BOHR, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise DirectOverlapError(
            f"This configured route requires maximum orbital rcut={EXPECTED_ORBITAL_RCUT_BOHR} Bohr "
            f"for cutoff={TABLE_CUTOFF_BOHR}, but the supplied orbitals give {maximum_rcut} Bohr"
        )
    return tuple(orbitals)


def validate_pair_compatibility(left: Structure, right: Structure) -> None:
    if left.species != right.species:
        raise DirectOverlapError(f"Species mismatch: left={left.species}, right={right.species}")
    if left.atoms_per_type != right.atoms_per_type:
        raise DirectOverlapError(
            f"Atom-count mismatch: left={left.atoms_per_type}, right={right.atoms_per_type}"
        )
    if not math.isclose(
        left.lattice_constant_bohr,
        right.lattice_constant_bohr,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ) or not np.allclose(left.lattice_vectors, right.lattice_vectors, rtol=0.0, atol=1.0e-8):
        raise DirectOverlapError("Left and right STRU lattice constants/vectors differ beyond 1e-8")


def physical_m_from_abacus_mm(mm: int) -> int:
    """Convert ABACUS's 0,1,2,... slot to physical 0,+1,-1,+2,-2,... m."""
    return -(mm // 2) if mm % 2 == 0 else (mm + 1) // 2


def build_basis_by_type(orbitals: Sequence[OrbitalSpec]) -> tuple[tuple[AODescriptor, ...], ...]:
    result: list[tuple[AODescriptor, ...]] = []
    for orbital in orbitals:
        descriptors: list[AODescriptor] = []
        for angular_momentum, zeta_count in enumerate(orbital.nzeta):
            for zeta in range(zeta_count):
                for mm in range(2 * angular_momentum + 1):
                    descriptors.append(
                        AODescriptor(
                            angular_momentum=angular_momentum,
                            zeta=zeta,
                            magnetic_m=physical_m_from_abacus_mm(mm),
                        )
                    )
        if len(descriptors) != orbital.nao_per_atom:
            raise DirectOverlapError(f"Internal AO-layout error for {orbital.species}")
        result.append(tuple(descriptors))
    return tuple(result)


def build_atom_records(
    structure: Structure,
    orbitals: Sequence[OrbitalSpec],
) -> tuple[tuple[AtomRecord, ...], int]:
    records: list[AtomRecord] = []
    ao_start = 0
    for type_index, (positions, orbital) in enumerate(zip(structure.positions_by_type_bohr, orbitals)):
        for atom_index, position in enumerate(positions):
            records.append(
                AtomRecord(
                    type_index=type_index,
                    atom_index_within_type=atom_index,
                    position_bohr=(float(position[0]), float(position[1]), float(position[2])),
                    ao_start=ao_start,
                    nao=orbital.nao_per_atom,
                )
            )
            ao_start += orbital.nao_per_atom
    return tuple(records), ao_start


def enumerate_neighbor_images(
    left: Structure,
    left_atoms: Sequence[AtomRecord],
    right_atoms: Sequence[AtomRecord],
    orbitals: Sequence[OrbitalSpec],
) -> dict[tuple[int, int, int], tuple[NeighborImage, ...]]:
    """Enumerate every periodic atom image whose orbital supports overlap."""
    cell = left.cell_bohr
    inverse_cell = np.linalg.inv(cell)
    reciprocal_column_norms = np.linalg.norm(inverse_cell, axis=0)
    mutable: dict[tuple[int, int, int], list[NeighborImage]] = {}

    for left_atom_index, left_atom in enumerate(left_atoms):
        left_position = np.asarray(left_atom.position_bohr, dtype=np.float64)
        left_rcut = orbitals[left_atom.type_index].rcut_bohr
        for right_atom_index, right_atom in enumerate(right_atoms):
            right_position = np.asarray(right_atom.position_bohr, dtype=np.float64)
            pair_cutoff = left_rcut + orbitals[right_atom.type_index].rcut_bohr
            delta_fractional = (right_position - left_position) @ inverse_cell

            fractional_radius = pair_cutoff * reciprocal_column_norms
            lower = np.floor(-delta_fractional - fractional_radius).astype(int) - 1
            upper = np.ceil(-delta_fractional + fractional_radius).astype(int) + 1
            spans = upper - lower + 1
            candidate_count = int(np.prod(spans, dtype=np.int64))
            if candidate_count > 1_000_000:
                raise DirectOverlapError(
                    "Periodic-image search became unreasonably large; check lattice units and orbital cutoffs"
                )

            ranges = [range(int(lower[axis]), int(upper[axis]) + 1) for axis in range(3)]
            for translation in itertools.product(*ranges):
                translation_vector = np.asarray(translation, dtype=np.float64)
                displacement = right_position + translation_vector @ cell - left_position
                if float(np.linalg.norm(displacement)) < pair_cutoff:
                    key = (int(translation[0]), int(translation[1]), int(translation[2]))
                    mutable.setdefault(key, []).append(
                        NeighborImage(
                            left_atom=left_atom_index,
                            right_atom=right_atom_index,
                            displacement_bohr=(
                                float(displacement[0]),
                                float(displacement[1]),
                                float(displacement[2]),
                            ),
                        )
                    )

    if not mutable:
        raise DirectOverlapError("No overlapping periodic atom images were found")
    return {translation: tuple(mutable[translation]) for translation in sorted(mutable)}


def prepare_inputs(
    left_path: Path,
    right_path: Path,
    orbital_dir: Path | None,
    species_element_map: dict[str, str] | None = None,
) -> PreparedInputs:
    left = parse_structure(left_path)
    right = parse_structure(right_path)
    validate_pair_compatibility(left, right)
    orbitals = resolve_orbitals(left, right, orbital_dir, species_element_map)
    basis_by_type = build_basis_by_type(orbitals)
    left_atoms, left_nao = build_atom_records(left, orbitals)
    right_atoms, right_nao = build_atom_records(right, orbitals)
    if left_nao != right_nao:
        raise DirectOverlapError(f"Left/right AO counts differ: {left_nao} vs {right_nao}")
    if EXPECTED_NAO is not None and left_nao != EXPECTED_NAO:
        raise DirectOverlapError(
            f"This configured route requires {EXPECTED_NAO} orbitals per frame, but the inputs give {left_nao}"
        )
    neighbors = enumerate_neighbor_images(left, left_atoms, right_atoms, orbitals)
    return PreparedInputs(
        left=left,
        right=right,
        orbitals=orbitals,
        basis_by_type=basis_by_type,
        left_atoms=left_atoms,
        right_atoms=right_atoms,
        neighbors_by_translation=neighbors,
        nao=left_nao,
    )


def validate_prepared_against_active_contract(prepared: PreparedInputs) -> None:
    if ACTIVE_CONTRACT is None:
        return
    basis = ACTIVE_CONTRACT.get("basis", {})
    expected = {
        "species": list(prepared.left.species),
        "atoms_per_type": list(prepared.left.atoms_per_type),
        "nao": prepared.nao,
        "max_rcut_bohr": max(orbital.rcut_bohr for orbital in prepared.orbitals),
        "species_element_map": {
            orbital.species: orbital.element
            for orbital in prepared.orbitals
            if orbital.species != orbital.element
        },
    }
    for key, actual in expected.items():
        declared = basis.get(key)
        if key == "max_rcut_bohr":
            if declared is None or not math.isclose(
                float(declared), float(actual), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise DirectOverlapError(
                    f"Contract {key}={declared!r} does not match inputs {actual!r}"
                )
        elif declared != actual:
            raise DirectOverlapError(
                f"Contract {key}={declared!r} does not match inputs {actual!r}"
            )
    declared_orbitals = basis.get("orbitals")
    actual_orbitals = [
        {
            "species": orbital.species,
            "element": orbital.element,
            "sha256": orbital.sha256,
            "rcut_bohr": orbital.rcut_bohr,
            "lmax": orbital.lmax,
            "nzeta": list(orbital.nzeta),
            "nao_per_atom": orbital.nao_per_atom,
        }
        for orbital in prepared.orbitals
    ]
    if declared_orbitals != actual_orbitals:
        raise DirectOverlapError("Orbital files/layout do not match the validated auto contract")


def import_pyabacus() -> tuple[Any, Any, str]:
    try:
        from pyabacus import ModuleBase as module_base
        from pyabacus import ModuleNAO as module_nao
    except Exception as exc:
        raise DirectOverlapError(
            "pyabacus with ModuleBase and ModuleNAO is unavailable in the active Python environment. "
            "Activate the hamnext-v0 environment after its pyabacus wheel has been installed. "
            f"Original import error: {exc}"
        ) from exc
    try:
        version = importlib.metadata.version("pyabacus")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return module_nao, module_base, version


def pyabacus_fingerprint() -> dict[str, Any]:
    """Identify the binary implementation covered by an auto contract."""
    module_nao, module_base, version = import_pyabacus()
    distribution = importlib.metadata.distribution("pyabacus")
    distribution_path = Path(distribution._path).resolve()
    direct_url_path = distribution_path / "direct_url.json"
    direct_url: dict[str, Any] | None = None
    if direct_url_path.is_file():
        try:
            direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DirectOverlapError(f"Invalid pyabacus direct_url.json: {direct_url_path}") from exc
    packages: list[dict[str, Any]] = []
    for name, module in (("ModuleNAO", module_nao), ("ModuleBase", module_base)):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            raise DirectOverlapError(f"Cannot locate pyabacus {name} package")
        package_init = resolved_input(Path(raw_path), f"pyabacus {name} package")
        relevant_files = sorted(
            (
                path
                for path in package_init.parent.iterdir()
                if path.is_file() and path.suffix in {".py", ".so"}
            ),
            key=lambda path: path.name,
        )
        if not relevant_files or not any(path.suffix == ".so" for path in relevant_files):
            raise DirectOverlapError(f"Cannot locate pyabacus {name} extension/shared library")
        packages.append(
            {
                "name": name,
                "directory": str(package_init.parent),
                "files": [
                    {
                        "name": path.name,
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in relevant_files
                ],
            }
        )
    return {
        "pyabacus_version": version,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "distribution": {
            "path": str(distribution_path),
            "direct_url": direct_url,
            "direct_url_sha256": sha256_file(direct_url_path) if direct_url is not None else None,
        },
        "packages": packages,
    }


def build_integrator(prepared: PreparedInputs) -> IntegratorContext:
    module_nao, module_base, version = import_pyabacus()
    runtime_fingerprint = pyabacus_fingerprint()
    if ACTIVE_CONTRACT is not None:
        calibrated_fingerprint = ACTIVE_CONTRACT.get("software")
        if calibrated_fingerprint != runtime_fingerprint:
            raise DirectOverlapError(
                "The active pyabacus/Python implementation differs from the one "
                "validated by the auto contract; recalibrate the contract"
            )
    orbital_paths = [str(orbital.path) for orbital in prepared.orbitals]

    radial_collection = module_nao.RadialCollection()
    radial_collection.build(len(orbital_paths), orbital_paths, "o")
    if int(radial_collection.ntype) != len(prepared.orbitals):
        raise DirectOverlapError(
            f"pyabacus loaded {radial_collection.ntype} orbital types, expected {len(prepared.orbitals)}"
        )

    for type_index, orbital in enumerate(prepared.orbitals):
        loaded_symbol = str(radial_collection.symbol(type_index))
        if loaded_symbol != orbital.element:
            raise DirectOverlapError(
                f"pyabacus type {type_index} is {loaded_symbol!r}, expected orbital "
                f"element {orbital.element!r} for ABACUS species {orbital.species!r}"
            )
        for angular_momentum, expected_nzeta in enumerate(orbital.nzeta):
            loaded_nzeta = int(radial_collection.nzeta(type_index, angular_momentum))
            if loaded_nzeta != expected_nzeta:
                raise DirectOverlapError(
                    f"pyabacus nzeta mismatch for {orbital.species} l={angular_momentum}: "
                    f"loaded {loaded_nzeta}, expected {expected_nzeta}"
                )

    loaded_rcut = float(radial_collection.rcut_max)
    if EXPECTED_ORBITAL_RCUT_BOHR is not None and not math.isclose(
        loaded_rcut, EXPECTED_ORBITAL_RCUT_BOHR, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise DirectOverlapError(
            f"pyabacus loaded maximum rcut={loaded_rcut}, expected {EXPECTED_ORBITAL_RCUT_BOHR} Bohr"
        )
    if (
        TABLE_CUTOFF_BOHR is None
        or (
            TABLE_MODE == "uniform_fft"
            and TABLE_CUTOFF_BOHR + 1.0e-12 < 2.0 * loaded_rcut
        )
    ):
        raise DirectOverlapError(
            f"Table cutoff {TABLE_CUTOFF_BOHR} is smaller than 2*loaded_rcut={2.0 * loaded_rcut}"
        )

    if TABLE_CUTOFF_BOHR is None or TABLE_NR is None:
        raise DirectOverlapError("Integral table contract was not configured")

    try:
        transformer = module_base.SphericalBesselTransformer(True)
    except TypeError:
        transformer = module_base.SphericalBesselTransformer()
    radial_collection.set_transformer(transformer)
    if TABLE_MODE == "uniform_fft":
        radial_collection.set_uniform_grid(True, TABLE_NR, TABLE_CUTOFF_BOHR, "i", True)
    elif TABLE_MODE == "legacy_kgrid":
        if LEGACY_PARAMETERS is None:
            raise DirectOverlapError("Missing legacy k-grid parameters")
        ecut = LEGACY_PARAMETERS["lcao_ecut"]
        dk = LEGACY_PARAMETERS["lcao_dk"]
        if not math.isfinite(ecut) or ecut <= 0.0 or not math.isfinite(dk) or dk <= 0.0:
            raise DirectOverlapError("Invalid legacy lcao_ecut/lcao_dk")
        nk = int(math.sqrt(ecut) / dk) + 4
        nk += 1 - nk % 2
        kgrid = np.ascontiguousarray(np.arange(nk, dtype=np.float64) * dk)
        radial_collection.set_grid(False, nk, kgrid, "t")
    else:
        raise DirectOverlapError(f"Unsupported table mode: {TABLE_MODE}")

    integrator = module_nao.TwoCenterIntegrator()
    integrator.tabulate(
        radial_collection,
        radial_collection,
        "S",
        TABLE_NR,
        TABLE_CUTOFF_BOHR,
    )
    contract_checks = validate_integrator_contract(
        prepared,
        radial_collection,
        integrator,
        require_fft=TABLE_MODE == "uniform_fft",
    )
    return IntegratorContext(
        module_nao=module_nao,
        module_base=module_base,
        radial_collection=radial_collection,
        transformer=transformer,
        integrator=integrator,
        pyabacus_version=version,
        contract_checks=contract_checks,
    )


def validate_integrator_contract(
    prepared: PreparedInputs,
    radial_collection: Any,
    integrator: Any,
    *,
    require_fft: bool,
) -> dict[str, Any]:
    """Reject numerically unsafe grids before any structure-sized calculation."""

    fft_channels: list[dict[str, Any]] = []
    zero_displacement: list[dict[str, Any]] = []
    failures: list[str] = []
    displacement = np.zeros(3, dtype=np.float64)
    for type_index, (orbital, basis) in enumerate(zip(prepared.orbitals, prepared.basis_by_type)):
        for angular_momentum, nzeta in enumerate(orbital.nzeta):
            for zeta in range(nzeta):
                compliant = bool(
                    radial_collection(type_index, angular_momentum, zeta).is_fft_compliant
                )
                fft_channels.append(
                    {
                        "type_index": type_index,
                        "species": orbital.species,
                        "l": angular_momentum,
                        "zeta": zeta,
                        "is_fft_compliant": compliant,
                    }
                )
                if require_fft and not compliant:
                    failures.append(
                        f"{orbital.species} l={angular_momentum} zeta={zeta} is not FFT compliant"
                    )

        block = np.zeros((len(basis), len(basis)), dtype=np.float64)
        for row, descriptor in enumerate(basis):
            values = np.asarray(
                integrator.snap(
                    type_index,
                    descriptor.angular_momentum,
                    descriptor.zeta,
                    descriptor.magnetic_m,
                    type_index,
                    displacement,
                    False,
                )[0],
                dtype=np.float64,
            )
            if values.shape != (len(basis),):
                raise DirectOverlapError(
                    f"Zero-displacement overlap for {orbital.species} has shape {values.shape}"
                )
            block[row] = values
        max_abs = float(np.max(np.abs(block - np.eye(len(basis))), initial=0.0))
        zero_displacement.append(
            {
                "type_index": type_index,
                "species": orbital.species,
                "shape": list(block.shape),
                "max_abs_from_identity": max_abs,
            }
        )
        if max_abs > ZERO_IDENTITY_ATOL:
            failures.append(
                f"{orbital.species} zero-displacement max_abs={max_abs:.9e} "
                f"> {ZERO_IDENTITY_ATOL:.9e}"
            )

    if failures:
        raise DirectOverlapError("Unsafe integral table contract: " + "; ".join(failures))
    return {
        "table_mode": TABLE_MODE,
        "require_fft": require_fft,
        "fft_channels": fft_channels,
        "zero_displacement": zero_displacement,
        "zero_identity_atol": ZERO_IDENTITY_ATOL,
    }


def calculate_overlap(
    prepared: PreparedInputs,
    context: IntegratorContext,
    quiet: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    overlap = np.zeros((prepared.nao, prepared.nao), dtype=np.float32, order="C")
    snap_calls = 0
    atom_image_pairs = 0
    translation_count = len(prepared.neighbors_by_translation)

    for translation_index, translation in enumerate(sorted(prepared.neighbors_by_translation), start=1):
        if not quiet:
            print(
                f"[direct-overlap] R {translation_index}/{translation_count}: {translation}",
                flush=True,
            )
        for neighbor in prepared.neighbors_by_translation[translation]:
            atom_image_pairs += 1
            left_atom = prepared.left_atoms[neighbor.left_atom]
            right_atom = prepared.right_atoms[neighbor.right_atom]
            right_basis = prepared.basis_by_type[right_atom.type_index]
            displacement = np.ascontiguousarray(neighbor.displacement_bohr, dtype=np.float64)

            for local_row, descriptor in enumerate(prepared.basis_by_type[left_atom.type_index]):
                result = context.integrator.snap(
                    left_atom.type_index,
                    descriptor.angular_momentum,
                    descriptor.zeta,
                    descriptor.magnetic_m,
                    right_atom.type_index,
                    displacement,
                    False,
                )
                snap_calls += 1
                if not result:
                    raise DirectOverlapError("pyabacus TwoCenterIntegrator.snap returned no value block")
                values = np.asarray(result[0], dtype=np.float64)
                if values.shape != (len(right_basis),):
                    raise DirectOverlapError(
                        f"pyabacus snap returned shape {values.shape}, expected {(len(right_basis),)} "
                        f"for right type {prepared.right.species[right_atom.type_index]}"
                    )
                if not np.isfinite(values).all():
                    raise DirectOverlapError("pyabacus produced a non-finite two-center integral")

                # Match the reference sparse writer's per-R threshold, then use
                # float32 R-ordered accumulation as in read_abacus.getHK().
                values[np.abs(values) <= SPARSE_THRESHOLD] = 0.0
                row = left_atom.ao_start + local_row
                column_slice = slice(right_atom.ao_start, right_atom.ao_start + right_atom.nao)
                overlap[row, column_slice] += values.astype(np.float32)

    if overlap.shape != (prepared.nao, prepared.nao):
        raise DirectOverlapError(f"Internal overlap shape error: {overlap.shape}")
    if overlap.dtype != np.float32 or not overlap.flags.c_contiguous:
        raise DirectOverlapError("Internal overlap array is not C-contiguous float32")
    if not np.isfinite(overlap).all():
        raise DirectOverlapError("Final overlap contains NaN or infinity")
    return overlap, {
        "translation_vectors": translation_count,
        "atom_image_pairs": atom_image_pairs,
        "snap_calls": snap_calls,
    }


def orbital_record(orbital: OrbitalSpec) -> dict[str, Any]:
    return {
        "species": orbital.species,
        "element": orbital.element,
        "path": str(orbital.path),
        "sha256": orbital.sha256,
        "rcut_bohr": orbital.rcut_bohr,
        "lmax": orbital.lmax,
        "nzeta": list(orbital.nzeta),
        "nao_per_atom": orbital.nao_per_atom,
    }


def build_input_signature(
    prepared: PreparedInputs,
    script_sha256: str,
    software_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "script_sha256": script_sha256,
        "left_stru": {"path": str(prepared.left.path), "sha256": prepared.left.sha256},
        "right_stru": {"path": str(prepared.right.path), "sha256": prepared.right.sha256},
        "orbitals": [orbital_record(orbital) for orbital in prepared.orbitals],
        "table": {
            "operator": "S",
            "mode": TABLE_MODE,
            "nr": TABLE_NR,
            "cutoff_bohr": TABLE_CUTOFF_BOHR,
            "spacing_bohr": TABLE_CUTOFF_BOHR / (TABLE_NR - 1),
            "orbital_grid_mode": "i",
            "enable_fft": TABLE_MODE == "uniform_fft",
            "sparse_threshold": SPARSE_THRESHOLD,
            "selection": (
                "validated automatic contract" if ACTIVE_CONTRACT is not None else "explicit manual settings"
            ),
            "legacy_parameters": LEGACY_PARAMETERS,
        },
        "auto_contract": (
            {
                "path": str(ACTIVE_CONTRACT_PATH),
                "sha256": ACTIVE_CONTRACT_SHA256,
                "contract_id": ACTIVE_CONTRACT.get("contract_id"),
                "status": ACTIVE_CONTRACT.get("status"),
            }
            if ACTIVE_CONTRACT is not None
            else None
        ),
        "software_fingerprint": software_fingerprint,
        "gamma_only": True,
        "shape": [prepared.nao, prepared.nao],
        "dtype": "float32",
        "atoms_per_type": list(prepared.left.atoms_per_type),
    }


def metadata_path_for(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def validated_existing_output(
    output: Path,
    metadata_path: Path,
    expected_signature: dict[str, Any],
) -> tuple[bool, str]:
    if not output.is_file() or not metadata_path.is_file():
        return False, "output or completion metadata is missing"
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read metadata: {exc}"
    if metadata.get("status") != "complete":
        return False, "metadata status is not complete"
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return False, "metadata schema version differs"
    if metadata.get("input_signature") != expected_signature:
        return False, "input signature differs"

    try:
        array = np.load(output, allow_pickle=False, mmap_mode="r")
        expected_shape = tuple(int(value) for value in expected_signature["shape"])
        if array.shape != expected_shape:
            return False, f"shape is {array.shape}"
        if array.dtype != np.dtype(np.float32):
            return False, f"dtype is {array.dtype}"
        if not array.flags.c_contiguous:
            return False, "array is not C-contiguous"
        if not np.isfinite(array).all():
            return False, "array contains a non-finite value"
    except (OSError, ValueError) as exc:
        return False, f"cannot load output: {exc}"

    expected_hash = metadata.get("output", {}).get("sha256")
    if not expected_hash or sha256_file(output) != expected_hash:
        return False, "output SHA256 does not match metadata"
    return True, "validated output and metadata match current inputs"


def create_temp_path(parent: Path, stem: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=parent)
    os.close(descriptor)
    return Path(name)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_result(output: Path, array: np.ndarray, metadata: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path_for(output)
    temporary_array = create_temp_path(output.parent, output.name)
    temporary_metadata = create_temp_path(output.parent, metadata_path.name)
    installed_array_hash: str | None = None
    write_started = time.perf_counter()

    try:
        with temporary_array.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        installed_array_hash = sha256_file(temporary_array)
        metadata["output"] = {
            "path": str(output),
            "sha256": installed_array_hash,
            "size_bytes": temporary_array.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "c_contiguous": bool(array.flags.c_contiguous),
        }
        metadata["timing_seconds"]["serialize_hash_and_prepare_metadata"] = time.perf_counter() - write_started
        metadata["timing_seconds"]["total_before_publish"] = (
            metadata["timing_seconds"]["before_atomic_write_total"]
            + metadata["timing_seconds"]["serialize_hash_and_prepare_metadata"]
        )
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_array, output)
        fsync_directory(output.parent)
        try:
            # Metadata is the completion marker and is deliberately installed last.
            os.replace(temporary_metadata, metadata_path)
            fsync_directory(output.parent)
        except Exception:
            if output.is_file() and installed_array_hash is not None and sha256_file(output) == installed_array_hash:
                output.unlink()
                fsync_directory(output.parent)
            raise
    finally:
        for temporary in (temporary_array, temporary_metadata):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_summary(prepared: PreparedInputs) -> dict[str, Any]:
    return {
        "left": str(prepared.left.path),
        "right": str(prepared.right.path),
        "species": list(prepared.left.species),
        "atoms_per_type": list(prepared.left.atoms_per_type),
        "nao_per_type_atom": [orbital.nao_per_atom for orbital in prepared.orbitals],
        "nao": prepared.nao,
        "table_nr": TABLE_NR,
        "table_mode": TABLE_MODE,
        "table_cutoff_bohr": TABLE_CUTOFF_BOHR,
        "table_spacing_bohr": TABLE_CUTOFF_BOHR / (TABLE_NR - 1),
        "translation_vectors": len(prepared.neighbors_by_translation),
        "atom_image_pairs": sum(len(images) for images in prepared.neighbors_by_translation.values()),
        "translation_bounds": {
            "minimum": [min(translation[axis] for translation in prepared.neighbors_by_translation) for axis in range(3)],
            "maximum": [max(translation[axis] for translation in prepared.neighbors_by_translation) for axis in range(3)],
        },
        "orbital_files": [orbital_record(orbital) for orbital in prepared.orbitals],
    }


def absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    return Path(os.path.abspath(expanded if expanded.is_absolute() else Path.cwd() / expanded))


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def reset_integration_state() -> None:
    global EXPECTED_NAO, EXPECTED_ORBITAL_RCUT_BOHR, TABLE_CUTOFF_BOHR, TABLE_NR
    global TABLE_MODE, LEGACY_PARAMETERS, ACTIVE_CONTRACT, ACTIVE_CONTRACT_PATH
    global ACTIVE_CONTRACT_SHA256, SPECIES_ELEMENT_MAP, ZERO_IDENTITY_ATOL

    EXPECTED_NAO = None
    EXPECTED_ORBITAL_RCUT_BOHR = None
    TABLE_CUTOFF_BOHR = None
    TABLE_NR = None
    TABLE_MODE = "uniform_fft"
    LEGACY_PARAMETERS = None
    ACTIVE_CONTRACT = None
    ACTIVE_CONTRACT_PATH = None
    ACTIVE_CONTRACT_SHA256 = None
    SPECIES_ELEMENT_MAP = {}
    ZERO_IDENTITY_ATOL = 1.0e-7


def input_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DirectOverlapError(f"Missing required ABACUS INPUT: {path}")
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("//", 1)[0].split("#", 1)[0].strip()
        fields = line.split()
        if len(fields) >= 2 and fields[0].upper() != "INPUT_PARAMETERS":
            result[fields[0].lower()] = fields[1]
    return result


def input_boolean(value: str | None) -> bool:
    return (value or "0").strip().lower() in {"1", "true", "t", "yes", "y"}


def format_pattern(pattern: str, label: str, number: int, context: str) -> str:
    try:
        rendered = pattern.format(index=label, label=label, number=number, num=number)
    except (IndexError, KeyError, ValueError) as exc:
        raise DirectOverlapError(f"Invalid {context} pattern {pattern!r}: {exc}") from exc
    if rendered == pattern and not any(token in pattern for token in ("{index", "{label", "{number", "{num")):
        raise DirectOverlapError(
            f"{context} pattern must vary by frame using {{index}} or {{number}}: {pattern!r}"
        )
    return rendered


def resolve_structure_path(scf_root: Path, label: str) -> Path:
    candidate = absolute_path(scf_root / label / "STRU")
    if not candidate.is_file():
        raise DirectOverlapError(
            f"Missing required SCF frame STRU: {candidate}. "
            "Production batch mode never substitutes a structure from another root."
        )
    return candidate


def reference_path(scf_root: Path, pattern: str, label: str) -> Path:
    rendered = Path(format_pattern(pattern, label, int(label), "reference"))
    return absolute_path(rendered if rendered.is_absolute() else scf_root / rendered)


def validate_scf_frame(scf_root: Path, label: str) -> dict[str, Any]:
    out = scf_root / label / "OUT.ABACUS"
    input_path = out / "INPUT"
    log_path = out / "running_scf.log"
    if not log_path.is_file():
        raise DirectOverlapError(f"Missing required SCF log: {log_path}")
    mapping = input_mapping(input_path)
    try:
        nspin = int(mapping.get("nspin", "1"))
    except ValueError as exc:
        raise DirectOverlapError(f"Invalid nspin in {input_path}: {mapping.get('nspin')!r}") from exc
    noncolin = input_boolean(mapping.get("noncolin"))
    lspinorb = input_boolean(mapping.get("lspinorb"))
    if nspin not in {1, 2} or noncolin or lspinorb:
        raise DirectOverlapError(
            "Direct tdoverlap supports scalar spatial AO overlap only; "
            f"nspin={nspin}, noncolin={noncolin}, lspinorb={lspinorb}: {input_path}"
        )

    text = log_path.read_text(encoding="utf-8", errors="replace")
    required_markers = {
        "SCF convergence": "charge density convergence is achieved",
        "final energy": "!FINAL_ETOT_IS",
        "normal finish": "Finish Time",
    }
    missing = [name for name, marker in required_markers.items() if marker not in text]
    if missing:
        raise DirectOverlapError(f"Incomplete SCF frame {label}; missing {missing}: {log_path}")
    version_match = re.search(r"\bABACUS\s+v([^\s]+)", text)
    return {
        "index": label,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "running_scf_log": str(log_path.resolve()),
        "running_scf_log_sha256": sha256_file(log_path),
        "abacus_version": version_match.group(1) if version_match else None,
        "nspin": nspin,
        "noncolin": noncolin,
        "lspinorb": lspinorb,
        "completed": True,
    }


def directory_has_orbitals(directory: Path, names: Sequence[str]) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in names)


def discover_orbital_dir(
    explicit: Path | None,
    scf_root: Path,
    first_label: str,
    orbital_names: Sequence[str],
) -> tuple[Path, dict[str, Any]]:
    attempts: list[dict[str, str]] = []
    if explicit is not None:
        candidate = absolute_path(explicit)
        if not directory_has_orbitals(candidate, orbital_names):
            raise DirectOverlapError(f"--orbital-dir does not contain every declared .orb file: {candidate}")
        return candidate, {"method": "explicit", "attempts": attempts}

    frame = scf_root / first_label
    input_path = frame / "OUT.ABACUS" / "INPUT"
    raw = input_mapping(input_path).get("orbital_dir")
    if raw:
        candidate_raw = Path(raw).expanduser()
        candidate = absolute_path(candidate_raw if candidate_raw.is_absolute() else frame / candidate_raw)
        attempts.append({"input": str(input_path), "declared": raw, "resolved": str(candidate)})
        if directory_has_orbitals(candidate, orbital_names):
            return candidate, {"method": "ABACUS INPUT orbital_dir", "attempts": attempts}

    for candidate in (frame, scf_root):
        resolved = absolute_path(candidate)
        attempts.append({"input": "fallback", "declared": str(candidate), "resolved": str(resolved)})
        if directory_has_orbitals(resolved, orbital_names):
            return resolved, {"method": "adjacent fallback", "attempts": attempts}
    raise DirectOverlapError(
        "Could not locate every NUMERICAL_ORBITAL file; checked: "
        + ", ".join(record["resolved"] for record in attempts)
    )


def basis_record(
    baseline: Structure,
    orbitals: Sequence[OrbitalSpec],
    nao: int,
) -> dict[str, Any]:
    return {
        "species": list(baseline.species),
        "atoms_per_type": list(baseline.atoms_per_type),
        "nao": nao,
        "max_rcut_bohr": max(orbital.rcut_bohr for orbital in orbitals),
        "species_element_map": {
            orbital.species: orbital.element
            for orbital in orbitals
            if orbital.species != orbital.element
        },
        "orbitals": [
            {
                "species": orbital.species,
                "element": orbital.element,
                "sha256": orbital.sha256,
                "rcut_bohr": orbital.rcut_bohr,
                "lmax": orbital.lmax,
                "nzeta": list(orbital.nzeta),
                "nao_per_atom": orbital.nao_per_atom,
            }
            for orbital in orbitals
        ],
    }


def discover_radial_support(
    scf_root: Path,
    first_label: str,
    orbital_rcut: float,
) -> dict[str, Any]:
    log_path = scf_root / first_label / "OUT.ABACUS" / "running_scf.log"
    input_path = scf_root / first_label / "OUT.ABACUS" / "INPUT"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float] = {"numerical_orbital": float(orbital_rcut)}
    patterns = {
        "logged_orbital": r"longest\s+orb\s+rcut\s*\(Bohr\)\s*=\s*([-+0-9.eE]+)",
        "nonlocal_projector": (
            r"longest\s+nonlocal\s+projector\s+rcut\s*\(Bohr\)\s*=\s*([-+0-9.eE]+)"
        ),
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if math.isfinite(value) and value > 0.0:
                values[name] = value
    mapping = input_mapping(input_path)
    return {
        "source_log": str(log_path.resolve()),
        "source_log_sha256": sha256_file(log_path),
        "source_input": str(input_path.resolve()),
        "source_input_sha256": sha256_file(input_path),
        "components_bohr": values,
        "rmax_hint_bohr": max(values.values()),
        "deepks_setorb": input_boolean(mapping.get("deepks_setorb")),
        "native_formula": {
            "rmax": "max(orbital rcut, nonlocal beta rcut, optional alpha rcut)",
            "dr_bohr": 0.01,
            "cutoff": "2*rmax",
            "nr": "int(rmax/0.01)+1",
        },
    }


def preflight_batch(
    args: argparse.Namespace,
    scf_root: Path,
    frame_labels: Sequence[tuple[str, int]],
    species_element_map: dict[str, str],
) -> tuple[
    dict[str, Any],
    dict[str, Structure],
    tuple[OrbitalSpec, ...],
    Path,
    dict[str, Any],
]:
    structures: dict[str, Structure] = {}
    frame_records: list[dict[str, Any]] = []
    scope_records: list[dict[str, Any]] = []
    baseline: Structure | None = None
    baseline_label = frame_labels[0][0]

    for label, _ in frame_labels:
        path = resolve_structure_path(scf_root, label)
        structure = parse_structure(path)
        if baseline is None:
            baseline = structure
        else:
            validate_pair_compatibility(baseline, structure)
            if structure.orbital_names != baseline.orbital_names:
                raise DirectOverlapError(f"NUMERICAL_ORBITAL declarations differ in frame {label}")
        structures[label] = structure
        scope_records.append(validate_scf_frame(scf_root, label))
        frame_records.append(
            {
                "index": label,
                "structure": str(path),
                "resolved_structure": str(path.resolve(strict=True)),
                "structure_sha256": structure.sha256,
                "coordinate_mode": structure.coordinate_mode,
            }
        )

    assert baseline is not None
    orbital_dir, discovery = discover_orbital_dir(
        args.orbital_dir, scf_root, baseline_label, baseline.orbital_names
    )
    orbitals = resolve_orbitals(baseline, baseline, orbital_dir, species_element_map)
    _, nao = build_atom_records(baseline, orbitals)
    basis = basis_record(baseline, orbitals, nao)
    radial_support = discover_radial_support(
        scf_root, baseline_label, float(basis["max_rcut_bohr"])
    )
    nspin_values = {record["nspin"] for record in scope_records}
    if len(nspin_values) != 1:
        raise DirectOverlapError(f"nspin differs across frames: {sorted(nspin_values)}")
    return (
        {
            "frame_count": len(frame_records),
            "frames": frame_records,
            "basis": basis,
            "orbital_dir": str(orbital_dir),
            "orbital_discovery": discovery,
            "scf_scope": scope_records,
            "consumer_scope": "fixed-cell scalar spatial-AO Gamma overlap",
            "abacus_radial_support": radial_support,
        },
        structures,
        orbitals,
        orbital_dir,
        radial_support,
    )


def next_nonempty(handle: Any, path: Path, context: str) -> str:
    for line in handle:
        if line.strip():
            return line
    raise DirectOverlapError(f"Unexpected EOF in {path} while reading {context}")


def final_integer(line: str, path: Path, context: str) -> int:
    try:
        return int(line.split()[-1])
    except (IndexError, ValueError) as exc:
        raise DirectOverlapError(f"Invalid {context} header in {path}: {line.rstrip()}") from exc


def read_gamma_csr(path: Path, expected_dimension: int) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise DirectOverlapError(f"Missing required single-frame S(R): {path}")
    digest = hashlib.sha256()
    with path.open("rb") as raw_handle:
        for block in iter(lambda: raw_handle.read(1024 * 1024), b""):
            digest.update(block)
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        first = next_nonempty(handle, path, "header")
        dimension_line = next_nonempty(handle, path, "matrix dimension") if "STEP" in first else first
        dimension = final_integer(dimension_line, path, "matrix dimension")
        block_count = final_integer(next_nonempty(handle, path, "block count"), path, "block count")
        if dimension != expected_dimension:
            raise DirectOverlapError(
                f"S(R) dimension is {dimension}, expected {expected_dimension}: {path}"
            )
        if block_count < 1:
            raise DirectOverlapError(f"S(R) block count must be positive: {path}")
        gamma = np.zeros((dimension, dimension), dtype=np.float32, order="C")
        total_nnz = 0
        nonzero_blocks = 0
        for block_index in range(block_count):
            header = next_nonempty(handle, path, f"block {block_index + 1}")
            fields = header.split()
            if len(fields) != 4:
                raise DirectOverlapError(f"Invalid S(R) block header in {path}: {header.rstrip()}")
            try:
                int(fields[0])
                int(fields[1])
                int(fields[2])
                nnz = int(fields[3])
            except ValueError as exc:
                raise DirectOverlapError(f"Non-integer S(R) block header in {path}") from exc
            if nnz < 0:
                raise DirectOverlapError(f"Negative nnz in {path}")
            if nnz == 0:
                continue
            values = np.fromstring(next_nonempty(handle, path, "values"), sep=" ", dtype=np.float32)
            columns = np.fromstring(next_nonempty(handle, path, "columns"), sep=" ", dtype=np.int64)
            row_pointer = np.fromstring(
                next_nonempty(handle, path, "row pointer"), sep=" ", dtype=np.int64
            )
            if values.size != nnz or columns.size != nnz or row_pointer.size != dimension + 1:
                raise DirectOverlapError(
                    f"CSR array length mismatch in {path}, block {block_index + 1}"
                )
            if (
                row_pointer[0] != 0
                or row_pointer[-1] != nnz
                or np.any(row_pointer[1:] < row_pointer[:-1])
            ):
                raise DirectOverlapError(f"Invalid CSR row pointer in {path}, block {block_index + 1}")
            if np.any(columns < 0) or np.any(columns >= dimension) or not np.isfinite(values).all():
                raise DirectOverlapError(f"Invalid CSR values/columns in {path}, block {block_index + 1}")
            for row in range(dimension):
                start = int(row_pointer[row])
                end = int(row_pointer[row + 1])
                if end > start:
                    gamma[row, columns[start:end]] += values[start:end]
            total_nnz += nnz
            nonzero_blocks += 1
    if not np.isfinite(gamma).all():
        raise DirectOverlapError(f"Gamma-folded S contains non-finite values: {path}")
    return gamma, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "dimension": dimension,
        "declared_blocks": block_count,
        "nonzero_blocks": nonzero_blocks,
        "total_nnz": total_nnz,
    }


def prepare_same_frame(
    structure: Structure,
    orbitals: tuple[OrbitalSpec, ...],
    expected_nao: int,
) -> PreparedInputs:
    basis_by_type = build_basis_by_type(orbitals)
    atoms, nao = build_atom_records(structure, orbitals)
    if nao != expected_nao:
        raise DirectOverlapError(
            f"Calibration AO dimension is {nao}, expected {expected_nao}: {structure.path}"
        )
    neighbors = enumerate_neighbor_images(structure, atoms, atoms, orbitals)
    return PreparedInputs(
        left=structure,
        right=structure,
        orbitals=orbitals,
        basis_by_type=basis_by_type,
        left_atoms=atoms,
        right_atoms=atoms,
        neighbors_by_translation=neighbors,
        nao=nao,
    )


def compare_matrices(
    candidate: np.ndarray,
    reference: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise DirectOverlapError(f"Matrix shape mismatch: {candidate.shape} vs {reference.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference_norm = float(np.linalg.norm(reference.astype(np.float64)))
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    relative_frobenius = float(np.linalg.norm(difference) / reference_norm) if reference_norm else math.inf
    allclose = bool(np.allclose(candidate, reference, rtol=args.rtol, atol=args.atol))
    return {
        "passed": bool(
            allclose
            and max_abs <= args.max_abs_tol
            and relative_frobenius <= args.relative_frobenius_tol
        ),
        "allclose": allclose,
        "max_abs": max_abs,
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_frobenius": relative_frobenius,
    }


def atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool = False) -> None:
    if path.is_symlink():
        raise DirectOverlapError(f"Refusing to replace a symlink: {path}")
    if path.exists() and not overwrite:
        raise DirectOverlapError(f"Refusing to replace existing JSON without explicit overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def calibration_context(scf_root: Path, labels: Sequence[str]) -> dict[str, Any]:
    keys = ("lcao_ecut", "lcao_dk", "lcao_dr", "lcao_rmax")
    records: list[dict[str, Any]] = []
    for label in labels:
        path = scf_root / label / "OUT.ABACUS" / "INPUT"
        mapping = input_mapping(path)
        values: dict[str, float] = {}
        for key in keys:
            if key in mapping:
                try:
                    values[key] = float(mapping[key])
                except ValueError as exc:
                    raise DirectOverlapError(f"Invalid {key} in {path}: {mapping[key]!r}") from exc
        records.append({"index": label, "path": str(path.resolve()), "values": values})
    distinct = {json.dumps(record["values"], sort_keys=True) for record in records}
    return {
        "records": records,
        "consistent_across_calibration_frames": len(distinct) <= 1,
        "note": "Modern native FFT two-center S uses the grid recorded in table, not legacy INPUT constants.",
    }


def select_calibration_labels(
    requested: Sequence[str] | None,
    left_labels: Sequence[str],
) -> list[str]:
    if requested:
        labels = list(dict.fromkeys(requested))
        if any(not label.isdigit() for label in labels):
            raise DirectOverlapError("--calibration-indices must contain only numeric frame labels")
    else:
        positions = (0, len(left_labels) // 2, len(left_labels) - 1)
        labels = list(dict.fromkeys(left_labels[position] for position in positions))
    unavailable = sorted(set(labels) - set(left_labels))
    if unavailable:
        raise DirectOverlapError(
            f"Calibration frames must be left frames inside the requested range: {unavailable}"
        )
    return labels


def install_native_grid(
    basis: dict[str, Any],
    radial_support: dict[str, Any],
    zero_identity_atol: float,
    max_cutoff_bohr: float,
) -> dict[str, Any]:
    global EXPECTED_NAO, EXPECTED_ORBITAL_RCUT_BOHR, TABLE_CUTOFF_BOHR, TABLE_NR
    global TABLE_MODE, LEGACY_PARAMETERS, ACTIVE_CONTRACT, ACTIVE_CONTRACT_PATH
    global ACTIVE_CONTRACT_SHA256, ZERO_IDENTITY_ATOL

    rmax = float(radial_support["rmax_hint_bohr"])
    cutoff = 2.0 * rmax
    if not math.isfinite(cutoff) or cutoff <= 0.0 or cutoff > max_cutoff_bohr + 1.0e-12:
        raise DirectOverlapError(
            f"Native table cutoff {cutoff} Bohr exceeds allowed {max_cutoff_bohr} Bohr"
        )
    nr = int(rmax / 0.01) + 1
    if nr < 2:
        raise DirectOverlapError(f"Invalid native table size nr={nr}")
    EXPECTED_NAO = int(basis["nao"])
    EXPECTED_ORBITAL_RCUT_BOHR = float(basis["max_rcut_bohr"])
    TABLE_CUTOFF_BOHR = cutoff
    TABLE_NR = nr
    TABLE_MODE = "uniform_fft"
    LEGACY_PARAMETERS = None
    ACTIVE_CONTRACT = None
    ACTIVE_CONTRACT_PATH = None
    ACTIVE_CONTRACT_SHA256 = None
    ZERO_IDENTITY_ATOL = float(zero_identity_atol)
    return {
        "name": "abacus-modern-exact",
        "mode": "uniform_fft",
        "cutoff_bohr": cutoff,
        "nr": nr,
        "spacing_bohr": cutoff / (nr - 1),
        "zero_identity_atol": float(zero_identity_atol),
        "legacy_parameters": None,
    }


def generate_contract(
    args: argparse.Namespace,
    scf_root: Path,
    labels: Sequence[str],
    structures: dict[str, Structure],
    orbitals: tuple[OrbitalSpec, ...],
    orbital_dir: Path,
    basis: dict[str, Any],
    radial_support: dict[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    table = install_native_grid(
        basis, radial_support, args.zero_identity_atol, args.max_table_cutoff_bohr
    )
    prepared = {
        label: prepare_same_frame(structures[label], orbitals, int(basis["nao"]))
        for label in labels
    }
    references: dict[str, np.ndarray] = {}
    reference_records: dict[str, dict[str, Any]] = {}
    for label in labels:
        path = reference_path(scf_root, args.reference_pattern, label)
        matrix, record = read_gamma_csr(path, int(basis["nao"]))
        references[label] = matrix
        reference_records[label] = record

    context = build_integrator(prepared[labels[0]])
    comparisons: list[dict[str, Any]] = []
    for label in labels:
        candidate, counters = calculate_overlap(prepared[label], context, True)
        comparison = compare_matrices(candidate, references[label], args)
        comparisons.append({"index": label, "comparison": comparison, "counters": counters})
        if not comparison["passed"]:
            raise DirectOverlapError(
                "Native two-center table failed same-frame S(R) calibration at "
                f"{label}: max_abs={comparison['max_abs']:.9e}, "
                f"relative_frobenius={comparison['relative_frobenius']:.9e}"
            )

    table["integrator_contract_checks"] = context.contract_checks
    table["abacus_radial_support"] = radial_support
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "validated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "ABACUS-two-center-native-grid-v3",
        "grid_policy": {
            "mode": "native_exact_fail_closed",
            "allow_compatibility_fallback": False,
        },
        "basis": basis,
        "software": pyabacus_fingerprint(),
        "table": table,
        "calibration": {
            "mode": "single-frame-data-SR",
            "indices": list(labels),
            "references": [reference_records[label] for label in labels],
            "thresholds": {
                "rtol": args.rtol,
                "atol": args.atol,
                "max_abs_tol": args.max_abs_tol,
                "relative_frobenius_tol": args.relative_frobenius_tol,
            },
            "candidate_trials": [
                {
                    "candidate": {
                        key: table[key]
                        for key in ("name", "mode", "cutoff_bohr", "nr", "spacing_bohr", "legacy_parameters")
                    },
                    "frames": comparisons,
                    "integrator_contract_checks": context.contract_checks,
                    "passed": True,
                }
            ],
        },
        "abacus_input_context": calibration_context(scf_root, labels),
        "consumer_scope": {
            "kpoint": "Gamma (k=0)",
            "spin": "scalar spatial AO overlap",
        },
        "provenance": {
            "generator": str(Path(__file__).resolve()),
            "worker": str(Path(__file__).resolve()),
            "worker_sha256": sha256_file(Path(__file__).resolve()),
            "scf_root": str(scf_root),
            "structure_source": "SCF_ROOT/{index}/STRU",
            "structure_inputs": [
                {
                    "index": label,
                    "path": str(scf_root / label / "STRU"),
                    "resolved_path": str((scf_root / label / "STRU").resolve(strict=True)),
                    "sha256": structures[label].sha256,
                }
                for label in sorted(structures, key=int)
            ],
            "orbital_dir": str(orbital_dir),
            "single_script": True,
            "no_staging_tree": True,
        },
    }
    contract["contract_id"] = contract_identifier(contract)
    atomic_write_json(contract_path, contract, overwrite=args.recalibrate)
    return contract


def load_contract_for_batch(
    args: argparse.Namespace,
    path: Path,
    scf_root: Path,
    labels: Sequence[str],
    structures: dict[str, Structure],
    basis: dict[str, Any],
    radial_support: dict[str, Any],
) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectOverlapError(f"Cannot read existing contract {path}: {exc}") from exc
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION or contract.get("status") != "validated":
        raise DirectOverlapError(f"Existing contract is not validated: {path}")
    if contract.get("algorithm") != "ABACUS-two-center-native-grid-v3":
        raise DirectOverlapError(f"Unsupported contract algorithm: {path}")
    if contract.get("grid_policy") != {
        "mode": "native_exact_fail_closed",
        "allow_compatibility_fallback": False,
    }:
        raise DirectOverlapError(f"Existing contract is not native-grid fail-closed: {path}")
    if contract.get("basis") != basis:
        raise DirectOverlapError(f"Existing contract basis differs; use a new output root: {path}")
    if contract.get("software") != pyabacus_fingerprint():
        raise DirectOverlapError(f"Existing contract pyabacus implementation differs: {path}")
    if contract.get("table", {}).get("abacus_radial_support") != radial_support:
        raise DirectOverlapError(f"Existing contract radial support differs: {path}")
    if contract.get("provenance", {}).get("worker_sha256") != sha256_file(Path(__file__).resolve()):
        raise DirectOverlapError(f"Existing contract generator script differs: {path}")
    if contract.get("calibration", {}).get("indices") != list(labels):
        raise DirectOverlapError(f"Existing contract calibration frames differ: {path}")
    expected_thresholds = {
        "rtol": args.rtol,
        "atol": args.atol,
        "max_abs_tol": args.max_abs_tol,
        "relative_frobenius_tol": args.relative_frobenius_tol,
    }
    if contract.get("calibration", {}).get("thresholds") != expected_thresholds:
        raise DirectOverlapError(f"Existing contract tolerances differ: {path}")
    recorded_references = contract.get("calibration", {}).get("references", [])
    current_references = [
        read_gamma_csr(reference_path(scf_root, args.reference_pattern, label), int(basis["nao"]))[1]
        for label in labels
    ]
    if recorded_references != current_references:
        raise DirectOverlapError(f"Existing contract same-frame S(R) references differ: {path}")
    if contract.get("provenance", {}).get("scf_root") != str(scf_root):
        raise DirectOverlapError(f"Existing contract SCF root differs: {path}")
    if contract.get("provenance", {}).get("structure_source") != "SCF_ROOT/{index}/STRU":
        raise DirectOverlapError(f"Existing contract does not require SCF-owned STRU inputs: {path}")
    current_structures = [
        {
            "index": label,
            "path": str(scf_root / label / "STRU"),
            "resolved_path": str((scf_root / label / "STRU").resolve(strict=True)),
            "sha256": structures[label].sha256,
        }
        for label in sorted(structures, key=int)
    ]
    if contract.get("provenance", {}).get("structure_inputs") != current_structures:
        raise DirectOverlapError(f"Existing contract SCF STRU inputs differ: {path}")
    if contract.get("contract_id") != contract_identifier(contract):
        raise DirectOverlapError(f"Existing contract content ID is invalid: {path}")
    return contract


def file_signature(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def bounded_text(value: str, limit: int = 20000) -> str:
    return value if len(value) <= limit else value[:limit] + f"\n...[truncated {len(value) - limit} characters]"


def pair_command(
    task: PairTask,
    orbital_dir: Path,
    contract_path: Path,
    species_element_map: dict[str, str],
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--left",
        str(task.left_stru),
        "--right",
        str(task.right_stru),
        "--output",
        str(task.output),
        "--orbital-dir",
        str(orbital_dir),
        "--contract",
        str(contract_path),
        "--quiet",
    ]
    if species_element_map:
        command.extend(["--species-element-map", *[f"{key}={value}" for key, value in species_element_map.items()]])
    if force:
        command.append("--force")
    return command


def run_pair_task(
    task: PairTask,
    command: Sequence[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    before = {"output": file_signature(task.output), "metadata": file_signature(task.metadata)}
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    after = {"output": file_signature(task.output), "metadata": file_signature(task.metadata)}
    if completed.returncode != 0:
        status = "failed"
    elif before == after:
        status = "skipped_validated"
    else:
        status = "completed"
    return {
        "order": task.order,
        "index": task.left_label,
        "right_index": task.right_label,
        "output": str(task.output),
        "metadata": str(task.metadata),
        "command": list(command),
        "returncode": completed.returncode,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": bounded_text(completed.stdout),
        "stderr": bounded_text(completed.stderr),
        "before": before,
        "after": after,
    }


def summarize_batch(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(records),
        "completed": sum(record["status"] == "completed" for record in records),
        "skipped_validated": sum(record["status"] == "skipped_validated" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "failed_indices": [record["index"] for record in records if record["status"] == "failed"],
        "passed": all(record["status"] != "failed" for record in records),
        "sum_item_elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in records),
    }


def validate_batch_arguments(args: argparse.Namespace) -> None:
    required = {
        "--scf-root": args.scf_root,
        "--output-root": args.output_root,
        "--start": args.start,
        "--end": args.end,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise DirectOverlapError("Batch mode requires " + ", ".join(missing))
    pair_only_values = {
        "--left": args.left,
        "--right": args.right,
        "--output": args.output,
        "--contract": args.contract,
        "--expected-nao": args.expected_nao,
        "--expected-orbital-rcut-bohr": args.expected_orbital_rcut_bohr,
        "--table-cutoff-bohr": args.table_cutoff_bohr,
        "--table-nr": args.table_nr,
    }
    mixed = [name for name, value in pair_only_values.items() if value is not None]
    if mixed or args.parse_only:
        raise DirectOverlapError(
            "Do not mix batch mode with single-pair/manual options: "
            + ", ".join(mixed + (["--parse-only"] if args.parse_only else []))
        )
    if args.start < 0 or args.end <= args.start:
        raise DirectOverlapError("Batch mode requires 0 <= --start < --end; endpoints are frames")
    if args.index_width < 1:
        raise DirectOverlapError("--index-width must be at least 1")
    if args.nproc < 1 or args.nproc > available_cpus():
        raise DirectOverlapError(
            f"--nproc must be between 1 and available CPUs ({available_cpus()}), got {args.nproc}"
        )
    positive = {
        "--rtol": args.rtol,
        "--atol": args.atol,
        "--max-abs-tol": args.max_abs_tol,
        "--relative-frobenius-tol": args.relative_frobenius_tol,
        "--zero-identity-atol": args.zero_identity_atol,
        "--max-table-cutoff-bohr": args.max_table_cutoff_bohr,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise DirectOverlapError(f"{name} must be finite and positive")
    if args.preflight_only and args.calibrate_only:
        raise DirectOverlapError("--preflight-only and --calibrate-only are mutually exclusive")


def default_report_path(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / "reports" / f"batch-{stamp}-{os.getpid()}.json"


def run_batch(args: argparse.Namespace) -> int:
    validate_batch_arguments(args)
    reset_integration_state()
    species_element_map = parse_species_element_map(args.species_element_map)
    scf_root = absolute_path(args.scf_root)
    output_root = absolute_path(args.output_root)
    if not scf_root.is_dir():
        raise DirectOverlapError(f"SCF root is not a directory: {scf_root}")
    if output_root == scf_root:
        raise DirectOverlapError("Output root must be independent of the SCF root")

    frame_labels = [
        (f"{number:0{args.index_width}d}", number)
        for number in range(args.start, args.end + 1)
    ]
    left_labels = [label for label, _ in frame_labels[:-1]]
    calibration_labels = select_calibration_labels(args.calibration_indices, left_labels)
    preflight_started = time.perf_counter()
    preflight, structures, orbitals, orbital_dir, radial_support = preflight_batch(
        args, scf_root, frame_labels, species_element_map
    )
    preflight_seconds = time.perf_counter() - preflight_started

    report_path = absolute_path(args.report_json) if args.report_json else default_report_path(output_root)
    if report_path.exists() and not args.overwrite_report:
        raise DirectOverlapError(
            f"Batch report already exists; choose another path or pass --overwrite-report: {report_path}"
        )
    if args.preflight_only:
        report = {
            "schema_version": BATCH_SCHEMA_VERSION,
            "complete": True,
            "passed": True,
            "mode": "preflight_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "request": {
                "start_frame": args.start,
                "end_frame": args.end,
                "scf_root": str(scf_root),
                "output_root": str(output_root),
                "species_element_map": species_element_map,
            },
            "preflight": preflight,
            "timing_seconds": {"preflight": preflight_seconds},
        }
        atomic_write_json(report_path, report, overwrite=args.overwrite_report)
        print(f"PREFLIGHT_OK frames={len(frame_labels)} pairs={len(left_labels)} report={report_path}")
        return 0

    contract_path = absolute_path(args.contract_json) if args.contract_json else output_root / "tdoverlap-contract.json"
    if contract_path.exists() and not args.recalibrate:
        contract = load_contract_for_batch(
            args,
            contract_path,
            scf_root,
            calibration_labels,
            structures,
            preflight["basis"],
            radial_support,
        )
        contract_action = "reused"
    else:
        contract = generate_contract(
            args,
            scf_root,
            calibration_labels,
            structures,
            orbitals,
            orbital_dir,
            preflight["basis"],
            radial_support,
            contract_path,
        )
        contract_action = "calibrated"
    print(
        f"CONTRACT_OK action={contract_action} status={contract['status']} "
        f"id={contract['contract_id']} path={contract_path}",
        flush=True,
    )
    if args.calibrate_only:
        return 0

    tasks = [
        PairTask(
            order=order,
            left_label=left_label,
            right_label=right_label,
            left_stru=scf_root / left_label / "STRU",
            right_stru=scf_root / right_label / "STRU",
            output=output_root / left_label / "SPIN0" / "tdoverlap.npy",
        )
        for order, ((left_label, _), (right_label, _)) in enumerate(
            zip(frame_labels[:-1], frame_labels[1:])
        )
    ]
    commands = {
        task.order: pair_command(
            task, orbital_dir, contract_path, species_element_map, args.force
        )
        for task in tasks
    }
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    records: list[dict[str, Any]] = []
    batch_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.nproc, len(tasks))) as executor:
        pending = {
            executor.submit(run_pair_task, task, commands[task.order], environment): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(pending):
            task = pending[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "order": task.order,
                    "index": task.left_label,
                    "right_index": task.right_label,
                    "output": str(task.output),
                    "metadata": str(task.metadata),
                    "command": commands[task.order],
                    "returncode": -1,
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}: {exc}",
                    "before": None,
                    "after": None,
                }
            records.append(record)
            if not args.quiet:
                print(
                    f"[{record['status']}] {record['index']}->{record['right_index']} "
                    f"rc={record['returncode']} elapsed={record['elapsed_seconds']:.3f}s",
                    flush=True,
                )
                if record["status"] == "failed" and record["stderr"]:
                    print(record["stderr"].strip(), file=sys.stderr, flush=True)

    records.sort(key=lambda record: int(record["order"]))
    summary = summarize_batch(records)
    report = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "complete": True,
        "passed": bool(summary["passed"]),
        "mode": "batch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "start_frame": args.start,
            "end_frame": args.end,
            "index_width": args.index_width,
            "nproc": args.nproc,
            "scf_root": str(scf_root),
            "structure_source": "SCF_ROOT/{index}/STRU",
            "output_root": str(output_root),
            "orbital_dir": str(orbital_dir),
            "species_element_map": species_element_map,
            "calibration_indices": calibration_labels,
            "contract": str(contract_path),
            "contract_id": contract["contract_id"],
            "contract_action": contract_action,
            "generator": str(Path(__file__).resolve()),
            "force": bool(args.force),
            "thread_environment": THREAD_ENVIRONMENT,
        },
        "preflight": preflight,
        "timing_seconds": {
            "preflight": preflight_seconds,
            "batch_wall": time.perf_counter() - batch_started,
        },
        "summary": summary,
        "pairs": records,
    }
    atomic_write_json(report_path, report, overwrite=args.overwrite_report)
    print(
        f"SUMMARY frames={len(frame_labels)} pairs={summary['pair_count']} "
        f"completed={summary['completed']} skipped={summary['skipped_validated']} "
        f"failed={summary['failed']} report={report_path}",
        flush=True,
    )
    return 0 if summary["passed"] else 1


def run(args: argparse.Namespace) -> int:
    if args.left is None or args.right is None:
        raise DirectOverlapError("Single-pair mode requires both --left and --right")
    configure_contract(args)
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    prepared = prepare_inputs(args.left, args.right, args.orbital_dir, SPECIES_ELEMENT_MAP)
    validate_prepared_against_active_contract(prepared)
    parsed_at = time.perf_counter()

    if args.parse_only:
        print(json.dumps(parse_summary(prepared), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise DirectOverlapError("--output is required unless --parse-only is used")

    output = args.output.expanduser().resolve(strict=False)
    metadata_path = metadata_path_for(output)
    if output.is_symlink() or metadata_path.is_symlink():
        raise DirectOverlapError(f"Refusing to replace an output or metadata symlink: {output}")
    for input_path in (prepared.left.path, prepared.right.path, *(orbital.path for orbital in prepared.orbitals)):
        if output == input_path or metadata_path == input_path:
            raise DirectOverlapError(f"Output path conflicts with required input: {input_path}")

    script_path = Path(__file__).resolve()
    software_fingerprint = pyabacus_fingerprint()
    input_signature = build_input_signature(
        prepared,
        sha256_file(script_path),
        software_fingerprint,
    )
    if not args.force:
        validated, reason = validated_existing_output(output, metadata_path, input_signature)
        if validated:
            if not args.quiet:
                print(f"[direct-overlap] skip: {reason}: {output}")
            return 0
        if output.exists() or metadata_path.exists():
            raise DirectOverlapError(
                f"Existing output is not reusable ({reason}); refusing to replace {output} "
                "without explicit --force"
            )

    if not args.quiet:
        summary = parse_summary(prepared)
        print(
            f"[direct-overlap] {summary['nao']}x{summary['nao']}, "
            f"{summary['translation_vectors']} R vectors, {summary['atom_image_pairs']} atom-image pairs",
            flush=True,
        )

    table_started = time.perf_counter()
    context = build_integrator(prepared)
    table_finished = time.perf_counter()
    overlap, counters = calculate_overlap(prepared, context, args.quiet)
    calculated_at = time.perf_counter()

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm": ALGORITHM_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_wall.isoformat(),
        "input_signature": input_signature,
        "geometry": {
            "species": list(prepared.left.species),
            "atoms_per_type": list(prepared.left.atoms_per_type),
            "lattice_constant_bohr": prepared.left.lattice_constant_bohr,
            "lattice_vectors": prepared.left.lattice_vectors.tolist(),
            "cell_bohr": prepared.left.cell_bohr.tolist(),
            "left_coordinate_mode": prepared.left.coordinate_mode,
            "right_coordinate_mode": prepared.right.coordinate_mode,
        },
        "periodic_images": {
            "translation_vectors": counters["translation_vectors"],
            "atom_image_pairs": counters["atom_image_pairs"],
            "translation_minimum": [
                min(translation[axis] for translation in prepared.neighbors_by_translation) for axis in range(3)
            ],
            "translation_maximum": [
                max(translation[axis] for translation in prepared.neighbors_by_translation) for axis in range(3)
            ],
        },
        "calculation": {
            "orientation": "rows=left/time-t, columns=right/time-(t+dt)",
            "displacement": "right + R @ cell - left",
            "gamma_fold": True,
            "snap_derivative": False,
            "snap_calls": counters["snap_calls"],
            "ao_order": "type, atom, l, zeta, m=(0,+1,-1,+2,-2,...)",
            "integrator_contract_checks": context.contract_checks,
        },
        "software": {
            "pyabacus_version": context.pyabacus_version,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
        },
        "timing_seconds": {
            "parse_validate_and_pbc": parsed_at - started,
            "build_two_center_table": table_finished - table_started,
            "calculate_overlap": calculated_at - table_finished,
            "before_atomic_write_total": calculated_at - started,
        },
    }

    atomic_write_result(output, overlap, metadata)
    finished = time.perf_counter()
    if not args.quiet:
        print(f"[direct-overlap] wrote {output} in {finished - started:.3f} s", flush=True)
        print(f"[direct-overlap] completion metadata: {metadata_path}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        batch_requested = any(
            value is not None
            for value in (
                args.scf_root,
                args.output_root,
                args.start,
                args.end,
                args.contract_json,
                args.report_json,
            )
        ) or args.calibrate_only or args.preflight_only or args.recalibrate
        pair_requested = args.left is not None or args.right is not None or args.output is not None
        if batch_requested and pair_requested:
            raise DirectOverlapError("Choose batch mode or --left/--right single-pair mode, not both")
        if batch_requested:
            return run_batch(args)
        if not pair_requested:
            raise DirectOverlapError(
                "Choose batch mode with --scf-root/--output-root/--start/--end, "
                "or single-pair mode with --left/--right"
            )
        return run(args)
    except DirectOverlapError as exc:
        print(f"direct_overlap.py: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("direct_overlap.py: interrupted; no completion metadata was written", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"direct_overlap.py: unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
