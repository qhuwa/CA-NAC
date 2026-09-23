# ABACUS SCF outputs to NAC

Prepare eigenstates from completed ABACUS SCF matrices and adjacent-frame AO
overlaps from the corresponding SCF structures and orbitals. Finish with
[common NAC projection](../common/README.md).

| Script | Role |
| --- | --- |
| `wfc_dft.py` | SCF `data-HR` / `data-SR` → `eigen.npy`, `wfc.npy` |
| `direct_overlap.py` | SCF structures and orbitals → calibrated `tdoverlap.npy` and sidecars |
| `abacus_csr.py` | Internal matrix reader used by `wfc_dft.py` |

## Dependencies and setup

Use Python 3.9+, NumPy and SciPy for WFC preparation and CA-NAC.
Direct AO overlap additionally requires `pyabacus` with `ModuleBase` and
`ModuleNAO`. Match its ABACUS source revision, compiler and floating-point
options to the SCF build, and retain the build manifest. The first production
invocation calibrates against ordinary same-frame `data-SR`; an import check
alone is insufficient.

Set these example paths and frame range for your calculation. The WFC and
pyabacus interpreters may be different.

```bash
WORKFLOWS=/path/to/CA-NAC/workflows
WFC_PYTHON=/path/to/numpy-scipy-environment/bin/python
OVERLAP_PYTHON=/path/to/pyabacus-environment/bin/python
SCF_ROOT=/path/to/scf
ORBITAL_DIR=/path/to/orbitals
OVERLAP_ROOT=/path/to/direct-overlap
ABACUS_ROUTE=/path/to/abacus-route
START=1
END=100
NPROC=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

See the [shared data and restart contract](../README.md#supported-data).

## Generate eigenstates

```bash
"$WFC_PYTHON" "$WORKFLOWS/abacus/wfc_dft.py" \
  --folder-pattern "$SCF_ROOT/{index}/OUT.ABACUS" \
  --save-pattern "$ABACUS_ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC"
```

The input filenames are `data-HR-sparse_SPIN0.csr` and
`data-SR-sparse_SPIN0.csr`. The existing float32 Gamma-folding, Rydberg-to-Hartree
conversion, generalized eigensolve and eV output convention are retained.
Neither `data-H0R` nor `data-S0R` is a substitute for these SCF matrices.

## Generate adjacent-frame AO overlaps

Every SCF frame must retain its own `STRU` and
`OUT.ABACUS/{INPUT,running_scf.log}` with normal SCF completion evidence.
The calibration frames also require ordinary `data-SR-sparse_SPIN0.csr` from
those same calculations. Do not replace a missing SCF structure with a trajectory
export or another calculation.

```bash
"$OVERLAP_PYTHON" "$WORKFLOWS/abacus/direct_overlap.py" \
  --scf-root "$SCF_ROOT" --orbital-dir "$ORBITAL_DIR" \
  --output-root "$OVERLAP_ROOT" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC"
```

If species labels differ from orbital `Element` labels, supply an explicit map,
for example `--species-element-map H1=H H2=H`. The script derives the native
integration grid, calibrates it and writes `tdoverlap-contract.json` automatically.
It publishes each pair under `<left-frame>/SPIN0/tdoverlap.npy` with
`tdoverlap.npy.meta.json`. Its own input/provenance checks gate continuation;
use a new output root when the basis, structures or implementation changes.
There is no separate calibration or numerical-validation script to run.

This overlap step also supplies the overlaps needed by the HamGNN route.

## Run NAC

Set `ROUTE="$ABACUS_ROUTE"` and `SOURCE=abacus`, then follow
[common NAC projection](../common/README.md) to link the overlaps and select
the physical band window and effective frame spacing.
