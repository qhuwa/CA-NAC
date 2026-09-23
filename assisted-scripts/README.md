# ABACUS and HamGNN outputs to NAC

These scripts start from completed ABACUS SCF outputs or existing HamGNN
predictions. They prepare Gamma-point eigenstates and adjacent-frame AO overlaps,
then call CA-NAC. Model training, graph generation and prediction remain in HamGNN.

The supported route is a fixed cell, real scalar orbitals, physical `SPIN0`, and
full-rank wavefunctions. SOC, other k-points, reduced-rank conditioning and other
physical spin channels are outside these entry points. The CA-NAC `HAMNET` reader
is a NumPy file format here; the `--source` argument records whether the
Hamiltonian came from ABACUS or HamGNN.

| Entry point | Input → output |
| --- | --- |
| `direct_overlap.py` | Completed SCF `STRU`, orbital files and calibration `data-SR` → `tdoverlap.npy`, sidecars and calibration contract |
| `wfc_dft.py` | ABACUS `data-HR` / `data-SR` → `eigen.npy`, `wfc.npy` |
| `wfc_hamgnn.py` | Per-frame `graph_data.npz` and `prediction_hamiltonian.npy` → `eigen.npy`, `wfc.npy` |
| `run_canac_route.py` | Prepared eigenstates and calibrated AO overlaps → raw or state-tracked NAC |

`abacus_csr.py`, `abacus_basis.py` and `hamgnn_wfc.py` are internal dependencies.
The CSR reader and CPU eigensolver are extracted from the existing ABACUS/HamGNN
helpers, and the HamGNN entry point incorporates the production streaming driver.
Only the orbital masks and unit constants used by this route are retained from
the HamGNN utility module; its attribution is preserved in `abacus_basis.py`.

## Dependencies

- Python 3.9+, NumPy and SciPy for WFC preparation and CA-NAC.
- HamGNN's environment, including PyTorch, PyTorch Geometric and pymatgen, for
  loading its pickled graph objects and the ABACUS basis masks. Use trusted graph
  files produced by your own HamGNN workflow.
- `pyabacus` with `ModuleBase` and `ModuleNAO` for direct AO overlap. Match its
  ABACUS source revision, compiler and floating-point options to the SCF build,
  and retain the build manifest. The first production invocation calibrates
  against ordinary same-frame `data-SR`; an import check alone is insufficient.

The pyabacus and HamGNN interpreters may be different. No host paths or cluster
environments are embedded in these scripts. Set BLAS threads explicitly and run
large trajectories in your scheduler allocation:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SCRIPTS=/path/to/CA-NAC/assisted-scripts
WFC_PYTHON=/path/to/hamgnn-environment/bin/python
OVERLAP_PYTHON=/path/to/pyabacus-environment/bin/python
SCF_ROOT=/path/to/scf
ORBITAL_DIR=/path/to/orbitals
OVERLAP_ROOT=/path/to/direct-overlap
ABACUS_ROUTE=/path/to/abacus-route
HAMGNN_ROUTE=/path/to/hamgnn-route
GRAPH_ROOT=/path/to/hamgnn-graphs
PREDICTION_ROOT=/path/to/hamgnn-predictions
MODEL_SOURCE=/path/to/checkpoint-used-for-these-predictions
START=1
END=100
NPROC=1
```

These are examples: choose the frame range, physical bands, orbital mask and
effective saved-frame interval from your own calculation. Frame endpoints are
inclusive; `START..END` gives `END-START+1` eigenstates and `END-START` NAC pairs.
Four-digit source labels are preserved by default.

## Shared adjacent-frame overlap

Every SCF frame must retain its own `STRU` and
`OUT.ABACUS/{INPUT,running_scf.log}` with normal SCF completion evidence.
The calibration frames also require ordinary `data-SR-sparse_SPIN0.csr` from
those same calculations. Do not replace a missing SCF structure with a trajectory
export or another calculation.

```bash
"$OVERLAP_PYTHON" "$SCRIPTS/direct_overlap.py" \
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

## ABACUS eigenstates

```bash
"$WFC_PYTHON" "$SCRIPTS/wfc_dft.py" \
  --folder-pattern "$SCF_ROOT/{index}/OUT.ABACUS" \
  --save-pattern "$ABACUS_ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC"
```

The input filenames are `data-HR-sparse_SPIN0.csr` and
`data-SR-sparse_SPIN0.csr`. The existing float32 Gamma-folding, Rydberg-to-Hartree
conversion, generalized eigensolve and eV output convention are retained.
Neither `data-H0R` nor `data-S0R` is a substitute for these SCF matrices.

## HamGNN eigenstates

Each graph file must contain exactly one graph. Predictions must store onsite
blocks followed by offsite blocks in the matching graph order. Hamiltonian
values are in Hartree; the overlap is dimensionless. Supported padded AO masks
are `13`, `15`, `27` and `40`; select the one used by the graph generator and
confirm it matches the actual orbital files. Unsupported masks are rejected.

```bash
NAO_MAX=13
"$WFC_PYTHON" "$SCRIPTS/wfc_hamgnn.py" \
  --graph-data-path "$GRAPH_ROOT/{index}/graph_data.npz" \
  --hamiltonian-path "$PREDICTION_ROOT/{index}/prediction_hamiltonian.npy" \
  --model-source "$MODEL_SOURCE" --nao-max "$NAO_MAX" --dtype float32 \
  --save-pattern "$HAMGNN_ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC" \
  --summary "$HAMGNN_ROUTE/wfc-summary.json"
```

The default expects a full predicted Hamiltonian. Use `--fix-edge-index` only
when the upstream model explicitly predicts a residual to be added to genuine
graph `Hon0/Hoff0` blocks. Do not add H0 again to a full prediction. Optionally
set `--expected-orbitals` to require the known full AO dimension.

Workers load one frame at a time and are recycled periodically. Both WFC entry
points publish a complete frame directory by rename, including
`wfc-source.json`; reruns skip only matching source paths, sizes, modification
times, parameters and valid output shapes/dtypes/finite values. Source files must
remain immutable during a run. Existing leaves that cannot be validated are
rejected; choose a new output root for changed inputs or precision. Generate WFCs
before linking the overlaps. No eigenmode removal or overlap conditioning occurs.

## Project either route into NAC

Choose the WFC route, link each matching pair together with its provenance
sidecar, and run CA-NAC. This example uses the ABACUS route; for HamGNN set
`ROUTE="$HAMGNN_ROUTE"` and `SOURCE=hamgnn`. Both routes must use the same SCF
geometry, AO basis and ordering as the declared overlap root.

```bash
ROUTE="$ABACUS_ROUTE"
SOURCE=abacus
BAND_MIN=1
BAND_MAX=2
POTIM_FS=1.0

for ((frame=START; frame<END; frame++)); do
  printf -v label '%04d' "$frame"
  for name in tdoverlap.npy tdoverlap.npy.meta.json; do
    ln -s "$OVERLAP_ROOT/$label/SPIN0/$name" "$ROUTE/$label/SPIN0/$name"
  done
done

"$WFC_PYTHON" "$SCRIPTS/run_canac_route.py" \
  --source "$SOURCE" --run-dir-pattern "$ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 \
  --band-min "$BAND_MIN" --band-max "$BAND_MAX" \
  --potim "$POTIM_FS" --nproc "$NPROC" --summary "$ROUTE/nac-summary.json"
```

Use absolute roots for the links. Already staged valid links can be reused.
Band indices are one-based physical indices in the full stored spectrum. The
driver finds `CAnac.py` in this checkout unless `--ca-nac-root` or `CA_NAC_ROOT`
selects another checkout. It checks arrays, adjacent-frame sidecars and available
WFC provenance, recomputes band projections instead of reusing stale projected
overlaps, and reports completion only after all requested NAC arrays exist.

Add `--state-tracking` to produce `nac_psrd.npy` instead of `nac_ps.npy`, using a
separate summary filename. Run each mode explicitly if both outputs are needed;
state tracking starts from the first selected frame and must run continuously.
Reinvoking the NAC driver regenerates the requested mode in the selected route.

`nac_ps.npy` and `nac_psrd.npy` are **dimensionless antisymmetrized overlap
numerators** (`ps` means pseudopotential). `--potim` records frame spacing and
does not scale these files. A derivative coupling is `nac / (2 * POTIM_FS)` in
`fs^-1`, or `1000 * nac / (2 * POTIM_FS)` in `ps^-1`; save such a conversion under
a separate filename and retain the original arrays.
