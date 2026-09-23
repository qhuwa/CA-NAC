# NAC workflows

[English](README.md) | [简体中文](README.zh-CN.md)

Choose the route matching your Hamiltonian source. These workflows start from
completed ABACUS SCF outputs or existing HamGNN predictions; model training,
graph generation and prediction remain in HamGNN.

| Route | Preparation | Final step |
| --- | --- | --- |
| [ABACUS](abacus/README.md) | SCF H/S → eigenstates; SCF structures and orbitals → adjacent-frame AO overlaps | [Common NAC projection](#shared-nac-projection) |
| [HamGNN](hamgnn/README.md) | Predicted H and graph S → eigenstates; reuse AO overlaps from the ABACUS workflow | [Common NAC projection](#shared-nac-projection) |

```text
workflows/
├── run_canac_route.py       # Either prepared route → NAC
├── abacus/
│   ├── wfc_dft.py           # ABACUS SCF H/S → eigen.npy, wfc.npy
│   ├── direct_overlap.py    # SCF STRU and orbitals → tdoverlap.npy
│   └── abacus_csr.py        # Internal ABACUS matrix reader
└── hamgnn/
    ├── wfc_hamgnn.py        # HamGNN predictions → eigen.npy, wfc.npy
    ├── hamgnn_wfc.py        # Internal Hamiltonian assembly and CPU solver
    └── abacus_basis.py      # Orbital masks used by the HamGNN graph format
```

The overlap generator belongs to `abacus/` because its inputs are completed
ABACUS SCF calculations. Its outputs can be shared by both routes when their
geometry, AO basis and ordering agree. `hamgnn/abacus_basis.py` is an internal
HamGNN dependency: its name describes the model's ABACUS orbital convention.

## Supported data

The supported route is a fixed cell, real scalar orbitals, physical `SPIN0`, and
full-rank wavefunctions. SOC, other k-points, reduced-rank conditioning and other
physical spin channels are outside these entry points. CA-NAC's `HAMNET` reader
name denotes the prepared NumPy format here; `--source abacus` or `--source hamgnn`
records the Hamiltonian source.

Frame endpoints are inclusive: `START..END` gives `END-START+1` eigenstates and
`END-START` adjacent pairs. Four-digit source labels are preserved by default.
Choose frame ranges, physical bands, orbital masks and effective saved-frame
spacing from your own calculation.

Both WFC entry points publish a complete frame directory by rename, including
`wfc-source.json`. Reruns skip only matching source paths, sizes, modification
times, parameters and valid output shapes/dtypes/finite values. Source files must
remain immutable during a run. Existing leaves that cannot be validated are
rejected; choose a new output root for changed inputs or precision. Generate WFCs
before linking overlaps. No eigenmode removal or overlap conditioning occurs.

The CSR reader and CPU eigensolver are extracted from the existing ABACUS/HamGNN
helpers. The HamGNN entry point incorporates the production streaming driver.
Only the orbital masks and unit constants used by this route are retained from
the HamGNN utilities; their attribution is preserved in `hamgnn/abacus_basis.py`.

No host paths or cluster environments are embedded in these scripts. Set BLAS
threads explicitly and run large trajectories in your scheduler allocation:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## Shared NAC projection

`run_canac_route.py` consumes eigenstates prepared by either the
[ABACUS](abacus/README.md) or [HamGNN](hamgnn/README.md) route and calls
CA-NAC. Both routes use the same NumPy contract and calibrated adjacent-frame
AO overlaps. This step requires Python 3.9+, NumPy and SciPy.

Keep `WORKFLOWS`, `WFC_PYTHON`, `OVERLAP_ROOT`, `START`, `END` and `NPROC`
from the chosen route's setup. Each frame provides `eigen.npy` and `wfc.npy`;
each left frame also needs `tdoverlap.npy` and `tdoverlap.npy.meta.json` for
its next frame.

### Link overlaps and run CA-NAC

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

"$WFC_PYTHON" "$WORKFLOWS/run_canac_route.py" \
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
