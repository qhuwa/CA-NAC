# Shared NAC projection

`run_canac_route.py` consumes eigenstates prepared by either the
[ABACUS](../abacus/README.md) or [HamGNN](../hamgnn/README.md) route and calls
CA-NAC. Both routes use the same NumPy contract and calibrated adjacent-frame
AO overlaps. This step requires Python 3.9+, NumPy and SciPy.

Keep `WORKFLOWS`, `WFC_PYTHON`, `OVERLAP_ROOT`, `START`, `END` and `NPROC`
from the chosen route's setup. Each frame provides `eigen.npy` and `wfc.npy`;
each left frame also needs `tdoverlap.npy` and `tdoverlap.npy.meta.json` for
its next frame.

## Link overlaps and run CA-NAC

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

"$WFC_PYTHON" "$WORKFLOWS/common/run_canac_route.py" \
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
