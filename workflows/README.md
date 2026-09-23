# NAC workflows

Choose the route matching your Hamiltonian source. These workflows start from
completed ABACUS SCF outputs or existing HamGNN predictions; model training,
graph generation and prediction remain in HamGNN.

| Route | Preparation | Final step |
| --- | --- | --- |
| [ABACUS](abacus/README.md) | SCF H/S → eigenstates; SCF structures and orbitals → adjacent-frame AO overlaps | [Common NAC projection](common/README.md) |
| [HamGNN](hamgnn/README.md) | Predicted H and graph S → eigenstates; reuse AO overlaps from the ABACUS workflow | [Common NAC projection](common/README.md) |

```text
workflows/
├── abacus/
│   ├── wfc_dft.py           # ABACUS SCF H/S → eigen.npy, wfc.npy
│   ├── direct_overlap.py    # SCF STRU and orbitals → tdoverlap.npy
│   └── abacus_csr.py        # Internal ABACUS matrix reader
├── hamgnn/
│   ├── wfc_hamgnn.py        # HamGNN predictions → eigen.npy, wfc.npy
│   ├── hamgnn_wfc.py        # Internal Hamiltonian assembly and CPU solver
│   └── abacus_basis.py      # Orbital masks used by the HamGNN graph format
└── common/
    └── run_canac_route.py   # Either prepared route → NAC
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
