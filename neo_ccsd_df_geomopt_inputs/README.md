# NEO-CCSD density-fitted geometry-optimization inputs

Input files for density-fitted NEO-CCSD (multicomponent CCSD) geometry
optimizations of every system in Tucker & Brorsen, *"An Analytic Nuclear
Energy Gradient for Multicomponent CCSD"*, J. Chem. Theory Comput.
(doi: [10.1021/acs.jctc.6c01090](https://doi.org/10.1021/acs.jctc.6c01090))
and its Supporting Information.

Each `.inp` file is a self-contained Python driver against the
`pyscf_neo_cc` production package of the `mc_coupled_cluster` repository
(see its `production/docs/manual.md` and `production/docs/STYLE.md` — the
production public API is Python, not a JSON dispatcher):

```bash
python HCN.neo-ccsd.aug-cc-pvdz.pb4f1.aug-cc-pvtz-ri.2026-08-25.inp
```

## File naming

```
<formula>.<method>.<electronic basis>.<nuclear basis>.<auxiliary basis>.<date>.inp
```

e.g. `HFDF.neo-ccsd.aug-cc-pvdz.pb4f1.aug-cc-pvtz-ri.2026-08-25.inp`.
The auxiliary field names the ee correlation-fitting (Cfit/RI) set; the
matching JKfit set for the DF-SCF Fock matrix (required by the DF gradient
Z-vector) is set inside each file at the same level. Isotopomers that share
a molecular formula are distinguished the way the paper labels them:
`HFDF` = (HF)(DF) (deuterium on the acceptor), `DFHF` = (DF)(HF) (deuterium
on the donor, i.e. the bridging position).

## Systems and starting geometries

| Files | Paper label | Starting geometry |
|---|---|---|
| HCN, DCN | XCN | SI Table S1 |
| HNC, DNC | XNC | campaign start, `cneo_ccsd_campaign/systems.py` |
| FHF, FDF | FXF⁻ (charge −1) | campaign start, `df_gradient/cluster/molecules.py` |
| HCCH, HCCD, DCCD | XCCX / XCCY | campaign start, `df_gradient/cluster/molecules.py` |
| HNNH, HNND, DNND | XNNX / XNNY | campaign start, `df_gradient/cluster/molecules.py` |
| HOOH, HOOD, DOOD | XOOX / XOOY | campaign start, `df_gradient/cluster/molecules.py` |
| HFHF, HFDF, DFHF, DFDF | (XF)₂ | SI Table S2 |
| HFHFHF | cyclic (HF)₃ | SI Table S3 |

The SI provides explicit Cartesian geometries only for HCN, (HF)₂, and
(HF)₃ (Tables S1–S3); those are used verbatim. For the remaining systems
the paper reports only optimized internal coordinates, so the starting
structures are the mc_coupled_cluster group's own campaign starts for the
identical systems (near-equilibrium and deliberately symmetry-free where a
symmetric start could trap the optimizer on a saddle).

## Conventions

- **Method**: NEO-CCSD via `DFCCSD` with analytic DF gradients
  (Bozkaya–Sherrill per-aux pipeline), geomeTRIC optimizer with the package
  default convergence criteria — identical to the criteria quoted in the
  paper (1e-6 Eh energy; 3e-4 / 4.5e-4 a.u. RMS/max gradient; 1.2e-3 /
  1.8e-3 a.u. RMS/max displacement).
- **Quantum nuclei**: all hydrogen nuclei, as in the paper. Deuterium is
  the pyscf NEO symbol `H+`; the protonic basis exponents follow the
  nuclear mass automatically.
- **Bases**: aug-cc-pVDZ / aug-cc-pVTZ / aug-cc-pVQZ electronic with the
  PB4-F1 protonic basis (the paper's main protocol). The cyclic (HF)₃
  system uses cc-pVDZ, as in SI Table S8. For the SI's PB4-D or 8s8p8d8f
  protonic-basis variants, change `NUC_BASIS` (and the filename field)
  accordingly.
- **Density fitting**: `cd_density_fit_unified` with the manual's
  recommended aux — Cfit and JKfit one zeta above the orbital basis
  (aug-cc-pVDZ → aug-cc-pVTZ-RI / aug-cc-pVTZ-JKfit, etc.), protonic aux
  via the Pavošević recipe. (The repo's DF campaign alternatively matches
  the Cfit set to the orbital basis; edit `AUXBASIS_E` / `JKFIT_AUX_E` for
  that convention.)
- **SCF**: ARH second-order solver via `use_arh`, inherited by the
  gradient scanner at every optimization step.
- **Frozen core**: none (`frozen=0`), matching the repository's paper
  campaigns.
- **Output**: per-step XYZ trajectory, final geometry, and the relaxed
  quantum-nucleus position expectation values ⟨r_p⟩ — the paper's
  preferred definition for bond distances and angles.
