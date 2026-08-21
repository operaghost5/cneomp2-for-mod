# Handoff — CNEO-MP2 refactor, analytic gradients, geomeTRIC optimization

Branch: `claude/cneomp2-refactor-efficiency-kvroth` (all work committed and
pushed there). Date: 2026-08-21.

## What this branch contains

Three bodies of work on top of the original CNEO-MP2 code:

1. **Performance/memory refactor of the CNEO-MP2 energy code.**
   The per-element Cython/Python loop kernels were replaced by
   NumPy/BLAS tensor contractions in `cymods/cneomp2_kernels.py`; the old
   import paths (`cymods.t_amps_e_only`, `cymods.hylleraas_e`, ...)
   re-export from there, so existing driver scripts run unchanged. The
   embedded driver class was turned into an importable library class
   `pyscf.neo.cneomp2.cNEOMP2`. The MO-integral setup only computes the
   cross AO blocks it needs (`neo/ao2mo.py:_cross_ovov`) instead of the
   full combined-molecule integral tensor.
   *Validated:* identical results to the original driver (energies to
   ~1e-12, Lagrange multipliers to the 1e-5 root-finder tolerance) with an
   ~800x speedup of the correlation stage on the mock benchmark; HCN test
   numbers reproduced against the user's records from the original code.

2. **Analytic nuclear gradients for unconstrained-amplitude (UCD) CNEO-MP2**
   (`neo/cneomp2_grad.py`), derived from scratch — see
   `docs/cneomp2_gradient_theory.md` for the derivation, and Section
   "Known issues" below for two upstream defects found on the way.
   *Validated:* against central finite differences at the FD noise floor
   (1e-8 to 1e-7 Ha/Bohr) on H2 (3 bases), HF (2), HeH+, H3+, LiH, N2, and
   HCN/aug-cc-pVTZ — the table in the theory doc has the numbers.

3. **geomeTRIC interface** (`neo/cneomp2_geomopt.py`) with analytic
   gradients (finite-difference fallback available:
   `CNEOMP2Engine(mol, fd=True)`).
   *Result:* HCN/aug-cc-pVTZ all-quantum optimizes in 2 gradient steps to
   r(H-C) = 1.0734 A, r(C-N) = 1.1591 A, E = -92.0870743 Ha.

## Environment requirements

- **The Yang-group PySCF fork** (github.com/theorychemyang/pyscf) built
  from source — the `neo` module needs its custom C kernels
  (`CVHFnrs4_incore_drv_diff_size_*` in libcvhf); stock PySCF wheels lack
  them. Overlay this repo's `neo/` as `pyscf/neo` (symlink works).
- `neo/hf.py` on this branch carries a small compatibility shim
  (`_get_err_vec_compat`) so the multicomponent DIIS works with current
  PySCF cores, which no longer accept lists in `scf.diis.get_err_vec`.
- `geometric` (pip) for the optimizer; NumPy/SciPy as usual.

## Running things

- Tests: `python -m unittest neo/test/test_cneomp2_grad.py` (5 tests,
  ~3 min; includes an all-quantum N2 regression test for the heavy-nucleus
  response defect below).
- Examples: `examples/hcn.aug-cc-pvtz.cneomp2.geometric.py` (analytic-
  gradient optimization), `examples/hcn.aug-cc-pvtz.cneomp2.constrained.BFGS.py`
  (energy-only optimization of the CCD energy).

## Known issues found in the inherited code (action items)

1. **`neo/cphf.py` falsely converges for heavy quantum nuclei.** The
   Krylov solve preconditions rows by 1/(eps_a - eps_i); mass-scaled
   nuclear gaps are huge, so nuclear-block residuals vanish from the
   convergence metric. On all-quantum N2 the returned response densities
   are wrong by up to 6% (constraint r.mo1 = 0 violated at 5e-7) at any
   tolerance, while the equations themselves are exactly satisfied by the
   true response — solver, not physics. **This also invalidates the
   analytic CNEO Hessian for heavy quantum nuclei** (H-only systems are
   fine, which is why tests pass). `neo/cneomp2_grad.py` works around it
   with a dense direct solve (`_solve_response_dense`); the same approach,
   or a properly scaled iterative solver, should be upstreamed into
   `neo/cphf.py` for the Hessian. Evidence and methodology are in
   docs/cneomp2_gradient_theory.md.

2. **The CCD (constrained-amplitude) residuals do not match Eq. 26 of the
   paper.** The constraint terms in `t_amps_en_only`/`t_amps_n_only`
   contract with row/column sums of the leading nocc x nocc and
   nvir x nvir corners of the AO-basis position matrix (this reproduces
   the original loop code exactly — the refactor preserved it), whereas
   Eq. 26 prescribes single sums over MO-basis position elements; the
   multiplier root-finding objective is a third contraction. Consequently
   no single Lagrangian is stationary at the implemented CCD solution and
   an analytic CCD gradient in the frozen-amplitude formulation is
   ill-defined (`Gradients(..., unconstrained=False)` raises with this
   explanation). Worth checking against the original authors' intent.
   Practical impact is small: UCD and CCD minima agree to ~0.0005 A on
   HCN, and the paper reports the same insensitivity.

3. Minor: the repo's `neo/hessian.py` calls into modern PySCF's
   `hessian.rhf.hess_elec` with arguments the current API no longer
   accepts (crashes with an einsum shape error) — a version-pinning issue
   independent of (1).

## Natural next steps

- **Z-vector (adjoint) gradient**: the current implementation solves the
  response equations once per nuclear displacement (3N dense solves; fine
  at HCN scale). A single adjoint solve replaces all of them — the
  Lagrangian pieces (D, Gamma, rotation matrices M^k) are already in
  place; only the transpose solve and the assembly of z-weighted
  right-hand sides are new.
- **CCD analytic gradient**, after resolving issue (2): the extra terms
  (multiplier-weighted correlated-density rotation of the position
  matrix) are already implemented and commented out in
  `_rotation_matrices`.
- Memory: the ee explicit-derivative contraction materializes the
  half-back-transformed Gamma (nao^2 x nocc x nvir); shell-blocked
  contraction would remove that ceiling for larger molecules.
- Upstream the DIIS shim and the dense response solver to the Yang fork.

## File map (new/changed on this branch)

| File | Role |
|---|---|
| `cymods/cneomp2_kernels.py` | vectorized amplitude/energy/density kernels |
| `neo/cneomp2.py` | cNEOMP2 library class (CCD energy driver) |
| `neo/cneomp2_grad.py` | analytic UCD gradients (+ dense response solver) |
| `neo/cneomp2_geomopt.py` | geomeTRIC engine + `optimize()` |
| `neo/hf.py` | multicomponent DIIS compatibility shim |
| `docs/cneomp2_gradient_theory.md` | derivation, solver notes, validation |
| `neo/test/test_cneomp2_grad.py` | FD regression tests (incl. N2 heavy) |
| `examples/hcn.aug-cc-pvtz.cneomp2.geometric.py` | analytic-gradient optimization |
| `examples/hcn.aug-cc-pvtz.cneomp2.constrained.BFGS.py` | energy-only optimization |
