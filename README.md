# cNEO-MP2

Implementation of the constrained nuclear–electronic orbital second-order
Møller–Plesset perturbation theory (CNEO-MP2) method; see
*J. Chem. Phys.* **164**, 184117 (2026), [doi:10.1063/5.0327643](https://doi.org/10.1063/5.0327643).

## Layout

- `neo/` — NEO/CNEO module (drop-in for `pyscf.neo`, based on the Yang-group
  PySCF fork). `neo/cneomp2.py` contains the `cNEOMP2` class implementing the
  double-loop iterative solution of the noncanonical MP2 amplitude equations
  with the correlated-density constraint.
- `cymods/` — the performance-critical CNEO-MP2 functions. All numerical
  kernels (amplitude updates, Hylleraas energies, MP2 nuclear density,
  amplitude RMSDs) live in `cymods/cneomp2_kernels.py` as vectorized
  NumPy/BLAS tensor contractions; the subpackages
  (`cymods.t_amps_e_only`, ...) re-export them under the historical import
  paths. **No Cython compilation is required.**
- `pymods/` — legacy pure-Python loop implementations of the same kernels
  (kept for reference; not imported by `neo/cneomp2.py`).
- `examples/` — self-contained geometry-optimization driver scripts.
- `numhess/` — numerical Hessian utilities.

## Notes on the refactored kernels

The kernels were rewritten from element-by-element loops into einsum/BLAS
contractions and validated element-wise against the original compiled
implementations (agreement to ~1e-13 relative, i.e. floating-point summation
order). Additional redundant work removed with no change to the iterates:

- the AO→MO transformations now compute only the electron–nucleus and
  nucleus–nucleus **cross** integral blocks with 4-fold symmetry instead of
  the full combined-system two-electron integral tensor, and are performed
  once per kernel invocation instead of once per amplitude update;
- λ-amplitudes and mirrored nuclear amplitude blocks are transpose views of
  the t-amplitude tensors rather than explicit copies;
- Fock matrices that were built but never referenced are no longer
  constructed.

Set `cymods.cneomp2_kernels.VERBOSE = True` to restore the per-call kernel
tracing output, and `cNEOMP2.verbose >= 4` for the per-subcycle diagnostics of
the Lagrange-multiplier constraint loops.
