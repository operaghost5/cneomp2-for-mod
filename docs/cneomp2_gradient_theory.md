# Analytic nuclear gradients for CNEO-MP2

This note derives the analytic-gradient working equations implemented in
`neo/cneomp2_grad.py` for the unconstrained-amplitude (UCD) CNEO-MP2 energy,
and records the validation results.  Equation numbers refer to the CNEO-MP2
paper (J. Chem. Phys. 164, 184117 (2026)).

## 1. Energy functional

The CNEO-MP2 total energy is

    E = E_CNEO-HF + E2,     E2 = J_e[t_e] + J_en[t_en] + J_n[t_n],

where the J's are the multicomponent Hylleraas functionals of Eqs. 17-19
evaluated with

* the CNEO-HF orbitals C^e, C^I (I = quantum nucleus), obtained from the
  constrained SCF: the nuclear Fock matrices carry the position-constraint
  term mu^(0)_I . (r - <r>_I S), and
* the *conventional* (unconstrained) NEO Fock matrices f^e, f^I of
  Eqs. 23/24, rebuilt from the CNEO orbitals; f^I is non-diagonal, so the
  amplitudes are obtained iteratively (non-canonical MP2, Eqs. 25-28).

In the UCD variant the correlated-density constraint of Eq. 22 is dropped
(mu^(2) = 0), so J is stationary with respect to every amplitude:
dJ/dt = 0 is exactly the residual equation solved by the amplitude
iterations (the Lambda amplitudes are the amplitude transposes throughout).

## 2. Gradient structure

Because J is stationary in t, the total derivative of E2 requires only the
derivatives of its *arguments*:

    dE2/dx = sum_k  D^k : df^k_MO/dx  +  sum_c Gamma_c : dg^c_MO/dx     (G1)

with

* unrelaxed one-particle densities `D^k = dJ/df^k` (occupied-occupied and
  virtual-virtual blocks only; k = e or a quantum nucleus), obtained by
  freeing the Fock index pair in each term of Eqs. 17-19, and
* two-particle densities `Gamma_c = dJ/dg^c` for the three integral classes
  c in {ee, eI, IJ}.  With Lambda = t these reduce to

      Gamma_ee(iajb) = 4 t(iajb) - 2 t(ibja)
      Gamma_eI(iaIA) = -4 t_eI(iaIA)
      Gamma_IJ(IAJB) = +2 t_IJ(IAJB)      (each unordered pair counted once)

  matching the +/- signs with which the g-terms enter Eqs. 17-19 (the code
  verifies these contractions against the energy kernels at run time).

The MO-matrix derivatives split into three pieces:

1. **Explicit AO derivatives (frozen orbitals).**
   For the Fock matrices these are exactly the `h1ao` matrices assembled by
   the CNEO Hessian module (`hessian.make_h1`): core-Hamiltonian and
   two-particle integral derivatives contracted with the frozen SCF
   densities.  The explicit derivative of the nuclear constraint term
   vanishes: the operator r - <r>_I S is translation invariant, and moving
   the nuclear basis center together with the constraint value R_I leaves
   it unchanged.  For the ERI classes the back-transformed Gamma's are
   contracted with `int2e_ip1` derivative integrals over the appropriate
   same- or cross-fragment blocks, with the same nuclear-charge factors as
   the energy integrals (Z_I for eI, Z_I Z_J for IJ).

2. **Fock response through the first-order densities.**
   df^k also contains the change of every component's SCF density.  With
   the first-order occupied orbitals mo1 (below), the response Fock
   matrices are obtained from the same multicomponent response function
   used by the Hessian (`_gen_neo_response`), and contracted with D^k.

3. **Orbital rotations.**
   Writing dC = C U, orthonormality fixes the symmetric part of U
   (U_pq + U_qp = -S1_pq, with S1 = 0 for nuclear components, whose basis
   functions move rigidly), while the occupied-virtual blocks follow from
   the coupled-perturbed CNEO-HF equations *including the response of the
   position-constraint multipliers* (`hessian.solve_mo1_rks`, which wraps
   `cphf.solve` with `with_f1n=True`).  The antisymmetric parts of the
   occupied-occupied and virtual-virtual blocks are arbitrary: a unitary
   rotation of the orbitals can be absorbed into a counter-rotation of the
   amplitude tensors, under which J is invariant at amplitude stationarity,
   so those rotations contribute nothing.  The rotation contribution is
   assembled as sum_tp U^k_tp M^k_tp, where the coefficient matrices M^k
   collect (a) f_MO D contractions and (b) Gamma contracted with MO ERIs
   carrying one running full index.

The CNEO-HF part of the gradient is the existing analytic
`neo.grad.Gradients` (Hellmann-Feynman plus basis-center terms; the nuclear
orbital response cancels at the SCF level by the same translational-
invariance argument, which does *not* hold for the correlation part -
hence the CPHF solves above).

This is a *forward-response* formulation: one constrained-CPHF solve per
nuclear displacement (3N per gradient).  A single-solve Z-vector (adjoint)
formulation is the natural future optimization; it was not needed for the
molecule sizes targeted here, where the SCF itself dominates the wall time.

### How the response equations are solved (and why not neo.cphf)

The right-hand sides are built with `hessian.make_h1` (verified element by
element against finite differences of the frozen-density Fock matrices),
but the linear equations are solved by a direct dense factorization inside
`cneomp2_grad` rather than by `neo.cphf.solve`.  The Krylov solver in
`neo.cphf` preconditions every row by 1/(eps_a - eps_i); for heavy quantum
nuclei the mass-scaled nuclear orbital-energy gaps are so large that the
nuclear-block residuals become invisibly small in the solver's convergence
metric, and it reports convergence while the actual solution error is
large.  On all-quantum N2/cc-pVDZ the returned first-order densities
deviate from finite-difference density responses by up to 6% (and the
first-order position constraint r.mo1 = 0 is violated at 5e-7), even at
Krylov tolerance 1e-13, while the *equations themselves* are satisfied by
the finite-difference response to 4e-8 - i.e. the equations are complete
and correct, only the linear solve is at fault.  Systems with only light
quantum nuclei (H, He) have small gaps and are unaffected, which is why
H-only tests of the Hessian machinery pass.  The same false convergence
therefore affects the analytic CNEO Hessian for heavy quantum nuclei;
this should be reported upstream.

The dense solve builds the coupled operator by batched applications of
the (finite-difference-verified) multicomponent response function over
unit vectors, LU-factorizes it once per gradient, and back-substitutes
all 3N right-hand sides.  System size is n_vir^e n_occ^e + sum_I n_vir^I
+ 3 N_quantum (about 1000 for HCN/aug-cc-pVTZ), negligible next to the
SCF.

## 3. Why the constrained-amplitude (CCD) gradient is not provided

The CCD energy implementation does not correspond to the paper's Eqs.
22/26/27: its residual constraint terms contract amplitude pairs with the
leading nocc x nocc / nvir x nvir corners of the *atomic-orbital* position
integral matrix over two independent virtual indices, whereas Eq. 26
prescribes a single sum over molecular-orbital <C|r|I> elements; moreover
the multiplier root-finding drives a *different* contraction (MO-space
gamma^(2) with shifted AO position integrals) to zero.  The stationary
point produced by kernel() is therefore not the saddle point of any single
Lagrangian, and a frozen-amplitude gradient of it is not well defined.
Since the paper (and our tests) find UCD and CCD minima to be nearly
identical, geometry optimization uses the UCD surface.

## 4. Validation (finite differences, central, Ha/Bohr)

| system | basis | quantum nuclei | max |analytic - FD| (MP2 part) |
|---|---|---|---|
| H2 | sto-3g/pb4d | both | 9.2e-09 |
| H2 | cc-pVDZ/pb4d | both | 1.5e-08 |
| HF | sto-3g/pb4d | H only | 3.2e-09 |
| HeH+ | cc-pVDZ/pb4d | both | 3.2e-08 |

(The residual differences are dominated by the finite-difference noise of
the constrained SCF itself; the HF-part agreement of the pre-existing
analytic SCF gradient sets the same scale.)

geomeTRIC optimizations driven by these gradients (`neo/cneomp2_geomopt.py`)
reproduce the energy-only optimization results; see the examples directory.
