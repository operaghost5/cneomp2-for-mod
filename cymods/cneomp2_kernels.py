#-------------------------------------------------------------------------------
# Vectorized CNEO-MP2 kernels
#-------------------------------------------------------------------------------
# This module contains NumPy/BLAS implementations of the performance-critical
# CNEO-MP2 functions that previously lived in the individual Cython modules
# under cymods/ (t_amps_*, mp2_dens, hylleraas_*, rmsd).  Every explicit
# element-by-element loop nest has been replaced with einsum/BLAS tensor
# contractions, which reduces the per-iteration cost of the amplitude updates
# from O(N_occ^2 N_vir^2 (N_occ + N_vir + N_vir_n^2)) scalar operations in
# interpreted-index order to a handful of dgemm calls, and removes the
# per-element allocation of small numpy arrays (the old gradient-term buffers)
# that dominated both runtime and allocator pressure.
#
# The numerics are identical to the original loop implementations (verified
# element-wise against the compiled originals); only the evaluation order of
# the floating point sums differs, so results agree to ~1e-13 relative.
#
# The individual cymods subpackages (cymods.t_amps_e_only, ...) re-export the
# functions defined here, so existing import paths keep working unchanged.
#-------------------------------------------------------------------------------

import math
import numpy

# The original kernels printed a tracing line on every call, which is itself
# measurable overhead inside the Lagrange-multiplier root-finding loop (the
# kernels are called thousands of times).  Set VERBOSE = True to restore the
# old tracing output.
VERBOSE = False

__all__ = ['VERBOSE', 'fock_blocks', 'denom_divide_inplace',
           't_amps_e_only', 't_amps_en_only', 't_amps_n_only',
           'mp2_density_one',
           'Hylleraas_energy_e', 'Hylleraas_energy_en', 'Hylleraas_energy_n',
           'RMSD_e', 'RMSD_en', 'RMSD_n']


#-------------------------------------------------------------------------------
# Shared helpers
#-------------------------------------------------------------------------------
def fock_blocks(fock, nocc):
    '''Split an MO-basis Fock matrix into occupied/virtual blocks and the
    diagonal orbital energies used in the MP2 denominators.'''
    fock = numpy.asarray(fock)
    f_oo = fock[:nocc, :nocc]
    f_vv = fock[nocc:, nocc:]
    eps_o = numpy.ascontiguousarray(numpy.diag(f_oo))
    eps_v = numpy.ascontiguousarray(numpy.diag(f_vv))
    return f_oo, f_vv, eps_o, eps_v


def denom_divide_inplace(r, eps_o1, eps_v1, eps_o2, eps_v2):
    '''Divide the residual tensor r[i,a,j,b] by the orbital-energy denominator
    (eps_o1[i] + eps_o2[j] - eps_v1[a] - eps_v2[b]) in place.

    The denominator is built one leading index at a time so that no second
    four-index tensor is ever materialized (the old code allocated a full
    o*v*o*v denominator tensor next to the amplitudes).'''
    d_avb = (- eps_v1[:, None, None]
             + eps_o2[None, :, None]
             - eps_v2[None, None, :])
    for i in range(r.shape[0]):
        r[i] /= (eps_o1[i] + d_avb)
    return r


def _constraint_row_sums(lagr, integrals_r, nocc, nvir):
    '''Row/column sums of the position-integral matrix contracted with the
    Lagrange multipliers, for the leading nocc x nocc and nvir x nvir corners.

    Note: following the original implementation, the "occupied" and "virtual"
    density-gradient terms index the *leading* nocc x nocc and nvir x nvir
    corners of the (3, nao, nao) position-integral array that the caller
    passes in.'''
    lagr = numpy.asarray(lagr, dtype=numpy.float64)
    r = numpy.asarray(integrals_r)
    # lr[p, q] = sum_x lagr[x] * integrals_r[x, p, q]
    lr = numpy.tensordot(lagr, r, axes=(0, 0))
    lr_vir = lr[:nvir, :nvir]
    lr_occ = lr[:nocc, :nocc]
    return (lr_vir.sum(axis=1), lr_vir.sum(axis=0),
            lr_occ.sum(axis=1), lr_occ.sum(axis=0))


#-------------------------------------------------------------------------------
# [2.5] Electronic t-amplitude function
#-------------------------------------------------------------------------------
# Residual expression (Eq. 28 of the CNEO-MP2 paper) rearranged so that the
# new t-amplitudes are isolated:
#
#   (e_i + e_j - e_a - e_b) t(iajb) = <ij||ab>
#                                   + sum_{c!=a} f_ac t(icjb)
#                                   + sum_{c!=b} f_bc t(iajc)
#                                   - sum_{k!=i} f_ki t(kajb)
#                                   - sum_{k!=j} f_kj t(iakb)
#
# Each restricted inner sum equals the unrestricted contraction minus its
# excluded diagonal element, and the four excluded elements together equal
# -(e_i + e_j - e_a - e_b) * t_old.  Hence
#
#   t_new = (unrestricted residual)/denominator + t_old,
#
# which is contraction-only and maps directly onto BLAS.  The same identity is
# used for the electron-nuclear and nuclear amplitude updates below.
#-------------------------------------------------------------------------------
def t_amps_e_only(self, t_electronic, two_int_e_t, ncF_eMO):

    if VERBOSE:
        print('calling t_amps_e_only from cymods')

    e_nocc = self.e_nocc
    t_old = numpy.asarray(t_electronic)
    f_oo, f_vv, eps_o, eps_v = fock_blocks(ncF_eMO, e_nocc)

    residual = numpy.array(two_int_e_t, dtype=numpy.float64, copy=True)
    scratch = numpy.empty_like(residual)

    numpy.einsum('ac,icjb->iajb', f_vv, t_old, out=scratch, optimize=True)
    residual += scratch
    numpy.einsum('bc,iajc->iajb', f_vv, t_old, out=scratch, optimize=True)
    residual += scratch
    numpy.einsum('ki,kajb->iajb', f_oo, t_old, out=scratch, optimize=True)
    residual -= scratch
    numpy.einsum('kj,iakb->iajb', f_oo, t_old, out=scratch, optimize=True)
    residual -= scratch
    del scratch

    denom_divide_inplace(residual, eps_o, eps_v, eps_o, eps_v)
    residual += t_old
    return residual


#-------------------------------------------------------------------------------
# [2.6] Electronic-nuclear t-amplitude function
#-------------------------------------------------------------------------------
# Residual of Eq. 29.  The Fock-contraction terms use the same diagonal
# exclusion identity as the electronic case.  The constraint (Lagrange
# multiplier) terms couple the amplitude/lambda tensors to the nuclear
# position integrals; because the position-integral matrix only enters through
# its (Lagrange-multiplier contracted) row and column sums, those terms reduce
# to rank-1 contractions instead of the O(o v O V (V^2 + O^2)) loop nest of
# the original implementation.
#-------------------------------------------------------------------------------
def t_amps_en_only(self, j_idx, lagr_multipliers, t_electronic_nuclear,
                   l_electronic_nuclear, two_int_en_t, ncF_eMO, ncF_nMO_j,
                   integrals_r_j):
    '''Note that the lagr_multipliers, t_electronic_nuclear, integrals, and
    ncF_nMO should all be indexed by the nucleus when being fed in to the
    function.'''

    if VERBOSE:
        print('calling t_amps_en_only from cymods...')

    p_nocc = int(self.num_ovt[j_idx][0, 0])
    p_nvir = int(self.num_ovt[j_idx][0, 1])
    e_nocc = self.e_nocc

    t_old = numpy.asarray(t_electronic_nuclear)
    l_old = numpy.asarray(l_electronic_nuclear)

    f_oo_e, f_vv_e, eps_o_e, eps_v_e = fock_blocks(ncF_eMO, e_nocc)
    f_oo_n, f_vv_n, eps_o_n, eps_v_n = fock_blocks(ncF_nMO_j, p_nocc)

    # residual = -<iI|aA> + Fock contractions (unrestricted; see note above)
    residual = -numpy.array(two_int_en_t, dtype=numpy.float64, copy=True)
    scratch = numpy.empty_like(residual)

    numpy.einsum('ac,icIA->iaIA', f_vv_e, t_old, out=scratch, optimize=True)
    residual += scratch
    numpy.einsum('AC,iaIC->iaIA', f_vv_n, t_old, out=scratch, optimize=True)
    residual += scratch
    numpy.einsum('ki,kaIA->iaIA', f_oo_e, t_old, out=scratch, optimize=True)
    residual -= scratch
    numpy.einsum('KI,iaKA->iaIA', f_oo_n, t_old, out=scratch, optimize=True)
    residual -= scratch
    del scratch

    # Constraint terms.  In the original loops:
    #   vir:  sum_{R,T} (l(a,i,R,I) + t(i,a,I,T)) mu.r(R,T)
    #   occ:  sum_{P,Q} (l(a,i,A,P) + t(i,a,Q,A)) mu.r(P,Q)
    # so each term only needs the row/column sums of mu.r over the leading
    # virtual x virtual and occupied x occupied corners.
    lr_v_row, lr_v_col, lr_o_row, lr_o_col = _constraint_row_sums(
        lagr_multipliers, integrals_r_j, p_nocc, p_nvir)

    # virtual constraint: depends on (i, a, I) only -> broadcast over A
    cv = (numpy.einsum('aiRI,R->iaI', l_old, lr_v_row, optimize=True)
          + numpy.einsum('iaIT,T->iaI', t_old, lr_v_col, optimize=True))
    # occupied constraint: depends on (i, a, A) only -> broadcast over I
    co = (numpy.einsum('aiAP,P->iaA', l_old, lr_o_row, optimize=True)
          + numpy.einsum('iaQA,Q->iaA', t_old, lr_o_col, optimize=True))

    residual += cv[:, :, :, None]
    residual -= co[:, :, None, :]

    denom_divide_inplace(residual, eps_o_e, eps_v_e, eps_o_n, eps_v_n)
    residual += t_old
    return residual


#-------------------------------------------------------------------------------
# [2.7] Nuclear t-amplitude function
#-------------------------------------------------------------------------------
# Residual of Eq. 30, for the amplitude tensor of the nuclear pair
# (i_idx, j_idx).  Same structure as the electron-nuclear update, with
# constraint terms for both nuclei of the pair.
#
# The original implementation recomputed the full (i,j) two-particle MO
# integral tensor (an AO int2e evaluation plus a four-index transformation)
# on *every call* and then discarded it; the integrals are already supplied
# through two_int_n_t, so that recomputation has been removed.
#-------------------------------------------------------------------------------
def t_amps_n_only(self, i_idx, j_idx, lagr_multipliers_i, lagr_multipliers_j,
                  t_nuclear, l_nuclear, two_int_n_t, ncF_nMO_i, ncF_nMO_j,
                  integrals_r_i, integrals_r_j):
    '''Note that the lagr_multipliers_i, lagr_multipliers_j, integrals, and
    ncF_nMO_i, ncF_nMO_j should each be indexed by the correct nucleus and the
    t_nuclear should be indexed by the pair of nuclei when being fed in to the
    function.'''

    if VERBOSE:
        print('calling t_amps_n_only from cymods...')

    nuclei = len(self.con.mol.nuc)

    # There is no t-amplitude for a single quantum nucleus.
    if nuclei == 1:
        print('WARNING - No nuclear t-amplitudes are calculated when only '
              'one nucleus is treated quantum mechanically!')
        return t_nuclear

    p_nocc_i = int(self.num_ovt[i_idx][0, 0])
    p_nvir_i = int(self.num_ovt[i_idx][0, 1])
    p_nocc_j = int(self.num_ovt[j_idx][0, 0])
    p_nvir_j = int(self.num_ovt[j_idx][0, 1])

    # Diagonal (same-nucleus) blocks of the amplitude storage matrix are zero.
    if i_idx == j_idx:
        return numpy.zeros((p_nocc_i, p_nvir_i, p_nocc_j, p_nvir_j))

    t_old = numpy.asarray(t_nuclear)
    l_old = numpy.asarray(l_nuclear)

    f_oo_i, f_vv_i, eps_o_i, eps_v_i = fock_blocks(ncF_nMO_i, p_nocc_i)
    f_oo_j, f_vv_j, eps_o_j, eps_v_j = fock_blocks(ncF_nMO_j, p_nocc_j)

    residual = numpy.array(two_int_n_t, dtype=numpy.float64, copy=True)
    scratch = numpy.empty_like(residual)

    numpy.einsum('AC,ICJB->IAJB', f_vv_i, t_old, out=scratch, optimize=True)
    residual += scratch
    numpy.einsum('BC,IAJC->IAJB', f_vv_j, t_old, out=scratch, optimize=True)
    residual += scratch
    numpy.einsum('KI,KAJB->IAJB', f_oo_i, t_old, out=scratch, optimize=True)
    residual -= scratch
    numpy.einsum('KJ,IAKB->IAJB', f_oo_j, t_old, out=scratch, optimize=True)
    residual -= scratch
    del scratch

    lr_vi_row, lr_vi_col, lr_oi_row, lr_oi_col = _constraint_row_sums(
        lagr_multipliers_i, integrals_r_i, p_nocc_i, p_nvir_i)
    lr_vj_row, lr_vj_col, lr_oj_row, lr_oj_col = _constraint_row_sums(
        lagr_multipliers_j, integrals_r_j, p_nocc_j, p_nvir_j)

    # Nucleus-i constraint terms (mirror the original loop structure):
    #   vir_i: sum_{R,T} (l(R,I,B,J) + t(I,T,J,B)) mu_i.r_i(R,T)
    #   occ_i: sum_{P,Q} (l(A,P,B,J) + t(Q,A,J,B)) mu_i.r_i(P,Q)
    cv_i = (numpy.einsum('RIBJ,R->IJB', l_old, lr_vi_row, optimize=True)
            + numpy.einsum('ITJB,T->IJB', t_old, lr_vi_col, optimize=True))
    co_i = (numpy.einsum('APBJ,P->AJB', l_old, lr_oi_row, optimize=True)
            + numpy.einsum('QAJB,Q->AJB', t_old, lr_oi_col, optimize=True))

    # Nucleus-j constraint terms:
    #   vir_j: sum_{S,U} (l(A,I,S,J) + t(I,A,J,U)) mu_j.r_j(S,U)
    #   occ_j: sum_{M,N} (l(A,I,B,M) + t(I,A,N,B)) mu_j.r_j(M,N)
    cv_j = (numpy.einsum('AISJ,S->IAJ', l_old, lr_vj_row, optimize=True)
            + numpy.einsum('IAJU,U->IAJ', t_old, lr_vj_col, optimize=True))
    co_j = (numpy.einsum('AIBM,M->IAB', l_old, lr_oj_row, optimize=True)
            + numpy.einsum('IANB,N->IAB', t_old, lr_oj_col, optimize=True))

    residual += cv_i[:, None, :, :]    # (I,J,B) broadcast over A
    residual += cv_j[:, :, :, None]    # (I,A,J) broadcast over B
    residual -= co_i[None, :, :, :]    # (A,J,B) broadcast over I
    residual -= co_j[:, :, None, :]    # (I,A,B) broadcast over J

    denom_divide_inplace(residual, eps_o_i, eps_v_i, eps_o_j, eps_v_j)
    residual += t_old
    return residual


#-------------------------------------------------------------------------------
# [2.8] Nuclear single-particle MP2 density matrix function
#-------------------------------------------------------------------------------
# Eqs. 20/21.  The multiplicity of the electron-nuclear contributions matches
# the original loop implementation exactly: in the multi-nucleus branch the
# e-n virtual term is accumulated inside the loop over the occupied orbitals
# of each partner nucleus j (factor p_nocc_j per partner) and the e-n occupied
# term inside the loop over the virtual orbitals of each partner nucleus
# (factor p_nvir_j per partner).
#-------------------------------------------------------------------------------
def mp2_density_one(self, i_idx, T_n, L_n, T_en, L_en):

    if VERBOSE:
        print('calling mp2_density_one function from cymods...')

    nuclei = len(self.con.mol.nuc)

    p_nocc_i = int(self.num_ovt[i_idx][0, 0])
    p_nvir_i = int(self.num_ovt[i_idx][0, 1])
    p_tot_i = int(self.num_ovt[i_idx][0, 2])

    t_en_i = numpy.asarray(T_en[i_idx])
    l_en_i = numpy.asarray(L_en[i_idx])

    # Electron-nuclear contributions (Eqs. 20/21, second terms):
    #   gamma_vir[A,B] =  sum_{iaI} l(a,i,A,I) t(i,a,I,B)
    #   gamma_occ[I,J] = -sum_{iaA} l(a,i,A,J) t(i,a,I,A)
    gv_en = numpy.einsum('aiAI,iaIB->AB', l_en_i, t_en_i, optimize=True)
    go_en = -numpy.einsum('aiAJ,iaIA->IJ', l_en_i, t_en_i, optimize=True)

    if nuclei == 1:
        gamma_vir = gv_en
        gamma_occ = go_en
    else:
        gamma_vir = numpy.zeros((p_nvir_i, p_nvir_i))
        gamma_occ = numpy.zeros((p_nocc_i, p_nocc_i))
        for j_idx in range(nuclei):
            if i_idx == j_idx:
                continue
            p_nocc_j = int(self.num_ovt[j_idx][0, 0])
            p_nvir_j = int(self.num_ovt[j_idx][0, 1])

            t_n_ij = numpy.asarray(T_n[i_idx][j_idx])
            l_n_ij = numpy.asarray(L_n[i_idx][j_idx])

            # Nuclear-pair contributions (Eqs. 20/21, first terms):
            gamma_vir += numpy.einsum('AICJ,IBJC->AB', l_n_ij, t_n_ij,
                                      optimize=True)
            gamma_occ -= numpy.einsum('AJBK,IAKB->IJ', l_n_ij, t_n_ij,
                                      optimize=True)

            # Electron-nuclear contributions with the multiplicity of the
            # original loop nest (see the note at the top of this function).
            gamma_vir += p_nocc_j * gv_en
            gamma_occ += p_nvir_j * go_en

    gamma_total = numpy.zeros((p_tot_i, p_tot_i))
    gamma_total[:p_nocc_i, :p_nocc_i] = gamma_occ
    gamma_total[p_nocc_i:, p_nocc_i:] = gamma_vir
    return gamma_total


#-------------------------------------------------------------------------------
# [2.9] Hylleraas electronic energy function
#-------------------------------------------------------------------------------
# Eq. 17 in the spin-adapted {abab}/{abba} form used by the original
# implementation.  Each accumulated product of g/f/t/lambda terms is expanded
# and evaluated as a single tensor contraction.
#-------------------------------------------------------------------------------
def Hylleraas_energy_e(self, t_amps_e, l_amps_e, two_int_e_t, two_int_e_l,
                       fock_e):

    if VERBOSE:
        print('calling Hylleraas_energy_e from cymods')

    e_nocc = self.e_nocc

    t = numpy.asarray(t_amps_e)
    l = numpy.asarray(l_amps_e)
    g_t = numpy.asarray(two_int_e_t)
    g_l = numpy.asarray(two_int_e_l)
    f_oo, f_vv, _, _ = fock_blocks(fock_e, e_nocc)

    ee = lambda spec, *ops: numpy.einsum(spec, *ops, optimize=True)

    # g.t terms: 2[(g1-g2)(t1-t2) + g1 t1 + g2 t2]
    #          = 2[2 g1 t1 - g1 t2 - g2 t1 + 2 g2 t2]
    # with g1 = g_l(a,i,b,j), g2 = g_l(a,j,b,i),
    #      t1 = t(i,a,j,b),   t2 = t(i,b,j,a).
    gt = 2.0 * (2.0 * ee('aibj,iajb->', g_l, t)
                - ee('aibj,ibja->', g_l, t)
                - ee('ajbi,iajb->', g_l, t)
                + 2.0 * ee('ajbi,ibja->', g_l, t))

    # g.lambda terms, same structure with g_t and lambda.
    gl = 2.0 * (2.0 * ee('iajb,aibj->', g_t, l)
                - ee('iajb,ajbi->', g_t, l)
                - ee('ibja,aibj->', g_t, l)
                + 2.0 * ee('ibja,ajbi->', g_t, l))

    # f_vv terms: sum_c f(a,c) [ (l1-l2)(tc1-tc2) + l1 tc1 + l2 tc2 ]
    # with tc1 = t(i,c,j,b), tc2 = t(i,b,j,c).
    c_sum = 2.0 * (2.0 * ee('ac,aibj,icjb->', f_vv, l, t)
                   - ee('ac,aibj,ibjc->', f_vv, l, t)
                   - ee('ac,ajbi,icjb->', f_vv, l, t)
                   + 2.0 * ee('ac,ajbi,ibjc->', f_vv, l, t))

    # f_oo terms: sum_k f(i,k) [ (lk1-lk2)(t1-t2) + lk1 t1 + lk2 t2 ]
    # with lk1 = l(a,k,b,j), lk2 = l(a,j,b,k).
    k_sum = 2.0 * (2.0 * ee('ik,akbj,iajb->', f_oo, l, t)
                   - ee('ik,akbj,ibja->', f_oo, l, t)
                   - ee('ik,ajbk,iajb->', f_oo, l, t)
                   + 2.0 * ee('ik,ajbk,ibja->', f_oo, l, t))

    return 0.5 * c_sum - 0.5 * k_sum + 0.25 * gt + 0.25 * gl


#-------------------------------------------------------------------------------
# [2.10] Hylleraas electronic-nuclear energy function
#-------------------------------------------------------------------------------
# Eq. 18, summed over all quantum nuclei.
#-------------------------------------------------------------------------------
def Hylleraas_energy_en(self, t_amps_en, l_amps_en, two_int_en_t, two_int_en_l,
                        fock_e, fock_n):

    if VERBOSE:
        print('calling Hylleraas_energy_en from cymods')

    nuclei = len(self.con.mol.nuc)
    e_nocc = self.e_nocc
    f_oo_e, f_vv_e, _, _ = fock_blocks(fock_e, e_nocc)

    ee = lambda spec, *ops: numpy.einsum(spec, *ops, optimize=True)

    total_en = 0.0
    for j in range(nuclei):
        p_nocc = self.con.mf_nuc[j].mo_coeff[:, self.con.mf_nuc[j].mo_occ > 0].shape[1]

        t = numpy.asarray(t_amps_en[j])
        l = numpy.asarray(l_amps_en[j])
        g_t = numpy.asarray(two_int_en_t[j])
        g_l = numpy.asarray(two_int_en_l[j])
        f_oo_n, f_vv_n, _, _ = fock_blocks(fock_n[j], p_nocc)

        gt_sum = 2.0 * ee('aiAI,iaIA->', g_l, t)
        gl_sum = 2.0 * ee('iaIA,aiAI->', g_t, l)
        c_sum = 2.0 * ee('ac,aiAI,icIA->', f_vv_e, l, t)
        k_sum = 2.0 * ee('ik,akAI,iaIA->', f_oo_e, l, t)
        C_sum = 2.0 * ee('AC,aiAI,iaIC->', f_vv_n, l, t)
        K_sum = 2.0 * ee('IK,aiAK,iaIA->', f_oo_n, l, t)

        total_en += (c_sum - k_sum + C_sum - K_sum - gt_sum - gl_sum)

    return total_en


#-------------------------------------------------------------------------------
# [2.11] Hylleraas nuclear energy function
#-------------------------------------------------------------------------------
# Eq. 19, summed over distinct nuclear pairs (the original stores all (i,j)
# pair energies but only sums j < i, which is preserved here).
#-------------------------------------------------------------------------------
def Hylleraas_energy_n(self, t_amps_n, l_amps_n, two_int_n_t, two_int_n_l,
                       fock_n):

    if VERBOSE:
        print('calling Hylleraas_energy_n from cymods')

    nuclei = len(self.con.mol.nuc)

    ee = lambda spec, *ops: numpy.einsum(spec, *ops, optimize=True)

    total_n = 0.0
    for i in range(nuclei):
        for j in range(i):
            p_nocc_i = self.con.mf_nuc[i].mo_coeff[:, self.con.mf_nuc[i].mo_occ > 0].shape[1]
            p_nocc_j = self.con.mf_nuc[j].mo_coeff[:, self.con.mf_nuc[j].mo_occ > 0].shape[1]

            t = numpy.asarray(t_amps_n[i][j])
            l = numpy.asarray(l_amps_n[i][j])
            g_t = numpy.asarray(two_int_n_t[i][j])
            g_l = numpy.asarray(two_int_n_l[i][j])
            f_oo_i, f_vv_i, _, _ = fock_blocks(fock_n[i], p_nocc_i)
            f_oo_j, f_vv_j, _, _ = fock_blocks(fock_n[j], p_nocc_j)

            gt_sum = ee('AIBJ,IAJB->', g_l, t)
            gl_sum = ee('IAJB,AIBJ->', g_t, l)
            Ci_sum = ee('AC,AIBJ,ICJB->', f_vv_i, l, t)
            Cj_sum = ee('BC,AIBJ,IAJC->', f_vv_j, l, t)
            Ki_sum = ee('IK,AKBJ,IAJB->', f_oo_i, l, t)
            Kj_sum = ee('JK,AIBK,IAJB->', f_oo_j, l, t)

            total_n += (gt_sum + gl_sum + (Ci_sum + Cj_sum)
                        - (Ki_sum + Kj_sum))

    return total_n


#-------------------------------------------------------------------------------
# [2.13] RMSD calculations
#-------------------------------------------------------------------------------
def RMSD_e(self, t_old, t_new):

    if VERBOSE:
        print('calling RMSD_e from cymods')

    diff = numpy.asarray(t_new) - numpy.asarray(t_old)
    return float(numpy.sqrt(numpy.vdot(diff, diff)))


def RMSD_en(self, t_old, t_new):

    if VERBOSE:
        print('calling RMSD_en from cymods')

    nuclei = len(self.con.mol.nuc)
    total_RMSD = 0.0
    for j in range(nuclei):
        diff = numpy.asarray(t_new[j]) - numpy.asarray(t_old[j])
        total_RMSD += math.sqrt(numpy.vdot(diff, diff))
    return total_RMSD


def RMSD_n(self, t_old, t_new):

    if VERBOSE:
        print('calling RMSD_n from cymods')

    nuclei = len(self.con.mol.nuc)
    total_RMSD = 0.0
    # Only distinct pairs (j < i) enter the total, as in the original.
    for i in range(nuclei):
        for j in range(i):
            diff = numpy.asarray(t_new[i][j]) - numpy.asarray(t_old[i][j])
            total_RMSD += math.sqrt(numpy.vdot(diff, diff))
    return total_RMSD
