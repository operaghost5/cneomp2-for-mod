#-------------------------------------------------------------------------------
# cNEOMP2 Module
#-------------------------------------------------------------------------------
# Implementation of constrained nuclear-electronic orbital second-order
# Moller-Plesset perturbation theory (CNEO-MP2); see
# J. Chem. Phys. 164, 184117 (2026), doi: 10.1063/5.0327643.
#
# This is a refactored version of the original implementation.  The iteration
# scheme, convergence criteria, and all working equations are unchanged; the
# numerical kernels now live in cymods.cneomp2_kernels as BLAS/einsum tensor
# contractions and the setup stage computes only the integral blocks that are
# actually used.  Specific redundancies removed relative to the original:
#
#   * Only the noncanonical NEO Fock matrices (Eqs. 23/24 evaluated with the
#     CNEO-HF orbitals) are constructed.  The canonical NEO-HF and CNEO-HF
#     Fock matrices that were previously also built (several full Coulomb
#     builds each) were never referenced afterwards.
#   * The nucleus-nucleus and electron-nucleus MO integrals (ia|IA), (IA|JB)
#     are computed once per kernel() call from the cross AO blocks only (see
#     neo.ao2mo), and the (JB|IA) tensor is obtained as a transpose view of
#     (IA|JB) rather than through a second AO->MO transformation.  The
#     same-nucleus diagonal blocks, which are never used, are not computed.
#   * The per-call DIIS objects were constructed fresh at every use, so their
#     kernel() always took the iteration-0 branch and returned its input
#     unchanged; those identity round-trips (full amplitude copies) have been
#     removed.  This does not change any iterate.
#   * Lambda amplitudes are transpose *views* of the t-amplitudes (the
#     relation used throughout the original: l = t.T with axes (0,2) and
#     (1,3) swapped, i.e. numpy.transpose(t, (1, 0, 3, 2))), so they cost no
#     memory and no copies.
#   * The mirrored nuclear amplitude blocks t[j][i] are transpose views of
#     t[i][j] instead of explicitly swapped copies.
#-------------------------------------------------------------------------------

import sys

import numpy
import scipy

from pyscf import ao2mo, neo, lib

from cymods.t_amps_e_only.t_amps_e_only import t_amps_e_only
from cymods.t_amps_en_only.t_amps_en_only import t_amps_en_only
from cymods.t_amps_n_only.t_amps_n_only import t_amps_n_only
from cymods.mp2_dens.mp2_density import mp2_density_one
from cymods.hylleraas_e.hylleraas_e import Hylleraas_energy_e
from cymods.hylleraas_en.hylleraas_en import Hylleraas_energy_en
from cymods.hylleraas_n.hylleraas_n import Hylleraas_energy_n
from cymods.rmsd.rmsd import RMSD_e, RMSD_en, RMSD_n

numpy.set_printoptions(suppress=True, precision=8,
                       linewidth=sys.maxsize, threshold=sys.maxsize)

line = ('---------------------------------------------')
asterisk = ('*********************************************')


def _l_view(t):
    '''Lambda amplitudes as a transpose view of the t-amplitudes:
    l(a,i,b,j) = t(i,a,j,b) (equivalently t.T with axes (0,2),(1,3) swapped,
    exactly the relation used by the original implementation).'''
    return numpy.transpose(t, (1, 0, 3, 2))


def _mirror(t):
    '''Amplitudes of the swapped particle pair as a transpose view:
    t_{ji}(J,B,I,A) = t_{ij}(I,A,J,B).'''
    return numpy.transpose(t, (2, 3, 0, 1))


#-------------------------------------------------------------------------------
# The cNEO-MP2 Class
#-------------------------------------------------------------------------------
class cNEOMP2(lib.StreamObject):

    #---------------------------------------------------------------------------
    # [1] Initialization and instantiation of class attributes
    #---------------------------------------------------------------------------
    # The reg object is the regular NEO-HF class treated mole object.
    # The con object is the NEO-c-DFT(HF) class treated mole object.
    # The base_energy is the cHF energy used as input to the cNEOMP2 class.
    #---------------------------------------------------------------------------
    def __init__(self, reg, con, base_energy):

        self.reg = reg
        self.con = con
        self.base_energy = base_energy

        # Verbosity of the MP2 iterations: >= 4 restores the per-subcycle
        # diagnostic output of the constraint (Lagrange-multiplier) loops.
        self.verbose = 0

        nuc_num = len(con.mol.nuc)
        self.mp2_density_converged = [False] * nuc_num

        # The electronic and nuclear density matrices and MO coefficients of
        # the reg mole object are set to those of the con (constrained) mole
        # object, so all Fock matrices built through reg are the noncanonical
        # NEO Fock matrices of Eqs. 23/24.
        self.reg.dm_elec = self.con.dm_elec
        self.reg.mf_elec.mo_coeff = self.con.mf_elec.mo_coeff
        for i in range(nuc_num):
            self.reg.dm_nuc[i] = self.con.dm_nuc[i]
            self.reg.mf_nuc[i].mo_coeff = self.con.mf_nuc[i].mo_coeff

        self.R_mp2 = self.con.mol.atom_coords(unit='ANG')

        # Position integrals shifted by the constrained expectation values,
        # used to evaluate the correlated-density constraint (Eq. 22).
        self.posn_ints = [None] * nuc_num
        for i in range(nuc_num):
            hk = self.con.mf_nuc[i]
            s1n = hk.get_ovlp(hk.mol)
            r_int = hk.mol.intor_symmetric('int1e_r', comp=3)
            shift = numpy.array([hk.nuclei_expect_position[x] * s1n
                                 for x in range(3)])
            self.posn_ints[i] = r_int - shift

        # Lagrange multipliers for the constrained optimization of the nuclear
        # and electronic-nuclear MP2 t-amplitudes.
        self.lagr = numpy.zeros((nuc_num, 3))
        self.try_lagr = numpy.zeros((nuc_num, 3))

        # Electronic dimensions.
        self.e_nocc = con.mf_elec.mo_coeff[:, con.mf_elec.mo_occ > 0].shape[1]
        self.e_tot = con.mf_elec.mo_coeff[0, :].shape[0]
        self.e_nvir = self.e_tot - self.e_nocc

        # Nuclear dimensions: number of occupied, virtual, and total orbitals
        # for each nucleus (kept in the original (1,3) array layout for
        # compatibility with the kernels).
        self.num_ovt = []
        for i in range(nuc_num):
            nuc_occ = self.con.mf_nuc[i].mo_coeff[:, self.con.mf_nuc[i].mo_occ > 0].shape[1]
            nuc_tot = self.con.mf_nuc[i].mo_coeff[0, :].shape[0]
            ovt_array = numpy.zeros((1, 3))
            ovt_array[0, 0] = nuc_occ
            ovt_array[0, 1] = nuc_tot - nuc_occ
            ovt_array[0, 2] = nuc_tot
            self.num_ovt.append(ovt_array)

        # Electronic amplitudes.
        self.t_elec = numpy.zeros((self.e_nocc, self.e_nvir,
                                   self.e_nocc, self.e_nvir))
        self.l_elec = self.t_elec.T

        # Electronic-nuclear amplitudes (one tensor per nucleus).
        self.t_elecnuc = []
        for i in range(nuc_num):
            O = int(self.num_ovt[i][0, 0])
            V = int(self.num_ovt[i][0, 1])
            self.t_elecnuc.append(numpy.zeros((self.e_nocc, self.e_nvir, O, V)))
        self.l_elecnuc = [_l_view(t) for t in self.t_elecnuc]

        # Nuclear amplitudes (one tensor per ordered pair of nuclei; the
        # (j,i) block is a transpose view of the (i,j) block).
        self.t_nuc = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(i + 1):
                Oi = int(self.num_ovt[i][0, 0])
                Vi = int(self.num_ovt[i][0, 1])
                Oj = int(self.num_ovt[j][0, 0])
                Vj = int(self.num_ovt[j][0, 1])
                self.t_nuc[i][j] = numpy.zeros((Oi, Vi, Oj, Vj))
                if i != j:
                    self.t_nuc[j][i] = _mirror(self.t_nuc[i][j])
        self.l_nuc = self._l_nuc_views(self.t_nuc)

        # Working copies used inside the constraint optimization.
        self.t_opt_en = list(self.t_elecnuc)
        self.l_opt_en = list(self.l_elecnuc)
        self.t_elecnuc_test = list(self.t_elecnuc)
        self.l_elecnuc_test = list(self.l_elecnuc)
        self.t_opt_n = self.t_nuc.copy()
        self.l_opt_n = self.l_nuc.copy()
        self.t_nuc_test = self.t_nuc.copy()
        self.l_nuc_test = self.l_nuc.copy()

    def _l_nuc_views(self, t_nuc):
        nuc_num = len(self.con.mol.nuc)
        l_nuc = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(nuc_num):
                l_nuc[i][j] = _l_view(t_nuc[i][j])
        return l_nuc

    #---------------------------------------------------------------------------
    # [2] Kernel (class method)
    #---------------------------------------------------------------------------
    def kernel(self):

        reg = self.reg
        con = self.con

        reg.dm_elec = con.dm_elec
        reg.mf_elec.mo_coeff = con.mf_elec.mo_coeff
        for i in range(len(reg.mol.nuc)):
            reg.dm_nuc[i] = con.dm_nuc[i]
            reg.mf_nuc[i].mo_coeff = con.mf_nuc[i].mo_coeff

        e_nocc = self.e_nocc
        e_tot = self.e_tot
        e_nvir = self.e_nvir
        nuc_num = len(con.mol.nuc)

        #-----------------------------------------------------------------------
        # [2.2] Convergence criteria
        #-----------------------------------------------------------------------
        t_conv_tol = getattr(self, 't_conv_tol', 1e-8)
        e_conv_tol = getattr(self, 'e_conv_tol', 1e-8)
        lagr_tol = getattr(self, 'lagr_tol', 1e-5)

        #-----------------------------------------------------------------------
        # [2.3] Construction of the noncanonical NEO Fock matrices (Eqs. 23/24)
        #-----------------------------------------------------------------------
        # The conventional NEO-HF Fock operator is evaluated with the CNEO-HF
        # orbitals/densities, giving the (non-diagonal) Fock matrices that
        # define the noncanonical MP2 iterations.
        ncH_e = reg.mf_elec.get_hcore(reg.mol.elec)
        ncV_e = reg.mf_elec.get_veff(reg.mol.elec, con.dm_elec)
        ncF_eAO = ncH_e + ncV_e
        ncF_eMO = con.mf_elec.mo_coeff.T @ ncF_eAO @ con.mf_elec.mo_coeff

        ncF_nMO = [None] * nuc_num
        for i in range(nuc_num):
            H_n = reg.mf_nuc[i].get_hcore(reg.mol.nuc[i])
            ncV_n = reg.mf_nuc[i].get_veff(reg.mol.nuc[i], con.dm_nuc[i])
            ncF_nAO = H_n + ncV_n
            ncF_nMO[i] = con.mf_nuc[i].mo_coeff.T @ ncF_nAO @ con.mf_nuc[i].mo_coeff

        #-----------------------------------------------------------------------
        # MO two-particle integrals (computed once per kernel() call)
        #-----------------------------------------------------------------------
        # Electron-electron (ov|ov) block.
        eri_ee_full = reg.mol.elec.intor('int2e', aosym='s8')
        co_e = con.mf_elec.mo_coeff[:, :e_nocc]
        cv_e = con.mf_elec.mo_coeff[:, e_nocc:e_tot]
        eri_ee = ao2mo.incore.general(eri_ee_full, (co_e, cv_e, co_e, cv_e),
                                      compact=False)
        del eri_ee_full
        TEIMO_t = eri_ee.reshape(e_nocc, e_nvir, e_nocc, e_nvir)
        TEIMO_l = TEIMO_t.T

        # Electron-nucleus (ia|IA) blocks; the lambda-ordered tensor is a
        # transpose view (l ordering: g(a,i,A,I) = g(i,a,I,A)).
        TPIMO_t = []
        TPIMO_l = []
        for j in range(nuc_num):
            O = int(self.num_ovt[j][0, 0])
            V = int(self.num_ovt[j][0, 1])
            eri_ep = neo.ao2mo.ep_ovov(con, con, j)
            tp = eri_ep.reshape(e_nocc, e_nvir, O, V)
            TPIMO_t.append(tp)
            TPIMO_l.append(_l_view(tp))

        # Nucleus-nucleus (IA|JB) blocks for distinct pairs.  (JB|IA) is the
        # exact mirror of (IA|JB), so only the lower triangle is transformed;
        # the diagonal blocks are never referenced (no t-amplitudes exist for
        # a nucleus with itself) and are left as zero tensors.
        TNIMO_t = numpy.empty((nuc_num, nuc_num), dtype=object)
        TNIMO_l = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(i + 1):
                Oi = int(self.num_ovt[i][0, 0])
                Vi = int(self.num_ovt[i][0, 1])
                Oj = int(self.num_ovt[j][0, 0])
                Vj = int(self.num_ovt[j][0, 1])
                if i == j:
                    TNIMO_t[i, j] = numpy.zeros((Oi, Vi, Oj, Vj))
                else:
                    eri_pp = neo.ao2mo.pp_ovov(con, con, i, j)
                    TNIMO_t[i, j] = eri_pp.reshape(Oi, Vi, Oj, Vj)
                    TNIMO_t[j, i] = _mirror(TNIMO_t[i, j])
                    TNIMO_l[j, i] = _l_view(TNIMO_t[j, i])
                TNIMO_l[i, j] = _l_view(TNIMO_t[i, j])

        #-----------------------------------------------------------------------
        # [2.4] Lagrangian constraint upon the MP2 nuclear densities
        #-----------------------------------------------------------------------
        # Position integrals of each nuclear basis (AO basis, as in the
        # original implementation).
        integrals_r = [con.mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3)
                       for i in range(nuc_num)]

        def Lagrangian_constraint(lagr_update, self, nuc_idx, t_nuclear,
                                  t_electronic_nuclear, position_ints):
            '''Given trial Lagrange multipliers for nucleus nuc_idx, relax the
            electronic-nuclear and nuclear t-amplitudes involving that nucleus
            and return the resulting correlated-density constraint residual
            (Eq. 22).  Used as the objective of the root finder.'''

            self.try_lagr[nuc_idx] = lagr_update

            cycles = 100
            for t_inner in range(cycles):

                # First amplitude update from the outer-loop amplitudes with
                # the trial multipliers.
                t_update_en = t_amps_en_only(
                    self, nuc_idx, lagr_update,
                    t_electronic_nuclear[nuc_idx], self.l_elecnuc[nuc_idx],
                    TPIMO_t[nuc_idx], ncF_eMO, ncF_nMO[nuc_idx],
                    integrals_r[nuc_idx])
                self.t_opt_en[nuc_idx] = t_update_en
                self.l_opt_en[nuc_idx] = _l_view(t_update_en)

                for nuc_2_idx in range(nuc_num):
                    t_update_n = t_amps_n_only(
                        self, nuc_idx, nuc_2_idx, lagr_update,
                        self.try_lagr[nuc_2_idx],
                        t_nuclear[nuc_idx][nuc_2_idx],
                        self.l_nuc[nuc_idx][nuc_2_idx],
                        TNIMO_t[nuc_idx][nuc_2_idx],
                        ncF_nMO[nuc_idx], ncF_nMO[nuc_2_idx],
                        integrals_r[nuc_idx], integrals_r[nuc_2_idx])
                    self.t_opt_n[nuc_idx][nuc_2_idx] = t_update_n
                    self.t_opt_n[nuc_2_idx][nuc_idx] = _mirror(t_update_n)
                    self.l_opt_n[nuc_idx][nuc_2_idx] = _l_view(t_update_n)
                    self.l_opt_n[nuc_2_idx][nuc_idx] = _l_view(self.t_opt_n[nuc_2_idx][nuc_idx])

                # Previous-iterate snapshot for the convergence tests.
                self.t_elecnuc_test[nuc_idx] = self.t_opt_en[nuc_idx]
                self.l_elecnuc_test[nuc_idx] = self.l_opt_en[nuc_idx]
                for nuc_2_idx in range(nuc_num):
                    self.t_nuc_test[nuc_idx][nuc_2_idx] = self.t_opt_n[nuc_idx][nuc_2_idx]
                    self.t_nuc_test[nuc_2_idx][nuc_idx] = self.t_opt_n[nuc_2_idx][nuc_idx]
                    self.l_nuc_test[nuc_idx][nuc_2_idx] = self.l_opt_n[nuc_idx][nuc_2_idx]
                    self.l_nuc_test[nuc_2_idx][nuc_idx] = self.l_opt_n[nuc_2_idx][nuc_idx]

                # Second amplitude update, from the refreshed amplitudes with
                # the trial multipliers.
                t_new_lambda_en = t_amps_en_only(
                    self, nuc_idx, self.try_lagr[nuc_idx],
                    self.t_opt_en[nuc_idx], self.l_opt_en[nuc_idx],
                    TPIMO_t[nuc_idx], ncF_eMO, ncF_nMO[nuc_idx],
                    integrals_r[nuc_idx])
                self.t_opt_en[nuc_idx] = t_new_lambda_en
                self.l_opt_en[nuc_idx] = _l_view(t_new_lambda_en)

                for nuc_2_idx in range(nuc_num):
                    t_new_lambda_n = t_amps_n_only(
                        self, nuc_idx, nuc_2_idx, self.try_lagr[nuc_idx],
                        self.lagr[nuc_2_idx],
                        self.t_opt_n[nuc_idx][nuc_2_idx],
                        self.l_opt_n[nuc_idx][nuc_2_idx],
                        TNIMO_t[nuc_idx][nuc_2_idx],
                        ncF_nMO[nuc_idx], ncF_nMO[nuc_2_idx],
                        integrals_r[nuc_idx], integrals_r[nuc_2_idx])
                    self.t_opt_n[nuc_idx][nuc_2_idx] = t_new_lambda_n
                    self.t_opt_n[nuc_2_idx][nuc_idx] = _mirror(t_new_lambda_n)
                    self.l_opt_n[nuc_idx][nuc_2_idx] = _l_view(t_new_lambda_n)
                    self.l_opt_n[nuc_2_idx][nuc_idx] = _l_view(self.t_opt_n[nuc_2_idx][nuc_idx])

                # Convergence tests on the energies and amplitude RMSDs.
                hyll_en_opt = Hylleraas_energy_en(self, self.t_opt_en, self.l_opt_en,
                                                  TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
                hyll_n_opt = Hylleraas_energy_n(self, self.t_opt_n, self.l_opt_n,
                                                TNIMO_t, TNIMO_l, ncF_nMO)
                hyll_en_prev = Hylleraas_energy_en(self, self.t_elecnuc_test,
                                                   self.l_elecnuc_test,
                                                   TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
                hyll_n_prev = Hylleraas_energy_n(self, self.t_nuc_test, self.l_nuc_test,
                                                 TNIMO_t, TNIMO_l, ncF_nMO)
                diff_E_opt_en = abs(hyll_en_opt - hyll_en_prev)
                diff_E_opt_n = abs(hyll_n_opt - hyll_n_prev)

                opt_RMSD_en = RMSD_en(self, self.t_opt_en, self.t_elecnuc_test)
                opt_RMSD_n = RMSD_n(self, self.t_opt_n, self.t_nuc_test)

                if self.verbose >= 4:
                    print('SUBCYCLE:', nuc_idx, 'ITERATION:', t_inner, '-------',
                          'E(MP2)_en:', hyll_en_opt, 'E(MP2)_n:', hyll_n_opt)
                    print('RMSD: ', ' en: ', opt_RMSD_en, ' n: ', opt_RMSD_n)
                    print('diff_E: ', ' en: ', diff_E_opt_en, ' n: ', diff_E_opt_n)
                    print(line)

                self.t_elecnuc_test = self.t_opt_en
                self.t_nuc_test = self.t_opt_n
                self.l_elecnuc_test = self.l_opt_en
                self.l_nuc_test = self.l_opt_n

                if (opt_RMSD_en < t_conv_tol) and (opt_RMSD_n < t_conv_tol) and \
                   (diff_E_opt_en < e_conv_tol) and (diff_E_opt_n < e_conv_tol):
                    if self.verbose >= 4:
                        print('SUCCESSFUL CONVERGENCE INNER LOOP')
                    self.mp2_density_converged[nuc_idx] = True
                    break
                elif t_inner > 98:
                    print('WARNING, NOT CONVERGED (inner Lagrange-multiplier loop)')

            density_matrix = mp2_density_one(self, nuc_idx, self.t_nuc_test,
                                             self.l_nuc_test, self.t_elecnuc_test,
                                             self.l_elecnuc_test)

            constraint = numpy.einsum('xij,ji->x', position_ints[nuc_idx],
                                      density_matrix)

            # (Preserved from the original implementation: accept the relaxed
            # amplitudes into the outer-loop storage when the constraint check
            # passes.)
            if (constraint.all() < 0.002):
                self.t_elecnuc = self.t_opt_en
                self.t_nuc = self.t_opt_n

            if self.verbose >= 4:
                print('Constraint: ', constraint)

            return constraint

        #-----------------------------------------------------------------------
        # [2.12] SCF optimization procedure
        #-----------------------------------------------------------------------
        max_iteration_cycles = 300
        constraint_opt = [None] * nuc_num
        hyll_e = hyll_en = hyll_n = 0.0

        for t in range(max_iteration_cycles):

            print(asterisk)
            print('THIS IS SCF CYCLE NO: ', t)

            # Refresh the lambda-amplitude views of the current t-amplitudes.
            self.l_elecnuc = [_l_view(x) for x in self.t_elecnuc]
            self.l_nuc = self._l_nuc_views(self.t_nuc)

            t_old_e = self.t_elec
            t_old_en = self.t_elecnuc
            t_old_n = self.t_nuc

            old_hyll_e = Hylleraas_energy_e(self, t_old_e, self.l_elec,
                                            TEIMO_t, TEIMO_l, ncF_eMO)
            old_hyll_en = Hylleraas_energy_en(self, t_old_en, self.l_elecnuc,
                                              TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
            old_hyll_n = Hylleraas_energy_n(self, t_old_n, self.l_nuc,
                                            TNIMO_t, TNIMO_l, ncF_nMO)

            # Amplitude updates (Eqs. 28-30).
            t_new_e = t_amps_e_only(self, self.t_elec, TEIMO_t, ncF_eMO)
            l_new_e = t_new_e.T

            t_new_en = [None] * nuc_num
            l_new_en = [None] * nuc_num
            for i in range(nuc_num):
                t_new_en[i] = t_amps_en_only(self, i, self.lagr[i],
                                             self.t_elecnuc[i], self.l_elecnuc[i],
                                             TPIMO_t[i], ncF_eMO, ncF_nMO[i],
                                             integrals_r[i])
                l_new_en[i] = _l_view(t_new_en[i])

            # Only the lower-triangle pair blocks are updated explicitly; the
            # (j,i) block is the mirror of the (i,j) block, as maintained (via
            # explicit swapped copies) by the original implementation.
            t_new_n = numpy.empty((nuc_num, nuc_num), dtype=object)
            for i in range(nuc_num):
                for j in range(i + 1):
                    t_new_n[i][j] = t_amps_n_only(self, i, j, self.lagr[i],
                                                  self.lagr[j],
                                                  self.t_nuc[i][j], self.l_nuc[i][j],
                                                  TNIMO_t[i][j],
                                                  ncF_nMO[i], ncF_nMO[j],
                                                  integrals_r[i], integrals_r[j])
                    if i != j:
                        t_new_n[j][i] = _mirror(t_new_n[i][j])
            l_new_n = self._l_nuc_views(t_new_n)

            #-------------------------------------------------------------------
            # [2.13] RMSD calculations
            #-------------------------------------------------------------------
            check_RMSD_e = RMSD_e(self, t_old_e, t_new_e)
            check_RMSD_en = RMSD_en(self, t_old_en, t_new_en)
            check_RMSD_n = RMSD_n(self, t_old_n, t_new_n)

            #-------------------------------------------------------------------
            # [2.14] Check all convergence criteria
            #-------------------------------------------------------------------
            self.t_elec = t_new_e
            self.t_elecnuc = t_new_en
            self.t_nuc = t_new_n
            self.l_elec = t_new_e.T
            self.l_elecnuc = l_new_en
            self.l_nuc = l_new_n

            self.t_opt_en = self.t_elecnuc
            self.t_opt_n = self.t_nuc
            self.l_opt_en = self.l_elecnuc
            self.l_opt_n = self.l_nuc

            hyll_e = Hylleraas_energy_e(self, t_new_e, l_new_e,
                                        TEIMO_t, TEIMO_l, ncF_eMO)
            hyll_en = Hylleraas_energy_en(self, t_new_en, l_new_en,
                                          TPIMO_t, TPIMO_l, ncF_eMO, ncF_nMO)
            hyll_n = Hylleraas_energy_n(self, t_new_n, l_new_n,
                                        TNIMO_t, TNIMO_l, ncF_nMO)

            diff_E_e = hyll_e - old_hyll_e
            diff_E_en = hyll_en - old_hyll_en
            diff_E_n = hyll_n - old_hyll_n

            print(asterisk)
            print('ITERATION: ', t, '-------', 'E(MP2)_e: ', hyll_e,
                  'E(MP2)_en: ', hyll_en, 'E(MP2)_n: ', hyll_n)
            print(line)
            print('RMSD: ', ' e :', check_RMSD_e, ' en: ', check_RMSD_en,
                  ' n: ', check_RMSD_n)
            print(line)
            print('diff_E: ', ' e: ', diff_E_e, ' en: ', diff_E_en,
                  ' n: ', diff_E_n)
            print(line)

            if (check_RMSD_e < t_conv_tol) and (check_RMSD_n < t_conv_tol) and \
               (check_RMSD_en < t_conv_tol) and (abs(diff_E_n) < e_conv_tol) and \
               (abs(diff_E_e) < e_conv_tol) and (abs(diff_E_en) < e_conv_tol):

                print('t-amplitudes and mp2 energies are converged!')

                for i in range(nuc_num):

                    constraint_opt[i] = scipy.optimize.root(
                        Lagrangian_constraint, self.try_lagr[i].flatten(),
                        args=(self, i, self.t_nuc, self.t_elecnuc, self.posn_ints),
                        method='lm', jac=False, tol=lagr_tol,
                        options={'col_deriv': True, 'ftol': lagr_tol,
                                 'xtol': lagr_tol, 'gtol': lagr_tol,
                                 'maxiter': 1000})
                    print(asterisk)
                    print(constraint_opt[i].status)
                    print(constraint_opt[i].message)
                    print(asterisk)

                    if (constraint_opt[i].status == 1) or (constraint_opt[i].status == 2):
                        self.lagr[i] = constraint_opt[i].x

                    opt_RMSD_en_mloop = RMSD_en(self, self.t_elecnuc, t_new_en)
                    opt_RMSD_n_mloop = RMSD_n(self, self.t_nuc, t_new_n)

                    hyll_en_opt = Hylleraas_energy_en(self, self.t_elecnuc,
                                                      self.l_elecnuc,
                                                      TPIMO_t, TPIMO_l,
                                                      ncF_eMO, ncF_nMO)
                    hyll_n_opt = Hylleraas_energy_n(self, self.t_nuc, self.l_nuc,
                                                    TNIMO_t, TNIMO_l, ncF_nMO)

                    diff_E_opt_en = hyll_en_opt - hyll_en
                    diff_E_opt_n = hyll_n_opt - hyll_n

                    if (constraint_opt[i].status == 1) or (constraint_opt[i].status == 2):
                        if (opt_RMSD_en_mloop < t_conv_tol) and \
                           (opt_RMSD_n_mloop < t_conv_tol) and \
                           (diff_E_opt_en < e_conv_tol) and \
                           (diff_E_opt_n < e_conv_tol):
                            print('SUCCESSFUL CONVERGENCE OUTER LOOP')
                            self.mp2_density_converged[i] = True
                            hyll_en = hyll_en_opt
                            hyll_n = hyll_n_opt
                    else:
                        self.mp2_density_converged[i] = False

                if all(self.mp2_density_converged):
                    for i in range(nuc_num):
                        print(constraint_opt[i])
                    break

            else:
                if t < (max_iteration_cycles - 1):
                    print('Not converged, continuing to next iteration...')
                    print(asterisk)
                else:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('WARNING! One or more sets of t-amplitudes has failed to converge!')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        #-----------------------------------------------------------------------
        # [2.16] Return variables and complete kernel function processes
        #-----------------------------------------------------------------------
        return self.base_energy, hyll_n, hyll_e, hyll_en, self.lagr
