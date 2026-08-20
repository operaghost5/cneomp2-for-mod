#!/usr/bin/env python
'''
Analytic nuclear gradients for CNEO-MP2.

Theory (see docs/cneomp2_gradient_theory.md for the full derivation)
--------------------------------------------------------------------
The CNEO-MP2 correlation energy is the multicomponent Hylleraas functional
J[t; f, g] of the paper (Eqs. 17-19), evaluated with the noncanonical NEO
Fock matrices f and the MO two-particle integrals g built from the CNEO-HF
orbitals.  J is stationary with respect to the amplitudes t (the residual
equations Eqs. 25-27 are exactly dJ/dt = 0), and for the constrained-
amplitude (CCD) variant the full Lagrangian is additionally stationary with
respect to the correlated-density multipliers mu^(2).  Therefore the total
derivative of the correlation energy requires no amplitude response:

    dE2/dx = sum_k D^k : df^k/dx  +  sum_classes Gamma : dg/dx
             (+ CCD: mu^(2) . gamma^(2) : d r~_MO/dx)

with the unrelaxed one-particle densities D^k = dJ/df^k (oo and vv blocks
only) and the two-particle densities Gamma = dJ/dg, both simple frozen-
amplitude contractions.

The total derivatives of the MO-basis Fock and ERI matrices decompose into
    (a) explicit AO-derivative integrals with frozen orbitals,
    (b) two-particle response of the Fock matrices through the first-order
        densities of every component, and
    (c) orbital-rotation terms driven by the coupled-perturbed CNEO-HF
        (CPHF) solution, including the response of the position-constraint
        Lagrange multipliers.
Terms (b) and (c) use the first-order orbitals mo1 from the same constrained
CPHF machinery as the CNEO Hessian (hessian.make_h1 + hessian.solve_mo1_rks),
i.e. this is a forward-response formulation: one CPHF solve per nuclear
displacement.  Occupied-occupied and virtual-virtual orbital rotations are
only needed through their symmetric (-S1/2) parts: their antisymmetric parts
leave J unchanged at amplitude stationarity because a unitary rotation of the
orbitals can be absorbed into a counter-rotation of the amplitudes.

The CNEO-HF part of the gradient is the existing neo.grad.Gradients.
'''

import numpy
from functools import reduce
from pyscf import gto, lib, neo
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.neo import cphf as neo_cphf
from pyscf.neo import hessian as neo_hessian

from cymods.cneomp2_kernels import (t_amps_e_only, t_amps_en_only,
                                    t_amps_n_only, RMSD_e, RMSD_en, RMSD_n,
                                    Hylleraas_energy_e, Hylleraas_energy_en,
                                    Hylleraas_energy_n)
from pyscf.neo.cneomp2 import _l_view, _mirror

einsum = lambda spec, *ops: numpy.einsum(spec, *ops, optimize=True)


#-------------------------------------------------------------------------------
# MO integral helpers
#-------------------------------------------------------------------------------
def _cross_eri_mo(mol1, mol2, c1_bra, c1_ket, c2_bra, c2_ket):
    '''(p q | P Q) cross-fragment MO Coulomb integrals with arbitrary
    coefficient blocks on each of the four indices (bare integrals, no charge
    factors).'''
    atm, bas, env = gto.conc_env(mol1._atm, mol1._bas, mol1._env,
                                 mol2._atm, mol2._bas, mol2._env)
    intor_name = 'int2e_sph'
    if getattr(mol1, 'cart', False):
        intor_name = 'int2e_cart'
    nbas1 = mol1._bas.shape[0]
    nbas2 = mol2._bas.shape[0]
    eri = gto.moleintor.getints(intor_name, atm, bas, env,
                                shls_slice=(0, nbas1, 0, nbas1,
                                            nbas1, nbas1 + nbas2,
                                            nbas1, nbas1 + nbas2),
                                aosym='s4')
    npair1 = mol1.nao_nr() * (mol1.nao_nr() + 1) // 2
    npair2 = mol2.nao_nr() * (mol2.nao_nr() + 1) // 2
    assert eri.shape == (npair1, npair2)
    # unpack fragment-2 pair (last axis), transform it, then fragment-1 pair
    eri2 = lib.unpack_tril(eri)                    # (npair1, nao2, nao2)
    half = einsum('xPQ,PA,QB->xAB', eri2, c2_bra, c2_ket)
    del eri2
    n2a, n2b = c2_bra.shape[1], c2_ket.shape[1]
    half = lib.unpack_tril(numpy.ascontiguousarray(
        half.reshape(npair1, n2a * n2b).T))        # (n2a*n2b, nao1, nao1)
    out = einsum('xpq,pa,qb->xab', half, c1_bra, c1_ket)
    n1a, n1b = c1_bra.shape[1], c1_ket.shape[1]
    return out.reshape(n2a, n2b, n1a, n1b).transpose(2, 3, 0, 1)


def _elec_eri_mo(mol, c_bra, c_ket, c_bra2, c_ket2, eri_s8=None):
    '''(p q | r s) same-fragment MO integrals with arbitrary blocks.'''
    if eri_s8 is None:
        eri_s8 = mol.intor('int2e', aosym='s8')
    from pyscf import ao2mo as pyscf_ao2mo
    out = pyscf_ao2mo.incore.general(eri_s8, (c_bra, c_ket, c_bra2, c_ket2),
                                     compact=False)
    return out.reshape(c_bra.shape[1], c_ket.shape[1],
                       c_bra2.shape[1], c_ket2.shape[1])


#-------------------------------------------------------------------------------
# Explicit AO-derivative ERI contractions
#-------------------------------------------------------------------------------
def _grad_eri_bra_side(mol_bra, mol_ket, gamma_half_sym, cket_bra, cket_ket,
                       de, atom_of_braAO, comp_offset=0, prefac=1.0):
    '''Contract Gamma with (nabla mu nu | kappa lambda) where the derivative
    acts on the bra fragment.  gamma_half_sym[mu, nu, K, L] must already be
    symmetrized over the bra AO pair (G[mu,nu] + G[nu,mu]) and carries the MO
    ket pattern (K, L).  The ket AO pair of the integral is transformed with
    (cket_bra, cket_ket).  Contributions are accumulated into de[atom, x]
    according to the atom that hosts each bra AO.

    Signs: pyscf int2e_ip1 returns -(nabla mu nu|kl); dE/dR = -sum over AOs
    on the displaced atom of that integral contracted with Gamma, hence the
    overall minus sign below (identical convention to grad.rhf).
    '''
    same_frag = mol_bra is mol_ket
    if same_frag:
        atm, bas, env = mol_bra._atm, mol_bra._bas, mol_bra._env
        nbas1 = mol_bra._bas.shape[0]
        ket0, ket1 = 0, nbas1
    else:
        atm, bas, env = gto.conc_env(mol_bra._atm, mol_bra._bas, mol_bra._env,
                                     mol_ket._atm, mol_ket._bas, mol_ket._env)
        nbas1 = mol_bra._bas.shape[0]
        ket0, ket1 = nbas1, nbas1 + mol_ket._bas.shape[0]
    intor_name = 'int2e_ip1_sph'
    if getattr(mol_bra, 'cart', False):
        intor_name = 'int2e_ip1_cart'

    nao_bra = mol_bra.nao_nr()
    ao_loc_bra = mol_bra.ao_loc_nr()
    # loop over bra shells in blocks to bound memory
    blk = max(1, int(2e8 / (3 * nao_bra * mol_ket.nao_nr()**2 * 8)))
    for sh0 in range(0, nbas1, blk):
        sh1 = min(sh0 + blk, nbas1)
        p0, p1 = ao_loc_bra[sh0], ao_loc_bra[sh1]
        ints = gto.moleintor.getints(intor_name, atm, bas, env,
                                     shls_slice=(sh0, sh1, 0, nbas1,
                                                 ket0, ket1, ket0, ket1),
                                     comp=3, aosym='s1')
        # ints: (3, p1-p0, nao_bra, naoK, naoK)
        ints_mo = einsum('xmnPQ,PK,QL->xmnKL', ints, cket_bra, cket_ket)
        del ints
        contrib = einsum('xmnKL,mnKL->xm', ints_mo,
                         gamma_half_sym[p0:p1])
        del ints_mo
        for m in range(p0, p1):
            de[atom_of_braAO[m]] -= prefac * contrib[:, m - p0]
    return de


#-------------------------------------------------------------------------------
# cNEOMP2 gradient class
#-------------------------------------------------------------------------------
class Gradients(lib.StreamObject):
    '''Analytic CNEO-MP2 nuclear gradients (response formulation).

    Args:
        mp2 : a cNEOMP2 object.  The CNEO-HF (con) calculation must be
            converged.  For unconstrained=True the (unconstrained) amplitudes
            are converged internally; for unconstrained=False the mp2 object
            must already hold converged constrained amplitudes and
            multipliers from mp2.kernel().

    Kwargs:
        unconstrained : bool
            True (default): gradient of the unconstrained-amplitude
            CNEO-MP2 energy (UCD variant).  False: constrained (CCD).
    '''
    def __init__(self, mp2, unconstrained=True):
        self.mp2 = mp2
        if not unconstrained:
            # The energy kernel's correlated-density constraint terms do not
            # correspond to the paper's Eqs. 22/26/27 (the implementation
            # contracts amplitude pairs with the leading corners of the
            # *atomic-orbital* position-integral matrix over independent
            # virtual index pairs, whereas Eq. 26 has a single sum over
            # molecular-orbital <C|r|I> elements), and the multiplier
            # root-finding targets yet another functional.  No consistent
            # Lagrangian exists to differentiate, so analytic constrained-
            # amplitude (CCD) gradients are not available.  Use the
            # unconstrained-amplitude (UCD) variant, whose optimized
            # geometries are nearly identical (see the CNEO-MP2 paper).
            raise NotImplementedError(
                'Analytic gradients are only available for the '
                'unconstrained-amplitude CNEO-MP2 variant '
                '(see source comment for why)')
        self.unconstrained = unconstrained
        self.verbose = getattr(mp2.con, 'verbose', 0)
        self.stdout = getattr(mp2.con.mol.super_mol
                              if hasattr(mp2.con.mol, 'super_mol') else
                              mp2.con.mol, 'stdout', None)
        self.t_conv_tol = 1e-10
        self.max_t_cycles = 500
        self.de = None
        self.e2 = None   # correlation energy consistent with the gradient

    #---------------------------------------------------------------------
    # amplitude solution at fixed multipliers
    #---------------------------------------------------------------------
    def _prepare(self):
        '''Build MO Fock matrices/integrals and converge the amplitudes at
        the multipliers appropriate for the selected variant.'''
        mp2 = self.mp2
        con = mp2.con
        reg = mp2.reg
        nuc_num = len(con.mol.nuc)

        # noncanonical NEO Fock matrices at the CNEO orbitals (Eqs. 23/24)
        ncH_e = reg.mf_elec.get_hcore(reg.mol.elec)
        ncV_e = reg.mf_elec.get_veff(reg.mol.elec, con.dm_elec)
        ncF_eAO = ncH_e + ncV_e
        c_e = con.mf_elec.mo_coeff
        ncF_eMO = c_e.T @ ncF_eAO @ c_e
        ncF_nMO = []
        for i in range(nuc_num):
            h_n = reg.mf_nuc[i].get_hcore(reg.mol.nuc[i])
            v_n = reg.mf_nuc[i].get_veff(reg.mol.nuc[i], con.dm_nuc[i])
            c_n = con.mf_nuc[i].mo_coeff
            ncF_nMO.append(c_n.T @ (h_n + v_n) @ c_n)

        # MO two-particle integrals (t-ordering)
        from pyscf import ao2mo as pyscf_ao2mo
        e_nocc, e_nvir, e_tot = mp2.e_nocc, mp2.e_nvir, mp2.e_tot
        co_e = c_e[:, :e_nocc]
        cv_e = c_e[:, e_nocc:e_tot]
        eri_s8 = reg.mol.elec.intor('int2e', aosym='s8')
        TEIMO_t = pyscf_ao2mo.incore.general(
            eri_s8, (co_e, cv_e, co_e, cv_e),
            compact=False).reshape(e_nocc, e_nvir, e_nocc, e_nvir)
        TPIMO_t = []
        for j in range(nuc_num):
            O = int(mp2.num_ovt[j][0, 0]); V = int(mp2.num_ovt[j][0, 1])
            TPIMO_t.append(neo.ao2mo.ep_ovov(con, con, j).reshape(
                e_nocc, e_nvir, O, V))
        TNIMO_t = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(i + 1):
                Oi = int(mp2.num_ovt[i][0, 0]); Vi = int(mp2.num_ovt[i][0, 1])
                Oj = int(mp2.num_ovt[j][0, 0]); Vj = int(mp2.num_ovt[j][0, 1])
                if i == j:
                    TNIMO_t[i, j] = numpy.zeros((Oi, Vi, Oj, Vj))
                else:
                    TNIMO_t[i, j] = neo.ao2mo.pp_ovov(con, con, i, j).reshape(
                        Oi, Vi, Oj, Vj)
                    TNIMO_t[j, i] = _mirror(TNIMO_t[i, j])

        integrals_r = [con.mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3)
                       for i in range(nuc_num)]

        if self.unconstrained:
            lagr = numpy.zeros((nuc_num, 3))
        else:
            # gradient of the constrained (CCD) energy: freeze the converged
            # multipliers from mp2.kernel() and re-converge the amplitudes at
            # exactly these multipliers, because kernel() leaves the amplitude
            # attributes at the root finder's last trial evaluation
            lagr = numpy.array(self.mp2.lagr, copy=True)
        if True:
            t_e = numpy.zeros_like(TEIMO_t)
            t_en = [numpy.zeros_like(x) for x in TPIMO_t]
            t_n = numpy.empty((nuc_num, nuc_num), dtype=object)
            for i in range(nuc_num):
                for j in range(nuc_num):
                    t_n[i][j] = numpy.zeros_like(TNIMO_t[i][j])
            for cyc in range(self.max_t_cycles):
                t_e_new = t_amps_e_only(mp2, t_e, TEIMO_t, ncF_eMO)
                rmsd = RMSD_e(mp2, t_e, t_e_new)
                t_e = t_e_new
                t_en_new = []
                for j in range(nuc_num):
                    t_en_new.append(t_amps_en_only(
                        mp2, j, lagr[j], t_en[j], _l_view(t_en[j]),
                        TPIMO_t[j], ncF_eMO, ncF_nMO[j], integrals_r[j]))
                rmsd += RMSD_en(mp2, t_en, t_en_new)
                t_en = t_en_new
                t_n_new = numpy.empty((nuc_num, nuc_num), dtype=object)
                for i in range(nuc_num):
                    for j in range(nuc_num):
                        t_n_new[i][j] = t_n[i][j]
                for i in range(nuc_num):
                    for j in range(i):
                        tij = t_amps_n_only(
                            mp2, i, j, lagr[i], lagr[j], t_n[i][j],
                            _l_view(t_n[i][j]), TNIMO_t[i][j],
                            ncF_nMO[i], ncF_nMO[j],
                            integrals_r[i], integrals_r[j])
                        t_n_new[i][j] = tij
                        t_n_new[j][i] = _mirror(tij)
                rmsd += RMSD_n(mp2, t_n, t_n_new)
                t_n = t_n_new
                if rmsd < self.t_conv_tol:
                    break
            else:
                logger.warn(self, 'cNEOMP2 amplitudes not fully converged '
                            'in gradient preparation: rmsd=%g', rmsd)
        l_e = _l_view(t_e)
        l_en = [_l_view(x) for x in t_en]
        l_n = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(nuc_num):
                l_n[i][j] = _l_view(t_n[i][j])

        # Hylleraas correlation energy consistent with these amplitudes
        TEIMO_l = TEIMO_t.T
        TPIMO_l = [_l_view(x) for x in TPIMO_t]
        TNIMO_l = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(nuc_num):
                TNIMO_l[i][j] = _l_view(TNIMO_t[i][j])
        e2 = (Hylleraas_energy_e(mp2, t_e, l_e, TEIMO_t, TEIMO_l, ncF_eMO)
              + Hylleraas_energy_en(mp2, t_en, l_en, TPIMO_t, TPIMO_l,
                                    ncF_eMO, ncF_nMO)
              + Hylleraas_energy_n(mp2, t_n, l_n, TNIMO_t, TNIMO_l, ncF_nMO))

        return dict(ncF_eMO=ncF_eMO, ncF_nMO=ncF_nMO, TEIMO_t=TEIMO_t,
                    TPIMO_t=TPIMO_t, TNIMO_t=TNIMO_t, t_e=t_e, t_en=t_en,
                    t_n=t_n, l_e=l_e, l_en=l_en, l_n=l_n, lagr=lagr,
                    integrals_r=integrals_r, eri_s8=eri_s8, e2=e2)

    #---------------------------------------------------------------------
    # unrelaxed densities D = dJ/df and two-particle densities Gamma = dJ/dg
    #---------------------------------------------------------------------
    def _densities(self, ing):
        mp2 = self.mp2
        nuc_num = len(mp2.con.mol.nuc)
        e_nocc, e_nvir = mp2.e_nocc, mp2.e_nvir
        t, l = ing['t_e'], ing['l_e']

        # ----- electronic 1PDM from J_e (Eq. 17 terms; factors 0.5 in front
        # of c_sum/k_sum are part of Hylleraas_energy_e)
        D_e_vv = 0.5 * 2.0 * (
            2.0 * einsum('aibj,icjb->ac', l, t)
            - einsum('aibj,ibjc->ac', l, t)
            - einsum('ajbi,icjb->ac', l, t)
            + 2.0 * einsum('ajbi,ibjc->ac', l, t))
        D_e_oo = -0.5 * 2.0 * (
            2.0 * einsum('akbj,iajb->ik', l, t)
            - einsum('akbj,ibja->ik', l, t)
            - einsum('ajbk,iajb->ik', l, t)
            + 2.0 * einsum('ajbk,ibja->ik', l, t))

        # ----- Gamma_ee in t-ordering: J_e g-terms reduce to
        # 0.25*(gt+gl) = sum_iajb g_iajb (4 t_iajb - 2 t_ibja)
        G_ee = 4.0 * t - 2.0 * t.transpose(0, 3, 2, 1)
        # numerical consistency check against the kernel expression
        g = ing['TEIMO_t']
        gt_check = Hylleraas_energy_e(mp2, t, l, g, g.T,
                                      numpy.zeros_like(ing['ncF_eMO']))
        assert abs(einsum('iajb,iajb->', G_ee, g) - gt_check) < 1e-8

        D_n_vv = []
        D_n_oo = []
        G_en = []
        for j in range(nuc_num):
            tj, lj = ing['t_en'][j], ing['l_en'][j]
            # e-n contributions to the electronic density
            D_e_vv += 2.0 * einsum('aiAI,icIA->ac', lj, tj)
            D_e_oo += -2.0 * einsum('akAI,iaIA->ik', lj, tj)
            # e-n contributions to the nuclear density
            D_n_vv.append(2.0 * einsum('aiAI,iaIC->AC', lj, tj))
            D_n_oo.append(-2.0 * einsum('aiAK,iaIA->IK', lj, tj))
            # Gamma_en: J_en g-terms are -(gt+gl) = sum g_iaIA (-4 t_iaIA)
            G_en.append(-4.0 * tj)

        G_nn = numpy.empty((nuc_num, nuc_num), dtype=object)
        for i in range(nuc_num):
            for j in range(i):
                tij, lij = ing['t_n'][i][j], ing['l_n'][i][j]
                D_n_vv[i] += einsum('AIBJ,ICJB->AC', lij, tij)
                D_n_vv[j] += einsum('AIBJ,IAJC->BC', lij, tij)
                D_n_oo[i] += -einsum('AKBJ,IAJB->IK', lij, tij)
                D_n_oo[j] += -einsum('AIBK,IAJB->JK', lij, tij)
                # J_n g-terms are +(gt+gl) = sum g_IAJB (2 t_IAJB)
                G_nn[i, j] = 2.0 * tij

        return D_e_oo, D_e_vv, D_n_oo, D_n_vv, G_ee, G_en, G_nn

    #---------------------------------------------------------------------
    # main driver
    #---------------------------------------------------------------------
    def kernel(self, atmlst=None):
        mp2 = self.mp2
        con = mp2.con
        mol = con.mol
        natm = mol.natm
        nuc_num = len(mol.nuc)
        if atmlst is None:
            atmlst = range(natm)

        log = logger.new_logger(self, self.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        # ---- CNEO-HF part
        g_hf = neo.grad.Gradients(con)
        g_hf.verbose = 0
        de = g_hf.kernel()
        log.timer('CNEO-HF gradient', *t0)

        # ---- ingredients
        ing = self._prepare()
        self.e2 = ing['e2']
        (D_e_oo, D_e_vv, D_n_oo, D_n_vv,
         G_ee, G_en, G_nn) = self._densities(ing)

        c_e = con.mf_elec.mo_coeff
        e_nocc, e_tot = mp2.e_nocc, mp2.e_tot
        co_e, cv_e = c_e[:, :e_nocc], c_e[:, e_nocc:e_tot]
        c_n = [con.mf_nuc[i].mo_coeff for i in range(nuc_num)]
        nocc_n = [int(mp2.num_ovt[i][0, 0]) for i in range(nuc_num)]
        co_n = [c_n[i][:, :nocc_n[i]] for i in range(nuc_num)]
        cv_n = [c_n[i][:, nocc_n[i]:] for i in range(nuc_num)]
        Z = [mol.atom_charge(mol.nuc[i].atom_index) for i in range(nuc_num)]

        # padded MO-basis densities and their AO representations
        nmo_e = e_tot
        D_e = numpy.zeros((nmo_e, nmo_e))
        D_e[:e_nocc, :e_nocc] = 0.5 * (D_e_oo + D_e_oo.T)
        D_e[e_nocc:, e_nocc:] = 0.5 * (D_e_vv + D_e_vv.T)
        D_e_ao = c_e @ D_e @ c_e.T
        D_n = []
        D_n_ao = []
        for i in range(nuc_num):
            nmo = c_n[i].shape[1]
            Dn = numpy.zeros((nmo, nmo))
            Dn[:nocc_n[i], :nocc_n[i]] = 0.5 * (D_n_oo[i] + D_n_oo[i].T)
            Dn[nocc_n[i]:, nocc_n[i]:] = 0.5 * (D_n_vv[i] + D_n_vv[i].T)
            D_n.append(Dn)
            D_n_ao.append(c_n[i] @ Dn @ c_n[i].T)

        # =====================================================================
        # 1) explicit Fock derivatives:  sum_k D^k : (dF^k_AO/dx)|frozen dm
        #    make_h1 provides exactly these matrices (the explicit derivative
        #    of the constraint term vanishes by translational invariance).
        # =====================================================================
        hessobj = neo_hessian.Hessian(con)
        h1ao_e, h1ao_n = neo_hessian.make_h1(hessobj, atmlst=atmlst)
        for i0, ia in enumerate(atmlst):
            de[i0] += einsum('xij,ij->x', h1ao_e[ia], D_e_ao)
            for j in range(nuc_num):
                de[i0] += einsum('xij,ij->x', h1ao_n[ia][j], D_n_ao[j])
        log.timer('cNEOMP2 explicit Fock derivative terms', *t0)

        # =====================================================================
        # 2) explicit ERI derivatives, contracted with Gamma
        # =====================================================================
        aoslices_e = mol.elec.aoslice_by_atom()
        atom_of_eAO = numpy.empty(mol.elec.nao_nr(), dtype=int)
        for ia in range(natm):
            p0, p1 = aoslices_e[ia, 2:]
            atom_of_eAO[p0:p1] = ia

        de_eri = numpy.zeros((natm, 3))
        # ---- ee class: derivative on bra pair; ket pair covered by the
        # particle-exchange symmetry of G_ee (factor 2)
        Gh = einsum('iajb,mi,na->mnjb', G_ee, co_e, cv_e)
        Gh = Gh + Gh.transpose(1, 0, 2, 3)
        _grad_eri_bra_side(mol.elec, mol.elec, Gh, co_e, cv_e,
                           de_eri, atom_of_eAO, prefac=2.0)
        del Gh
        # ---- en classes
        for j in range(nuc_num):
            ja = mol.nuc[j].atom_index
            atom_of_nAO = numpy.full(mol.nuc[j].nao_nr(), ja, dtype=int)
            # derivative on the electronic bra pair
            Gh = einsum('iaIA,mi,na->mnIA', G_en[j], co_e, cv_e) * Z[j]
            Gh = Gh + Gh.transpose(1, 0, 2, 3)
            _grad_eri_bra_side(mol.elec, mol.nuc[j], Gh, co_n[j], cv_n[j],
                               de_eri, atom_of_eAO)
            del Gh
            # derivative on the nuclear pair (nuclear mol as bra fragment)
            Gh = einsum('iaIA,MI,NA->MNia', G_en[j], co_n[j], cv_n[j]) * Z[j]
            Gh = Gh + Gh.transpose(1, 0, 2, 3)
            _grad_eri_bra_side(mol.nuc[j], mol.elec, Gh, co_e, cv_e,
                               de_eri, atom_of_nAO)
            del Gh
        # ---- nn classes
        for i in range(nuc_num):
            for j in range(i):
                zz = Z[i] * Z[j]
                iaat = mol.nuc[i].atom_index
                jaat = mol.nuc[j].atom_index
                atom_of_iAO = numpy.full(mol.nuc[i].nao_nr(), iaat, dtype=int)
                atom_of_jAO = numpy.full(mol.nuc[j].nao_nr(), jaat, dtype=int)
                Gh = einsum('IAJB,MI,NA->MNJB', G_nn[i, j],
                            co_n[i], cv_n[i]) * zz
                Gh = Gh + Gh.transpose(1, 0, 2, 3)
                _grad_eri_bra_side(mol.nuc[i], mol.nuc[j], Gh,
                                   co_n[j], cv_n[j], de_eri, atom_of_iAO)
                del Gh
                Gh = einsum('IAJB,MJ,NB->MNIA', G_nn[i, j],
                            co_n[j], cv_n[j]) * zz
                Gh = Gh + Gh.transpose(1, 0, 2, 3)
                _grad_eri_bra_side(mol.nuc[j], mol.nuc[i], Gh,
                                   co_n[i], cv_n[i], de_eri, atom_of_jAO)
                del Gh
        de += de_eri[list(atmlst)]
        log.timer('cNEOMP2 explicit ERI derivative terms', *t0)

        # =====================================================================
        # 3) rotation-coefficient matrices M^k (coefficient of U^k_tp)
        # =====================================================================
        M_e, M_n = self._rotation_matrices(ing, D_e, D_n, G_ee, G_en, G_nn)
        log.timer('cNEOMP2 rotation intermediates', *t0)

        # =====================================================================
        # 4) CPHF response: mo1 for every displacement, then assemble the
        #    rotation and Fock-response terms
        # =====================================================================
        mo1s_e, e1s_e, mo1s_n, f1s_n = neo_hessian.solve_mo1_rks(
            con, h1ao_e, h1ao_n, atmlst=list(atmlst),
            max_memory=4000, verbose=self.verbose)

        s_e = con.mf_elec.get_ovlp()
        s1a = -mol.elec.intor('int1e_ipovlp', comp=3)
        vresp = con.gen_response(hermi=1)
        s_n = [con.mf_nuc[i].get_ovlp() for i in range(nuc_num)]

        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices_e[ia, 2:]
            s1ao = numpy.zeros((3, s_e.shape[0], s_e.shape[0]))
            s1ao[:, p0:p1] += s1a[:, p0:p1]
            s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
            s1mo = einsum('pm,xmn,nq->xpq', c_e.T, s1ao, c_e)

            mo1e = mo1s_e[ia]                      # (3, nao, nocc), AO basis
            u_e = einsum('pm,mn,xnq->xpq', c_e.T, s_e, mo1e)  # (3,nmo,nocc)

            dm1e = einsum('xmi,ni->xmn', mo1e * 2.0, co_e)
            dm1e_symm = dm1e + dm1e.transpose(0, 2, 1)
            dm1n = []
            u_n = []
            for j in range(nuc_num):
                mo1nj = mo1s_n[ia][j]
                dm1n.append(einsum('xmi,ni->xmn', mo1nj, co_n[j]))
                u_n.append(einsum('pm,mn,xnq->xpq', c_n[j].T, s_n[j], mo1nj))

            # ---- Fock response through the first-order densities
            v1e, v1n = vresp(dm1e_symm, dm1e, dm1n)
            de[i0] += einsum('xij,ij->x', v1e, D_e_ao)
            for j in range(nuc_num):
                de[i0] += einsum('xij,ij->x', v1n[j], D_n_ao[j])

            # ---- orbital rotation terms
            for x in range(3):
                U = self._full_U(u_e[x], s1mo[x], e_nocc)
                de[i0, x] += einsum('tp,tp->', U, M_e)
                for j in range(nuc_num):
                    Un = self._full_U(u_n[j][x], None, nocc_n[j])
                    de[i0, x] += einsum('tp,tp->', Un, M_n[j])
        log.timer('cNEOMP2 response terms', *t0)

        self.de = de
        return de

    @staticmethod
    def _full_U(u_occ, s1mo, nocc):
        '''Complete the orbital-rotation matrix from its occupied columns.
        u_occ: (nmo, nocc) response (occupied columns, including the -S1/2
        oo choice made by the CPHF solver).  s1mo: full first-order MO
        overlap (None for nuclear components, whose S1 vanishes).'''
        nmo = u_occ.shape[0]
        U = numpy.zeros((nmo, nmo))
        U[:, :nocc] = u_occ
        if s1mo is None:
            U[:nocc, nocc:] = -u_occ[nocc:].T
        else:
            U[:nocc, nocc:] = -s1mo[:nocc, nocc:] - u_occ[nocc:].T
            U[nocc:, nocc:] = -0.5 * s1mo[nocc:, nocc:]
        return U

    #---------------------------------------------------------------------
    # rotation-coefficient matrices
    #---------------------------------------------------------------------
    def _rotation_matrices(self, ing, D_e, D_n, G_ee, G_en, G_nn):
        '''M^k[t,p] such that the orbital-rotation part of dE2 equals
        sum_k sum_tp U^k_tp M^k[t,p].'''
        mp2 = self.mp2
        con = mp2.con
        nuc_num = len(con.mol.nuc)
        e_nocc, e_tot = mp2.e_nocc, mp2.e_tot
        c_e = con.mf_elec.mo_coeff
        co_e, cv_e = c_e[:, :e_nocc], c_e[:, e_nocc:e_tot]
        c_n = [con.mf_nuc[i].mo_coeff for i in range(nuc_num)]
        nocc_n = [int(mp2.num_ovt[i][0, 0]) for i in range(nuc_num)]
        co_n = [c_n[i][:, :nocc_n[i]] for i in range(nuc_num)]
        cv_n = [c_n[i][:, nocc_n[i]:] for i in range(nuc_num)]
        Z = [con.mol.atom_charge(con.mol.nuc[i].atom_index)
             for i in range(nuc_num)]

        # ---- Fock-matrix rotation terms: sum_pq D_pq d(f_pq)
        f_e = ing['ncF_eMO']
        M_e = einsum('pq,tq->tp', D_e, f_e) + einsum('pq,pt->tq', D_e, f_e)
        M_n = []
        for j in range(nuc_num):
            f_n = ing['ncF_nMO'][j]
            M_n.append(einsum('pq,tq->tp', D_n[j], f_n)
                       + einsum('pq,pt->tq', D_n[j], f_n))

        # ---- ERI rotation terms
        # ee class: g with one full index; pair symmetry gives factor 2
        g_Pajb = _elec_eri_mo(con.mol.elec, c_e, cv_e, co_e, cv_e,
                              eri_s8=ing['eri_s8'])
        M_e[:, :e_nocc] += 2.0 * einsum('iajb,Pajb->Pi', G_ee, g_Pajb)
        del g_Pajb
        g_iPjb = _elec_eri_mo(con.mol.elec, co_e, c_e, co_e, cv_e,
                              eri_s8=ing['eri_s8'])
        M_e[:, e_nocc:] += 2.0 * einsum('iajb,iPjb->Pa', G_ee, g_iPjb)
        del g_iPjb

        for j in range(nuc_num):
            Gj = G_en[j]
            g1 = _cross_eri_mo(con.mol.elec, con.mol.nuc[j],
                               c_e, cv_e, co_n[j], cv_n[j]) * Z[j]
            M_e[:, :e_nocc] += einsum('iaIA,PaIA->Pi', Gj, g1)
            del g1
            g2 = _cross_eri_mo(con.mol.elec, con.mol.nuc[j],
                               co_e, c_e, co_n[j], cv_n[j]) * Z[j]
            M_e[:, e_nocc:] += einsum('iaIA,iPIA->Pa', Gj, g2)
            del g2
            g3 = _cross_eri_mo(con.mol.elec, con.mol.nuc[j],
                               co_e, cv_e, c_n[j], cv_n[j]) * Z[j]
            M_n[j][:, :nocc_n[j]] += einsum('iaIA,iaPA->PI', Gj, g3)
            del g3
            g4 = _cross_eri_mo(con.mol.elec, con.mol.nuc[j],
                               co_e, cv_e, co_n[j], c_n[j]) * Z[j]
            M_n[j][:, nocc_n[j]:] += einsum('iaIA,iaIP->PA', Gj, g4)
            del g4

        for i in range(nuc_num):
            for j in range(i):
                Gij = G_nn[i, j]
                zz = Z[i] * Z[j]
                g1 = _cross_eri_mo(con.mol.nuc[i], con.mol.nuc[j],
                                   c_n[i], cv_n[i], co_n[j], cv_n[j]) * zz
                M_n[i][:, :nocc_n[i]] += einsum('IAJB,PAJB->PI', Gij, g1)
                del g1
                g2 = _cross_eri_mo(con.mol.nuc[i], con.mol.nuc[j],
                                   co_n[i], c_n[i], co_n[j], cv_n[j]) * zz
                M_n[i][:, nocc_n[i]:] += einsum('IAJB,IPJB->PA', Gij, g2)
                del g2
                g3 = _cross_eri_mo(con.mol.nuc[i], con.mol.nuc[j],
                                   co_n[i], cv_n[i], c_n[j], cv_n[j]) * zz
                M_n[j][:, :nocc_n[j]] += einsum('IAJB,IAPB->PJ', Gij, g3)
                del g3
                g4 = _cross_eri_mo(con.mol.nuc[i], con.mol.nuc[j],
                                   co_n[i], cv_n[i], co_n[j], c_n[j]) * zz
                M_n[j][:, nocc_n[j]:] += einsum('IAJB,IAJP->PB', Gij, g4)
                del g4

        # ---- CCD variant: rotation of the correlated-density constraint
        if not self.unconstrained:
            from cymods.cneomp2_kernels import mp2_density_one
            for j in range(nuc_num):
                gamma = mp2_density_one(mp2, j, ing['t_n'],
                                        ing['l_n'], ing['t_en'], ing['l_en'])
                r_shift = mp2.posn_ints[j]
                r_mo = numpy.einsum('pm,xmn,nq->xpq', c_n[j].T, r_shift,
                                    c_n[j], optimize=True)
                mu_r = numpy.tensordot(ing['lagr'][j], r_mo, axes=(0, 0))
                M_n[j] += (einsum('pq,tq->tp', gamma, mu_r)
                           + einsum('pq,pt->tq', gamma, mu_r))

        return M_e, M_n

    grad = lib.alias(kernel, alias_name='grad')
