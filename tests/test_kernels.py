"""Regression test: vectorized CNEO-MP2 kernels (cymods.cneomp2_kernels)
against the straightforward loop implementations kept in pymods/.

Run from the repository root:

    python tests/test_kernels.py

Only numpy is strictly required for the vectorized kernels; pymods imports
pyscf at module level, so pyscf must be importable to run the comparison.
"""
import io
import os
import sys
import contextlib

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cymods import cneomp2_kernels as K

from pymods.t_amps_e_only.t_amps_e_only import t_amps_e_only as ref_t_e
from pymods.t_amps_en_only.t_amps_en_only import t_amps_en_only as ref_t_en
from pymods.t_amps_n_only import t_amps_n_only as _ref_t_n_mod
from pymods.t_amps_n_only.t_amps_n_only import t_amps_n_only as ref_t_n
# NOTE: the pymods copy of mp2_density_one predates the production (cymods)
# version and has a different accumulation structure, so the loop reference
# for the density is transcribed below (ref_dens) from the original
# cymods/mp2_dens/mp2_density.pyx instead.
from pymods.hylleraas_e.hylleraas_e import Hylleraas_energy_e as ref_h_e
from pymods.hylleraas_en.hylleraas_en import Hylleraas_energy_en as ref_h_en
from pymods.hylleraas_n.hylleraas_n import Hylleraas_energy_n as ref_h_n
from pymods.rmsd.rmsd import RMSD_e as ref_r_e, RMSD_en as ref_r_en, RMSD_n as ref_r_n

TOL = 1e-10


def ref_dens(self, i_idx, T_n, L_n, T_en, L_en):
    '''Loop reference for the nuclear MP2 single-particle density,
    transcribed verbatim from the original cymods/mp2_dens/mp2_density.pyx.'''
    nuclei = len(self.con.mol.nuc)
    e_nocc = self.e_nocc
    e_nvir = self.e_nvir
    p_nocc_i = int(self.num_ovt[i_idx][0, 0])
    p_nvir_i = int(self.num_ovt[i_idx][0, 1])
    p_tot_i = int(self.num_ovt[i_idx][0, 2])

    gamma_vir = numpy.zeros((p_nvir_i, p_nvir_i))
    gamma_occ = numpy.zeros((p_nocc_i, p_nocc_i))

    if nuclei == 1:
        for A in range(p_nvir_i):
            for B in range(p_nvir_i):
                for I in range(p_nocc_i):
                    for ie in range(e_nocc):
                        for ae in range(e_nvir):
                            gamma_vir[A, B] += L_en[i_idx][ae, ie, A, I] * T_en[i_idx][ie, ae, I, B]
        for I in range(p_nocc_i):
            for J in range(p_nocc_i):
                for A in range(p_nvir_i):
                    for ie in range(e_nocc):
                        for ae in range(e_nvir):
                            gamma_occ[I, J] -= L_en[i_idx][ae, ie, A, J] * T_en[i_idx][ie, ae, I, A]
    else:
        for j_idx in range(nuclei):
            if i_idx == j_idx:
                continue
            p_nocc_j = int(self.num_ovt[j_idx][0, 0])
            p_nvir_j = int(self.num_ovt[j_idx][0, 1])
            for A in range(p_nvir_i):
                for B in range(p_nvir_i):
                    s = 0.0
                    en = 0.0
                    for I in range(p_nocc_i):
                        for J in range(p_nocc_j):
                            for C in range(p_nvir_j):
                                s += L_n[i_idx][j_idx][A, I, C, J] * T_n[i_idx][j_idx][I, B, J, C]
                            for ie in range(e_nocc):
                                for ae in range(e_nvir):
                                    en += L_en[i_idx][ae, ie, A, I] * T_en[i_idx][ie, ae, I, B]
                    gamma_vir[A, B] += s + en
            for I in range(p_nocc_i):
                for J in range(p_nocc_i):
                    s = 0.0
                    en = 0.0
                    for A in range(p_nvir_i):
                        for B in range(p_nvir_j):
                            for KK in range(p_nocc_j):
                                s -= L_n[i_idx][j_idx][A, J, B, KK] * T_n[i_idx][j_idx][I, A, KK, B]
                            for ie in range(e_nocc):
                                for ae in range(e_nvir):
                                    en -= L_en[i_idx][ae, ie, A, J] * T_en[i_idx][ie, ae, I, A]
                    gamma_occ[I, J] += s + en

    gamma_total = numpy.zeros((p_tot_i, p_tot_i))
    gamma_total[:p_nocc_i, :p_nocc_i] = gamma_occ
    gamma_total[p_nocc_i:, p_nocc_i:] = gamma_vir
    return gamma_total


# ---------------------------------------------------------------------------
# Minimal stand-in for the cNEOMP2 instance the kernels reach into.
# ---------------------------------------------------------------------------
class _MF:
    def __init__(self, nocc, tot):
        self.mo_coeff = numpy.zeros((tot, tot))
        self.mo_occ = numpy.zeros(tot)
        self.mo_occ[:nocc] = 1.0


class _Mol:
    def __init__(self, n):
        self.nuc = [object()] * n


class _Con:
    def __init__(self, nuc_dims):
        self.mol = _Mol(len(nuc_dims))
        self.mf_nuc = [_MF(o, t) for (o, t) in nuc_dims]


class Stub:
    def __init__(self, e_nocc, e_tot, nuc_dims):
        self.e_nocc = e_nocc
        self.e_tot = e_tot
        self.e_nvir = e_tot - e_nocc
        self.con = _Con(nuc_dims)
        self.reg = self.con
        self.num_ovt = []
        for (o, t) in nuc_dims:
            a = numpy.zeros((1, 3))
            a[0, 0], a[0, 1], a[0, 2] = o, t - o, t
            self.num_ovt.append(a)


def build_case(e_nocc, e_tot, nuc_dims, seed=0):
    rng = numpy.random.default_rng(seed)
    st = Stub(e_nocc, e_tot, nuc_dims)
    o, v = st.e_nocc, st.e_nvir
    n = len(nuc_dims)
    O = [d[0] for d in nuc_dims]
    V = [d[1] - d[0] for d in nuc_dims]
    T = [d[1] for d in nuc_dims]
    R = lambda *s: rng.standard_normal(s)

    c = {'self': st, 'O': O, 'V': V}

    F = R(e_tot, e_tot)
    F = 0.5 * (F + F.T)
    F[numpy.diag_indices(e_tot)] = numpy.arange(e_tot) * 3.0 - 10.0
    c['ncF_eMO'] = F

    c['ncF_nMO'] = []
    c['integrals_r'] = []
    for j in range(n):
        Fn = R(T[j], T[j])
        Fn = 0.5 * (Fn + Fn.T)
        Fn[numpy.diag_indices(T[j])] = numpy.arange(T[j]) * 2.0 - 5.0
        c['ncF_nMO'].append(Fn)
        r = R(3, T[j], T[j])
        c['integrals_r'].append(0.5 * (r + numpy.swapaxes(r, 1, 2)))

    c['lagr'] = R(n, 3)

    c['t_elec'] = R(o, v, o, v)
    c['l_elec'] = R(v, o, v, o)
    c['t_elec_2'] = R(o, v, o, v)
    c['TEIMO_t'] = R(o, v, o, v)
    c['TEIMO_l'] = R(v, o, v, o)

    c['t_en'] = [R(o, v, O[j], V[j]) for j in range(n)]
    c['l_en'] = [R(v, o, V[j], O[j]) for j in range(n)]
    c['t_en_2'] = [R(o, v, O[j], V[j]) for j in range(n)]
    c['TPIMO_t'] = [R(o, v, O[j], V[j]) for j in range(n)]
    c['TPIMO_l'] = [R(v, o, V[j], O[j]) for j in range(n)]

    def objmat(fn):
        m = numpy.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                m[i][j] = fn(i, j)
        return m

    c['t_n'] = objmat(lambda i, j: R(O[i], V[i], O[j], V[j]))
    c['l_n'] = objmat(lambda i, j: R(V[i], O[i], V[j], O[j]))
    c['t_n_2'] = objmat(lambda i, j: R(O[i], V[i], O[j], V[j]))
    c['TNIMO_t'] = objmat(lambda i, j: R(O[i], V[i], O[j], V[j]))
    c['TNIMO_l'] = objmat(lambda i, j: R(V[i], O[i], V[j], O[j]))
    return c


def check(name, a, b, worst):
    a = numpy.asarray(a, dtype=float)
    b = numpy.asarray(b, dtype=float)
    scale = max(numpy.max(numpy.abs(a)), 1.0)
    dev = numpy.max(numpy.abs(a - b)) / scale
    status = 'ok' if dev < TOL else 'FAIL'
    print('  %-22s scaled max deviation = %.3e  [%s]' % (name, dev, status))
    return max(worst, dev)


def main():
    # The pymods reference t_amps_n_only recomputes (and discards) an integral
    # transformation through pyscf.neo.ao2mo; stub it out so the test needs no
    # SCF objects.
    def _pp_stub(mf, mf2, i=0, j=0):
        def ov(m):
            occ = m.mo_coeff[:, m.mo_occ > 0].shape[1]
            tot = m.mo_coeff[0, :].shape[0]
            return occ, tot - occ
        Oi, Vi = ov(mf.mf_nuc[i])
        Oj, Vj = ov(mf.mf_nuc[j])
        return numpy.zeros(Oi * Vi * Oj * Vj)
    _ref_t_n_mod.neo.ao2mo.pp_ovov = _pp_stub

    worst = 0.0
    for (e_nocc, e_tot, nuc_dims, seed) in [
            (3, 12, [(1, 8), (1, 7)], 1),
            (4, 18, [(1, 10), (1, 9), (1, 8)], 2)]:
        print('case: e_nocc=%d e_tot=%d nuc_dims=%s' % (e_nocc, e_tot, nuc_dims))
        c = build_case(e_nocc, e_tot, nuc_dims, seed)
        s = c['self']
        n = len(nuc_dims)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pairs = []
            pairs.append(('t_amps_e_only',
                          ref_t_e(s, c['t_elec'], c['TEIMO_t'], c['ncF_eMO']),
                          K.t_amps_e_only(s, c['t_elec'], c['TEIMO_t'], c['ncF_eMO'])))
            for j in range(n):
                args = (s, j, c['lagr'][j], c['t_en'][j], c['l_en'][j],
                        c['TPIMO_t'][j], c['ncF_eMO'], c['ncF_nMO'][j],
                        c['integrals_r'][j])
                pairs.append(('t_amps_en_only[%d]' % j,
                              ref_t_en(*args), K.t_amps_en_only(*args)))
            for i in range(n):
                for j in range(n):
                    args = (s, i, j, c['lagr'][i], c['lagr'][j],
                            c['t_n'][i][j], c['l_n'][i][j], c['TNIMO_t'][i][j],
                            c['ncF_nMO'][i], c['ncF_nMO'][j],
                            c['integrals_r'][i], c['integrals_r'][j])
                    pairs.append(('t_amps_n_only[%d,%d]' % (i, j),
                                  ref_t_n(*args), K.t_amps_n_only(*args)))
            for i in range(n):
                args = (s, i, c['t_n'], c['l_n'], c['t_en'], c['l_en'])
                pairs.append(('mp2_density_one[%d]' % i,
                              ref_dens(*args), K.mp2_density_one(*args)))
            args = (s, c['t_elec'], c['l_elec'], c['TEIMO_t'], c['TEIMO_l'],
                    c['ncF_eMO'])
            pairs.append(('Hylleraas_energy_e', ref_h_e(*args), K.Hylleraas_energy_e(*args)))
            args = (s, c['t_en'], c['l_en'], c['TPIMO_t'], c['TPIMO_l'],
                    c['ncF_eMO'], c['ncF_nMO'])
            pairs.append(('Hylleraas_energy_en', ref_h_en(*args), K.Hylleraas_energy_en(*args)))
            args = (s, c['t_n'], c['l_n'], c['TNIMO_t'], c['TNIMO_l'], c['ncF_nMO'])
            pairs.append(('Hylleraas_energy_n', ref_h_n(*args), K.Hylleraas_energy_n(*args)))
            pairs.append(('RMSD_e', ref_r_e(s, c['t_elec'], c['t_elec_2']),
                          K.RMSD_e(s, c['t_elec'], c['t_elec_2'])))
            pairs.append(('RMSD_en', ref_r_en(s, c['t_en'], c['t_en_2']),
                          K.RMSD_en(s, c['t_en'], c['t_en_2'])))
            pairs.append(('RMSD_n', ref_r_n(s, c['t_n'], c['t_n_2']),
                          K.RMSD_n(s, c['t_n'], c['t_n_2'])))

        for (name, a, b) in pairs:
            worst = check(name, a, b, worst)

    print('worst scaled deviation: %.3e' % worst)
    assert worst < TOL, 'kernel regression FAILED'
    print('all kernels agree with the reference loop implementations')


if __name__ == '__main__':
    main()
