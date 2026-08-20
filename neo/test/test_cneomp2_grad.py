#!/usr/bin/env python
'''Finite-difference tests of the analytic CNEO-MP2 gradient
(unconstrained-amplitude variant).'''

import unittest
import numpy
from pyscf import neo
from pyscf.neo.cneomp2 import cNEOMP2
from pyscf.neo import cneomp2_grad

BOHR = 0.52917721092


def energy_and_grad(atoms_bohr, charge, qnuc, ebasis, do_grad=True):
    mol = neo.Mole()
    mol.build(atom=[[s, numpy.array(c) * BOHR] for s, c in atoms_bohr],
              basis=ebasis, charge=charge, quantum_nuc=qnuc,
              nuc_basis='pb4d')
    reg = neo.HF(mol)
    con = neo.cdft.CDFT(mol)
    reg.verbose = con.verbose = 0
    con.conv_tol = 1e-12
    e_chf = con.scf()
    assert con.converged
    mp2 = cNEOMP2(reg, con, e_chf)
    g = cneomp2_grad.Gradients(mp2, unconstrained=True)
    if do_grad:
        de = g.kernel()
        return e_chf + g.e2, de
    return e_chf + g._prepare()['e2'], None


def fd_grad(atoms_bohr, charge, qnuc, ebasis, h=4e-4):
    natm = len(atoms_bohr)
    de = numpy.zeros((natm, 3))
    for ia in range(natm):
        for x in range(3):
            cp = [[s, list(c)] for s, c in atoms_bohr]
            cm = [[s, list(c)] for s, c in atoms_bohr]
            cp[ia][1][x] += h
            cm[ia][1][x] -= h
            ep, _ = energy_and_grad(cp, charge, qnuc, ebasis, do_grad=False)
            em, _ = energy_and_grad(cm, charge, qnuc, ebasis, do_grad=False)
            de[ia, x] = (ep - em) / (2 * h)
    return de


class KnownValues(unittest.TestCase):
    def test_h2_all_quantum(self):
        atoms = [['H1', (0., 0., .70)], ['H2', (0., 0., -.70)]]
        e, de = energy_and_grad(atoms, 0, [0, 1], 'sto-3g')
        de_fd = fd_grad(atoms, 0, [0, 1], 'sto-3g')
        self.assertLess(abs(de - de_fd).max(), 5e-7)

    def test_hf_one_quantum(self):
        atoms = [['F1', (0., 0., 0.)], ['H2', (0., 0., 1.75)]]
        e, de = energy_and_grad(atoms, 0, [1], 'sto-3g')
        de_fd = fd_grad(atoms, 0, [1], 'sto-3g')
        self.assertLess(abs(de - de_fd).max(), 5e-7)

    def test_hehp_unequal_charges(self):
        atoms = [['He1', (0., 0., 0.)], ['H2', (0., 0., 1.45)]]
        e, de = energy_and_grad(atoms, 1, [0, 1], 'ccpvdz')
        de_fd = fd_grad(atoms, 1, [0, 1], 'ccpvdz')
        self.assertLess(abs(de - de_fd).max(), 5e-7)

    def test_ccd_gradient_raises(self):
        mol = neo.Mole()
        mol.build(atom='H1 0 0 0.37; H2 0 0 -0.37', basis='sto-3g',
                  quantum_nuc=[0, 1], nuc_basis='pb4d')
        reg = neo.HF(mol)
        con = neo.cdft.CDFT(mol)
        reg.verbose = con.verbose = 0
        e = con.scf()
        mp2 = cNEOMP2(reg, con, e)
        with self.assertRaises(NotImplementedError):
            cneomp2_grad.Gradients(mp2, unconstrained=False)


if __name__ == '__main__':
    unittest.main()
