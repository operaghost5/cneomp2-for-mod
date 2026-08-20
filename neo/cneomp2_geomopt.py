#!/usr/bin/env python
'''
Geometry optimization of CNEO-MP2 energies with the geomeTRIC optimizer.

Provides a geomeTRIC custom engine driven by the analytic CNEO-MP2 gradients
(pyscf.neo.cneomp2_grad, unconstrained-amplitude variant), plus an optional
finite-difference fallback for cross-checking.

Example::

    from pyscf import neo
    from pyscf.neo import cneomp2_geomopt
    mol = neo.Mole()
    mol.build(atom='H 0 0 0; C 0 0 1.07; N 0 0 2.23', basis='aug-cc-pvtz',
              quantum_nuc=[0, 1, 2], nuc_basis='pb4d')
    mol_eq, engine = cneomp2_geomopt.optimize(mol)
    print(mol_eq.atom_coords())
'''

import os
import tempfile
import numpy
from pyscf import neo
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.neo.cneomp2 import cNEOMP2
from pyscf.neo import cneomp2_grad

try:
    import geometric
    import geometric.molecule
    import geometric.engine
    _GeomEngineBase = geometric.engine.Engine
except ImportError:
    geometric = None
    _GeomEngineBase = object

BOHR = nist.BOHR  # Angstrom per Bohr


def _pure_symbols(mol):
    return [mol.atom_pure_symbol(i) for i in range(mol.natm)]


class CNEOMP2Engine(_GeomEngineBase):
    '''geomeTRIC engine evaluating CNEO-MP2 energies and analytic gradients.

    Args:
        mol : neo.Mole
    Kwargs:
        fd : bool
            Use central finite-difference gradients of the same energy
            instead of the analytic ones (validation fallback; 6N energy
            evaluations per step).
        fd_step : float
            Finite-difference step in Bohr.
        conv_tol : float
            CNEO-HF SCF convergence tolerance.
    '''
    def __init__(self, mol, fd=False, fd_step=5e-4, conv_tol=1e-11,
                 verbose=0):
        if geometric is None:
            raise ImportError('geomeTRIC is required: pip install geometric')
        molecule = geometric.molecule.Molecule()
        molecule.elem = _pure_symbols(mol)
        molecule.xyzs = [mol.atom_coords() * BOHR]  # Angstrom
        super().__init__(molecule)
        self.mol = mol
        self.fd = fd
        self.fd_step = fd_step
        self.conv_tol = conv_tol
        self.verbose = verbose
        self.dm_elec0 = None    # SCF restart data between steps
        self.nsteps = 0
        self.history = []       # (energy, coords_bohr) per evaluation

    def _energy_gradient(self, coords_bohr, do_grad=True):
        mol = self.mol.set_geom_(coords_bohr, unit='Bohr', inplace=False)
        reg = neo.HF(mol)
        con = neo.cdft.CDFT(mol)
        reg.verbose = self.verbose
        con.verbose = self.verbose
        con.conv_tol = self.conv_tol
        e_chf = con.scf()
        if not con.converged:
            raise RuntimeError('CNEO-HF did not converge during geometry '
                               'optimization')
        mp2 = cNEOMP2(reg, con, e_chf)
        g = cneomp2_grad.Gradients(mp2, unconstrained=True)
        if do_grad:
            de = g.kernel()
            e2 = g.e2
            return e_chf + e2, de
        ing = g._prepare()
        return e_chf + ing['e2'], None

    def calc_new(self, coords, dirname):
        coords_bohr = numpy.asarray(coords).reshape(-1, 3)
        if self.fd:
            e0, _ = self._energy_gradient(coords_bohr, do_grad=False)
            de = numpy.zeros_like(coords_bohr)
            h = self.fd_step
            for ia in range(coords_bohr.shape[0]):
                for x in range(3):
                    cp = coords_bohr.copy(); cp[ia, x] += h
                    cm = coords_bohr.copy(); cm[ia, x] -= h
                    ep, _ = self._energy_gradient(cp, do_grad=False)
                    em, _ = self._energy_gradient(cm, do_grad=False)
                    de[ia, x] = (ep - em) / (2 * h)
        else:
            e0, de = self._energy_gradient(coords_bohr)
        self.nsteps += 1
        self.history.append((e0, coords_bohr.copy()))
        logger.info(self.mol, 'CNEO-MP2 geomeTRIC step %d  E = %.10f',
                    self.nsteps, e0)
        return {'energy': e0, 'gradient': numpy.asarray(de).ravel()}


def optimize(mol, fd=False, maxsteps=100, convergence_set='GAU',
             verbose=0, **engine_kwargs):
    '''Optimize the geometry of a neo.Mole on the CNEO-MP2
    (unconstrained-amplitude) surface with geomeTRIC.

    Returns:
        (mol_eq, engine): the optimized neo.Mole and the engine (whose
        .history holds every (energy, coords_bohr) evaluated).
    '''
    if geometric is None:
        raise ImportError('geomeTRIC is required: pip install geometric')
    engine = CNEOMP2Engine(mol, fd=fd, verbose=verbose, **engine_kwargs)
    tmpdir = tempfile.mkdtemp(prefix='cneomp2_geomopt')
    inp = os.path.join(tmpdir, 'cneomp2.txt')
    with open(inp, 'w') as f:
        f.write('\n')
    m = geometric.optimize.run_optimizer(
        customengine=engine, input=inp, maxiter=maxsteps,
        convergence_set=convergence_set, qccnv=False)
    coords_eq = m.xyzs[-1] / BOHR
    mol_eq = mol.set_geom_(coords_eq, unit='Bohr', inplace=False)
    return mol_eq, engine
