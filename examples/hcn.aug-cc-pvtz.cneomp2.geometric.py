'''Geometry optimization of HCN on the CNEO-MP2 surface with geomeTRIC,
driven by the analytic nuclear gradients (unconstrained-amplitude variant).

Electronic basis aug-cc-pVTZ; all nuclei quantum (PB4-D for H, 12s12p12d
even-tempered for C/N).  Converges in a handful of gradient evaluations
(2 steps from a good starting structure) to

    r(H-C) = 1.0734 A,  r(C-N) = 1.1591 A,  E = -92.0870743 hartree

in agreement with the energy-only BFGS optimization of the constrained-
amplitude energy (examples/hcn.aug-cc-pvtz.cneomp2.constrained.BFGS.py:
1.0729 / 1.1586 A).
'''
import numpy
from pyscf import neo
from pyscf.neo import cneomp2_geomopt

BOHR = 0.52917721092

mol = neo.Mole()
mol.build(atom=[['H1', (0., 0., -1.073)], ['C2', (0., 0., 0.)],
                ['N3', (0., 0., 1.159)]],
          basis='aug-cc-pvtz', charge=0, quantum_nuc=[0, 1, 2],
          nuc_basis='pb4d')
mol.max_memory = 10000

mol_eq, engine = cneomp2_geomopt.optimize(mol)

c = mol_eq.atom_coords() * BOHR
print('converged in %d gradient evaluations' % engine.nsteps)
print('final E = %.10f hartree' % engine.history[-1][0])
print('r(H-C) = %.4f A' % numpy.linalg.norm(c[0] - c[1]))
print('r(C-N) = %.4f A' % numpy.linalg.norm(c[2] - c[1]))
