"""CNEO-MP2 geometry optimization of HCN.

Electronic basis: aug-cc-pVTZ.  Nuclear basis: PB4-D (hydrogen); heavier
quantum nuclei automatically receive the 12s12p12d even-tempered basis
(alpha = 2*sqrt(2)*m, beta = sqrt(3)) exactly as in the CNEO-MP2 paper.
All nuclei are treated quantum mechanically.

The molecule is linear, so the geometry is parametrized by the two bond
lengths (r_HC, r_CN) with C fixed at the origin and the molecule on the
z axis; BFGS minimizes the CNEO-MP2 total energy over these two variables.
"""
import os
import sys
import time

import numpy
import scipy.optimize

from pyscf import neo
import pyscf
print('pyscf:', pyscf.__version__, pyscf.__file__, flush=True)

from pyscf.neo.cneomp2 import cNEOMP2

EBASIS = 'aug-cc-pvtz'
NBASIS = 'pb4d'
MAXMEM = 10000

neval = [0]


def energy(x):
    r_hc, r_cn = float(x[0]), float(x[1])
    neval[0] += 1
    t0 = time.time()
    mol = neo.Mole()
    mol.build(atom=[['H1', (0.0, 0.0, -r_hc)],
                    ['C2', (0.0, 0.0, 0.0)],
                    ['N3', (0.0, 0.0, r_cn)]],
              basis=EBASIS, charge=0, quantum_nuc=[0, 1, 2],
              nuc_basis=NBASIS)
    mol.max_memory = MAXMEM

    reg = neo.HF(mol)
    con = neo.cdft.CDFT(mol)
    reg.verbose = 0
    con.verbose = 0

    e_chf = con.scf()
    t1 = time.time()

    # silence the per-cycle output of the MP2 driver
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mp2 = cNEOMP2(reg, con, e_chf)
        base, hyll_n, hyll_e, hyll_en, lagr = mp2.kernel()
    log = buf.getvalue()
    conv = log.count('SUCCESSFUL CONVERGENCE OUTER LOOP')
    t2 = time.time()

    e_tot = e_chf + hyll_e + hyll_en + hyll_n
    print('EVAL %3d  r_HC=%.6f  r_CN=%.6f  E(cHF)=%.10f  '
          'Ee=%.8f Een=%.8f En=%.8f  E(tot)=%.10f  scf_conv=%s outer_conv=%d  '
          '[scf %.0fs, mp2 %.0fs]'
          % (neval[0], r_hc, r_cn, e_chf, hyll_e, hyll_en, hyll_n, e_tot,
             getattr(con, 'converged', '?'), conv, t1 - t0, t2 - t1),
          flush=True)
    return e_tot


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'opt'
    x0 = numpy.array([1.077, 1.160])   # start near the expected minimum
    if mode == 'single':
        energy(x0)
    else:
        # eps and gtol chosen to sit above the ~1e-7 Ha noise floor of the
        # converged CNEO-MP2 energies (SCF 1e-9, t-amplitudes 1e-8, LM 1e-5):
        # finite-difference gradient noise ~ 1e-7/1e-3 = 1e-4 Ha/Angstrom.
        res = scipy.optimize.minimize(
            energy, x0, method='BFGS', jac=None,
            options={'gtol': 2e-4, 'eps': 1e-3, 'maxiter': 30, 'disp': True})
        print(res, flush=True)
        print('FINAL r_HC = %.4f Angstrom, r_CN = %.4f Angstrom'
              % (res.x[0], res.x[1]), flush=True)
