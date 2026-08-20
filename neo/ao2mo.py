#!/usr/bin/env python
# Author: Kurt Brorsen (brorsenk@missouri.edu)

import pyscf.gto
import numpy
import pyscf.ao2mo as ao2mo
from pyscf import gto
from pyscf.ao2mo import _ao2mo
from timeit import default_timer as timer


def _cross_ovov(mol1, mol2, mo1, mo2, nocc1, nocc2):
    '''(o1 v1 | o2 v2) MO Coulomb integrals between two fragments.

    Only the cross AO block (mol1 mol1 | mol2 mol2) is computed, with 4-fold
    permutational symmetry, instead of the full 2-electron integral tensor of
    the combined molecule.  For fragments with n1 and n2 AOs this reduces the
    integral storage from (n1+n2)^4/8 to n1^2 n2^2/4 doubles and skips the
    (11|11), (22|22), and (11|12)-type integral blocks entirely, which is
    both much faster and the dominant memory saving of the CNEO-MP2 setup
    stage.

    Returns a 2D array of shape (nocc1*nvir1, nocc2*nvir2), the same layout
    that ao2mo.incore.general produced in the original implementation.
    '''
    atm, bas, env = gto.conc_env(mol1._atm, mol1._bas, mol1._env,
                                 mol2._atm, mol2._bas, mol2._env)
    intor_name = 'int2e_sph'
    if getattr(mol1, 'cart', False):
        intor_name = 'int2e_cart'
    nbas1 = mol1._bas.shape[0]
    nbas2 = mol2._bas.shape[0]
    # Packed cross block: shape (n1*(n1+1)/2, n2*(n2+1)/2)
    eri = gto.moleintor.getints(intor_name, atm, bas, env,
                                shls_slice=(0, nbas1, 0, nbas1,
                                            nbas1, nbas1 + nbas2,
                                            nbas1, nbas1 + nbas2),
                                aosym='s4')

    mo1 = numpy.asarray(mo1, order='F')
    mo2 = numpy.asarray(mo2, order='F')
    tot1 = mo1.shape[1]
    tot2 = mo2.shape[1]

    # Half-transform the (packed) fragment-2 pair index: -> (npair1, o2*v2)
    half = _ao2mo.nr_e2(eri, mo2, (0, nocc2, nocc2, tot2), 's4', 's1')
    del eri
    # Transform the (packed) fragment-1 pair index: -> (o2*v2, o1*v1)
    half = numpy.ascontiguousarray(half.T)
    out = _ao2mo.nr_e2(half, mo1, (0, nocc1, nocc1, tot1), 's4', 's1')
    del half
    return numpy.ascontiguousarray(out.T)

def ep_setup(mf, mf2,  i=0, j=0, ep=True):

    if(ep==True):

        mol_tot = mf.mol.elec + mf.mol.nuc[i] 
        tot1  = mf.mf_elec.mo_coeff[0,:].shape[0]
        tot2  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]

    else:

        mol_tot = mf.mol.nuc[i] + mf.mol.nuc[j] 
        tot1  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
        tot2  = mf.mf_nuc[j].mo_coeff[0,:].shape[0]

    eri = mol_tot.intor('int2e',aosym='s8')

    mo_coeff_tot = numpy.zeros((tot1+tot2,tot1+tot2))

    if(ep==True):

        mo_coeff_tot[:tot1,:tot1] = mf2.mf_elec.mo_coeff
        mo_coeff_tot[tot1:,tot1:] = mf2.mf_nuc[i].mo_coeff

    else:

        mo_coeff_tot[:tot1,:tot1] = mf2.mf_nuc[i].mo_coeff
        mo_coeff_tot[tot1:,tot1:] = mf2.mf_nuc[j].mo_coeff

    return eri, mo_coeff_tot


def pp_setup(mf, mf2, i=0, j=0, pp=True):

    if(pp==True):

        mol_tot_p = mf.mol.nuc[i] + mf.mol.nuc[j]
        ptot1 = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
        ptot2 = mf.mf_nuc[j].mo_coeff[0,:].shape[0]
   
    else:

        pass   
         
    eri = mol_tot_p.intor('int2e',aosym='s8')

    mo_coeff_tot_p = numpy.zeros((ptot1+ptot2,ptot1+ptot2))

    if(pp==True):

        mo_coeff_tot_p[:ptot1,:ptot1] = mf2.mf_nuc[i].mo_coeff
        mo_coeff_tot_p[ptot1:,ptot1:] = mf2.mf_nuc[j].mo_coeff   

    else:

        pass

    return eri, mo_coeff_tot_p

#end1

def ep_full(mf, mf2, i=0):

    print('calling ep_full')
    eri, mo_coeff_tot = ep_setup(mf,mf2,i)

    e_tot  = mf.mf_elec.mo_coeff[0,:].shape[0]
    p_tot  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]

#    c_e= mo_coeff_tot[:,:e_tot]
#    c_n= mo_coeff_tot[:,e_tot:]

    eri_ep = ao2mo.incore.full(eri, mo_coeff_tot,compact=False)
    charge_i_ep =  mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)
    scaled_eri_ep = eri_ep*(charge_i_ep)

    return scaled_eri_ep

#start2

def pp_full(mf, mf2, i=0, j=0):

   eri, mo_coeff_tot_p = pp_setup(mf, mf2, i, j)

#   p_tot_i = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
#   p_tot_j = mf.mf_nuc[j].mo_coeff[0,:].shape[0]

   eri_pp = ao2mo.incore.full(eri, mo_coeff_tot_p, compact=False)
  
   charge_i_pp = mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)
   charge_j_pp = mf.mol.nuc[j].super_mol.atom_charge(mf.mol.nuc[j].atom_index)

   scaled_eri_pp = eri_pp*(charge_i_pp*charge_j_pp)
 
   return scaled_eri_pp

#end2

def ep_ovov(mf, mf2, i=0):
    '''(ia|IA) electron-nucleus MO integrals, scaled by the nuclear charge.

    Only the cross (elec elec|nuc nuc) AO block is computed and transformed
    (see _cross_ovov); the result is identical to transforming the full
    combined-molecule integrals as done previously, at a fraction of the
    memory and time.'''

    e_nocc = mf.mf_elec.mo_coeff[:,mf.mf_elec.mo_occ>0].shape[1]
    p_nocc = mf.mf_nuc[i].mo_coeff[:,mf.mf_nuc[i].mo_occ>0].shape[1]

    charge_i_ep =  mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)

    eri_ep = _cross_ovov(mf.mol.elec, mf.mol.nuc[i],
                         mf2.mf_elec.mo_coeff, mf2.mf_nuc[i].mo_coeff,
                         e_nocc, p_nocc)

    eri_ep *= charge_i_ep
    return eri_ep


def pp_ovov(mf, mf2, i=0, j=0):
    '''(IA|JB) nucleus-nucleus MO integrals, scaled by the nuclear charges.

    Only the cross (nuc_i nuc_i|nuc_j nuc_j) AO block is computed and
    transformed (see _cross_ovov).'''

    p_nocc_i = mf.mf_nuc[i].mo_coeff[:,mf.mf_nuc[i].mo_occ>0].shape[1]
    p_nocc_j = mf.mf_nuc[j].mo_coeff[:,mf.mf_nuc[j].mo_occ>0].shape[1]

    charge_i_pp = mf.mol.nuc[i].super_mol.atom_charge(mf.mol.nuc[i].atom_index)
    charge_j_pp = mf.mol.nuc[j].super_mol.atom_charge(mf.mol.nuc[j].atom_index)

    eri_pp = _cross_ovov(mf.mol.nuc[i], mf.mol.nuc[j],
                         mf2.mf_nuc[i].mo_coeff, mf2.mf_nuc[j].mo_coeff,
                         p_nocc_i, p_nocc_j)

    eri_pp *= (charge_i_pp*charge_j_pp)
    return eri_pp



