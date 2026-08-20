import tracemalloc
import numpy
import pyscf
from pyscf import neo, ao2mo

#-------------------------------------------------------------------------------        
# [2.7] Nuclear t-amplitude function 
#-------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def t_amps_n_only(self, i_idx, j_idx, lagr_multipliers_i, lagr_multipliers_j, t_nuclear, l_nuclear, two_int_n_t, ncF_nMO_i, ncF_nMO_j, integrals_r_i, integrals_r_j):

    '''
    Note that the lagr_multipliers_i, lagr_multipliers_j, integrals,and ncF_nMO_i, ncF_nMO_j should each be indexed by the correct nucleus
    and the t_nuclear should be indexed by the pair of nuclei when being fed in to the function.
    '''

    print('calling t_amps_n_only from pymods...', flush=True)
    print(tracemalloc.get_traced_memory(), flush=True)
    # Initializing variables for nuclear t terms.
    nuclei = len(self.con.mol.nuc) 

    coulomb_term_n = 0
    sum_over_Ci_not_A_n = 0
    sum_over_Cj_not_B_n = 0
    sum_over_Ki_not_I_n = 0
    sum_over_Kj_not_J_n = 0
    sum_pieces_IJAB = 0



    virtual_gradient_term_ni = numpy.zeros((3,))
    occupied_gradient_term_ni = numpy.zeros((3,))
    virtual_gradient_term_nj = numpy.zeros((3,))
    occupied_gradient_term_nj = numpy.zeros((3,))
 

    T_old_n = t_nuclear 

    L_old_n = l_nuclear

    # This if statement is because there is no T amplitude for a single nucleus.
    if nuclei == 1:

        print('WARNING - No nuclear t-amplitudes are calculated when only one nucleus is treated quantum mechanically!')

    else:

        # Set all diagonal T tensors in the nuclear T amplitude storage matrix to zero.
        if (i_idx == j_idx):
            p_nocc_i = self.con.mf_nuc[i_idx].mo_coeff[:,self.con.mf_nuc[i_idx].mo_occ>0].shape[1]
            p_tot_i  = self.con.mf_nuc[i_idx].mo_coeff[0,:].shape[0]
            p_nvir_i = p_tot_i - p_nocc_i
            p_nocc_j = self.con.mf_nuc[j_idx].mo_coeff[:,self.con.mf_nuc[j_idx].mo_occ>0].shape[1]
            p_tot_j  = self.con.mf_nuc[j_idx].mo_coeff[0,:].shape[0]
            p_nvir_j = p_tot_j - p_nocc_j

            T_new_n = numpy.zeros((p_nocc_i, p_nvir_i, p_nocc_j, p_nvir_j))

        # Calculate the nuclear T amplitudes which are actually possible (the off diagonal T tensors in the storage matrix).
        else:

            # Define occupied and virtual by nuclei:
            p_nocc_i = self.con.mf_nuc[i_idx].mo_coeff[:,self.con.mf_nuc[i_idx].mo_occ>0].shape[1]
            p_tot_i  = self.con.mf_nuc[i_idx].mo_coeff[0,:].shape[0]
            p_nvir_i = p_tot_i - p_nocc_i
            p_nocc_j = self.con.mf_nuc[j_idx].mo_coeff[:,self.con.mf_nuc[j_idx].mo_occ>0].shape[1]
            p_tot_j  = self.con.mf_nuc[j_idx].mo_coeff[0,:].shape[0]
            p_nvir_j = p_tot_j - p_nocc_j

            # Defining the occupied and virtual orbital energies for each nuclei.
            moe_n_occupied_i = ncF_nMO_i[:p_nocc_i,:p_nocc_i]
            moe_n_occupied_j = ncF_nMO_j[:p_nocc_j,:p_nocc_j]
            moe_n_virtual_i = ncF_nMO_i[p_nocc_i:,p_nocc_i:]
            moe_n_virtual_j = ncF_nMO_j[p_nocc_j:,p_nocc_j:]

            # Initialization and building of the oribtal energy difference tensor.
            e_n = numpy.zeros((p_nocc_i, p_nvir_i, p_nocc_j, p_nvir_j))
            for I in range(p_nocc_i):
                for A in range(p_nvir_i):
                    for J in range(p_nocc_j):
                        for B in range(p_nvir_j):
                            e_n[I,A,J,B] = (moe_n_occupied_i[I,I] + moe_n_occupied_j[J,J] - moe_n_virtual_i[A,A] - moe_n_virtual_j[B,B])

            # Initializing T(abij) {abab} tensors to perform iterations with.
            T_new_n = numpy.zeros((p_nocc_i, p_nvir_i, p_nocc_j, p_nvir_j))

#  n nnn
#  nn   n   #------------------------------------------------
#  n    n   # Beginning of nuclear t-amplitude sub-cycle:
#  n    n   #------------------------------------------------ 
#  n    n
            # Two particle integral tensor:
            eri_pp = neo.ao2mo.pp_ovov(self.con, self.con, i_idx, j_idx)
            TNIMO = eri_pp.reshape(p_nocc_i, p_nvir_i, p_nocc_j, p_nvir_j)

            T_old_lambda_n_partial = numpy.swapaxes(T_old_n.T, 0,2)
            T_old_lambda_n = numpy.swapaxes(T_old_lambda_n_partial, 1,3)

            #-----------------------------------------------------------------------
            # Set/reset all elements of T tensor and RMSD inner summation to zero:
            #-----------------------------------------------------------------------
            T_new_n = numpy.zeros((p_nocc_i,p_nvir_i,p_nocc_j,p_nvir_j))

            print(tracemalloc.get_traced_memory(), flush=True)
            #----------------------------------------------------------------
            # Outer Summation (indexed by I,J occupied, A,B virtual)
            #----------------------------------------------------------------
            for I in range(p_nocc_i):
                for A in range(p_nvir_i):
                    for J in range(p_nocc_j):
                        for B in range (p_nvir_j):

                            #------------------------------------------------------------------------------
                            # Set/reset inner summation terms equal to zero to begin the iteration:
                            #------------------------------------------------------------------------------
                            sum_over_Ci_not_A_n = 0
                            sum_over_Cj_not_B_n = 0
                            sum_over_Ki_not_I_n = 0
                            sum_over_Kj_not_J_n = 0

                            virtual_gradient_term_ni = numpy.zeros((3,))
                            occupied_gradient_term_ni = numpy.zeros((3,))
                            virtual_gradient_term_nj = numpy.zeros((3,))
                            occupied_gradient_term_nj = numpy.zeros((3,))


                            #------------------------------------------------------------------------------
                            # Constant Term:
                            #------------------------------------------------------------------------------
                            coulomb_term_n = (two_int_n_t[I,A,J,B])

                            #------------------------------------------------------------------------------
                            # Inner Virtual Summations (indexed by Ci and Cj)
                            #------------------------------------------------------------------------------ 
                            for Ci in range(p_nvir_i):
                                if (Ci != A):
                                    sum_over_Ci_not_A_n += (ncF_nMO_i[A+p_nocc_i,Ci+p_nocc_i]*(T_old_n[I,Ci,J,B]))

                            for Cj in range(p_nvir_j):
                                if (Cj != B):
                                    sum_over_Cj_not_B_n += (ncF_nMO_j[B+p_nocc_j,Cj+p_nocc_j]*(T_old_n[I,A,J,Cj]))

                            for R in range(p_nvir_i):
                                for T in range(p_nvir_i):
                                    for x in range(3):
                                       virtual_gradient_term_ni[x] += (L_old_n[R,I,B,J] + T_old_n[I,T,J,B]) * integrals_r_i[x,R,T]
                            lagr_dot_gradient_vir_ni = numpy.dot(lagr_multipliers_i, virtual_gradient_term_ni)

                            for S in range(p_nvir_j):
                                for U in range(p_nvir_j):
                                    for x in range(3):
                                        virtual_gradient_term_nj[x] += (L_old_n[A,I,S,J] + T_old_n[I,A,J,U]) * integrals_r_j[x,S,U]
                            lagr_dot_gradient_vir_nj = numpy.dot(lagr_multipliers_j, virtual_gradient_term_nj)
 
                            #----------------------------------------------------------------------------------
                            # Inner Occupied Summations (indexed by Ki and Kj)
                            #----------------------------------------------------------------------------------
                            for Ki in range(p_nocc_i):
                                if (Ki != I):
                                    sum_over_Ki_not_I_n += (ncF_nMO_i[Ki,I]*(T_old_n[Ki,A,J,B]))

                            for Kj in range(p_nocc_j):
                                if (Kj != I):
                                    sum_over_Kj_not_J_n += (ncF_nMO_j[Kj,J]*(T_old_n[I,A,Kj,B]))


                            for P in range(p_nocc_i):
                                for Q in range(p_nocc_i):
                                    for x in range(3):
                                        occupied_gradient_term_ni[x] += (L_old_n[A,P,B,J] + T_old_n[Q,A,J,B]) * integrals_r_i[x,P,Q]
                            lagr_dot_gradient_occ_ni = numpy.dot(lagr_multipliers_i, occupied_gradient_term_ni)

                            for M in range(p_nocc_j):
                                for N in range(p_nocc_j):
                                    for x in range(3):
                                        occupied_gradient_term_nj[x] += (L_old_n[A,I,B,M] + T_old_n[I,A,N,B]) * integrals_r_j[x,M,N]
                            lagr_dot_gradient_occ_nj = numpy.dot(lagr_multipliers_j, occupied_gradient_term_nj)

                            total_dot_product_of_lm_and_grad = (lagr_dot_gradient_vir_ni + lagr_dot_gradient_vir_nj) - (lagr_dot_gradient_occ_ni + lagr_dot_gradient_occ_nj)

                            #------------------------------------------------------------------------------
                            # Build elements of T(abij){abab}:
                            #------------------------------------------------------------------------------
                            T_new_n[I,A,J,B] = (((coulomb_term_n + sum_over_Ci_not_A_n + sum_over_Cj_not_B_n - sum_over_Ki_not_I_n - sum_over_Kj_not_J_n + total_dot_product_of_lm_and_grad))/((e_n[I,A,J,B])))

    print(tracemalloc.get_traced_memory(), flush=True)
    return T_new_n

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

