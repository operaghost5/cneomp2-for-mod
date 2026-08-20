# Thin re-export: the vectorized implementations live in cymods.cneomp2_kernels.
from cymods.cneomp2_kernels import RMSD_e, RMSD_en, RMSD_n

__all__ = ['RMSD_e', 'RMSD_en', 'RMSD_n']
