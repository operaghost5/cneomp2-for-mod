# Thin re-export: the vectorized implementation lives in cymods.cneomp2_kernels.
# This keeps the historical import path
#   from cymods.t_amps_e_only.t_amps_e_only import t_amps_e_only
# working without requiring Cython compilation.
from cymods.cneomp2_kernels import t_amps_e_only

__all__ = ['t_amps_e_only']
