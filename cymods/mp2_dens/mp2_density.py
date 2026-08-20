# Thin re-export: the vectorized implementation lives in cymods.cneomp2_kernels.
from cymods.cneomp2_kernels import mp2_density_one

__all__ = ['mp2_density_one']
