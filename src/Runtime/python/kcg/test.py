# from Runtime.python.kcg.Loader import driver
# if __name__ == "__main__":
#     print("hello")
#     print(driver)
#     print(driver.loader)

from typing import List,Type
from kcg.Utils import *
from kcg.Kernel import *
from kcg.CompiledKernelFactory import *
from kcg.Operators import matmul

hsacoPath='/home/pangyunfei/xushilong/KernelCodeGen/src/Runtime/python/kcg/amd_triton_kernel-762981.hsaco'
funName = 'matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c'

o = CompiledKernelFactory.getKernel(EnumOperator.Matmul)
o.run()