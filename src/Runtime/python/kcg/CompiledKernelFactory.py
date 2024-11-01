from kcg.CompiledKernel import *
from kcg.Operators.matmul import *

class EnumOperator(Enum):
    Matmul = 1
    Convolution = 2
    Poll = 3
    def __str__(self):
        return f'{self.name}'


class CompiledKernelFactory :
    @staticmethod
    def getKernel(kind : EnumOperator) -> CompiledKernel:
        if kind==EnumOperator.Matmul :
            signature = getMatmulSignature()
            return CompiledKernel(
                "/home/pangyunfei/xushilong/KernelCodeGen/src/Runtime/python/kcg/amd_triton_kernel-762981.hsaco",
                "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c",
                50000,
                signature
            )
        if kind==EnumOperator.Convolution :
            return None
        if kind==EnumOperator.Poll:
            return None