# 存放 Kernel 相关的类
from kcg.common.Utils import EnumBackendType, calculate_file_hash
import functools

class KernelParam :
    def __init__(self,index,param):
        self.m_index = index
        self.m_param = param

class KernelInfo :
    def __init__(self):
        self.m_module = None
        self.m_function = None
        self.m_nRegs = 0
        self.m_nSpills = 0

class KernelLibFile :
    def __init__(self,filePath : str,backendType : EnumBackendType):
        self.m_file = filePath
        self.m_backendType = backendType
        self.m_kernelInfo = None
    
    def getInfo(self) -> KernelInfo:
        return self.m_kernelInfo
    
    def getType(self) -> EnumBackendType:
        return self.m_backendType
    
    def __hash__(self) -> str:
        return self.hash()
    
    @functools.lru_cache
    def hash(self)->str :
        return "lib_"+calculate_file_hash(self.m_file) + calculate_file_hash(self.m_backendType)

