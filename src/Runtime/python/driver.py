# 根据当前环境，创建Hip 或 Cuda的Loader，用于加载二进制文件

import abc
import hashlib
import os
import tempfile
from pathlib import Path

import build
from build import build_cmd
# from .cache import get_cache_manager


def get_default_cache_path() ->str :
    print("[D] use default cache path. to be modified")
    return "/home/pangyunfei/xushilong/KernelCodeGen/src/Runtime"
    

class DriverBase(metaclass=abc.ABCMeta):
    CUDA = 0
    HIP = 1

    @staticmethod
    def third_party_dir():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")

    def __init__(self) -> None:
        pass


# -----------------------------
# CUDA
# -----------------------------


class CudaUtils(object):
    m_cache_path = None
    # singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "Loader", "cuda.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        # cache = get_cache_manager(key)
        fname = "cuda_utils.so"
        
        # cache_path = cache.get_file(fname)
        if self.m_cache_path is None:
            # with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(get_default_cache_path(), "__main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = build_cmd("cuda_utils", src_path, self.m_cache_path)
            self.m_cache_path = so

            # with open(so, "rb") as f:
            #     cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("cuda_utils", self.m_cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.CUtensorMapDataType = mod.CUtensorMapDataType
        self.CUtensorMapInterleave = mod.CUtensorMapInterleave
        self.CUtensorMapSwizzle = mod.CUtensorMapSwizzle
        self.CUtensorMapL2promotion = mod.CUtensorMapL2promotion
        self.CUtensorMapFloatOOBfill = mod.CUtensorMapFloatOOBfill
        self.cuTensorMapEncodeTiled = mod.cuTensorMapEncodeTiled
        self.cuMemAlloc = mod.cuMemAlloc
        self.cuMemcpyHtoD = mod.cuMemcpyHtoD
        self.cuMemFree = mod.cuMemFree


class CudaDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = CudaUtils()
        self.backend = self.CUDA


# -----------------------------
# HIP
# -----------------------------


class HIPUtils(object):
    m_cache_path = None
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "Loader", "hip.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        # cache = get_cache_manager(key)
        fname = "hip_utils.so"
        # cache_path = cache.get_file(fname)
        if self.m_cache_path is None:
            # with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(get_default_cache_path(), "__main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = build_cmd("hip_utils", src_path, self.m_cache_path)
            self.m_cache_path = so
            # with open(so, "rb") as f:
            #     cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("hip_utils", self.m_cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class HIPDriver(DriverBase):
    # singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = HIPUtils()
        self.backend = self.HIP


class UnsupportedDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(UnsupportedDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = None
        self.backend = None


# -----------------------------
# Driver
# -----------------------------


class LazyProxy:

    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._obj = None

    def _initialize_obj(self):
        if self._obj is None:
            self._obj = self._init_fn()

    def __getattr__(self, name):
        self._initialize_obj()
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            self._initialize_obj()
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        self._initialize_obj()
        delattr(self._obj, name)

    def __repr__(self):
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self):
        self._initialize_obj()
        return str(self._obj)


def initialize_driver():
    import torch

    if torch.version.hip is not None:
        return HIPDriver()
    elif torch.cuda.is_available():
        return CudaDriver()
    else:
        return UnsupportedDriver()


driver = LazyProxy(initialize_driver)

