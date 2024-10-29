'''
common functions, some basic type definitions in kcg
'''

import hashlib
from enum import Enum


class EnumBackendType(Enum):
    CUDA = 1
    HIP = 2
    UNKNOWN = 3



def calculate_file_hash(file_path ,algorithm='md5',hash_len=8) -> str:
    # 以二进制只读模式打开文件
    ret = ""
    with open(file_path, 'rb') as file:
        # 选择哈希算法
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError("Unsupported algorithm. Please choose from 'md5', 'sha1', or 'sha256'.")

        # 逐块更新哈希值
        for chunk in iter(lambda: file.read(4096), b''):
            hasher.update(chunk)

        # 返回计算得到的哈希值
        ret = str(hasher.hexdigest())
        return ret[:hash_len]

# # 要计算哈希值的文件路径
# file_path = '/home/pangyunfei/xushilong/KernelCodeGen/src/Runtime/python/kcg/common/__init__.py'

# # 计算文件的 SHA256 哈希值
# file_hash = calculate_file_hash(file_path, 'sha256')
# print(f"The SHA256 hash of the file is: {file_hash}")


