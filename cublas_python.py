# -*- coding: utf-8 -*-
"""Cublas_Python.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_Rnz1D3KcWqPT1vrDjHHU5Fwbqj1MFn3
"""

#!nvcc --version

import numpy as np
import cupy as cp
import time

N = 10000  # Matrix size

# CPU (NumPy)
A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

start_cpu = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
end_cpu = time.time()

# GPU (cuBLAS with cupy)
A_gpu = cp.array(A_cpu)
B_gpu = cp.array(B_cpu)

start_gpu = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Device(0).synchronize()  # Ensure GPU computation is finished
end_gpu = time.time()

print(f"NumPy (CPU) time: {end_cpu - start_cpu:.5f} sec")
print(f"cuBLAS (GPU) time: {end_gpu - start_gpu:.5f} sec")