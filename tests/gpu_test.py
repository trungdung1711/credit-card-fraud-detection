import numpy as np
import cudf
from sklearn.linear_model import LinearRegression
from cuml.linear_model import LinearRegression as cuLinearRegression
import time

n_samples, n_features = 10_000_000, 50
X_cpu = np.random.rand(n_samples, n_features)
y_cpu = np.random.rand(n_samples)

# CPU
t0 = time.time()
model_cpu = LinearRegression().fit(X_cpu, y_cpu)
print("CPU time:", time.time() - t0, "s")

# GPU
X_gpu = cudf.DataFrame(X_cpu)
y_gpu = cudf.Series(y_cpu)

t0 = time.time()
model_gpu = cuLinearRegression().fit(X_gpu, y_gpu)
print("GPU time:", time.time() - t0, "s")
