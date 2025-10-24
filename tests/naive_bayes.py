import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import pandas as pd

np.random.seed(42)

NUM0 = 5000
NUM1 = 5000
bins = 50

# given class 0
x0 = np.random.normal(2, 1, NUM0)

# given class 1
x1 = np.random.normal(6, 1.5, NUM1)


X = np.concatenate([x0, x1]).reshape(-1, 1)
y = np.array([0] * NUM0 + [1] * NUM1)

df = pd.DataFrame({"X": np.concatenate([x0, x1]), "class": y})

model = GaussianNB()
model.fit(X, y)

print("Class prior probabilities P(C):", model.class_prior_)
print("Means (μ):", model.theta_.ravel())
print("Variance: ", model.var_.ravel())

x_values = np.linspace(-1, 10, 300)
mu0, mu1 = model.theta_.ravel()

# Compute Gaussian densities

# Plot the class-conditional densities
plt.figure(figsize=(8, 4))

sns.histplot(
    data=df, x="X", hue="class", bins=bins, kde=True, palette=["skyblue", "pink"]
)

# sns.histplot(x=x0, bins=30, kde=True, color="skyblue", label="Class 0")
# sns.histplot(x=x1, bins=30, kde=True, color="pink", label="Class 1")

plt.title("Histogram of values given the class (0 or 1)")
plt.xlabel("Feature x")
plt.ylabel("Density")
plt.show()
