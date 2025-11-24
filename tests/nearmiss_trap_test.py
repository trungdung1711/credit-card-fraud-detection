import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=2,
    weights=[0.9, 0.1],  # Imbalanced
    class_sep=0.3,  # Low separation -> overlapping classes
    random_state=17112004,
)

# Apply NearMiss undersampling
nm = NearMiss(version=1, n_neighbors=3)
X_res, y_res = nm.fit_resample(X, y)

# Plot before and after NearMiss
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original dataset
axes[0].scatter(
    X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.3, label="Majority (Normal)", c="tab:blue"
)
axes[0].scatter(
    X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.8, label="Minority (Fraud)", c="tab:red"
)
axes[0].set_title("Before NearMiss (Overlapping classes)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# After NearMiss
axes[1].scatter(
    X_res[y_res == 0][:, 0],
    X_res[y_res == 0][:, 1],
    alpha=0.6,
    label="Majority (Kept)",
    c="tab:blue",
)
axes[1].scatter(
    X_res[y_res == 1][:, 0],
    X_res[y_res == 1][:, 1],
    alpha=0.9,
    label="Minority",
    c="tab:red",
)
axes[1].set_title("After NearMiss (Boundary points kept)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
