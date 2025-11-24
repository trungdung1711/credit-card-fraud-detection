from imblearn.under_sampling import ClusterCentroids
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 1️⃣ Create an imbalanced 2D dataset
X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.9, 0.1],  # 900/100
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_samples=1000,
    random_state=17112004,
)

# 2️⃣ Apply Cluster Centroids undersampling
# Example: Keep 200 centroids for the majority class
cc = ClusterCentroids(sampling_strategy={0: 500, 1: sum(y == 1)}, random_state=42)
# cc = ClusterCentroids(sampling_strategy="majority", random_state=42)

X_res, y_res = cc.fit_resample(X, y)

# 3️⃣ Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: Before undersampling
axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Majority", alpha=0.4)
axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Minority", alpha=0.8)
axes[0].set_title("Before ClusterCentroids")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# --- Right: After undersampling
axes[1].scatter(
    X_res[y_res == 0][:, 0],
    X_res[y_res == 0][:, 1],
    label="Majority (centroids)",
    alpha=0.8,
)
axes[1].scatter(
    X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label="Minority", alpha=0.8
)
axes[1].set_title("After ClusterCentroids Undersampling")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

# --- Adjust layout and show
plt.tight_layout()
plt.show()

print("Original dataset shape:", {0: sum(y == 0), 1: sum(y == 1)})
print("Resampled dataset shape:", {0: sum(y_res == 0), 1: sum(y_res == 1)})
