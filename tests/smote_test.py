import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Create imbalanced dataset (2D for visualization)
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    random_state=666,
)

# Apply SMOTE
sm = SMOTE(random_state=42, sampling_strategy="minority")
X_res, y_res = sm.fit_resample(X, y)

# Plot before and after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
axes[0].scatter(
    X[y == 0][:, 0], X[y == 0][:, 1], label="Majority", alpha=0.6, edgecolor="k"
)
axes[0].scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    label="Minority",
    alpha=0.9,
    color="red",
    edgecolor="k",
)
axes[0].set_title("Before SMOTE")
axes[0].legend()

# After SMOTE
axes[1].scatter(
    X_res[y_res == 0][:, 0],
    X_res[y_res == 0][:, 1],
    label="Majority",
    alpha=0.6,
    edgecolor="k",
)
axes[1].scatter(
    X_res[y_res == 1][:, 0],
    X_res[y_res == 1][:, 1],
    label="Minority (synthetic)",
    alpha=0.9,
    color="red",
    edgecolor="k",
)
axes[1].set_title("After SMOTE")
axes[1].legend()

plt.tight_layout()
plt.show()
