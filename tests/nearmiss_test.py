import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss

X, y = make_classification(
    n_classes=2,
    class_sep=1.5,
    weights=[0.9, 0.1],
    n_informative=2,
    n_redundant=0,
    flip_y=0,
    n_features=2,
    n_clusters_per_class=1,
    n_samples=300,
    random_state=43,
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    label="Majority class (0)",
    alpha=0.6,
    edgecolor="k",
)
plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    label="Minority class (1)",
    alpha=0.9,
    color="red",
    edgecolor="k",
)
plt.title("Before NearMiss Undersampling")
plt.legend()
plt.grid(True)

nm = NearMiss(version=1)
X_res, y_res = nm.fit_resample(X, y)

plt.subplot(1, 2, 2)
plt.scatter(
    X_res[y_res == 0][:, 0],
    X_res[y_res == 0][:, 1],
    label="Majority class (0)",
    alpha=0.6,
    edgecolor="k",
)
plt.scatter(
    X_res[y_res == 1][:, 0],
    X_res[y_res == 1][:, 1],
    label="Minority class (1)",
    alpha=0.9,
    color="red",
    edgecolor="k",
)
plt.title("After NearMiss Undersampling (Version 1)")
plt.legend()
plt.grid(True)
plt.show()
