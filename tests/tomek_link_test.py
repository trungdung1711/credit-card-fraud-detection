# Tomek Links visualization
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.80, 0.20],
    class_sep=0.3,
    random_state=101014,
)

tl = TomekLinks(sampling_strategy="auto")
X_res, y_res = tl.fit_resample(X, y)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Before
axs[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Majority", alpha=0.6)
axs[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Minority", alpha=0.9)
axs[0].set_title("Before Tomek Links")
axs[0].legend()

# After
axs[1].scatter(
    X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1], label="Majority", alpha=0.6
)
axs[1].scatter(
    X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1], label="Minority", alpha=0.9
)
axs[1].set_title("After Tomek Links (cleaner boundary)")
axs[1].legend()

plt.show()
