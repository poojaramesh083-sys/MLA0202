import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
X = np.array([
    1.0, 1.5, 2.0, 2.2, 2.8,
    7.0, 7.5, 8.0, 8.5, 9.0
]).reshape(-1, 1)
model = GaussianMixture(
    n_components=2,
    random_state=42
)
model.fit(X)
labels = model.predict(X)
means = model.means_.flatten()
weights = model.weights_.flatten()
x_range = np.linspace(0, 10, 400).reshape(-1, 1)
density = np.exp(model.score_samples(x_range))
fig, ax = plt.subplots(figsize=(9, 6))
for cluster in np.unique(labels):
    ax.scatter(
        X[labels == cluster],
        np.zeros(np.sum(labels == cluster)),
        label=f"Cluster {cluster + 1}",
        s=70
    )
ax.set_title("Expectation-Maximization Algorithm")
ax.set_xlabel("Data Values")
ax.set_yticks([])
ax.legend()
ax.grid(True, axis="x")
ax2 = ax.twinx()
ax2.plot(
    x_range.flatten(),
    density,
    linestyle="--",
    label="Gaussian Mixture Density"
)
ax2.set_ylabel("Probability Density")
ax2.legend(loc="upper right")
output = (
    "CONSOLE OUTPUT\n\n"
    "Algorithm: Expectation-Maximization\n"
    "Model: Gaussian Mixture Model\n"
    f"Number of Clusters: {model.n_components}\n"
    f"Component Means: {means[0]:.3f}, {means[1]:.3f}\n"
    f"Component Weights: {weights[0]:.3f}, {weights[1]:.3f}\n"
    f"Converged: {model.converged_}\n"
    f"Iterations: {model.n_iter_}"
)
fig.text(
    0.5, 0.02, output,
    ha="center", va="bottom",
    fontsize=10, family="monospace",
    bbox=dict(boxstyle="round", facecolor="white")
)
plt.tight_layout(rect=[0, 0.25, 1, 1])
plt.show()
