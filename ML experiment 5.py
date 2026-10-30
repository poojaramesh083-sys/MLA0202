import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
X, y = make_classification(n_samples=240, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=1.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.25, 0.9, 1.1]})
for cls, color, label in [(0, "#A8DADC", "Class 0"), (1, "#F4A7A7", "Class 1")]:
    axes[0].scatter(X_train[y_train == cls, 0], X_train[y_train == cls, 1], c=color, edgecolors="#666666", linewidths=0.5, s=43, label=f"{label} (train)")
    axes[0].scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], facecolors="none", edgecolors=color, linewidths=1.5, s=75, label=f"{label} (test)")
axes[0].set_title("K-NN Classification (K=5)")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].legend(frameon=True, fontsize=8)
axes[0].grid(alpha=0.2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(ax=axes[1], cmap="Pastel2", colorbar=False)
axes[1].set_title("Confusion Matrix")
axes[2].axis("off")
console_text = f"""K-Nearest Neighbours Classification
Number of training samples: {len(X_train)}
Number of testing samples: {len(X_test)}
Value of K: {k}
Accuracy: {accuracy:.2%}

Confusion Matrix:
{cm[0, 0]:2d}  {cm[0, 1]:2d}
{cm[1, 0]:2d}  {cm[1, 1]:2d}"""
axes[2].text(0.02, 0.96, "Console Output", fontsize=14, fontweight="bold", va="top", color="#4A5568")
axes[2].text(0.02, 0.84, console_text, fontsize=10.5, va="top", family="monospace", linespacing=1.7, bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F1F7", edgecolor="#D8C6DC"))
fig.suptitle(f"K-Nearest Neighbours Results | Accuracy: {accuracy:.2%}", fontsize=15)
fig.tight_layout()
plt.savefig("knn_visual_output.png", dpi=180, bbox_inches="tight")
plt.show()
print("K-Nearest Neighbours Classification")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Value of K: {k}")
print(f"Accuracy: {accuracy:.2%}")
print("Confusion Matrix:")
print(cm)
