import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
plot_model = LogisticRegression(max_iter=5000)
plot_model.fit(X_train_pca, y_train)
x_min = min(X_train_pca[:, 0].min(), X_test_pca[:, 0].min()) - 1
x_max = max(X_train_pca[:, 0].max(), X_test_pca[:, 0].max()) + 1
y_min = min(X_train_pca[:, 1].min(), X_test_pca[:, 1].min()) - 1
y_max = max(X_train_pca[:, 1].max(), X_test_pca[:, 1].max()) + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = plot_model.predict_proba(grid)[:, 1].reshape(xx.shape)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=["#F8D7DA", "#D6EAF8"], alpha=0.8)
axes[0].contour(xx, yy, Z, levels=[0.5], colors=["#9B7EBD"], linewidths=2)
axes[0].scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], color="#E5989B", edgecolors="white", s=45, label="Malignant")
axes[0].scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], color="#A8DADC", edgecolors="white", s=45, label="Benign")
axes[0].set_title("Logistic Regression Decision Boundary")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend()
sns.heatmap(cm, annot=True, fmt="d", cmap="Pastel2", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"], linewidths=1, linecolor="white", cbar=False, ax=axes[1], annot_kws={"size": 13})
axes[1].set_title(f"Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("Actual Label")
fig.tight_layout()
plt.show()
print("Logistic Regression Classification")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Confusion Matrix:")
print(cm)
