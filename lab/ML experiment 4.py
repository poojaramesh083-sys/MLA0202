import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd', learning_rate_init=0.1, max_iter=2000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, ax=axes[0], cmap='Blues')
axes[0].set_title("ANN Confusion Matrix")
axes[1].plot(model.loss_curve_)
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("Loss")
axes[1].set_title("Backpropagation Training Loss")
axes[1].grid(True)
plt.tight_layout()
plt.show()
