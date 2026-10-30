import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Iris Flower Classification using KNN")
print("Accuracy:", accuracy * 100, "%")

print("\nActual and Predicted Values:")
for actual, predicted in zip(y_test, y_pred):
    print(iris.target_names[actual], "->",
          iris.target_names[predicted])

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted Flower")
plt.ylabel("Actual Flower")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(y_test, 'o-', label="Actual")
plt.plot(y_pred, 'x--', label="Predicted")

plt.title("Actual vs Predicted Iris Classification")
plt.xlabel("Test Samples")
plt.ylabel("Flower Class")
plt.yticks([0, 1, 2], iris.target_names)
plt.legend()
plt.grid()
plt.show()

plt.bar(["KNN Accuracy"], [accuracy * 100])
plt.title("KNN Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()
