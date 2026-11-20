import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Flower")
plt.ylabel("Actual Flower")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(["Naive Bayes"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Iris Classification Accuracy")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(y_test, 'o-', label="Actual")
plt.plot(y_pred, 'x--', label="Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Flower Class")
plt.yticks([0, 1, 2], iris.target_names)
plt.title("Actual vs Predicted Iris Flowers")
plt.legend()
plt.grid()
plt.show()

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("\nNew Flower Prediction:",
      iris.target_names[prediction[0]])
