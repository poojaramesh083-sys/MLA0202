import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")

print("\nActual Values:", y_test)
print("Predicted Values:", y_pred)

plt.figure(figsize=(8, 5))
plt.plot(y_test, 'o-', label='Actual Values')
plt.plot(y_pred, 's--', label='Predicted Values')
plt.title("Perceptron - Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Iris Class")
plt.legend()
plt.grid()
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.title("Iris Classification Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

plt.figure(figsize=(5, 4))
plt.bar(["Perceptron"], [accuracy * 100])
plt.title("Perceptron Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()
