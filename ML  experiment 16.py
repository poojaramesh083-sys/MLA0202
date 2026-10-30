import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(), SVC()
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("-" * 40)

result_df = pd.DataFrame({
    "Algorithm": results.keys(),
    "Accuracy (%)": [v * 100 for v in results.values()]
})

print("\nClassification Performance:")
print(result_df.to_string(index=False))

plt.figure(figsize=(10, 5))

plt.bar(
    results.keys(),
    [v * 100 for v in results.values()]
)

plt.title("Comparison of Classification Algorithms")
plt.xlabel("Classification Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

metrics = {
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

for name, model in models.items():
    y_pred = model.predict(X_test)

    metrics["Precision"].append(
        precision_score(y_test, y_pred, average="weighted")
    )
    metrics["Recall"].append(
        recall_score(y_test, y_pred, average="weighted")
    )
    metrics["F1 Score"].append(
        f1_score(y_test, y_pred, average="weighted")
    )

metric_df = pd.DataFrame(metrics, index=models.keys())

metric_df.plot(kind="bar", figsize=(11, 6))

plt.title("Performance Comparison of Classification Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

best_model = max(results, key=results.get)

print("\nBest Classification Algorithm:", best_model)
print("Best Accuracy:", round(results[best_model] * 100, 2), "%")
