import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Age': [22, 25, 30, 35, 40, 45, 50, 55, 28, 32,
            38, 42, 48, 52, 60],
    'Income': [20000, 25000, 30000, 40000, 50000,
               60000, 70000, 80000, 28000, 35000,
               45000, 55000, 65000, 75000, 90000],
    'Loan': [50000, 60000, 40000, 30000, 20000,
             25000, 15000, 10000, 70000, 65000,
             35000, 30000, 20000, 15000, 10000],
    'Credit_Score': ['Poor', 'Poor', 'Standard',
                     'Standard', 'Good', 'Good',
                     'Good', 'Good', 'Poor', 'Standard',
                     'Standard', 'Good', 'Good',
                     'Good', 'Good']
}

df = pd.DataFrame(data)

X = df[['Age', 'Income', 'Loan']]
y = df['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Credit Score Classification")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 8: Graph 1 - Credit Score Distribution
plt.figure(figsize=(7, 4))
sns.countplot(x='Credit_Score', data=df)
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of People")
plt.show()

# Step 9: Graph 2 - Actual vs Predicted
plt.figure(figsize=(7, 4))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=model.classes_,
            yticklabels=model.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Credit Score")
plt.ylabel("Actual Credit Score")
plt.show()

plt.figure(figsize=(7, 4))
sns.barplot(x=model.feature_importances_,
            y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

new_customer = pd.DataFrame({
    'Age': [30],
    'Income': [45000],
    'Loan': [20000]
})

prediction = model.predict(new_customer)
print("\nNew Customer Credit Score:", prediction[0])
