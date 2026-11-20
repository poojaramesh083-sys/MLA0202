import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = {
    'Income': [25000, 30000, 45000, 50000, 60000,
               70000, 80000, 90000, 35000, 55000,
               75000, 40000, 85000, 65000, 28000,
               95000, 48000, 72000, 32000, 88000],

    'Credit_Score': [550, 580, 620, 650, 680,
                     700, 750, 780, 590, 660,
                     720, 610, 760, 690, 560,
                     800, 640, 730, 570, 790],

    'Loan_Amount': [100000, 120000, 150000, 180000,
                    200000, 220000, 250000, 280000,
                    110000, 190000, 230000, 130000,
                    270000, 210000, 105000, 300000,
                    160000, 240000, 115000, 290000],

    'Loan_Status': [0, 0, 0, 1, 1,
                    1, 1, 1, 0, 1,
                    1, 0, 1, 1, 0,
                    1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Income', 'Credit_Score', 'Loan_Amount']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Bank Loan Prediction")
print("--------------------")
print("Actual Values:", list(y_test))
print("Predicted Values:", list(y_pred))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            zero_division=0))

new_customer = pd.DataFrame(
    [[75000, 720, 200000]],
    columns=['Income', 'Credit_Score', 'Loan_Amount']
)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nLoan Prediction: Eligible")
else:
    print("\nLoan Prediction: Not Eligible")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Eligible', 'Eligible'],
            yticklabels=['Not Eligible', 'Eligible'])

plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.show()

plt.figure(figsize=(7, 4))

plt.plot(range(len(y_test)), list(y_test),
         marker='o', label='Actual')

plt.plot(range(len(y_pred)), list(y_pred),
         marker='x', label='Predicted')

plt.title("Actual vs Predicted Loan Status")
plt.xlabel("Test Customer")
plt.ylabel("Loan Status (0 = No, 1 = Yes)")
plt.yticks([0, 1], ['Not Eligible', 'Eligible'])
plt.legend()
plt.grid()
plt.show()

df['Loan_Status'].value_counts().sort_index().plot(
    kind='bar', color=['skyblue', 'orange']
)

plt.title("Bank Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ['Not Eligible', 'Eligible'],
           rotation=0)
plt.show()
