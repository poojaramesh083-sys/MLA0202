import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    'Month': np.arange(1, 13),
    'Sales': [100, 120, 130, 150, 170, 180,
              200, 220, 210, 240, 260, 280]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

future_months = pd.DataFrame({
    'Month': [13, 14, 15, 16, 17, 18]
})

future_sales = model.predict(future_months)

print("Future Sales Prediction")
for month, sales in zip(future_months['Month'], future_sales):
    print("Month", month, "Predicted Sales:", round(sales, 2))

plt.figure(figsize=(10, 5))

plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red',
         label='Regression Line')

plt.plot(future_months['Month'], future_sales,
         marker='o', linestyle='--', color='green',
         label='Future Sales Prediction')

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
