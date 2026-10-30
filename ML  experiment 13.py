import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 10000, 15000, 20000, 25000,
                30000, 35000, 40000, 45000, 50000],
    'Engine': [1500, 1500, 1200, 1200, 1000,
               1000, 1000, 800, 800, 800],
    'Price': [900000, 850000, 750000, 700000, 600000,
              550000, 450000, 400000, 300000, 250000]
}

df = pd.DataFrame(data)

print("Car Dataset:")
print(df)

X = df[['Age', 'Mileage', 'Engine']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Prices:", list(y_test))
print("Predicted Prices:", y_pred.astype(int))

print("\nR2 Score:", r2_score(y_test, y_pred))

new_car = pd.DataFrame({
    'Age': [3],
    'Mileage': [18000],
    'Engine': [1500]
})

predicted_price = model.predict(new_car)

print("\nPredicted Car Price: Rs.", int(predicted_price[0]))
                                        
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df['Age'], df['Price'], marker='o')
plt.xlabel("Car Age (Years)")
plt.ylabel("Car Price (Rs.)")
plt.title("Car Age vs Car Price")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Mileage'], df['Price'])
plt.xlabel("Mileage (KM)")
plt.ylabel("Car Price (Rs.)")
plt.title("Mileage vs Car Price")
plt.grid()
plt.show()
