import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Area': [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3200],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Bathrooms': [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'Price': [2000000, 2500000, 3000000, 3500000,
              4000000, 4500000, 5000000, 5500000,
              6000000, 6500000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("House Price Prediction")
print("----------------------")
print("Actual Prices:", list(y_test))
print("Predicted Prices:", np.round(y_pred, 2))

print("\nMean Squared Error:",
      mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.plot(range(len(y_test)), y_test.values,
         marker='o', label='Actual Price')
plt.plot(range(len(y_pred)), y_pred,
         marker='s', label='Predicted Price')

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Test Data")
plt.ylabel("House Price")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Area'], df['Price'])
plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Area vs House Price")
plt.grid()
plt.show()
new_house = pd.DataFrame({
    'Area': [1800],
    'Bedrooms': [3],
    'Bathrooms': [2]
})

predicted_price = model.predict(new_house)

print("\nNew House Details:")
print("Area: 1800 sq.ft")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Predicted House Price: Rs.", round(predicted_price[0], 2))
