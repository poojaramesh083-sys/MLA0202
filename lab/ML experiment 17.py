import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'RAM': [2, 3, 4, 4, 6, 6, 8, 8, 12, 12,
            4, 6, 8, 12, 16],
    'Storage': [32, 32, 64, 128, 64, 128, 128, 256,
                256, 512, 64, 128, 256, 512, 512],
    'Battery': [3000, 3500, 4000, 4500, 4000, 5000,
                4500, 5000, 5000, 5500, 4000, 4500,
                5000, 5500, 6000],
    'Camera': [8, 12, 13, 16, 16, 20, 24, 32, 48,
               64, 13, 20, 32, 50, 108],
    'Screen': [5.0, 5.2, 5.5, 6.0, 5.8, 6.1, 6.2,
               6.4, 6.5, 6.7, 5.6, 6.0, 6.3, 6.7, 6.9],
    'Price': [5000, 7000, 9000, 12000, 15000, 18000,
              22000, 25000, 30000, 40000, 10000,
              17000, 26000, 38000, 50000]
}

df = pd.DataFrame(data)

X = df[['RAM', 'Storage', 'Battery', 'Camera', 'Screen']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mobile Price Prediction")
print("-----------------------")
print("Mean Absolute Error:",
      mean_absolute_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))
new_mobile = pd.DataFrame({
    'RAM': [8],
    'Storage': [128],
    'Battery': [5000],
    'Camera': [50],
    'Screen': [6.5]
})

predicted_price = model.predict(new_mobile)

print("\nNew Mobile Details:")
print(new_mobile)
print("Predicted Mobile Price: Rs.",
      round(predicted_price[0], 2))

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, color='blue',
            label='Predicted Prices')

plt.plot([y.min(), y.max()],
         [y.min(), y.max()],
         color='red', linestyle='--',
         label='Perfect Prediction')

plt.xlabel("Actual Price (Rs.)")
plt.ylabel("Predicted Price (Rs.)")
plt.title("Actual vs Predicted Mobile Prices")
plt.legend()
plt.grid()
plt.show()

importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(X.columns, importance, color='green')

plt.xlabel("Mobile Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Mobile Price Prediction")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.show()
