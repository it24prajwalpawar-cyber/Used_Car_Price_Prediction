import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# 2. Create car age
df["car_age"] = 2026 - df["year"]

# 3. Separate input features and target
X = df[[
    "car_age",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]]

y = df["selling_price"]

# 4. Convert categorical features into numerical features
X_encoded = pd.get_dummies(X, drop_first=True)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create and train Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")
print("--------------------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 9. Save model and feature columns
joblib.dump(model, "car_price_model.pkl")
joblib.dump(list(X_encoded.columns), "model_columns.pkl")

print("--------------------------------")
print("Model saved successfully!")

# 10. Visualize Actual vs Predicted Prices
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")

plt.title("Actual vs Predicted Car Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()