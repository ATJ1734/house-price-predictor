import pandas as pd                  # For loading and handling the CSV data
import matplotlib.pyplot as plt      # For plotting results
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression     # The ML model
from sklearn.metrics import mean_squared_error        # For checking accuracy

# 1. Load the data
data = pd.read_csv("data.csv")

# 2. Separate inputs (X) and output (y)
X = data[["Size", "Bedrooms", "Age"]]  # Features
y = data["Price"]                      # Target

# 3. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Print model accuracy (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 7. Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# 8. Custom prediction (example)
print("\n--- Try Your Own Prediction ---")
example = [[2200, 3, 10]]  # Size, Bedrooms, Age
predicted_price = model.predict(example)
print(f"Predicted price for a 2200 sq.m, 3 bed, 10 yr old house: ${predicted_price[0]:,.2f}")
