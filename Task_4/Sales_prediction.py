# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your sales data from a CSV file 
data = pd.read_csv('car_purchasing.csv')

# Data preprocessing: Extract features and target variable
X = data[['AdvertisingExpenditure', 'TargetAudienceSegmentation']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the actual vs. predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()

# Now, you can use this trained model to make sales predictions for new data
new_data = pd.DataFrame({'AdvertisingExpenditure': [1000], 'TargetAudienceSegmentation': [2]})
predicted_sales = model.predict(new_data)
print("Predicted Sales for new data:", predicted_sales)
