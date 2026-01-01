import pandas as pd
from sklearn.linear_model import LinearRegression

# Dummy dataset
data = {
    'square_feet': [1000, 1500, 2000, 2500],
    'bedrooms': [2, 3, 3, 4],
    'price': [200000, 250000, 300000, 350000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['square_feet', 'bedrooms']]
y = df['price']

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
predicted_price = model.predict([[1800, 3]])[0]
print("Model trained successfully")
print("Predicted price for 1800 sqft, 3 bedrooms:", predicted_price)

# Simple CI check: fail if prediction not in expected range
if not (200000 <= predicted_price <= 300000):
    raise ValueError(f"Prediction {predicted_price} is out of expected range! Failing CI.")
