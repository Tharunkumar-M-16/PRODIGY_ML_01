import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset (CSV file must be in same folder)
data = pd.read_csv("train.csv")

# Select features and target
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

# Handle missing values
X = X.fillna(X.median())

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Take user input
sqft = int(input("Enter house square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Predict price
price = model.predict([[sqft, bedrooms, bathrooms]])

print("\nEstimated House Price:", int(price[0]))


