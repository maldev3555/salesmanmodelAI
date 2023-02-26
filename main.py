import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_excel("salesreport.xls")

# Clean the data
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Create a new feature for total sales per product
df["Total Sales"] = df["Qty"] * df["Gross Sales"]

# Split the data into training and testing sets
X = df[["Qty", "Gross Sales", "Total Sales"]]
y = df["product code"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a linear regression model
reg = LinearRegression()
reg.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = reg.score(X_train_scaled, y_train)
test_score = reg.score(X_test_scaled, y_test)
print("Training R^2 score: {:.3f}".format(train_score))
print("Testing R^2 score: {:.3f}".format(test_score))

