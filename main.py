import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the original data
df_original = pd.read_excel("salesreport.xls")

# Load the new Excel files into separate dataframes
new_data1 = pd.read_excel('fastmovingprouct2days.xls')
new_data2 = pd.read_excel('prdctslsvatlstng10days.xls')
new_data3 = pd.read_excel('priftbltyreport.xls')

# Concatenate the new and existing dataframes along the appropriate axis
all_data1 = pd.concat([df_original, new_data1])
all_data2 = pd.concat([df_original, new_data2])
all_data3 = pd.concat([df_original, new_data3])

# Combine the dataframes horizontally if needed
all_data = pd.concat([all_data1, all_data2, all_data3], axis=1)

# Drop any duplicate rows or missing values
all_data.drop_duplicates(inplace=True)
all_data.dropna(inplace=True)

# Create a new feature for total sales per product
all_data["Total Sales"] = all_data["Qty"] * all_data["Gross Sales"]

# Check that the target variable has at least one sample
if all_data["product code"].isnull().all():
    print("Target variable is empty. Cannot split data.")
else:
    # Split the data into training and testing sets
    X = all_data[["Qty", "Gross Sales", "Total Sales"]]
    y = all_data["product code"]
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
