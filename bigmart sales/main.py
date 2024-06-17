import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load and inspect the dataset
data = pd.read_csv('house_tiny.csv')
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Identify categorical features
# - Item_Identifier
# - Item_Fat_Content
# - Item_Type
# - Outlet_Identifier
# - Outlet_Size
# - Outlet_Location_Type
# - Outlet_Type

# Handle missing values
# Fill missing values in 'Item_Weight' with the column's mean
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

# Fill missing values in 'Outlet_Size' with the mode
mode_outlet_size = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
missing_outlet_size = data['Outlet_Size'].isnull()
data.loc[missing_outlet_size, 'Outlet_Size'] = data.loc[missing_outlet_size, 'Outlet_Type'].apply(lambda x: mode_outlet_size[x])

# Verify missing values have been handled
print(data.isnull().sum())

# Visualization of features
sns.set()

# Distribution of 'Item_Weight'
plt.figure(figsize=(6,6))
sns.histplot(data['Item_Weight'], kde=True)
plt.title('Distribution of Item Weight')
plt.show()

# Distribution of 'Item_Visibility'
plt.figure(figsize=(6,6))
sns.histplot(data['Item_Visibility'], kde=True)
plt.title('Distribution of Item Visibility')
plt.show()

# Distribution of 'Item_MRP'
plt.figure(figsize=(6,6))
sns.histplot(data['Item_MRP'], kde=True)
plt.title('Distribution of Item MRP')
plt.show()

# Distribution of 'Item_Outlet_Sales'
plt.figure(figsize=(6,6))
sns.histplot(data['Item_Outlet_Sales'], kde=True)
plt.title('Distribution of Item Outlet Sales')
plt.show()

# Count plot for 'Outlet_Establishment_Year'
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=data)
plt.title('Outlet Establishment Year')
plt.show()

# Count plot for 'Item_Fat_Content'
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=data)
plt.title('Item Fat Content')
plt.show()

# Count plot for 'Item_Type'
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=data)
plt.title('Item Type')
plt.show()

# Count plot for 'Outlet_Size'
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=data)
plt.title('Outlet Size')
plt.show()

# Standardize values in 'Item_Fat_Content'
data['Item_Fat_Content'].replace({'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}, inplace=True)
print(data['Item_Fat_Content'].value_counts())

# Encode categorical features
label_encoder = LabelEncoder()
categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

print(data.head())

# Prepare data for modeling
features = data.drop(columns='Item_Outlet_Sales')
target = data['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)
print(features.shape, X_train.shape, X_test.shape)

# Train XGBRegressor model
model = XGBRegressor()
model.fit(X_train, y_train)

# Evaluate model performance on training data
train_predictions = model.predict(X_train)
r2_train = metrics.r2_score(y_train, train_predictions)
print('Training R-squared value: ', r2_train)

# Evaluate model performance on test data
test_predictions = model.predict(X_test)
r2_test = metrics.r2_score(y_test, test_predictions)
print('Test R-squared value: ', r2_test)
