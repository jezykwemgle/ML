import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from services import convert_ram, convert_price, convert_storage, convert_processor, save_model, compare_models
import joblib


"""
-------------------------------------------------------------------------------------------------------------
Preparing data
-------------------------------------------------------------------------------------------------------------
"""
df = pd.read_csv("datasets/Laptops.csv")
df = df.drop(columns=['Unnamed: 0', 'Screen Size', 'Model Name', 'Operating System', 'Touch_Screen', ])
df = df.dropna()

df['Storage'] = df['Storage'].apply(convert_storage)
df['RAM'] = df['RAM'].apply(convert_ram)
df['Price'] = df['Price'].apply(convert_price)
df['Processor'] = df['Processor'].apply(convert_processor)
df = pd.get_dummies(df, columns=['Brand', 'Processor', ])

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
df_normalized = pd.DataFrame(df_normalized, columns=df.columns)

X = df_normalized.drop(columns=['Price', ])
y = df_normalized['Price']

"""
-------------------------------------------------------------------------------------------------------------
Gradient Descent
-------------------------------------------------------------------------------------------------------------
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

save_model(model, 'linear_regression_model_test.pkl')

"""
-------------------------------------------------------------------------------------------------------------
Normal Equation
-------------------------------------------------------------------------------------------------------------
"""
X_train_ne = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_ne = np.c_[np.ones((X_test.shape[0], 1)), X_test]

y_train_ne = y_train
y_test_ne = y_test

# 0 = (X^T * X)^-1 * X^T * y
# np.linalg.inv() обернена матриця
theta = np.linalg.inv(X_train_ne.T.dot(X_train_ne)).dot(X_train_ne.T).dot(y_train_ne)
y_pred_ne = X_test_ne.dot(theta)

mse_ne = mean_squared_error(y_test_ne, y_pred_ne)

"""
-------------------------------------------------------------------------------------------------------------
Model compare
-------------------------------------------------------------------------------------------------------------
"""
print(f'Mean Squared Error (Gradient Descent): {mse}')
print(f'Mean Squared Error (Normal Equation): {mse_ne}')
