import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

from L3.services import save_model
from L4.services import convert_quality, best_degree

# Data preparing
df = pd.read_csv('../datasets/apple_quality.csv')
df_normal = df.drop(['A_id', 'Acidity'], axis=1)
df_normal['Quality'] = df_normal['Quality'].astype(str).apply(convert_quality)
df_normal = df_normal.dropna()

# Splitting the dataset into train/test
X = df_normal.drop(['Quality'], axis=1)
y = df_normal['Quality']

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=0.001, max_iter=1000, ),
    'Lasso': Lasso(alpha=0.001, max_iter=1000, ),
    'ElasticNet': ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=1000, )
}

# Models
for name, model in models.items():
    model.fit(X_Train, y_Train)
    y_pred = model.predict(X_Test)
    mse = mean_squared_error(y_Test, y_pred)
    mae = mean_absolute_error(y_Test, y_pred)
    rmse = np.sqrt(mse)
    print(f"{name}: \nMSE: {mse}\nMAE: {mae}\nRMSE: {rmse}")

print("\n--------------------\nCV dataset\n--------------------")

# Train/test/validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_val)
X_test = scaler.transform(X_test)

degree = best_degree(np.arange(1, 5), X, y)
polynomial = PolynomialFeatures(degree=degree, include_bias=False)
polynomial_X_train = polynomial.fit_transform(X_train)
polynomial_X_test = polynomial.transform(X_test)

model = Ridge(alpha=1)
model.fit(polynomial_X_train, y_train)
predict = model.predict(polynomial_X_test)

mse = mean_squared_error(y_test, predict)
mae = mean_absolute_error(y_test, predict)
rmse = np.sqrt(mse)
print(f"Best degree: {degree}")
print(f"Ridge model: \nMSE: {mse}\nMAE: {mae}\nRMSE: {rmse}")

save_model(model, "best_ridge_model.pkl")


