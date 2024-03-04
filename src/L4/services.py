import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def convert_quality(value: str) -> int:
    if value.lower() == 'good':
        return 1
    if value.lower() == 'bad':
        return 0


def best_degree_param(d_range: np.arange,
                      X_train: np.ndarray, X_val: np.ndarray,
                      y_train: np.ndarray, y_val: np.ndarray, ):
    eval_rmse_errors = []
    for d in d_range:
        poly_converter = PolynomialFeatures(degree=d, include_bias=False)
        poly_features_train = poly_converter.fit_transform(X_train)
        poly_features_eval = poly_converter.fit_transform(X_val)

        model = Ridge(alpha=1)
        model.fit(poly_features_train, y_train)
        pred = model.predict(poly_features_eval)

        MSE = mean_squared_error(y_val, pred)

        eval_rmse_errors.append(MSE * (-1))

    return d_range[np.argmax(np.array(eval_rmse_errors))]


def best_degree(d_range: np.ndarray, X: pd.DataFrame, y: np.ndarray):
    mse_dict = {}

    for d in d_range:
        model = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1))
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mse_dict[d] = np.mean(scores)
    return max(mse_dict, key=mse_dict.get)
