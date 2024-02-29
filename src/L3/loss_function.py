import numpy as np


def loss_function(y: np.ndarray, h_x: np.ndarray) -> float:
    """
    Calculates the loss
    :param y: Target values
    :param h_x: Predicted values
    :return: Loss
    """
    squared_error = np.square(y - h_x)
    sum_squared_error = np.sum(squared_error)
    return sum_squared_error / y.size


if __name__ == "__main__":
    y_true = np.random.randint(1, 100, 100)
    y_pred = np.random.randint(1, 100, 100)
    loss = loss_function(y_true, y_pred)
    print(f"Function loss: {loss}")
