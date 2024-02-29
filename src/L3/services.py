import joblib
from sklearn.linear_model import LinearRegression


def convert_storage(storage: str) -> float:
    if isinstance(storage, float):
        return 0
    elif 'GB' in storage:
        return float(storage.replace(' GB', '').strip())
    elif 'TB' in storage:
        return float(storage.replace(' TB', '').strip()) * 1024
    else:
        return float('NaN')


def convert_ram(value: str) -> float:
    if isinstance(value, float):
        return 0
    else:
        return float(value.replace('GB', '').strip())


def convert_price(price: str) -> float:
    return float(price.replace('₹', '').replace(',', ''))


def convert_processor(value: str) -> str:
    if value.startswith('Core'):
        return 'Core'
    elif value.startswith('Ryzen'):
        return 'Ryzen'
    elif value.startswith('MediaTek '):
        return 'MediaTek '
    elif value.startswith('Athlon'):
        return 'Athlon'
    elif value.startswith('M1'):
        return 'M1'
    elif value.startswith('M2'):
        return 'M2'
    else:
        return value


def compare_models(mse_gd: float, mse_ne: float) -> None:
    print(f'Mean Squared Error (Gradient Descent): {mse_gd}')
    print(f'Mean Squared Error (Normal Equation): {mse_ne}')
    if mse_gd > mse_ne:
        print('The model with Gradient Descent is better than the model with Normal Equation')
    else:
        print('The model with Normal Equation is better than the model with Gradient Descent')


def save_model(model: LinearRegression, file_name: str) -> None:
    joblib.dump(model, file_name)
