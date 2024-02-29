import numpy as np

print("Task 1")
random_list = np.random.randint(low=1, high=100, size=10)
print(f"Створіть масив NumPy із 10 випадкових цілих чисел"
      f"\n{random_list}"
      f"\nВиконайте наступні операції:")

average = np.average(random_list)
median = np.median(random_list)
std = np.std(random_list)
print(f"Середнє значення:{average}"
      f"\nМедіана: {median}"
      f"\nСтандартне відхилення масиву: {std}.")

random_list[random_list % 2 == 0] = 0
print(f"Замініть всі парні числа у масиві на 0.\n{random_list}")

print("\nTask 2")
matrix = np.random.randint(low=1, high=10, size=(3, 3))
print(f"Створіть 2D масив NumPy (матрицю) розміром (3, 3) із випадковими цілими числами.\n{matrix}")

first_row = matrix[0]
print(f"Виведіть перший рядок матриці: {first_row}")
last_row = matrix[-1]
print(f"Виведіть останній стовпець матриці: {last_row}")
diagonal = np.diag(matrix)
print(f"Виведіть діагональні елементи матриці: {diagonal}")

print("\nTask 3")
matrix1 = np.random.randint(low=1, high=10, size=(3, 3))
matrix2 = np.random.randint(low=1, high=10, size=(3, ))

print(f"Створіть 2D масив NumPy розміром (3, 3)"
      f"\n{matrix1}"
      f"\nта 1D масив розміром (3,)"
      f"\n{matrix2}\n=>\n{matrix2[:, np.newaxis]}")
sum_matrix = matrix1 + matrix2[:, np.newaxis]
print(f"Додавання 1D масиву до кожного рядка 2D масиву:\n{sum_matrix}")

print("\nTask 4")
matrix4 = np.random.randint(low=1, high=100, size=(5, 5))
print(f"Створіть 2D масив NumPy розміром (5, 5) з випадковими цілими числами\n{matrix4}")
unique = np.unique(matrix4)
print(f"Знайдіть та виведіть всі унікальні елементи у масиві:\n{unique}")

c = 250
rows = matrix4[np.sum(matrix4, axis=1) > c]
print(f"Виведіть всі рядки, сума елементів у яких більша за {c}:\n{rows}")

print("\nTask 5")
array = np.arange(1, 21)
print(f"Створіть 1D масив NumPy, що містить цілі числа від 1 до 20 (включно)\n{array}")

matrix5_2d = array.reshape(4, 5)
print(f"Використайте оператор shape, щоб перетворити 1D масив у 2D масив розміром (4, 5):"
      f"\n{matrix5_2d}"
      f"\nРозмір: {matrix5_2d.shape}")

import pandas as pd

df = pd.DataFrame(
    {
        "Ім'я": ["Анастасія", "Михайло", "Анна", "Олександр", "Наталія", "Марія",],
        "Вік": [22, 38, 20, 21, 48, 60],
        "Місто": ["Київ", "Одеса", "Київ", "Харків", "Миколаїв", "Херсон"]
    }
)
print(f"Створіть DataFrame Pandas із щонайменше 5 рядками та 3 стовпцями.\n{df}")
df["Заробіток"] = [10, 23, 0, 7, 15, 6]
f = 10
filtered = df[df["Заробіток"] >= f]
print(f"Відфільтруйте DataFrame, щоб показати лише рядки, де числове значення більше {f}.\n{filtered}")

# wine.csv
print("\nwine.csv")
wine_df = pd.read_csv('../datasets/wine.csv')
print(f"Відобразіть перші 5 рядків набору даних.\n{wine_df.head(5)}")
print(f"Розрахуйте та виведіть загальну статистику для числових стовпців у наборі даних\n{wine_df.describe()}")
unique_values_w = wine_df["1"].unique().tolist()
print(f"Знайдіть та виведіть унікальні значення у категорійному стовпці.\n{unique_values_w}")

# Laptop.csv
print("\nLaptop.csv")
csv_df = pd.read_csv('../datasets/Laptops.csv')
print(f"Відобразіть перші 5 рядків набору даних.\n{csv_df.head(5)}")
print(f"Розрахуйте та виведіть загальну статистику для числових стовпців у наборі даних\n{csv_df.describe()}")
unique_values = csv_df["Brand"].unique().tolist()
print(f"Знайдіть та виведіть унікальні значення у категорійному стовпці.\n{unique_values}")
