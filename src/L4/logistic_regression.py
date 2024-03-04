import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

from L4.services import convert_quality
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data preparing
df = pd.read_csv('../datasets/apple_quality.csv')
df_normal = df.drop(['A_id', 'Acidity'], axis=1)
df_normal['Quality'] = df_normal['Quality'].astype(str).apply(convert_quality)
df_normal = df_normal.dropna()

# Splitting the dataset into train/test
X = df_normal.drop(['Quality'], axis=1)
y = df_normal['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000, penalty='l2',).fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_pred = model.predict_proba(X_train)
y_test_pred = model.predict_proba(X_test)

print('accuracy: ', accuracy_score(y_test, y_pred))
print('loss train: ', log_loss(y_train, y_train_pred))
print('loss test: ', log_loss(y_test, y_test_pred))
print('confusion_matrix: \n', confusion_matrix(y_test, y_pred))
print('classification_report: \n', classification_report(y_test, y_pred), '\n')
