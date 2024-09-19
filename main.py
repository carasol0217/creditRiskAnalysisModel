# file_pth = r'/Users/samkim/Downloads/credit_risk.csv'   # to get file path user Finder -> view -> show path bar 
# credit_data = pd.read_csv(file_pth)
# print(credit_data.head())

# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt

# loading data
file_pth = r'/Users/samkim/Downloads/credit_risk.csv'
data = pd.read_csv(file_pth)
print(data.head())

# processing data/handling missing inputs
data.ffill(inplace=True)  # Forward filling missing values

# encoding categorical variables
categorical_columns = ['Home', 'Intent', 'Default']  # correct categorical columns from your dataset
label_encoder = LabelEncoder()

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])


# creating new feature
data['loan_to_income_ratio'] = data['Amount'] / data['Income']

# spltting data
X = data.drop(columns=['Default'])  # replacing target column w target variable
y = data['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training random forest classifier for now
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# evaluation
y_pred = rf.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
print(f"ROC AUC Score: {roc_auc}")
print(classification_report(y_test, y_pred))

# visualizing
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# feature importance
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
print("Feature Importance Ranking:")
for i in sorted_indices:
    print(f"{X.columns[i]}: {importances[i]}")
