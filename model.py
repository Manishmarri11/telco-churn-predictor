import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
df = pd.read_csv(r'C:\Users\manis\customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns = ['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df[service_cols] = df[service_cols].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
x = df.drop(columns=['Churn'])
y = df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {x_train.shape}")
print(f"Test size: {x_test.shape}")

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in x_train.columns if c not in num_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight = 'balanced'))
])
model.fit(x_train, y_train)

prob = model.predict_proba(x_test)[:, 1]
predictions = model.predict(x_test)
print("ROC AUC Score:", roc_auc_score(y_test, prob))
print("Average Precision Score:", average_precision_score(y_test, prob))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))