# Customer Churn Prediction — IBM Telco Dataset

A machine learning pipeline that predicts customer churn using logistic regression on the IBM Telco Customer Churn dataset. Built with scikit-learn, the model identifies at-risk customers with **0.84 ROC-AUC**, enabling targeted retention strategies before customers leave.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.8418 |
| Average Precision | 0.6339 |
| Churn Recall | 0.79 |
| Churn Precision | 0.51 |
| Accuracy | 0.74 |

The model catches **79% of customers who will churn**. Accuracy is deprioritized — the dataset is imbalanced (73% No Churn / 27% Churn), so recall on the minority class is the metric that matters.

### Confusion Matrix

```
                 Predicted No Churn    Predicted Churn
Actual No Churn        749                  286
Actual Churn            80                  294
```

- **294** churners correctly flagged before leaving ✅
- **80** churners missed ❌
- **286** false alarms (retention offer sent unnecessarily)
- **749** non-churners correctly left alone

---

## Dataset

**IBM Telco Customer Churn** — 7,043 rows, 20 features (after dropping `customerID`).

Each row represents a customer. Features include demographic info, account details, and subscribed services. Target column is `Churn` (0 = stayed, 1 = churned).

Download: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Structure

```
customer_churn/
├── churn.py                              # Full pipeline: cleaning → training → evaluation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (download separately from Kaggle)
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Data Cleaning

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df[service_cols] = df[service_cols].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
```

- `TotalCharges` loads as a string due to blank values for new customers — converted to float, blanks filled with 0 (tenure = 0 customers have no charges yet)
- `customerID` dropped — unique identifier with no predictive value
- `Churn` encoded: Yes → 1, No → 0
- Redundant category `"No internet service"` collapsed to `"No"` across 6 service columns — same for `"No phone service"` in `MultipleLines`

### 2. Train/Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
```

80/20 split, stratified on `y` to preserve the 73/27 class ratio in both splits.

### 3. Preprocessing Pipeline

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
```

- **Numeric columns** (`tenure`, `MonthlyCharges`, `TotalCharges`): scaled to mean 0, std 1 — prevents large-magnitude features from dominating logistic regression coefficients
- **Categorical columns** (all others): one-hot encoded into binary columns — converts strings to numbers the model can learn from
- `handle_unknown='ignore'` — safely handles any unseen categories at prediction time
- Preprocessor is fit on training data only, applied to test data. No leakage.

### 4. Model

```python
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
```

Logistic regression with `class_weight='balanced'` — automatically upweights the minority class (churners) during training to correct for the 73/27 imbalance. `max_iter=1000` ensures convergence on the expanded one-hot feature space.

The full `Pipeline` chains preprocessing and model into one object — calling `model.fit()` fits both steps, calling `model.predict()` applies both steps in order.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/customer-churn-prediction
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project folder.

Update the file path in `churn.py`:
```python
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

Run:
```bash
python churn.py
```
