"""
Step 1: Train two scikit-learn models and save them with joblib.

  - Regression:     LinearRegression on California housing (predict house value)
  - Classification: LogisticRegression on Iris (predict flower species)

Run from the project root:   python train_models.py
"""
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# Folder where the trained models will be saved: <project>/models
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1) REGRESSION: predict median house value
# ---------------------------------------------------------------
housing = fetch_california_housing(as_frame=True)  # downloads once, then cached

# We use 4 easy-to-understand features so the web form stays small
reg_features = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
X_reg = housing.data[reg_features].values   # input table  (rows x 4 columns)
y_reg = housing.target.values               # target: value in units of $100,000

# Hold back 20% of the data to test the model on rows it has never seen
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)             # "learn" from the training rows

r2 = r2_score(y_test, reg_model.predict(X_test))
print(f"Regression R^2 on test data: {r2:.3f}")

joblib.dump(reg_model, MODELS_DIR / "regression_model.joblib")

# ---------------------------------------------------------------
# 2) CLASSIFICATION: predict iris species
# ---------------------------------------------------------------
iris = load_iris()
X_clf, y_clf = iris.data, iris.target       # 4 measurements -> class 0, 1 or 2

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf_model = LogisticRegression(max_iter=200)
clf_model.fit(X_train, y_train)

acc = accuracy_score(y_test, clf_model.predict(X_test))
print(f"Classification accuracy on test data: {acc:.3f}")

joblib.dump(clf_model, MODELS_DIR / "classification_model.joblib")

print("Saved models to:", MODELS_DIR)
