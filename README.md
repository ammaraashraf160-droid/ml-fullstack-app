# Full-Stack Machine Learning App

A small ML web app: two scikit-learn models, a FastAPI backend, and an HTML frontend.

- **Regression:** Linear Regression on the California housing dataset (predicts house value).
- **Classification:** Logistic Regression on the Iris dataset (predicts flower species).

## Results

| Model | Metric | Score |
|---|---|---|
| Linear Regression | R² (test set) | 0.498 |
| Logistic Regression | Accuracy (test set) | 0.967 |

## Project structure

```
ml-fullstack-app/
├── backend/main.py        # FastAPI app with the two POST endpoints
├── frontend/index.html    # Page with one form per model
├── models/                # Saved .joblib models (created by train_models.py)
├── train_models.py        # Trains and saves both models
├── requirements.txt
└── README.md
```

## How to run

1. Clone the repo and open a terminal in the project folder.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Train and save the models (needs internet the first time to download the housing data):
   ```
   python train_models.py
   ```
5. Start the backend:
   ```
   uvicorn backend.main:app --reload
   ```
   Interactive API docs: http://127.0.0.1:8000/docs
6. Open `frontend/index.html` in your browser, fill in a form and click the button.

## API

`POST /predict/regression`
```json
{"med_inc": 5, "house_age": 20, "ave_rooms": 6, "ave_occup": 3}
```
returns `{"predicted_value_usd": 312345.67}`

`POST /predict/classification`
```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```
returns `{"species": "setosa", "confidence": 0.97}`
