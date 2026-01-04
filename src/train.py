import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from data_loader import load_data
from preprocess import get_splits
import os

os.makedirs("models", exist_ok=True)

mlflow.set_experiment("heart_disease_mlops")

df = load_data()

(X_train, X_test, y_train, y_test), preprocessor = get_splits(df)

models = {
    "log_reg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipe, name)

        joblib.dump(pipe, f"models/{name}.pkl")
        print(name, acc, auc)
