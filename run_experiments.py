import mlflow
import mlflow.sklearn
from model import train_model

params = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 4},
]

for param in params:
    with mlflow.start_run():
        mlflow.log_param("n_estimators", param["n_estimators"])
        mlflow.log_param("max_depth", param["max_depth"])

        accuracy = train_model(param["n_estimators"], param["max_depth"])
        mlflow.log_metric("accuracy", accuracy)
