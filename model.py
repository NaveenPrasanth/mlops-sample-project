import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load the Adult dataset
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
data = pd.read_csv(
    "data/adult.csv", names=columns, na_values=" ?", skipinitialspace=True
)

# Preprocess the data
data.dropna(inplace=True)
label_encoders = {}
for column in data.select_dtypes(include=["object"]).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

X = data.drop("income", axis=1)
y = data["income"]


def train_model(n_estimators, max_depth, min_samples_split=2, min_samples_leaf=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    accuracy = train_model(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best parameters: ", study.best_params)
    print("Best accuracy: ", study.best_value)

    # Train the best model
    best_params = study.best_params
    best_accuracy = train_model(**best_params)
    print("Best accuracy after training with best parameters: ", best_accuracy)

    # Train and save the best model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "best_model.pkl")
    print("Best model saved as best_model.pkl")
