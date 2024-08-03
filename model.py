# model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(n_estimators=100, max_depth=3):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    joblib.dump(model, f'model_{n_estimators}_{accuracy}.pkl')
    print(f"Model saved as model_{n_estimators}_{accuracy}.pkl with accuracy: {accuracy}")

    return accuracy


if __name__ == "__main__":
    train_model()
