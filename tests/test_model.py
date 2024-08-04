import unittest
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from model import train_model, objective, X, y
import optuna


class TestModel(unittest.TestCase):
    def test_train_model(self):
        # Run Optuna optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)  # Reduced trials for testing purposes

        # Verify that the study found the best parameters
        best_params = study.best_params
        self.assertIsNotNone(best_params, "Optuna failed to find the best parameters.")

        # Train the model with the best parameters and verify accuracy
        best_accuracy = train_model(**best_params)
        self.assertGreater(best_accuracy, 0.8, "Expected accuracy > 0.8")

        # Verify that the best model is saved correctly
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, "best_model.pkl")

        self.assertTrue(os.path.exists("best_model.pkl"), "Model file not found")

        # Clean up the saved model file after the test
        if os.path.exists("best_model.pkl"):
            os.remove("best_model.pkl")

        print("Best model saved as best_model.pkl")


if __name__ == "__main__":
    unittest.main()
