import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load features and labels
X = np.load("data/features.npy")
y = np.load("data/labels.npy")

# Train-test split (stratified = balanced classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ML pipeline: Scaling + SVM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# Hyperparameter search space
param_grid = {
    "svm__kernel": ["rbf"],
    "svm__C": [1, 10, 50, 100],
    "svm__gamma": ["scale", 0.01, 0.001]
}

print("🔍 Searching for best model parameters...")

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train model
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

print("✅ Best parameters found:")
print(grid.best_params_)

# Test performance
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Final Accuracy:", round(accuracy * 100, 2), "%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/intent_model.pkl")

print("💾 Model saved as models/intent_model.pkl")
