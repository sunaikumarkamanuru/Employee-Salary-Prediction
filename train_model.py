import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def main():
    print("Loading dataset...")

    df = pd.read_csv("job_salary_prediction_dataset.csv")

    target = "salary"

    categorical_features = [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work"
    ]

    numerical_features = [
        "experience_years",
        "skills_count",
        "certifications"
    ]

    X = df[categorical_features + numerical_features]
    y = df[target]

    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Creating preprocessor...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    print("Creating model...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=5,
                    max_depth=10,
                    min_samples_leaf=5,
                    n_jobs=1,
                    random_state=42
                )
            )
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    print("\nSaving model...")

    joblib.dump(
        pipeline,
        "model.pkl",
        compress=3
    )

    model_size_mb = os.path.getsize("model.pkl") / (1024 * 1024)

    print(f"Model size: {model_size_mb:.2f} MB")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()