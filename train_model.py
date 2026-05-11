import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('job_salary_prediction_dataset.csv')
    
    # Define features and target
    target = 'salary'
    categorical_features = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']
    numerical_features = ['experience_years', 'skills_count', 'certifications']
    
    X = df[categorical_features + numerical_features]
    y = df[target]
    
    # Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough'
    )
    
    # We use a RandomForestRegressor, but limit n_estimators for faster training
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=42))
    ])
    
    # Train
    print("Training Random Forest model (this might take a minute)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Save the pipeline
    print("Saving model to model.pkl...")
    joblib.dump(pipeline, 'model.pkl')
    print("Model saved. Ready for Streamlit!")

if __name__ == '__main__':
    main()
