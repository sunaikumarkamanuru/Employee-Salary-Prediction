import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Generate Synthetic Data
# (Assuming no data was provided, we create a dataset to train and demonstrate)
np.random.seed(42)
n_samples = 1000

# Experience (Years)
experience = np.random.randint(1, 26, n_samples)

# Education Level
education_levels = ['Bachelors', 'Masters', 'PhD']
education = np.random.choice(education_levels, n_samples, p=[0.5, 0.35, 0.15])

# Role
roles = ['Developer', 'Data Scientist', 'Manager', 'Director']
role = np.random.choice(roles, n_samples, p=[0.4, 0.3, 0.2, 0.1])

# Base salary and multipliers for synthetic salary generation
base_salary = 50000
edu_multiplier = {'Bachelors': 1.0, 'Masters': 1.2, 'PhD': 1.5}
role_multiplier = {'Developer': 1.0, 'Data Scientist': 1.3, 'Manager': 1.6, 'Director': 2.2}

# Calculate Salary with some random noise
salary = []
for exp, edu, r in zip(experience, education, role):
    # Base + Exp bonus + Edu multiplier + Role multiplier + noise
    calculated_salary = (base_salary + (exp * 2500)) * edu_multiplier[edu] * role_multiplier[r]
    noise = np.random.normal(0, 5000)
    salary.append(calculated_salary + noise)

# Create DataFrame
df = pd.DataFrame({
    'Experience': experience,
    'Education': education,
    'Role': role,
    'Salary': salary
})

print("First 5 rows of the generated dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 2: Data Preprocessing
# We need to encode the categorical features ('Education' and 'Role')
categorical_features = ['Education', 'Role']
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# Numeric features pass through as they are
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough'
)

# Step 3: Define Models
# We'll compare a simple Linear Regression and a Random Forest Regressor
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 4: Train-Test Split
X = df.drop('Salary', axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train & Evaluate Models
print("\n--- Training Linear Regression ---")
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
print(f"Linear Regression MAE: ${mean_absolute_error(y_test, lr_pred):,.2f}")
print(f"Linear Regression R2 Score: {r2_score(y_test, lr_pred):.4f}")

print("\n--- Training Random Forest ---")
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
print(f"Random Forest MAE: ${mean_absolute_error(y_test, rf_pred):,.2f}")
print(f"Random Forest R2 Score: {r2_score(y_test, rf_pred):.4f}")

# Step 6: Sample Predictions
print("\n--- Example Predictions (Using Random Forest) ---")
# Let's predict salary for a few new profiles
new_data = pd.DataFrame({
    'Experience': [3, 10, 15],
    'Education': ['Bachelors', 'Masters', 'PhD'],
    'Role': ['Developer', 'Data Scientist', 'Director']
})

predictions = rf_pipeline.predict(new_data)
for i, row in new_data.iterrows():
    print(f"Profile: {row['Experience']} yrs Experience, {row['Education']}, {row['Role']} -> Predicted Salary: ${predictions[i]:,.2f}")
