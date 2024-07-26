import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Corrected file path
file_path ="C:/Users/KIIT/Desktop/powerbi_projects/ml1/financial_risk_assessment.csv" 
data = pd.read_csv(file_path)

# Drop the 'Marital Status Change' column if it exists
if 'Marital Status Change' in data.columns:
    data = data.drop(columns=['Marital Status Change'])

# Print columns in the dataset to verify
print("Columns in dataset:", data.columns)

# Separate features and target variables
target_columns = ['Risk Rating', 'Loan Approval', 'Credit Score', 'Income Level', 'Loan Default']

# Check which target columns are present in the dataset
available_target_columns = [col for col in target_columns if col in data.columns]

# Handle missing target values only for available target columns
data = data.dropna(subset=available_target_columns)

# Prepare features
X = data.drop(columns=available_target_columns)

# Prepare targets
y_targets = {target: data[target] for target in available_target_columns}

# Explicitly list numerical and categorical features
categorical_features = ['Gender', 'Education Level', 'Marital Status', 'Loan Purpose', 'Employment Status', 'Payment History']
numerical_features = [col for col in X.columns if col not in categorical_features and pd.api.types.is_numeric_dtype(X[col])]

# Encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Train models
models = {}
for target, y in y_targets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if target in ['Risk Rating', 'Loan Approval', 'Loan Default']:
        model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    else:
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])
    
    model.fit(X_train, y_train)
    models[target] = model
    
    # Save the model using pickle
    with open(f'model_{target.lower().replace(" ", "_")}.pkl', 'wb') as file:
        pickle.dump(model, file)

print("Models trained and saved successfully.")
