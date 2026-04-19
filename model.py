import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def main():
    print("Loading data...")
    df = pd.read_csv('StudentsPerformance.csv')

    # Features and Target
    X = df.drop('math score', axis=1)
    y = df['math score']

    # Define categorical and numerical features
    num_features = ['reading score', 'writing score']
    nominal_features = ['gender', 'race/ethnicity']
    ordinal_features = ['parental level of education', 'lunch', 'test preparation course']

    # Define ordinal categories
    education_order = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
    lunch_order = ['free/reduced', 'standard']
    test_prep_order = ['none', 'completed']

    # Preprocessing pipelines
    num_transformer = StandardScaler()
    nominal_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    ordinal_transformer = OrdinalEncoder(categories=[education_order, lunch_order, test_prep_order])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

    # Define the Random Forest Regressor Model
    rf_model = RandomForestRegressor(random_state=42)

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])

    # Split the data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # GridSearchCV to find the best hyperparameters
    print("Training model with GridSearchCV...")
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Evaluate the best pipeline
    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance -> MSE: {mse:.2f}, R2 Score: {r2:.4f}")

    # Extract feature names to save for the dashboard along with the model
    # Num Features
    feature_names = num_features.copy()
    
    # Nominal Features (OneHot)
    # Get the fitted OneHotEncoder
    ohe = best_pipeline.named_steps['preprocessor'].transformers_[1][1]
    nom_names = list(ohe.get_feature_names_out(nominal_features))
    feature_names.extend(nom_names)
    
    # Ordinal Features
    feature_names.extend(ordinal_features)

    # Dictionary to export and use in App
    export_data = {
        'pipeline': best_pipeline,
        'feature_names': feature_names,
        'y_test': y_test.values,
        'y_pred': y_pred
    }

    print("Exporting the trained model pipeline to rf_model_pipeline.pkl...")
    with open('rf_model_pipeline.pkl', 'wb') as f:
        pickle.dump(export_data, f)
        
    print("Optimization and Export complete!")

if __name__ == '__main__':
    main()
