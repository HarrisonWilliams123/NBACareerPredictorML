import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb

from load_data import load_outcomes, load_college, build_training_set, label_encode
from features import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES

#Includes random seed to ensure reproducibility
np.random.seed(42)

def train_model():
    #Load data 
    outcomes = load_outcomes()
    college = load_college()

    #Build training set
    train_df = build_training_set(college, outcomes)
    train_df, label_map = label_encode(train_df)


    #Features and target
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    y = train_df["label"]

    #Preprocessor
    preprocessor = build_preprocessor()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        random_state=42,
        class_weight="balanced",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=50,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8
    )


    #Pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    #Hyperparameter grid
    #param_grid = {
        #"model__C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        #"model__solver": ["lbfgs", "newton-cg"]
    #}

    #Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    #GridSearchCV
    #grid = GridSearchCV(
        #estimator=pipeline,
        #param_grid=param_grid,
        #cv=5,
        #scoring="f1_macro",
        #n_jobs=-1,
        #verbose=2
    #)

    #Training with grid search
    pipeline.fit(X_train, y_train)

    #Best model
    #clf = grid.best_estimator_

    #Evaluate
    y_pred = pipeline.predict(X_val)
    #print("Best Params:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Macro F1:", f1_score(y_val, y_pred, average="macro"))
    print(classification_report(y_val, y_pred))

    #Save model and label map
    joblib.dump(pipeline, "models/nba_outcome_lightgbm.pkl")
    joblib.dump(label_map, "models/label_lightgbm.pkl")

    print("Model was saved.")

if __name__ == "__main__":
    train_model()