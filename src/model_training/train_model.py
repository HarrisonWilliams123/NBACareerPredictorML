import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

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

    #Pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            multi_class="multinomial",
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    #Hyperparameter grid
    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        "model__solver": ["lbfgs", "newton-cg"]
    }

    #Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    #GridSearchCV
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    #Training with grid search
    grid.fit(X_train, y_train)

    #Best model
    clf = grid.best_estimator_

    #Evaluate
    y_pred = clf.predict(X_val)
    print("Best Params:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Macro F1:", f1_score(y_val, y_pred, average="macro"))
    print(classification_report(y_val, y_pred))

    #Save model and label map
    joblib.dump(clf, "models/nba_outcome_logreg2.pkl")
    joblib.dump(label_map, "models/label_logreg2.pkl")

    print("Model was saved.")

if __name__ == "__main__":
    train_model()