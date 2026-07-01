import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

from load_data import load_outcomes, load_college, build_training_set, label_encode
from features import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

np.random.seed(42)

def evaluate_models():
    outcomes = load_outcomes()
    college = load_college()

    train_df = build_training_set(college, outcomes)
    train_df, label_map = label_encode(train_df)

    #Feature_Engineering
    train_df['MPG'] = train_df["MP"] / train_df["G"]
    train_df['AST_TOV_Ratio'] = train_df["AST"] / (train_df["TOV"])
    train_df['NRTG'] = train_df["ORTG"] - train_df["DRTG"]
    train_df['3P_FG_Ratio'] = train_df["3PA"] / train_df["FGA"]
    train_df['2P_FG_Ratio'] = train_df["2PA"] / train_df["FGA"]
    train_df['FT_PTS_Ratio'] = train_df["FT"] / train_df["PTS"]
    train_df['2P_PTS_Ratio'] = (train_df["2P"] * 2) / train_df["PTS"]
    train_df['3P_PTS_Ratio'] = (train_df["3P"] * 3) / train_df["PTS"]

    ENGINEERED_FEATURES = [
        "MPG", "AST_TOV_Ratio", "NRTG", "3P_FG_Ratio", "2P_FG_Ratio", "FT_PTS_Ratio", "2P_PTS_Ratio", "3P_PTS_Ratio"
    ]

    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    y = train_df['label']

    preprocessor = build_preprocessor()

    models = {
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            multi_class="multinomial",
            max_iter=500,
            random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=5,
            scoring="f1_macro",
            n_jobs=1
        )

        results.append((name, np.mean(scores)))
    
    print(results)

if __name__ == "__main__":
    evaluate_models()