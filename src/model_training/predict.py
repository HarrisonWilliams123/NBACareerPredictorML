import joblib
import pandas as pd

from load_data import load_prospects
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def predict_prospects():
    #Loads model and label map
    clf = joblib.load("models/nba_outcome_model3.pkl")
    label_map = joblib.load("models/label_map3.pkl")
    inv_label_map = {v: k for k, v in label_map.items()}

    #Load Prospects
    prospects = load_prospects()

    #Extract Features
    X = prospects[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    #Predict 
    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    #Add predictions to dataframe
    prospects["PredictedOutcomeClass"] = [inv_label_map[p] for p in preds]
    prospects["P_Rotational"] = probs[:, 0]
    prospects["P_Starter"] = probs[:, 1]
    prospects["P_AllStar"] = probs[:, 2]
    prospects["P_Superstar"] = probs[:, 3]

    return prospects

if __name__ == "__main__":
    df = predict_prospects()
    df.to_csv("prospect_predictions3.csv", index=False)
