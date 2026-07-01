import joblib
import pandas as pd

from load_data import load_prospects
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def predict_prospects():
    #Loads model and label map
    clf = joblib.load("models/nba_outcome_logreg.pkl")
    label_map = joblib.load("models/label_logreg.pkl")
    inv_label_map = {v: k for k, v in label_map.items()}

    #Load Prospects
    prospects = load_prospects()

    #Feature Engineering
    prospects['MPG'] = prospects["MP"] / prospects["G"]
    prospects['AST_TOV_Ratio'] = prospects["AST"] / (prospects["TOV"])
    prospects['NRTG'] = prospects["ORTG"] - prospects["DRTG"]
    prospects['3P_FG_Ratio'] = prospects["3PA"] / prospects["FGA"]
    prospects['2P_FG_Ratio'] = prospects["2PA"] / prospects["FGA"]
    prospects['FT_PTS_Ratio'] = prospects["FT"] / prospects["PTS"]
    prospects['2P_PTS_Ratio'] = (prospects["2P"] * 2) / prospects["PTS"]
    prospects['3P_PTS_Ratio'] = (prospects["3P"] * 3) / prospects["PTS"]

    ENGINEERED_FEATURES = [
        "MPG", "AST_TOV_Ratio", "NRTG", "3P_FG_Ratio", "2P_FG_Ratio", "FT_PTS_Ratio", "2P_PTS_Ratio", "3P_PTS_Ratio"
    ]

    #Extract Features
    X = prospects[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]

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
    df.to_csv("prospect_predictions4.csv", index=False)
