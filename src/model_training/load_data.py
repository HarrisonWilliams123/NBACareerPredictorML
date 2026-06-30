import pandas as pd

#The methods that loads the .csv files
def load_outcomes():
    return pd.read_csv("data/New_NBA_Outcomes.csv")

def load_college():
    return pd.read_csv("data/Total_Filtered_NBA_College_Stats.csv")

def load_prospects():
    return pd.read_csv("data/NBAProspects2026.csv")

def build_training_set(college_df, outcomes_df):
    #Joins NBA Players' College Stats with their NBA Outcomes
    merged = college_df.merge(
        outcomes_df[["player_id", "OutcomeClass"]],
        on="player_id",
        how="inner"
    )
    return merged

def label_encode(df):
    label_map = {
        "Rotational": 0,
        "Starter": 1, 
        "All-Star": 2,
        "Superstar": 3
    }
    df["label"] = df["OutcomeClass"].map(label_map)
    return df, label_map

