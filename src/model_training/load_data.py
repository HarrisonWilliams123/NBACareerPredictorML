import pandas as pd

#The methods that loads the .csv files
def load_outcomes():
    return pd.read_csv("data/New_NBA_Outcomes.csv")

def load_college():
    df = pd.read_csv("data/ML_Ready_NBA_College_Stats.csv")

    df['MPG'] = df["MP"] / (df["G"] + 1)
    df['AST_TOV_Ratio'] = df["AST"] / (df["TOV"] + 1)
    df['NRTG'] = df["ORTG"] - df["DRTG"]
    df['3P_FG_Ratio'] = df["3PA"] / (df["FGA"] + 1)
    df['2P_FG_Ratio'] = df["2PA"] / (df["FGA"] + 1)
    df['FT_PTS_Ratio'] = df["FT"] / (df["PTS"] + 1)
    df['2P_PTS_Ratio'] = (df["2P"] * 2) / (df["PTS"] + 1)
    df['3P_PTS_Ratio'] = (df["3P"] * 3) / (df["PTS"] + 1)

    df['Poss'] = df["FGA"] + df["TOV"] + 0.44 * df["FTA"]

    df["PTS_per_poss"] = df["PTS"] / (df["Poss"] + 1)
    df["REB_per_poss"] = df["TRB"] / (df["Poss"] + 1)
    df["AST_per_poss"] = df["AST"] / (df["Poss"] + 1)
    df["STL_per_poss"] = df["STL"] / (df["Poss"] + 1)
    df["BLK_per_poss"] = df["BLK"] / (df["Poss"] + 1)

    df["TS_adj"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"] + 1))
    df["EFG_adj"] = (df["FG"] + 0.5 * df["3P"]) / (df["FGA"] + 1)
    df["Usage"] = (df["FGA"] + df["FTA"] + df["TOV"]) / (df["MP"] + 1)

    df["PTS_x_MPG"] = df["PTS"] * df["MPG"]
    df["Usage_x_TS"] = df["Usage"] * df["TS_adj"]
    df["AST_x_MPG"] = df["AST"] * df["MPG"]

    return df

def load_prospects():
    df = pd.read_csv("data/NBAProspects2026.csv")

    df['MPG'] = df["MP"] / (df["G"] + 1)
    df['AST_TOV_Ratio'] = df["AST"] / (df["TOV"] + 1)
    df['NRTG'] = df["ORTG"] - df["DRTG"]
    df['3P_FG_Ratio'] = df["3PA"] / (df["FGA"] + 1)
    df['2P_FG_Ratio'] = df["2PA"] / (df["FGA"] + 1)
    df['FT_PTS_Ratio'] = df["FT"] / (df["PTS"] + 1)
    df['2P_PTS_Ratio'] = (df["2P"] * 2) / (df["PTS"] + 1)
    df['3P_PTS_Ratio'] = (df["3P"] * 3) / (df["PTS"] + 1)

    df['Poss'] = df["FGA"] + df["TOV"] + 0.44 * df["FTA"]

    df["PTS_per_poss"] = df["PTS"] / (df["Poss"] + 1)
    df["REB_per_poss"] = df["TRB"] / (df["Poss"] + 1)
    df["AST_per_poss"] = df["AST"] / (df["Poss"] + 1)
    df["STL_per_poss"] = df["STL"] / (df["Poss"] + 1)
    df["BLK_per_poss"] = df["BLK"] / (df["Poss"] + 1)

    df["TS_adj"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"] + 1))
    df["EFG_adj"] = (df["FG"] + 0.5 * df["3P"]) / (df["FGA"] + 1)
    df["Usage"] = (df["FGA"] + df["FTA"] + df["TOV"]) / (df["MP"] + 1)

    df["PTS_x_MPG"] = df["PTS"] * df["MPG"]
    df["Usage_x_TS"] = df["Usage"] * df["TS_adj"]
    df["AST_x_MPG"] = df["AST"] * df["MPG"]

    return df

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

