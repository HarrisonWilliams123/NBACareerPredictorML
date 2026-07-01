from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Numeric Features in the dataset that are included in both datasets
NUMERIC_FEATURES = [
    "G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
    "FT", "FTA", "FT%", "ORB",
    "DRB", "TRB", "AST", "STL", "BLK",
    "TOV", "PF", "PTS", "ORTG", "DRTG", "PER", "TS%",
    "3PAr", "FTr", "OWS", "DWS", "WS/40"
]

CATEGORICAL_FEATURES = ["Pos", "Conf", "Class"]

ENGINEERED_FEATURES = [
    "MPG", "AST_TOV_Ratio", "NRTG",
    "3P_FG_Ratio", "2P_FG_Ratio", "FT_PTS_Ratio",
    "2P_PTS_Ratio", "3P_PTS_Ratio",
    "PTS_per_poss", "REB_per_poss", "AST_per_poss",
    "STL_per_poss", "BLK_per_poss",
    "TS_adj", "EFG_adj", "Usage",
    "PTS_x_MPG", "Usage_x_TS", "AST_x_MPG"
]

#Returns a Column Transformer that scales numeric features and one-hot encodes categorical features
def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor
