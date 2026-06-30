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

#Returns a Column Transformer that scales numeric features and one-hot encodes categorical features
def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor
