import pandas as pd

#Load Files
adv = pd.read_csv("data/Advanced.csv")
allstar = pd.read_csv("data/All-Star Selections.csv")
awards = pd.read_csv("data/Player Award Shares.csv")

#Career Totals
adv_totals = (
    adv.groupby(["player", "player_id"]).agg(Career_MP=("mp", "sum"), Career_WS=("ws", "sum"), Career_VORP=("vorp", "sum")).reset_index()
)

#All-Star Selections
allstar_counts = (
    allstar.groupby("player_id").size().reset_index(name="All_Star_Selections")
)

#All-NBA Selections
mvp = awards[awards["award"] == "nba mvp"]
mvp = mvp[mvp["pts_won"] > 0]

allnba = (
    mvp.groupby("player_id")["season"].nunique().reset_index(name="All_NBA_Selections")
)

#Merging Everything
df = adv_totals.merge(allstar_counts, on="player_id", how="left")
df = df.merge(allnba, on="player_id", how="left")

df["All_Star_Selections"] = df["All_Star_Selections"].fillna(0).astype(int)
df["All_NBA_Selections"] = df["All_NBA_Selections"].fillna(0).astype(int)

#Label Players
def label(row):
    ws = row.Career_WS
    mp = row.Career_MP
    allstar = row.All_Star_Selections
    allnba = row.All_NBA_Selections

    if allnba >= 3 or ws >= 100:
        return "Superstar"
    if allstar >= 1 or allnba >= 1 or ws >= 50:
        return "All-Star"
    if mp >= 10000 or ws >= 20:
        return "Starter"
    return "Rotational"

df["OutcomeClass"] = df.apply(label, axis=1)

df.to_csv("data/NBAOutcomes_Unfiltered.csv", index=False)

name_df = pd.read_csv("data/TotalNBACollegeStats.csv")
df['player'] = df['player'].str.strip()
name_df['Name'] = name_df['Name'].str.strip()


filtered_df = df[df['player'].isin(name_df['Name'])]

#Save
filtered_df.to_csv("data/NBAOutcomes_Filtered.csv", index=False)