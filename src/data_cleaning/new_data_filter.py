import pandas as pd

outcomes_df = pd.read_csv("data/FinalNBAOutcomes.csv")

#Filters irrelevant players or players that are too young
outcomes_df = outcomes_df[outcomes_df['Career_MP'] > 3000]

#outcomes_df.to_csv("data/New_NBA_Outcomes.csv", index=False)

#Reads the college stats dataframe
college_stats_df = pd.read_csv("data/FinishedTotalNBACollegeStats.csv")

#Filters the college stats dataframe
filtered_college_df = college_stats_df[college_stats_df["player_id"].isin(outcomes_df["player_id"])]

filtered_college_df.to_csv("data/ML_Ready_NBA_College_Stats.csv", index=False)
