import pandas as pd

college = pd.read_csv("data/FinalNBACollegeStats.csv")
college2 = pd.read_csv("data/FinalNBACollegeStats2.csv")
intl = pd.read_csv("data/FinalNBAInternationalStats.csv")

final_df = pd.concat([college, college2, intl], ignore_index=True)

final_df = final_df.drop(columns=['GS', 'PProd', 'League', 'Unnamed: 3', 'Est_Poss'])

final_df.loc[457:506, 'eFG%'] = final_df.loc[457:506].apply(lambda row: round(((row['FG'] * row['G']) + (0.5 * (row['3P'] * row['G']))) / (row['FGA'] * row['G']), 3), axis=1)
final_df['TOV%'] = final_df['TOV%'].fillna(final_df['TOV%'].mean())


final_df.to_csv("data/TotalNBACollegeStats.csv", index=False)
