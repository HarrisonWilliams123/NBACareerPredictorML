import pandas as pd

cleaned_nba_college_data = pd.read_csv("data/NBAPlayerCollegeStats.csv")
raw_nba_college_data = pd.read_csv("data/NBAPlayerCollegeStats.csv")
progress_nba_data = pd.read_csv("data/ProgressNBACollegeStats.csv")

#Removes the previous college years, keeps the latest year
cleaned_nba_college_data = cleaned_nba_college_data.drop_duplicates(subset=['player'], keep='last')

#Removes the wrong nba players in the databases
names_to_remove = ['Donovan Mitchell', 'Kyle Anderson', 'Gary Harris', 'Isaiah Jackson', 'Cameron Johnson', 'Jalen Johnson', 'Isaac Jones', 'Tre Jones', 'A.J. Lawson', ' Isaiah Stevens', 'Brandon Williams', 'Grant Williams', ' Jaylin Williams', 'Jalen Smith']
rows_to_add = [726, 26, 361, 429, 471, 485, 515, 527, 596, 1004, 1136, 1141, 1155]

#Removes the wrong nba player data 
for player in names_to_remove:
    cleaned_nba_college_data = cleaned_nba_college_data[cleaned_nba_college_data['player'] != player]

#Adds the correct nba player data
df_list = []
for row in rows_to_add:
    df1 = raw_nba_college_data.iloc[[row]]
    df_list.append(df1)

cleaned_new_players = pd.concat(df_list, ignore_index=True)
cleaned_nba_college_data = pd.concat([cleaned_nba_college_data, cleaned_new_players], ignore_index=True)

#Moves around the columns to match the nba prospects data
cleaned_nba_college_data = cleaned_nba_college_data[['player', 'team', 'conf', 'exp', 'pos', 'g', 'mpg', 'fgm', 'fga', 
                                                     'fg_pct', 'three_m', 'three_a', 'three_pct', 'two_m', 'two_a', 'two_pct', 
                                                     'efg', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'rpg', 'apg', 'spg', 'bpg', 'tov', 
                                                     'pfr', 'ppg', 'ortg', 'drtg', 'ts', 'ftr', 'oreb_rate', 'dreb_rate', 'ast', 'stl', 
                                                     'blk', 'to', 'usg', 'obpm', 'dbpm', 'bpm']]
cleaned_nba_college_data = cleaned_nba_college_data.rename(columns={"player" : "Name","team": "Team", "conf": "Conf", "pos": "Pos", "exp": "Class", 
                                                                    "g": "G", "mpg":"MP", "fgm":"FG", "fga":"FGA", "fg_pct":"FG%",  
                                                                    "three_m":"3P", "three_a":"3PA", "three_pct":"3P%", "two_m":"2P", 
                                                                    "two_a":"2PA", "two_pct":"2P%", "efg":"eFG%", "ftm": "FT", "fta":"FTA", 
                                                                    "ft_pct":"FT%", "oreb":"ORB", "dreb":"DRB", "rpg":"TRB", "apg":"AST", 
                                                                    "spg":"STL", "bpg":"BLK", "tov":"TOV", "pfr":"PF", "ppg":"PTS", "ortg":"ORTG", 
                                                                    "drtg":"DRTG", "ts":"TS%", "ftr":"FTr", "oreb_rate":"ORB%", "dreb_rate":"DRB%",
                                                                      "ast":"AST%", "stl":"STL%", "blk":"BLK%", "to":"TOV%", "usg":"USG%", 
                                                                      "obpm":"OBPM", "dbpm":"DBPM", "bpm": "BPM"})

#Converted to the right information of Minutes Played
cleaned_nba_college_data['MP'] = cleaned_nba_college_data.apply(lambda row: round(row['MP'] * row['G']), axis=1)

#Converts the fg and fga to the correct format
cleaned_nba_college_data['FG'] = cleaned_nba_college_data.apply(lambda row: round(row['FG'] / row['G'], 1), axis=1)
cleaned_nba_college_data['FGA'] = cleaned_nba_college_data.apply(lambda row: round(row['FGA'] / row['G'], 1), axis=1)

#Round the field goal percentages to the correct value
cleaned_nba_college_data['FG%'] = cleaned_nba_college_data['FG%'].round(3)

#Converts the 3P,3PA,2P, and 2PA to the correct format
cleaned_nba_college_data['3P'] = cleaned_nba_college_data.apply(lambda row: round(row['3P'] / row['G'], 1), axis=1)
cleaned_nba_college_data['3PA'] = cleaned_nba_college_data.apply(lambda row: round(row['3PA'] / row['G'], 1), axis=1)
cleaned_nba_college_data['2P'] = cleaned_nba_college_data.apply(lambda row: round(row['2P'] / row['G'], 1), axis=1)
cleaned_nba_college_data['2PA'] = cleaned_nba_college_data.apply(lambda row: round(row['2PA'] / row['G'], 1), axis=1)

#Converts the right eFG% value
cleaned_nba_college_data['eFG%'] = cleaned_nba_college_data.apply(lambda row: round(row['eFG%'] / 100, 3), axis=1)

#Converts the right FT & FTA values
cleaned_nba_college_data['FT'] = cleaned_nba_college_data.apply(lambda row: round(row['FT'] / row['G'], 1), axis=1)
cleaned_nba_college_data['FTA'] = cleaned_nba_college_data.apply(lambda row: round(row['FTA'] / row['G'], 1), axis=1)

#Rounds the values here, to match the format of the other data
cleaned_nba_college_data['ORB'] = cleaned_nba_college_data['ORB'].round(1)
cleaned_nba_college_data['DRB'] = cleaned_nba_college_data['DRB'].round(1)
cleaned_nba_college_data['TRB'] = cleaned_nba_college_data['TRB'].round(1)
cleaned_nba_college_data['AST'] = cleaned_nba_college_data['AST'].round(1)
cleaned_nba_college_data['STL'] = cleaned_nba_college_data['STL'].round(1)
cleaned_nba_college_data['BLK'] = cleaned_nba_college_data['BLK'].round(1)
cleaned_nba_college_data['TOV'] = cleaned_nba_college_data['TOV'].round(1)
cleaned_nba_college_data['PTS'] = cleaned_nba_college_data['PTS'].round(1)
cleaned_nba_college_data['ORTG'] = cleaned_nba_college_data['ORTG'].round(1)
cleaned_nba_college_data['DRTG'] = cleaned_nba_college_data['DRTG'].round(1)

#Converts to the right format for TS% & FTr
cleaned_nba_college_data['TS%'] = cleaned_nba_college_data.apply(lambda row: round(row['TS%'] / 100, 3), axis=1)
cleaned_nba_college_data['FTr'] = cleaned_nba_college_data.apply(lambda row: round(row['FTr'] / 100, 3), axis=1)

#Converts to the right format for OBPM, DBPM, & BPM
cleaned_nba_college_data['OBPM'] = cleaned_nba_college_data['OBPM'].round(1)
cleaned_nba_college_data['DBPM'] = cleaned_nba_college_data['DBPM'].round(1)
cleaned_nba_college_data['BPM'] = cleaned_nba_college_data['BPM'].round(1)

#Capitalizes the Class of the player's college data
cleaned_nba_college_data['Class'] = cleaned_nba_college_data['Class'].str.upper()

#Replaces specific examples for position with Guard, Forward or Center
cleaned_nba_college_data['Pos'] = cleaned_nba_college_data['Pos'].replace({'Combo G': 'G', 'Scoring PG': 'G', 'Pure PG': 'G'})
cleaned_nba_college_data['Pos'] = cleaned_nba_college_data['Pos'].replace({'Wing F' : 'F', 'Stretch 4' : 'F', 'Wing G' : 'F'})
cleaned_nba_college_data['Pos'] = cleaned_nba_college_data['Pos'].replace({'PF/C' : 'C'})

#Removes the extra duplicates that were not caught
progress_nba_data = progress_nba_data.drop_duplicates(subset=['Name'], keep='last')
#Removes an inactive player
progress_nba_data = progress_nba_data[progress_nba_data['Name'] != 'Jalen Smith']

#Merges to create the final spreadsheet
cleaned_nba_college_data = cleaned_nba_college_data.drop(columns= "Pos") 
unique_columns = ["Name"] + ["Pos"] + list(progress_nba_data.columns.difference(cleaned_nba_college_data.columns))
final_df = pd.merge(cleaned_nba_college_data, progress_nba_data[unique_columns], on='Name', how ='left')

#Gets rid of the duplicate players
final_df = final_df.drop_duplicates(subset=['Name'], keep='last')

#Rearranges the columns to match the nba prospects sheet
final_df = final_df[['Name', 'Team', 'Conf', 'Class', 'Pos', 'G', 'MP', 'FG', 'FGA', 
                                                     'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
                                                     'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 
                                                     'PF', 'PTS', 'ORTG', 'DRTG', 'PER ', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
                                                     'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/40', 'OBPM',
                                                     'DBPM', 'BPM']]

#Gets rid of the duplicate columns
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
#Writes the csv file
final_df.to_csv("data/testNBACollegeStats.csv", index=False)




