import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("data/NBAInternationalStats.csv")

# --- Helper Baseline Constants for International/FIBA Play ---
PACE_40 = 72.0       # Est. team possessions per 40 mins
LG_ORTG = 105.0      # Est. league average points per 100 possessions
LG_PTS_PER_FGM = 2.0 # Standard normalization factor

#Calculating the Per 36 Stats as Per 40 
df['MP'] = df['MP'].apply(lambda x: round(x * 40/36, 1))
df['FG'] = df['FG'].apply(lambda x: round(x * 40/36, 1))
df['FGA'] = df['FGA'].apply(lambda x: round(x * 40/36, 1))
df['3P'] = df['3P'].apply(lambda x: round(x * 40/36, 1))
df['3PA'] = df['3PA'].apply(lambda x: round(x * 40/36, 1))
df['2P'] = df['2P'].apply(lambda x: round(x * 40/36, 1))
df['2PA'] = df['2PA'].apply(lambda x: round(x * 40/36, 1))
df['FT'] = df['FT'].apply(lambda x: round(x * 40/36, 1))
df['FTA'] = df['FTA'].apply(lambda x: round(x * 40/36, 1))
df['ORB'] = df['ORB'].apply(lambda x: round(x * 40/36, 1))
df['DRB'] = df['DRB'].apply(lambda x: round(x * 40/36, 1))
df['TRB'] = df['TRB'].apply(lambda x: round(x * 40/36, 1))
df['AST'] = df['AST'].apply(lambda x: round(x * 40/36, 1))
df['STL'] = df['STL'].apply(lambda x: round(x * 40/36, 1))
df['BLK'] = df['BLK'].apply(lambda x: round(x * 40/36, 1))
df['TOV'] = df['TOV'].apply(lambda x: round(x * 40/36, 1))
df['PF'] = df['PF'].apply(lambda x: round(x * 40/36, 1))
df['PTS'] = df['PTS'].apply(lambda x: round(x * 40/36, 1))

# Pre-calculating total minutes per player based on games if not clean
df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
df['G'] = pd.to_numeric(df['G'], errors='coerce')

# 1. True Shooting & Attempts Rates
df['TS%'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
df['3PAr'] = df['3PA'] / np.where(df['FGA'] == 0, 1, df['FGA'])
df['FTr'] = df['FTA'] / np.where(df['FGA'] == 0, 1, df['FGA'])

# 2. Possession & Usage Estimations
# Est. individual possessions used
df['Est_Poss'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
# Usage% = Percentage of team plays used by the player while on the floor
df['USG%'] = (df['Est_Poss'] / (PACE_40 * (df['MP'] / (df['G'] * 40)))) * 100

# 3. Advanced Rebound, Assist, Steal, Block Percentages
# Approximating available rebounds based on baseline pacing split constants
df['ORB%'] = (df['ORB'] / (0.40 * PACE_40 * (df['MP'] / (df['G'] * 40)))) * 100
df['DRB%'] = (df['DRB'] / (0.60 * PACE_40 * (df['MP'] / (df['G'] * 40)))) * 100
df['TRB%'] = (df['TRB'] / (PACE_40 * (df['MP'] / (df['G'] * 40)))) * 100

df['AST%'] = (df['AST'] / (0.60 * df['FG'] + 1)) * 100 # Estimated assisted field goals
df['STL%'] = (df['STL'] / (PACE_40 * (df['MP'] / (df['G'] * 40)))) * 100
df['BLK%'] = (df['BLK'] / (0.45 * df['FGA'] + 1)) * 100 # Estimated opponent rim attempts

# 4. Ratings (ORTG & DRTG)
df['ORTG'] = (df['PTS'] / np.where(df['Est_Poss'] == 0, 1, df['Est_Poss'])) * 100
# Defensive Rating estimated via proxy of individual defensive interventions
df['DRTG'] = LG_ORTG - (df['STL'] * 2.5) - (df['BLK'] * 1.8) + (df['PF'] * 0.5)

# 5. Box Plus-Minus (OBPM, DBPM, BPM) via Multi-Variable Regression Proxies
df['OBPM'] = (df['PTS'] * 0.15) + (df['AST'] * 0.12) - (df['TOV'] * 0.20) + (df['3PAr'] * 1.5) - 2.0
df['DBPM'] = (df['DRB%'] * 0.08) + (df['STL'] * 0.6) + (df['BLK'] * 0.4) - 1.5
df['BPM'] = df['OBPM'] + df['DBPM']

# 6. PER (Player Efficiency Rating) - Normalized to baseline 15.00
raw_per = (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK'] 
           - (df['FGA'] - df['FG']) - (df['FTA'] - df['FT']) - df['TOV'])
df['PER '] = (raw_per / np.where(df['MP'] == 0, 1, df['MP'])) * 30 + 10

# 7. Win Shares (OWS, DWS, WS, WS/40)
df['OWS'] = (df['PTS'] - (df['Est_Poss'] * 0.95)) * 0.04
df['DWS'] = ((df['MP'] / 40) * 0.1) + (df['STL'] * 0.05) + (df['BLK'] * 0.05)
df['WS'] = df['OWS'] + df['DWS']
df['WS/40'] = df['WS'] / np.where(df['MP'] == 0, 1, df['MP']) * 40

# Clean up infinite numbers or NaNs from low minute anomalies
df = df.replace([np.inf, -np.inf], np.nan).fillna(0).round(3)

# Save to match the schema format of FinalNBACollegeStats.csv
df.to_csv("data/FinalNBAInternationalStats.csv", index=False)

