import pandas as pd
import unicodedata
import re

df1 = pd.read_csv("data/NBAOutcomes_Unfiltered.csv")
df2 = pd.read_csv("data/NBAOutcomes_Filtered.csv")


rows = [3, 5242, 1103, 3866, 5367, 5103, 4219, 3885, 1680, 1853, 3052, 1266, 5370, 3090, 4060, 1185, 4426, 3456, 2196, 3971, 3972, 3050, 3240, 4924, 1344, 145, 3986, 2919, 3976]

for row in rows:
    df2 = pd.concat([df2, df1.iloc[[row]]], ignore_index=True)

df3 = pd.read_csv("data/TotalNBACollegeStats.csv")

df3 = df3[df3['Name'] != 'Tamar Bates']
df3 = df3[df3['Name'] != 'Thomas Sorber']


df2.rename(columns={'player': 'Name'}, inplace=True)


df3.loc[426, 'Name'] = 'Jimmy Butler'
df2.loc[234, 'Name'] = 'Jimmy Butler'
df2.loc[484, 'Name'] = 'Moussa Diabate'
df2.loc[485, 'Name'] = 'Egor Demin'
df2.loc[486, 'Name'] = 'GG Jackson'
df2.loc[487, 'Name'] = 'Kasparas Jakucionis'
df2.loc[488, 'Name'] = 'David Jones Garcia'
df2.loc[489, 'Name'] = 'Yanic Konan Niederhauser'
df2.loc[490, 'Name'] = 'Kelly Oubre'
df2.loc[491, 'Name'] = 'PJ Washington'
df2.loc[493, 'Name'] = 'Robert Williams III'
df2.loc[494, 'Name'] = 'Luka Doncic'
df2.loc[495, 'Name'] = 'Hugo Gonzalez'
df2.loc[496, 'Name'] = 'Nikola Jokic'
df2.loc[497, 'Name'] = 'Nikola Jovic'
df2.loc[498, 'Name'] = 'Karlo Matkovic'
df2.loc[499, 'Name'] = 'Kristaps Porzingis'
df2.loc[500, 'Name'] = 'Tidjane Salaun'
df2.loc[501, 'Name'] = 'Dennis Schroder'
df2.loc[502, 'Name'] = 'Alperen Sengun'
df2.loc[503, 'Name'] = 'Nolan Traore'
df2.loc[504, 'Name'] = 'Jonas Valanciunas'
df2.loc[505, 'Name'] = 'Nikola Vucevic'
df2.loc[492, 'Name'] = 'Dario Saric'

def clean(s):
    if pd.isna(s):
        return s
    s = str(s)

    # Normalize unicode
    s = unicodedata.normalize('NFKC', s)

    # Remove zero‑width, non‑breaking, and weird spaces
    s = re.sub(r'[\u200b\u200c\u200d\uFEFF\xa0]', '', s)

    # Strip normal whitespace
    s = s.strip()

    return s

df2['Name'] = df2['Name'].apply(clean)
df3['Name'] = df3['Name'].apply(clean)

df3 = pd.merge(df3, df2[['Name', 'player_id']], on='Name', how='right')

df3['Class'] = df3['Class'].fillna('FR')
df3['Conf'] = df3['Conf'].fillna('Intl')
df3['Pos'] = df3['Pos'].fillna('F')


df2.to_csv("data/FinalNBAOutcomes.csv", index=False)
df3.to_csv("data/FinishedTotalNBACollegeStats.csv", index=False)