import pandas as pd

df = pd.read_csv("data/ML_Ready_NBA_College_Stats.csv")

group_0 = ['Intl']
group_1 = ['BW', 'BE', 'WCC', 'MVC', 'A-10', 'Patriot', 'SB', 'MWC', 'CUSA', 
      'Amer', 'MAC', 'BSky', 'BSth', 'OVC', 'Horz', 'Big East', 'Sum', 
      'Slnd', 'A10', 'WAC', 'Pat', 'SC', 'Ivy', 'AAC']
group_2 = ['Other', 'Other ']
group_3 = ['B12', 'P12', 'B10', 'ACC', 'SEC', 'Big Ten', 'Big 12', 'P10']

mapping = {c:0 for c in group_0} | \
          {c:1 for c in group_1} | \
          {c:2 for c in group_2} | \
          {c:3 for c in group_3}

df['Conf'] = df['Conf'].replace(mapping)

df.loc[6, 'Pos'] = 'C'
df.loc[20, 'Pos'] = 'G'
df.loc[21, 'Pos'] = 'C'
df.loc[46, 'Pos'] = 'C'
df.loc[83, 'Pos'] = 'G'
df.loc[95, 'Pos'] = 'C'
df.loc[105, 'Pos'] = 'C'
df.loc[192, 'Pos'] = 'G'
df.loc[244, 'Pos'] = 'G'
df.loc[246, 'Pos'] = 'C'
df.loc[262, 'Pos'] = 'G'
df.loc[285, 'Pos'] = 'C'
df.loc[306, 'Pos'] = 'G'
df.loc[307, 'Pos'] = 'C'
df.loc[309, 'Pos'] = 'C'
df.loc[310, 'Pos'] = 'G'
df.loc[311, 'Pos'] = 'C'
df.loc[312, 'Pos'] = 'C'
df.loc[313, 'Pos'] = 'C'

df['Pos'] = df['Pos'].replace({'G': 0, 'F': 1, 'C': 2})

df.loc[23, 'Class'] = 'JR'
df.loc[46, 'Class'] = 'SO'
df.loc[99, 'Class'] = 'SO'
df.loc[147, 'Class'] = 'JR'
df.loc[211, 'Class'] = 'SR'
df.loc[246, 'Class'] = 'SO'
df.loc[262, 'Class'] = 'SO'
df.loc[304, 'Class'] = 'SO'
df.loc[313, 'Class'] = 'SO'

df['Class'] = df['Class'].replace({'FR': 0, 'SO': 1, 'JR': 2, 'SR': 3})

df.to_csv("data/EDA_Ready_NBA_College_Stats.csv", index=False)