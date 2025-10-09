import pandas as pd
p='player_stats.csv'
df=pd.read_csv(p)
print('COLUMNS:', df.columns.tolist()[:30])
print('\nSAMPLE PLAYERS:')
print(df['player'].astype(str).unique()[:50])
# check contains
q='la'
mask = df['player'].astype(str).str.lower().str.contains(q, na=False)
print('\nCOUNT matches for',q, mask.sum())
print(df.loc[mask, 'player'].unique()[:20])
