import pandas as pd

df=pd.read_csv('player_stats.csv')
vals = df['player'].astype(str)
print('First 20 reprs:')
for v in vals.unique()[:20]:
    print(repr(v), 'lower->', repr(v.lower()))

print('\nCheck equality examples:')
print('Laz exact matches count:', (vals.str.lower()=='laz').sum())
print('Laz contains count:', vals.str.lower().str.contains('laz', na=False).sum())
print('\nSample rows where contains la:')
print(df[vals.str.lower().str.contains('la', na=False)][['player']].head(10).to_string())
