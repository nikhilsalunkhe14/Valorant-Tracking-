import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

df = app.df
players_cols = [c for c in df.columns if c.lower() in ("player", "player_name", "playerid", "player_id")]
print('players_cols:', players_cols)
if players_cols:
    pname_col = players_cols[0]
else:
    pname_col = 'player'
print('pname_col:', pname_col)

for q in ['Laz','laz','La','TenZ','TENNN']:
    ql = q.strip().lower()
    eq_mask = df[pname_col].astype(str).str.lower() == ql
    cont_mask = df[pname_col].astype(str).str.lower().str.contains(ql, na=False)
    print('\nQuery:', q)
    print(' exact match count:', eq_mask.sum())
    print(' contains match count:', cont_mask.sum())
    print(' exact sample:', df.loc[eq_mask, pname_col].unique()[:10])
    print(' contain sample:', df.loc[cont_mask, pname_col].unique()[:10])
