import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app as appmod
print('app module loaded, df columns:', list(appmod.df.columns)[:30])
print('sample players from app.df:', appmod.df['player'].astype(str).unique()[:10])
print('contains la count via app.df:', appmod.df['player'].astype(str).str.lower().str.contains('la').sum())
