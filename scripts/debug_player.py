import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app
print('df player sample:', app.df['player'].astype(str).unique()[:10])
print('\nCalling player_summary("Laz")')
print(app.player_summary('Laz'))
print('\nCalling player_summary("TenZ")')
print(app.player_summary('TenZ'))
print('\n/api/search("La") via app test client')
with app.app.test_client() as c:
    r=c.get('/api/search?q=La')
    print('status',r.status_code,'data',r.get_data(as_text=True))
    r2=c.get('/api/player?name=Laz')
    print('/api/player status',r2.status_code,'data',r2.get_data(as_text=True))
