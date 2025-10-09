import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
c = app.test_client()
for q in ['La','Ten','TENNN','Laz']:
    r = c.get(f'/api/search?q={q}')
    print('SEARCH',q,'status',r.status_code,'->', r.get_data(as_text=True)[:200])

r = c.get('/api/player?name=Laz')
print('\nPLAYER Laz status', r.status_code)
print(r.get_data(as_text=True)[:1000])
