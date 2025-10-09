import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

pairs = [
    ('TenZ','Aspas'),
    ('Laz','TENNN'),
    ('TenZ','TenZ'),
    ('Sayaplayer','Laz')
]
with app.test_client() as c:
    for p1,p2 in pairs:
        r = c.get(f'/api/compare?player1={p1}&player2={p2}')
        print(p1,p2,'status',r.status_code)
        print(r.get_data(as_text=True)[:400])
