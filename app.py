from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from flask import render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib
# Use a non-GUI backend so Flask can save figures without opening a display (prevents tkinter errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
# no additional helper imports required

app = Flask(__name__)

# Change this path if your CSV is elsewhere
CSV_PATH = "player_stats.csv"

# Load dataset once at startup for responsiveness
df = pd.read_csv(CSV_PATH)

# Normalize column names (lowercase) for safety
df.columns = [c.strip() for c in df.columns]


def find_player_column():
    """Return the best candidate column name for player names.
    Prefer exact 'player', then 'player_name', then other player-like columns.
    """
    cols_map = {c.lower(): c for c in df.columns}
    # preferred ordering
    for key in ('player', 'player_name'):
        if key in cols_map:
            return cols_map[key]
    # otherwise pick any column containing 'player'
    for c in df.columns:
        if 'player' in c.lower():
            return c
    # fallback to first column
    return df.columns[0]

# Helper functions
def player_summary(player_name):
    # Choose the best player-name column (prefer 'player' over 'player_id')
    pname_col = find_player_column()
    
    # CASE-INSENSITIVE search by 'player' column
    player_df = df[df[pname_col].astype(str).str.lower() == str(player_name).lower()]
    
    if player_df.empty:
        # attempt partial match
        player_df = df[df[pname_col].astype(str).str.lower().str.contains(str(player_name).lower(), na=False)]
    
    if player_df.empty:
        return None
    
    def to_numeric_series(s):
        return pd.to_numeric(s.astype(str).str.replace('\xa0','', regex=False), errors='coerce').fillna(0)

    kills = to_numeric_series(player_df.get('kill', player_df.get('kills', player_df.get('kills ', pd.Series([0]*len(player_df)))))).sum()
    deaths = to_numeric_series(player_df.get('death', player_df.get('deaths', pd.Series([0]*len(player_df))))).sum()
    assists = to_numeric_series(player_df.get('assist', player_df.get('assists', pd.Series([0]*len(player_df))))).sum()
    matches = player_df.shape[0]
    
    # ACS column (average combat score) might be 'acs' or 'rating' etc.
    acs_col = None
    for c in df.columns:
        if c.lower() in ("acs", "avgcombat", "averagecombat", "rating"):
            acs_col = c
            break
    
    acs = None
    if acs_col:
        acs_vals = pd.to_numeric(player_df[acs_col].astype(str).str.replace('\xa0','', regex=False), errors='coerce')
        acs = acs_vals.mean()
    
    kast_col = next((c for c in df.columns if c.lower().startswith("kast")), None)
    kast = None
    if kast_col:
        # try to parse percent if present like "75%"
        kast_vals = pd.to_numeric(player_df[kast_col].astype(str).str.replace('%','').str.replace('\xa0','', regex=False), errors='coerce')
        kast = kast_vals.mean()
    
    hs_col = next((c for c in df.columns if 'hs' in c.lower()), None)
    hs = None
    if hs_col:
        hs_vals = pd.to_numeric(player_df[hs_col].astype(str).str.replace('%','').str.replace('\xa0','', regex=False), errors='coerce')
        hs = hs_vals.mean()
    
    # K/D ratio
    kd = round(kills / deaths, 2) if deaths > 0 else float('inf')
    
    # top agents and maps
    agent_col = next((c for c in df.columns if 'agent' in c.lower()), None)
    top_agent = None
    if agent_col:
        top_agent = player_df[agent_col].value_counts().idxmax()
    
    map_col = next((c for c in df.columns if c.lower() in ('map','maps')), None)
    top_map = None
    if map_col:
        top_map = player_df[map_col].value_counts().idxmax()
    
    # simple consistency metric: std of ACS or kills
    consistency = None
    if acs_col:
        try:
            consistency = round(acs_vals.std(), 2)
        except Exception:
            consistency = None
    
    # Build summary
    summary = {
        "player": player_name,
        "matches": int(matches),
        "kills": int(kills),
        "deaths": int(deaths),
        "assists": int(assists),
        "kd_ratio": kd,
        "acs": round(float(acs),2) if acs is not None and not np.isnan(acs) else None,
        "kast_pct": round(float(kast),2) if kast is not None and not np.isnan(kast) else None,
        "hs_pct": round(float(hs),2) if hs is not None and not np.isnan(hs) else None,
        # Win rate: proportion of matches where player's team won (win_lose == 'team win')
        "win_rate": None,
        "top_agent": top_agent,
        "top_map": top_map,
        "consistency": consistency
    }
    
    # small time series sample: ACS per match (if acs_col exists) or kills per match
    timeseries = []
    if acs_col:
        timeseries = list(pd.to_numeric(player_df[acs_col].astype(str).str.replace('\xa0','', regex=False), errors='coerce').fillna(0).head(50))
    else:
        timeseries = list(to_numeric_series(player_df.get('kill', player_df.get('kills', pd.Series([0])))).head(50))
    
    summary['timeseries'] = timeseries
    
    # agent distribution (top 5)
    if agent_col:
        agent_counts = player_df[agent_col].value_counts().nlargest(6).to_dict()
        summary['agent_counts'] = agent_counts
    else:
        summary['agent_counts'] = {}

    # Compute win rate if win_lose-like column exists
    win_col = next((c for c in df.columns if 'win' in c.lower() or c.lower() == 'win_lose'), None)
    if win_col:
        try:
            wins = player_df[win_col].astype(str).str.lower().str.contains('team win', na=False).sum()
            summary['win_rate'] = round((wins / matches) * 100, 2) if matches > 0 else None
        except Exception:
            summary['win_rate'] = None
    
    return summary


# Static agent metadata used for the champions/agents pages and agent detail view
AGENTS = [
    { 'name': 'Jett', 'role': 'Duelist', 'image': '/static/images/agents/jett.png', 'description': 'An agile duelist who excels at entry and mobility.', 'abilities': ['Cloudburst', 'Updraft', 'Tailwind', 'Blade Storm'] },
    { 'name': 'Phoenix', 'role': 'Duelist', 'image': '/static/images/agents/phoenix.png', 'description': 'A self-sufficient duelist with healing and flashes.', 'abilities': ['Curveball', 'Blaze', 'Hot Hands', 'Run It Back'] },
    { 'name': 'Raze', 'role': 'Duelist', 'image': '/static/images/agents/raze.png', 'description': 'High explosive damage dealer focused on area denial.', 'abilities': ['Boom Bot', 'Blast Pack', 'Paint Shells', 'Showstopper'] },
    { 'name': 'Reyna', 'role': 'Duelist', 'image': '/static/images/agents/reyna.png', 'description': 'A self-sustaining fragger that snowballs off kills.', 'abilities': ['Leer', 'Devour', 'Dismiss', 'Empress'] },
    { 'name': 'Yoru', 'role': 'Duelist', 'image': '/static/images/agents/yoru.png', 'description': 'A deceiving agent who manipulates space and vision.', 'abilities': ['Fakeout', 'Gatecrash', 'Blindside', 'Dimensional Drift'] },
    { 'name': 'Neon', 'role': 'Duelist', 'image': '/static/images/agents/neon.png', 'description': 'High-speed duelist who excels at pushing and entry.', 'abilities': ['Fast Lane', 'Relay Bolt', 'High Gear', 'Overdrive'] },
    { 'name': 'Iso', 'role': 'Duelist', 'image': '/static/images/agents/iso.png', 'description': 'A precise entry fragger with unique utility (custom).', 'abilities': ['Flash', 'Smokes', 'Gadget', 'Ultimate'] },
    { 'name': 'Viper', 'role': 'Controller', 'image': '/static/images/agents/viper.png', 'description': 'A controller that controls space with toxic utility.', 'abilities': ['Snake Bite', 'Poison Cloud', 'Toxin Screen', 'Viper’s Pit'] },
    { 'name': 'Omen', 'role': 'Controller', 'image': '/static/images/agents/omen.png', 'description': 'A versatile controller who manipulates sightlines and positions.', 'abilities': ['Shrouded Step', 'Paranoia', 'Dark Cover', 'From the Shadows'] },
    { 'name': 'Harbor', 'role': 'Controller', 'image': '/static/images/agents/harbor.png', 'description': 'A sea-themed controller that blocks and drowns vision.', 'abilities': ['Cascade', 'Cove', 'High Tide', 'Reckoning'] },
    { 'name': 'Astra', 'role': 'Controller', 'image': '/static/images/agents/astra.png', 'description': 'A global controller with strategic map-wide utility.', 'abilities': ['Gravity Well', 'Nebula', 'Nova Pulse', 'Astral Form'] },
    { 'name': 'Sova', 'role': 'Initiator', 'image': '/static/images/agents/sova.png', 'description': 'An initiator who gathers information and reveals enemies.', 'abilities': ['Shock Bolt', 'Recon Bolt', 'Owl Drone', 'Hunter’s Fury'] },
    { 'name': 'Breach', 'role': 'Initiator', 'image': '/static/images/agents/breach.png', 'description': 'A mechanical initiator who clears angles with strong utility.', 'abilities': ['Aftershock', 'Flashpoint', 'Fault Line', 'Rolling Thunder'] },
    { 'name': 'Skye', 'role': 'Initiator', 'image': '/static/images/agents/skye.png', 'description': 'A team-focused initiator with healing and recon.', 'abilities': ['Regrowth', 'Trailblazer', 'Guiding Light', 'Seekers'] },
    { 'name': "KAY/O", 'role': 'Initiator', 'image': '/static/images/agents/kayo.png', 'description': 'An initiator built to suppress enemy abilities.', 'abilities': ['FRAG/ment', 'ZERO/point', 'FLASH/drive', 'NULL/cmd'] },
    { 'name': 'Fade', 'role': 'Initiator', 'image': '/static/images/agents/fade.png', 'description': 'An initiator who uses nightmares to track enemies.', 'abilities': ['Seize', 'Prowler', 'Haunt', 'Nightfall'] },
    { 'name': 'Gekko', 'role': 'Initiator', 'image': '/static/images/agents/gekko.png', 'description': 'A disruptive initiator with deployable gadgets.', 'abilities': ['Mosh Pit', 'Wingman', 'Dizzy', 'Thrash'] },
    { 'name': 'Sage', 'role': 'Sentinel', 'image': '/static/images/agents/sage.png', 'description': 'A healer and anchor for site holds.', 'abilities': ['Barrier Orb', 'Slow Orb', 'Healing Orb', 'Resurrection'] },
    { 'name': 'Cypher', 'role': 'Sentinel', 'image': '/static/images/agents/cypher.png', 'description': 'An information-gathering sentinel who traps and watches.', 'abilities': ['Trapwire', 'Neural Theft', 'Spycam', 'Cyber Cage'] },
    { 'name': 'Killjoy', 'role': 'Sentinel', 'image': '/static/images/agents/killjoy.png', 'description': 'A sentinel that locks down sites with gadgets.', 'abilities': ['Nanoswarm', 'Alarmbot', 'Turret', 'Lockdown'] },
    { 'name': 'Chamber', 'role': 'Sentinel', 'image': '/static/images/agents/chamber.png', 'description': 'A sentinel with precise weapons and traps.', 'abilities': ['Trademark', 'Headhunter', 'Rendezvous', 'Tour De Force'] },
    { 'name': 'Deadlock', 'role': 'Sentinel', 'image': '/static/images/agents/deadlock.png', 'description': 'A sentinel that denies movement and zone control.', 'abilities': ['Stamp', 'Torment', 'Annihilate', 'Barrier'] }
]


def agent_summary(agent_name):
    """Return simple statistics for an agent (counts, avg acs, top players)."""
    agent_col = next((c for c in df.columns if 'agent' in c.lower()), None)
    pname_col = find_player_column()
    acs_col = next((c for c in df.columns if c.lower() in ("acs", "avgcombat", "averagecombat", "rating")), None)

    result = {
        'matches': 0,
        'avg_acs': None,
        'top_players': {}
    }

    if not agent_col:
        return result

    agent_df = df[df[agent_col].astype(str).str.lower() == str(agent_name).lower()]
    matches = int(agent_df.shape[0])
    result['matches'] = matches

    if matches > 0 and acs_col in df.columns:
        try:
            acs_vals = pd.to_numeric(agent_df[acs_col].astype(str).str.replace('\xa0','', regex=False), errors='coerce')
            result['avg_acs'] = round(float(acs_vals.mean()), 2) if not np.isnan(acs_vals.mean()) else None
        except Exception:
            result['avg_acs'] = None

    if matches > 0:
        try:
            top_players = agent_df[pname_col].astype(str).value_counts().nlargest(8).to_dict()
            result['top_players'] = top_players
        except Exception:
            result['top_players'] = {}

    return result


@app.route('/agent/<path:agent_name>')
def agent_page(agent_name):
    # Case-insensitive match against AGENTS list
    found = None
    for a in AGENTS:
        if a['name'].strip().lower() == agent_name.strip().lower():
            found = a
            break

    if not found:
        # Try replacing common separators
        key = agent_name.replace('-', ' ').replace('_', ' ').strip().lower()
        for a in AGENTS:
            if a['name'].strip().lower() == key:
                found = a
                break

    if not found:
        return render_template('agent.html', agent=None), 404

    stats = agent_summary(found['name'])
    return render_template('agent.html', agent=found, stats=stats)


# Simple MAPS metadata for map detail pages
MAPS = [
    { 'name': 'Ascent', 'image': '/static/images/maps/ascent.jpeg', 'layout': '/static/images/maps/layouts/ascent_layout.png', 'description': 'A two-site map with mid control and verticality.' },
    { 'name': 'Bind', 'image': '/static/images/maps/bind.jpeg', 'layout': '/static/images/maps/layouts/bind_layout.png', 'description': 'A short rotation map with no mid and teleporters.' },
    { 'name': 'Haven', 'image': '/static/images/maps/haven.jpeg', 'layout': '/static/images/maps/layouts/haven_layout.png', 'description': 'Three-site map with tight chokepoints and site stacks.' },
    { 'name': 'Split', 'image': '/static/images/maps/split.jpeg', 'layout': '/static/images/maps/layouts/split_layout.png', 'description': 'A map with vertical play and tight corridors.' },
    { 'name': 'Icebox', 'image': '/static/images/maps/icebox.jpeg', 'layout': '/static/images/maps/layouts/icebox_layout.png', 'description': 'Vertical map with elevated positions and close-range fights.' },
    { 'name': 'Breeze', 'image': '/static/images/maps/breeze.jpeg', 'layout': '/static/images/maps/layouts/breeze_layout.png', 'description': 'Wide-open map favoring long-range engagements.' },
    { 'name': 'Fracture', 'image': '/static/images/maps/fracture.jpeg', 'layout': '/static/images/maps/layouts/fracture_layout.png', 'description': 'Asymmetric map with unique flanking routes.' },
    { 'name': 'Pearl', 'image': '/static/images/maps/pearl.jpeg', 'layout': '/static/images/maps/layouts/pearl_layout.png', 'description': 'Closed-map with tight lanes and close angles.' },
    { 'name': 'Lotus', 'image': '/static/images/maps/lotus.jpeg', 'layout': '/static/images/maps/layouts/lotus_layout.png', 'description': 'A newer map with open mid areas and multiple site approaches.' },
    { 'name': 'Sunset', 'image': '/static/images/maps/sunset.jpeg', 'layout': '/static/images/maps/layouts/sunset_layout.png', 'description': 'Stylized map with interesting sightlines.' },
    { 'name': 'Abyss', 'image': '/static/images/maps/abyss.jpeg', 'layout': '/static/images/maps/layouts/abyss_layout.png', 'description': 'A dark, moody map with narrow corridors and verticality.' }
]


@app.route('/map/<path:map_name>')
def map_page(map_name):
    # Case-insensitive match against MAPS
    found = None
    for m in MAPS:
        if m['name'].strip().lower() == map_name.strip().lower():
            found = m
            break

    if not found:
        key = map_name.replace('-', ' ').replace('_', ' ').strip().lower()
        for m in MAPS:
            if m['name'].strip().lower() == key:
                found = m
                break

    if not found:
        return render_template('map.html', map=None), 404

    # Attempt to detect a real layout image file in the static layouts folder.
    layout_file = None
    try:
        layouts_dir = os.path.join(app.static_folder, 'images', 'maps', 'layouts')
        if os.path.isdir(layouts_dir):
            # list files and match by map-name substring (case-insensitive)
            candidates = os.listdir(layouts_dir)
            # only consider png/jpg/jpeg/webp/svg
            candidates = [fn for fn in candidates if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg'))]
            key = found['name'].strip().lower()
            key_norm = re.sub('[^a-z0-9]', '', key)
            for fn in candidates:
                fn_low = fn.lower()
                fn_norm = re.sub('[^a-z0-9]', '', fn_low)
                # match if key is substring or normalized forms match
                if (key in fn_low) or (key_norm and key_norm in fn_norm) or fn_low.startswith(key):
                    layout_file = '/static/images/maps/layouts/' + fn
                    break
    # If a layout_file was not found on disk, verify whether the configured path exists
        if not layout_file and found.get('layout'):
            # convert configured layout path to filesystem path
            cfg = found['layout']
            # if it already exists exactly
            fs = os.path.join(app.root_path, cfg.lstrip('/').replace('/', os.sep))
            if os.path.exists(fs):
                layout_file = cfg
            else:
                layout_file = None
    except Exception:
        layout_file = None

    # Optionally, we could compute map-specific stats from the dataset (top agents, win rates)
    return render_template('map.html', map=found, layout_file=layout_file)


# Allowed upload extensions for map layouts
# upload/fetch functionality removed — map pages now display layouts from the layouts folder only

@app.route("/")
def home():
    # Build player list for dropdown using best player-name column
    pname_col = find_player_column()
    
    players = sorted(df[pname_col].astype(str).unique().tolist())
    return render_template("home.html", players=players)

@app.route("/showdown")
def showdown():
    pname_col = find_player_column()
    players = sorted(df[pname_col].astype(str).unique().tolist())
    return render_template("showdown.html", players=players)

@app.route("/champions")
def champions():
    return render_template("champions.html")

# CORRECTED API ENDPOINTS FOR CASE-INSENSITIVE SEARCH BY 'player' COLUMN

@app.route("/api/search")
def api_search():
    # Case-insensitive search by 'player' column
    q = request.args.get("q", "").strip().lower()
    
    if not q:
        return jsonify({"players": []})
    
    # Find the 'player' column (prefer readable player column)
    pname_col = find_player_column()
    
    # Search for players containing the query (case-insensitive)
    matching_players = df[df[pname_col].astype(str).str.lower().str.contains(q, na=False)][pname_col].unique().tolist()
    
    # Limit to top 10 matches
    matching_players = sorted(matching_players)[:10]
    
    return jsonify({"players": matching_players})

@app.route("/api/player")
def api_player():
    # Case-insensitive player lookup by 'player' column
    name = request.args.get("name", "").strip()
    
    if not name:
        return jsonify({"error": "no player specified"}), 400
    
    summary = player_summary(name)
    
    if summary is None:
        return jsonify({"error": "player not found"}), 404
    
    return jsonify(summary)

@app.route("/api/compare")
def api_compare():
    # Case-insensitive comparison of two players
    player1 = request.args.get("player1", "").strip()
    player2 = request.args.get("player2", "").strip()
    
    if not player1 or not player2:
        return jsonify({"error": "both players must be specified"}), 400
    
    summary1 = player_summary(player1)
    summary2 = player_summary(player2)
    
    if summary1 is None or summary2 is None:
        return jsonify({"error": "one or both players not found"}), 404
    
    return jsonify({
        "player1": summary1,
        "player2": summary2
    })

@app.route('/predict')
def predict():
    # Load and preprocess data
    df = pd.read_csv('player_stats.csv')
    df = df.drop_duplicates()
    df['acs'] = pd.to_numeric(df['acs'], errors='coerce')
    df = df.dropna(subset=['acs'])
    for col in ['kill', 'death', 'assist', 'kast%', 'adr', 'hs%']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    threshold = np.percentile(df['acs'], 80)
    df['High_Performer'] = (df['acs'] >= threshold).astype(int)
    
    features = df[['kill', 'death', 'assist', 'kast%', 'adr', 'hs%']]
    target = df['High_Performer']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    conf_mat = confusion_matrix(y_test, preds).tolist()  # Convert to list for easier rendering
    
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest (area = %.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Save plot image to static folder (ensure 'static' folder exists)
    roc_path = os.path.join('static', 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    
    # Pass results to template
    return render_template('predict.html',
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1,
                           confusion_matrix=conf_mat,
                           roc_image='roc_curve.png')

@app.route('/predict_player', methods=['POST'])
def predict_player():
    # Get form inputs
    try:
        values = [float(request.form['kill']),
                  float(request.form['death']),
                  float(request.form['assist']),
                  float(request.form['kast']),
                  float(request.form['adr']),
                  float(request.form['hs'])]
    except Exception as e:
        # You can log error details here if needed
        return render_template('predict.html',
            player_prediction=None,
            accuracy=0,
            precision=0,
            recall=0,
            f1=0,
            confusion_matrix=[[0,0],[0,0]],
            roc_image='roc_curve.png'
        )

    # Load and preprocess the data/train the model exactly as before
    df = pd.read_csv('player_stats.csv')
    df = df.drop_duplicates()
    df['acs'] = pd.to_numeric(df['acs'], errors='coerce')
    df = df.dropna(subset=['acs'])
    for col in ['kill', 'death', 'assist', 'kast%', 'adr', 'hs%']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    threshold = np.percentile(df['acs'], 80)
    df['High_Performer'] = (df['acs'] >= threshold).astype(int)
    features = df[['kill', 'death', 'assist', 'kast%', 'adr', 'hs%']]
    target = df['High_Performer']

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(features, target)

    # Make prediction
    player_pred = model.predict([values])[0]

    # Render the result on the same predict page
    # ALWAYS supply the required variables
    return render_template('predict.html',
        player_prediction=player_pred,
        accuracy=0,
        precision=0,
        recall=0,
        f1=0,
        confusion_matrix=[[0,0],[0,0]],
        roc_image='roc_curve.png'
    )


if __name__ == "__main__":
    # debug True for development; switch off for production
    app.run(debug=True, port=5000)
