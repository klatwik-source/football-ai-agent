import requests
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

API_TOKEN = ""
DISCORD_WEBHOOK = ""

HEADERS = {"X-Auth-Token": API_TOKEN}

def get_matches():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
    r = requests.get(url, headers=HEADERS)
    return r.json()['matches']

def build_df(matches):
    rows = []
    for m in matches:
        if m['score']['fullTime']['home'] is None:
            continue
        rows.append({
            "home": m['homeTeam']['name'],
            "away": m['awayTeam']['name'],
            "home_goals": m['score']['fullTime']['home'],
            "away_goals": m['score']['fullTime']['away']
        })
    df = pd.DataFrame(rows)
    df['over25'] = (df.home_goals + df.away_goals) > 2.5
    return df

def train_model(df):
    X = df[['home_goals', 'away_goals']]
    y = df['over25']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    df['confidence'] = model.predict_proba(X)[:, 1]
    return df

def send_discord(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

def run_agent():
    matches = get_matches()
    df = build_df(matches)
    df = train_model(df)

    high_conf = df[df.confidence >= 0.65]

    for _, row in high_conf.iterrows():
        msg = (
            f"⚽ **{row.home} vs {row.away}**\n"
            f"🎯 Typ: OVER 2.5\n"
            f"📊 Pewność: {round(row.confidence*100,2)}%\n"
            f"🧠 AI Agent"
        )
        send_discord(msg)

if __name__ == "__main__":
    while True:
        run_agent()
        time.sleep(86400)
