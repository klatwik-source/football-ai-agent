import requests
import pandas as pd
import time
import os
from sklearn.ensemble import RandomForestClassifier

# Pobranie sekretów z GitHub Actions
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not API_TOKEN:
    raise Exception("BRAK API_TOKEN – sprawdź secrets w repo")

if not DISCORD_WEBHOOK:
    raise Exception("BRAK DISCORD_WEBHOOK – sprawdź secrets w repo")

HEADERS = {"X-Auth-Token": API_TOKEN}

# Funkcja pobrania meczów
def get_matches():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
    r = requests.get(url, headers=HEADERS)
    return r.json()['matches']

# Funkcja budowania DataFrame
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

# Funkcja trenowania modelu i liczenia confidence
def train_model(df):
    X = df[['home_goals', 'away_goals']]
    y = df['over25']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    df['confidence'] = model.predict_proba(X)[:, 1]
    return df

# Funkcja wysyłki na Discord
def send_discord(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# Funkcja główna
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
