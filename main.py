import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# === SEKRETY Z GITHUB ACTIONS ===
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not API_TOKEN:
    raise Exception("BRAK API_TOKEN – sprawdź secrets w repo")
if not DISCORD_WEBHOOK:
    raise Exception("BRAK DISCORD_WEBHOOK – sprawdź secrets w repo")

HEADERS = {"X-Auth-Token": API_TOKEN}

# === LIGI DO ANALIZY ===
LEAGUES = ["PL", "PD", "SA", "BL1"]  # Premier, LaLiga, SerieA, Bundesliga

CONF_THRESHOLD = 0.65  # tylko pewne typy

# Pobranie meczów z każdej ligi
def get_matches():
    all_matches = []
    for league in LEAGUES:
        url = f"https://api.football-data.org/v4/competitions/{league}/matches?status=FINISHED"
        r = requests.get(url, headers=HEADERS)
        data = r.json()
        if 'matches' in data:
            all_matches.extend(data['matches'])
    return all_matches

# Budowa DataFrame
def build_df(matches):
    rows = []
    for m in matches:
        home_goals = m['score']['fullTime']['home']
        away_goals = m['score']['fullTime']['away']
        if home_goals is None or away_goals is None:
            continue
        rows.append({
            "league": m['competition']['name'],
            "home": m['homeTeam']['name'],
            "away": m['awayTeam']['name'],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "btts": (home_goals > 0 and away_goals > 0),
            "over25": (home_goals + away_goals) > 2.5,
            "over35": (home_goals + away_goals) > 3.5
        })
    df = pd.DataFrame(rows)
    return df

# Trening modelu AI i liczenie confidence
def train_model(df, target_col):
    X = df[['home_goals', 'away_goals']]
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    df[f"{target_col}_conf"] = model.predict_proba(X)[:, 1]
    return df

# Wysyłka na Discord
def send_discord(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# Funkcja główna
def run_agent():
    matches = get_matches()
    df = build_df(matches)

    for col in ["over25", "over35", "btts"]:
        df = train_model(df, col)

    # Wysyłamy tylko pewne typy
    for col in ["over25", "over35", "btts"]:
        high_conf = df[df[f"{col}_conf"] >= CONF_THRESHOLD]
        for _, row in high_conf.iterrows():
            type_name = col.upper()
            msg = (
                f"⚽ **{row.league}: {row.home} vs {row.away}**\n"
                f"🎯 Typ: {type_name}\n"
                f"📊 Pewność: {round(row[f'{col}_conf']*100,2)}%\n"
                f"🧠 AI Agent"
            )
            send_discord(msg)

# Start
if __name__ == "__main__":
    run_agent()
