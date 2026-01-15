import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# === SEKRETY Z GITHUB ACTIONS ===
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
BOOKMAKER_API = os.getenv("BOOKMAKER_API")  # np. darmowe kursy API

if not API_TOKEN:
    raise Exception("BRAK API_TOKEN – sprawdź secrets w repo")
if not DISCORD_WEBHOOK:
    raise Exception("BRAK DISCORD_WEBHOOK – sprawdź secrets w repo")

HEADERS = {"X-Auth-Token": API_TOKEN}

# === LIGI DO ANALIZY ===
LEAGUES = ["PL", "PD", "SA", "BL1"]

CONF_THRESHOLD = 0.65  # tylko pewne typy

# Pobranie meczów
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

# Trening AI
def train_model(df, target_col):
    X = df[['home_goals', 'away_goals']]
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    df[f"{target_col}_conf"] = model.predict_proba(X)[:, 1]
    return df

# Pobranie kursów bukmacherskich (przykład darmowy)
def get_odds():
    # Tutaj API lub plik CSV z kursami
    # Format: home, away, over25, over35, btts
    return pd.DataFrame(columns=["home","away","over25","over35","btts"])

# Obliczenie value betting
def filter_value_bets(df, odds_df, col):
    merged = pd.merge(df, odds_df, on=["home","away"], how="left", suffixes=("","_odds"))
    high_conf = merged[(merged[f"{col}_conf"] >= CONF_THRESHOLD) &
                       (merged[f"{col}_conf"] > 1/merged[f"{col}_odds"])]
    return high_conf

# Wysyłka Discord
def send_discord(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# Funkcja główna
def run_agent():
    matches = get_matches()
    df = build_df(matches)
    odds_df = get_odds()

    for col in ["over25", "over35", "btts"]:
        df = train_model(df, col)
        value_bets = filter_value_bets(df, odds_df, col)

        for _, row in value_bets.iterrows():
            type_name = col.upper()
            msg = (
                f"⚽ **{row.league}: {row.home} vs {row.away}**\n"
                f"🎯 Typ: {type_name}\n"
                f"📊 Pewność AI: {round(row[f'{col}_conf']*100,2)}%\n"
                f"💰 Kurs: {row[f'{col}_odds']}\n"
                f"🧠 AI Agent"
            )
            send_discord(msg)

    # Raport skuteczności
    report = ""
    for col in ["over25","over35","btts"]:
        report += f"{col.upper()} – średnia pewność: {df[f'{col}_conf'].mean():.2f}\n"
    send_discord(f"📊 **Raport dzienny AI**\n{report}")

if __name__ == "__main__":
    run_agent()
