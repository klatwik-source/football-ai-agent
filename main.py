import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === SEKRETY GITHUB ACTIONS ===
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not API_TOKEN:
    raise Exception("BRAK API_TOKEN – sprawdź secrets w repo")
if not DISCORD_WEBHOOK:
    raise Exception("BRAK DISCORD_WEBHOOK – sprawdź secrets w repo")

HEADERS = {"X-Auth-Token": API_TOKEN}

# === LIGI ===
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

# Trening AI z podziałem na train/test
def train_model(df, target_col):
    X = df[['home_goals', 'away_goals']]
    y = df[target_col]
    
    # Chronologiczny podział: ostatnie 20% jako test
    split_index = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Liczymy confidence na “niewidzianych” meczach
    df[f"{target_col}_conf"] = pd.Series([0]*len(df), index=df.index)
    df.loc[X_test.index, f"{target_col}_conf"] = model.predict_proba(X_test)[:, 1]
    
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
        high_conf = df[df[f"{col}_conf"] >= CONF_THRESHOLD]
        for _, row in high_conf.iterrows():
            type_name = col.upper()
            msg = (
                f"⚽ **{row.league}: {row.home} vs {row.away}**\n"
                f"🎯 Typ: {type_name}\n"
                f"📊 Pewność AI: {round(row[f'{col}_conf']*100,2)}%\n"
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
