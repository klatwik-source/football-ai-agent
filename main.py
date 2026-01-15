import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# === SEKRETY GITHUB ACTIONS ===
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not API_TOKEN:
    raise Exception("BRAK API_TOKEN – sprawdź secrets w repo")
if not DISCORD_WEBHOOK:
    raise Exception("BRAK DISCORD_WEBHOOK – sprawdź secrets w repo")

HEADERS = {"X-Auth-Token": API_TOKEN}

# === LIGI I PUCHARY ===
COMPETITIONS = [
    "PL",      # Premier League
    "PD",      # LaLiga
    "SA",      # Serie A
    "BL1",     # Bundesliga
    "CL",      # Liga Mistrzów
    "EL",      # Liga Europy
    "EC"       # Liga Konferencji
]

CONF_THRESHOLD = 0.65  # tylko pewne typy

# Pobranie meczów
def get_matches():
    all_matches = []
    for comp in COMPETITIONS:
        url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=FINISHED"
        r = requests.get(url, headers=HEADERS)
        data = r.json()
        if 'matches' in data:
            all_matches.extend(data['matches'])
    return all_matches

# Budowa DataFrame z dodatkowymi cechami
def build_df(matches):
    rows = []
    for m in matches:
        home_goals = m['score']['fullTime']['home']
        away_goals = m['score']['fullTime']['away']
        if home_goals is None or away_goals is None:
            continue
        rows.append({
            "competition": m['competition']['name'],  # liga/puchar
            "home": m['homeTeam']['name'],
            "away": m['awayTeam']['name'],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "btts": (home_goals > 0 and away_goals > 0),
            "over25": (home_goals + away_goals) > 2.5,
            "over35": (home_goals + away_goals) > 3.5
        })
    df = pd.DataFrame(rows)
    
    # Forma drużyny: średnia bramek w ostatnich 3 meczach
    df['home_form'] = df.groupby('home')['home_goals'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['away_form'] = df.groupby('away')['away_goals'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    # Różnica bramek
    df['goal_diff'] = df['home_goals'] - df['away_goals']
    
    return df

# Trening modelu z dodatkowymi cechami (train/test)
def train_model(df, target_col):
    X = df[['home_goals', 'away_goals', 'home_form', 'away_form', 'goal_diff']]
    y = df[target_col]
    
    # Chronologiczny podział na train/test
    split_index = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Liczymy confidence na niewidzianych meczach
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

    # Trenuj osobno dla każdej ligi/pucharu
    for comp in df['competition'].unique():
        comp_df = df[df['competition'] == comp].copy()
        for col in ["over25", "over35", "btts"]:
            comp_df = train_model(comp_df, col)
            high_conf = comp_df[comp_df[f"{col}_conf"] >= CONF_THRESHOLD]
            
            for _, row in high_conf.iterrows():
                type_name = col.upper()
                msg = (
                    f"⚽ **{row.competition}: {row.home} vs {row.away}**\n"
                    f"🎯 Typ: {type_name}\n"
                    f"📊 Pewność AI: {round(row[f'{col}_conf']*100,2)}%\n"
                    f"🧠 AI Agent"
                )
                send_discord(msg)
    
    # Raport dzienny – średnia pewność per liga/puchar
    report = ""
    for comp in df['competition'].unique():
        comp_df = df[df['competition'] == comp]
        for col in ["over25","over35","btts"]:
            mean_conf = comp_df[f"{col}_conf"].mean()
            report += f"{comp} – {col.upper()} – średnia pewność: {mean_conf:.2f}\n"
    send_discord(f"📊 **Raport dzienny AI**\n{report}")

if __name__ == "__main__":
    run_agent()
