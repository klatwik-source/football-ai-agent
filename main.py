import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# =================================
# Sekrety z GitHub
# =================================
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

if not API_TOKEN or not DISCORD_WEBHOOK or not ODDS_API_KEY:
    raise Exception("Upewnij się, że API_TOKEN, DISCORD_WEBHOOK i ODDS_API_KEY są ustawione w GitHub Secrets")

HEADERS = {"X-Auth-Token": API_TOKEN}

# =================================
# Ligi i Puchary
# =================================
COMPETITIONS = [
    "PL", "PD", "SA", "BL1",  # ligi
    "CL", "EL", "EC"           # LM, LE, LK
]

CONF_THRESHOLD = 0.65

# =================================
# Pobranie meczów
# =================================
def get_matches():
    all_matches = []
    for comp in COMPETITIONS:
        url = f"https://api.football-data.org/v4/competitions/{comp}/matches?status=FINISHED,SCHEDULED"
        r = requests.get(url, headers=HEADERS)
        data = r.json()
        if 'matches' in data:
            all_matches.extend(data['matches'])
    return all_matches

# =================================
# Budowa DataFrame
# =================================
def build_df(matches):
    rows = []
    for m in matches:
        home_goals = m['score']['fullTime'].get('home') if 'fullTime' in m['score'] else None
        away_goals = m['score']['fullTime'].get('away') if 'fullTime' in m['score'] else None
        rows.append({
            "competition": m['competition']['name'],
            "home": m['homeTeam']['name'],
            "away": m['awayTeam']['name'],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "status": m['status'],
            "btts": (home_goals > 0 and away_goals > 0) if home_goals is not None else None,
            "over25": (home_goals + away_goals > 2.5) if home_goals is not None else None,
            "over35": (home_goals + away_goals > 3.5) if home_goals is not None else None
        })
    df = pd.DataFrame(rows)

    # Obliczenia formy i różnicy bramek dla zakończonych meczów
    finished_df = df[df['status'] == 'FINISHED'].copy()
    finished_df['home_form'] = finished_df.groupby('home')['home_goals'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    finished_df['away_form'] = finished_df.groupby('away')['away_goals'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    finished_df['goal_diff'] = finished_df['home_goals'] - finished_df['away_goals']

    return df, finished_df

# =================================
# Trening modelu
# =================================
def train_model(df, target_col):
    df = df[df[target_col].notnull()].copy()
    X = df[['home_goals','away_goals','home_form','away_form','goal_diff']]
    y = df[target_col].astype(int)

    split_index = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # kolumna float na confidence
    df[f"{target_col}_conf"] = 0.0
    if len(X_test) > 0:
        df.loc[X_test.index, f"{target_col}_conf"] = model.predict_proba(X_test)[:,1]

    return df, model

# =================================
# Pobranie kursów bukmacherskich
# =================================
def get_odds():
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?regions=eu&markets=h2h,totals&apiKey={ODDS_API_KEY}"
    r = requests.get(url)
    data = r.json()
    
    odds_list = []
    for match in data:
        try:
            home = match['home_team']
            away = match['away_team']
            totals = match['bookmakers'][0]['markets'][1]['outcomes']
            over25 = next((o['price'] for o in totals if o['name'] == 'Over 2.5'), None)
            btts = next((o['price'] for o in totals if o['name'] == 'BTTS'), None)
            odds_list.append({"home": home,"away": away,"over25": over25,"btts": btts})
        except:
            continue
    return pd.DataFrame(odds_list)

# =================================
# Filtr value bets z zabezpieczeniem
# =================================
def filter_value_bets(upcoming_df, model_df, odds_df, col):
    df = upcoming_df.copy()

    # Dodaj confidence z modelu
    df = pd.merge(df, model_df[['home','away',f'{col}_conf']], on=['home','away'], how='left')

    # Zabezpieczenie: jeśli kolumna kursów nie istnieje, dodaj z None
    if col not in odds_df.columns:
        odds_df[col] = None

    # Dodaj kursy z bukmacherki
    df = pd.merge(df, odds_df[['home','away',col]], on=['home','away'], how='left')

    # 🔹 DODAJEMY KOLUMNĘ JEŻELI NIE ISTNIEJE
    if col not in df.columns:
        df[col] = None
    if f'{col}_conf' not in df.columns:
        df[f'{col}_conf'] = None

    # Usuń wiersze bez kursu lub confidence
    df = df[df[f'{col}_conf'].notnull() & df[col].notnull()]

    # Filtr value bets
    value_bets = df[(df[f"{col}_conf"] >= CONF_THRESHOLD) & (df[f"{col}_conf"] > 1/df[col])]
    return value_bets

# =================================
# Wysyłka na Discord
# =================================
def send_discord(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# =================================
# Funkcja główna
# =================================
def run_agent():
    matches = get_matches()
    df_all, finished_df = build_df(matches)
    odds_df = get_odds()
    today_str = datetime.now().strftime("%Y-%m-%d")
    upcoming_df = df_all[df_all['status'] == 'SCHEDULED'].copy()

    for comp in df_all['competition'].unique():
        comp_finished = finished_df[finished_df['competition']==comp].copy()
        if len(comp_finished) < 5:
            continue

        for col in ["over25","btts"]:
            comp_finished, model = train_model(comp_finished, col)
            comp_upcoming = upcoming_df[upcoming_df['competition']==comp].copy()
            if comp_upcoming.empty:
                continue

            value_bets = filter_value_bets(comp_upcoming, comp_finished, odds_df, col)
            for _, row in value_bets.iterrows():
                msg = (
                    f"⚽ **{row.competition}: {row.home} vs {row.away} ({today_str})**\n"
                    f"🎯 Typ: {col.upper()}\n"
                    f"📊 Pewność AI: {round(row[f'{col}_conf']*100,2)}%\n"
                    f"💰 Kurs: {row[col]}\n"
                    f"🧠 AI Agent"
                )
                send_discord(msg)

    # Raport dzienny
    report = ""
    for comp in df_all['competition'].unique():
        for col in ["over25","btts"]:
            mean_conf = finished_df[finished_df['competition']==comp][f"{col}_conf"].mean()
            report += f"{comp} – {col.upper()} – średnia pewność: {mean_conf:.2f}\n"
    send_discord(f"📊 **Raport dzienny AI**\n{report}")

if __name__ == "__main__":
    run_agent()
