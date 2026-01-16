import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not API_TOKEN or not DISCORD_WEBHOOK:
    raise Exception("Brak API_TOKEN lub DISCORD_WEBHOOK")

HEADERS = {"X-Auth-Token": API_TOKEN}

COMPETITIONS = ["PL", "PD", "SA", "BL1", "CL", "EL", "EC"]
CONF_THRESHOLD = 0.55

# ----------------------
# DISCORD
# ----------------------
def send(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# ----------------------
# FETCH MATCHES
# ----------------------
def get_matches():
    all_matches = []
    for c in COMPETITIONS:
        url = f"https://api.football-data.org/v4/competitions/{c}/matches?status=FINISHED,SCHEDULED"
        r = requests.get(url, headers=HEADERS).json()
        if "matches" in r:
            for m in r["matches"]:
                m["competition_code"] = c
                all_matches.append(m)
    return all_matches

# ----------------------
# BUILD DATAFRAME
# ----------------------
def build_features(matches):
    rows = []
    for m in matches:
        ft = m["score"]["fullTime"]
        hg = ft["home"] if ft else None
        ag = ft["away"] if ft else None
        rows.append({
            "competition": m["competition_code"],
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_goals": hg,
            "away_goals": ag,
            "status": m["status"]
        })
    df = pd.DataFrame(rows)
    
    # HISTORIA BRAMEK
    df["home_goals"] = df["home_goals"].fillna(0)
    df["away_goals"] = df["away_goals"].fillna(0)

    return df

# ----------------------
# CREATE FEATURES
# ----------------------
def compute_features(df):
    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    
    # ŚREDNIA goli gospodarzy w sezonie
    home_stats = df[df.status=="FINISHED"].groupby("home").agg({
        "home_goals": ["mean", "count"]
    })
    home_stats.columns = ["home_avg_goals", "home_games"]
    
    away_stats = df[df.status=="FINISHED"].groupby("away").agg({
        "away_goals": ["mean", "count"]
    })
    away_stats.columns = ["away_avg_goals", "away_games"]
    
    df = df.merge(home_stats, left_on="home", right_index=True, how="left")
    df = df.merge(away_stats, left_on="away", right_index=True, how="left")
    
    df["home_avg_goals"].fillna(0, inplace=True)
    df["away_avg_goals"].fillna(0, inplace=True)
    
    return df

# ----------------------
# TRAIN + PREDICT
# ----------------------
def train_and_predict(finished_df, upcoming_df, target):
    finished_df = compute_features(finished_df)
    upcoming_df = compute_features(upcoming_df)

    X_train = finished_df[["home_avg_goals","away_avg_goals","goal_diff","total_goals"]]
    y_train = finished_df[target].astype(int)

    if len(X_train) < 20:
        return None

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    X_pred = upcoming_df[["home_avg_goals","away_avg_goals","goal_diff","total_goals"]]
    upcoming_df[target+"_conf"] = model.predict_proba(X_pred)[:,1]
    
    return upcoming_df

# ----------------------
# AGENT
# ----------------------
def run_agent():
    matches = get_matches()
    df = build_features(matches)

    finished = df[df.status=="FINISHED"]
    upcoming = df[df.status=="SCHEDULED"]

    today = datetime.now().strftime("%Y-%m-%d")
    send(f"📊 **Predykcje AI Agent – {today}**")

    for comp in df.competition.unique():
        f = finished[finished.competition==comp]
        u = upcoming[upcoming.competition==comp]
        if f.empty or u.empty:
            continue

        send(f"\n🏆 **{comp}**")

        for target, label in [
            ("over25","Over 2.5"),
            ("btts","BTTS"),
            ("home_win","Zwycięstwo gospodarza")
        ]:
            preds = train_and_predict(f, u, target)
            if preds is None:
                send(f"⚠️ Za mało danych do modelu: {label}")
                continue

            # wybieramy wszystkie mecze z confidence >= threshold
            good = preds[preds[target+"_conf"] >= CONF_THRESHOLD]
            if good.empty:
                send(f"❌ Brak pewnych typów dla {label}")
                continue

            send(f"\n🎯 **{label}**")
            for _, r in good.iterrows():
                send(
                    f"⚽ {r.home} vs {r.away}\n"
                    f"🧠 Pewność AI: {round(r[target+'_conf']*100,1)}%"
                )

if __name__ == "__main__":
    run_agent()
