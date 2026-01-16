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

COMPETITIONS = ["PL","PD","SA","BL1","CL","EL","EC"]
CONF_THRESHOLD = 0.55

# =========================
# DISCORD
# =========================
def send(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# =========================
# MATCHES
# =========================
def get_matches():
    all_matches = []
    for c in COMPETITIONS:
        url = f"https://api.football-data.org/v4/competitions/{c}/matches?status=FINISHED,SCHEDULED"
        r = requests.get(url, headers=HEADERS).json()
        if "matches" in r:
            all_matches.extend(r["matches"])
    return all_matches

# =========================
# DATAFRAME
# =========================
def build_df(matches):
    rows = []
    for m in matches:
        ft = m["score"]["fullTime"]
        hg = ft["home"] if ft else None
        ag = ft["away"] if ft else None

        rows.append({
            "competition": m["competition"]["name"],
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_goals": hg,
            "away_goals": ag,
            "status": m["status"],
            "over25": (hg + ag > 2.5) if hg is not None else None,
            "btts": (hg > 0 and ag > 0) if hg is not None else None,
            "home_win": (hg > ag) if hg is not None else None,
            "draw": (hg == ag) if hg is not None else None,
            "away_win": (hg < ag) if hg is not None else None
        })

    df = pd.DataFrame(rows)
    finished = df[df.status == "FINISHED"].copy()

    finished["goal_diff"] = finished.home_goals - finished.away_goals
    finished["total_goals"] = finished.home_goals + finished.away_goals

    return df, finished

# =========================
# TRAINING
# =========================
def train(df, target):
    df = df[df[target].notnull()].copy()
    if len(df) < 10:
        return None, None

    X = df[["goal_diff","total_goals"]]
    y = df[target].astype(int)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    df[target+"_conf"] = model.predict_proba(X)[:,1]
    return df, model

# =========================
# AGENT
# =========================
def run_agent():
    matches = get_matches()
    df_all, finished = build_df(matches)
    upcoming = df_all[df_all.status == "SCHEDULED"]

    today = datetime.now().strftime("%Y-%m-%d")
    send(f"📊 **AI FOOTBALL AGENT – {today}**")

    for comp in df_all.competition.unique():
        f = finished[finished.competition == comp]
        u = upcoming[upcoming.competition == comp]
        if f.empty or u.empty:
            continue

        send(f"\n🏆 **{comp}**")

        for target, label in [
            ("over25","Over 2.5"),
            ("btts","BTTS"),
            ("home_win","1X / X2 (Winner)")
        ]:
            trained, model = train(f, target)
            if model is None:
                send(f"⚠️ Brak danych dla {label}")
                continue

            top = trained.sort_values(target+"_conf", ascending=False).head(3)

            send(f"\n🎯 **{label} – TOP CONFIDENCE**")
            for _, r in top.iterrows():
                send(
                    f"⚽ {r.home} vs {r.away}\n"
                    f"🧠 Pewność AI: {round(r[target+'_conf']*100,1)}%"
                )

if __name__ == "__main__":
    run_agent()
