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
            for m in r["matches"]:
                m["competition_code"] = c
                all_matches.append(m)
    return all_matches

# =========================
# DATA
# =========================
def build_df(matches):
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
            "status": m["status"],
            "over25": (hg + ag > 2.5) if hg is not None else None,
            "btts": (hg > 0 and ag > 0) if hg is not None else None,
            "home_win": (hg > ag) if hg is not None else None,
        })

    df = pd.DataFrame(rows)
    return df

# =========================
# FEATURES (BEZ WYNIKU)
# =========================
def add_features(df):
    df["is_home"] = 1
    return df

# =========================
# TRAIN + PREDICT
# =========================
def train_and_predict(finished, upcoming, target):
    finished = finished[finished[target].notnull()]
    if len(finished) < 20:
        return None

    X_train = finished[["is_home"]]
    y_train = finished[target].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    X_pred = upcoming[["is_home"]]
    upcoming[target+"_conf"] = model.predict_proba(X_pred)[:,1]

    return upcoming

# =========================
# AGENT
# =========================
def run_agent():
    df = build_df(get_matches())
    df = add_features(df)

    finished = df[df.status == "FINISHED"]
    upcoming = df[df.status == "SCHEDULED"]

    today = datetime.now().strftime("%Y-%m-%d")
    send(f"📊 **AI FOOTBALL AGENT – {today}**")

    for comp in df.competition.unique():
        f = finished[finished.competition == comp]
        u = upcoming[upcoming.competition == comp]

        if f.empty or u.empty:
            continue

        send(f"\n🏆 **{comp}**")

        for target, label in [
            ("over25","Over 2.5"),
            ("btts","BTTS"),
            ("home_win","Zwycięzca (1)")
        ]:
            preds = train_and_predict(f, u, target)
            if preds is None:
                send(f"⚠️ Brak danych dla {label}")
                continue

            value = preds[preds[target+"_conf"] >= CONF_THRESHOLD]
            if value.empty:
                send(f"❌ Brak pewnych typów dla {label}")
                continue

            send(f"\n🎯 **{label}**")
            for _, r in value.head(3).iterrows():
                send(
                    f"⚽ {r.home} vs {r.away}\n"
                    f"🧠 Pewność AI: {round(r[target+'_conf']*100,1)}%"
                )

if __name__ == "__main__":
    run_agent()
