import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timezone

# ======================
# ENV
# ======================
API_TOKEN = os.getenv("API_TOKEN")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

if not API_TOKEN or not DISCORD_WEBHOOK:
    raise Exception("Brak API_TOKEN lub DISCORD_WEBHOOK")

HEADERS = {"X-Auth-Token": API_TOKEN}

COMPETITIONS = ["PL", "PD", "SA", "BL1", "CL", "EL", "EC"]
CONF_THRESHOLD = 0.6

# ======================
# DISCORD
# ======================
def send(msg):
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

# ======================
# FETCH MATCHES
# ======================
def get_matches():
    all_matches = []
    for c in COMPETITIONS:
        url = f"https://api.football-data.org/v4/competitions/{c}/matches?status=FINISHED,SCHEDULED"
        r = requests.get(url, headers=HEADERS).json()
        for m in r.get("matches", []):
            m["competition_code"] = c
            all_matches.append(m)
    return all_matches

# ======================
# BUILD DATAFRAME
# ======================
def build_dataframe(matches):
    rows = []
    for m in matches:
        ft = m["score"]["fullTime"]
        rows.append({
            "competition": m["competition_code"],
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_goals": ft["home"] if ft else None,
            "away_goals": ft["away"] if ft else None,
            "status": m["status"],
            "date": m["utcDate"][:10]
        })
    return pd.DataFrame(rows)

# ======================
# TARGETS (TYLKO FINISHED)
# ======================
def add_targets(df):
    df = df.copy()
    df["total_goals"] = df.home_goals + df.away_goals
    df["over25"] = (df.total_goals > 2.5).astype(int)
    df["btts"] = ((df.home_goals > 0) & (df.away_goals > 0)).astype(int)
    df["home_win"] = (df.home_goals > df.away_goals).astype(int)
    df["draw"] = (df.home_goals == df.away_goals).astype(int)
    return df

# ======================
# TEAM FEATURES (BRAKUJĄCA FUNKCJA)
# ======================
def add_team_features(df, reference_df):
    """
    df – dataframe do którego dodajemy cechy
    reference_df – tylko FINISHED (historia)
    """
    home_avg = reference_df.groupby("home")["home_goals"].mean()
    away_avg = reference_df.groupby("away")["away_goals"].mean()

    df = df.copy()
    df["home_avg_goals"] = df["home"].map(home_avg)
    df["away_avg_goals"] = df["away"].map(away_avg)

    return df.dropna()

# ======================
# TRAIN + PREDICT
# ======================
def train_and_predict(finished_df, upcoming_df, target):
    if len(finished_df) < 40:
        return None

    finished_df = add_team_features(finished_df, finished_df)
    upcoming_df = add_team_features(upcoming_df, finished_df)

    X_train = finished_df[["home_avg_goals", "away_avg_goals"]]
    y_train = finished_df[target]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    X_pred = upcoming_df[["home_avg_goals", "away_avg_goals"]]
    upcoming_df["conf"] = model.predict_proba(X_pred)[:, 1]

    return upcoming_df[upcoming_df.conf >= CONF_THRESHOLD]

# ======================
# AGENT
# ======================
def run_agent():
    matches = get_matches()
    df = build_dataframe(matches)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    finished = df[df.status == "FINISHED"]
    upcoming = df[(df.status == "SCHEDULED") & (df.date == today)]

    if upcoming.empty:
        send("❌ Brak meczów dzisiaj")
        return

    finished = add_targets(finished)

    send(f"📊 **AI Football Agent – {today}**")

    for comp in COMPETITIONS:
        f = finished[finished.competition == comp]
        u = upcoming[upcoming.competition == comp]

        if f.empty or u.empty:
            continue

        send(f"\n🏆 **{comp}**")

        for target, label in [
            ("over25", "Over 2.5"),
            ("btts", "BTTS"),
            ("home_win", "Gospodarz wygra"),
            ("draw", "Remis")
        ]:
            preds = train_and_predict(f, u, target)

            if preds is None or preds.empty:
                send(f"❌ Brak pewnych typów: {label}")
                continue

            send(f"\n🎯 **{label}**")
            for _, r in preds.iterrows():
                send(
                    f"⚽ {r.home} vs {r.away}\n"
                    f"🧠 Pewność: {round(r.conf * 100, 1)}%"
                )

if __name__ == "__main__":
    run_agent()
