# scripts/update_records_nfl.py
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import argparse

import requests
from dotenv import load_dotenv
import pandas as pd

# --- chemins projet ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RECORDS_CSV = DATA_DIR / "nfl_team_records.csv"

# Mapping éventuel (alias d'équipes)
TEAM_ALIAS_MAP: dict[str, str] = {}
try:
    sys.path.insert(0, str(ROOT))
    from utils.teams import TEAM_ALIASES  # si dispo
    TEAM_ALIAS_MAP = {k.lower(): v for k, v in TEAM_ALIASES.items()}
except Exception:
    TEAM_ALIAS_MAP = {}

def canonical_team_name(name: str) -> str:
    if not name:
        return name
    key = name.strip().lower()
    return TEAM_ALIAS_MAP.get(key, name.strip())

def nfl_week_start(ts_local: pd.Timestamp) -> pd.Timestamp:
    weekday = ts_local.weekday()  # Mon=0 ... Thu=3
    delta_days = (weekday - 3) % 7
    return (ts_local - pd.to_timedelta(delta_days, unit="D")).normalize()

def pick_season_for_today(today: datetime) -> int:
    return today.year

def load_env_key(explicit_api_key: str | None = None) -> str:
    """
    Ordre de priorité :
    1) argument --api-key si fourni
    2) .env à la racine
    3) .env dans le CWD
    4) variable d'environnement existante
    """
    if explicit_api_key:
        return explicit_api_key.strip()

    env_root = ROOT / ".env"
    env_cwd = Path(".") / ".env"

    loaded = False
    if env_root.exists():
        load_dotenv(env_root, override=True)
        loaded = True
    if env_cwd.exists():
        load_dotenv(env_cwd, override=True)
        loaded = True

    key = os.getenv("ODDS_API_KEY")
    if key:
        return key.strip()

    if not loaded:
        load_dotenv(override=True)
        key = os.getenv("ODDS_API_KEY")
        if key:
            return key.strip()

    raise SystemExit("ODDS_API_KEY manquant : ajoute-le dans SmartOdds/.env ou passe --api-key.")

def fetch_completed_scores(api_key: str) -> list[dict]:
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores"
    params = {"apiKey": api_key, "daysFrom": 365}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Réponse inattendue: {data!r}")
    return data

def parse_game_row(g: dict) -> dict | None:
    try:
        home = canonical_team_name(g["home_team"])
        away = canonical_team_name(g["away_team"])
        commence_utc = pd.to_datetime(g["commence_time"], utc=True)
        completed = bool(g.get("completed", False))
        scores = g.get("scores", [])
        home_score = None
        away_score = None
        for s in scores or []:
            n = canonical_team_name(s.get("name", ""))
            if n == home:
                home_score = float(s.get("score")) if s.get("score") is not None else None
            elif n == away:
                away_score = float(s.get("score")) if s.get("score") is not None else None
        return {
            "home": home,
            "away": away,
            "kickoff_utc": commence_utc,
            "completed": completed,
            "home_score": home_score,
            "away_score": away_score,
        }
    except Exception:
        return None

def compute_records(games: list[dict], season_year: int, tz: str = "America/Toronto") -> pd.DataFrame:
    if not games:
        return pd.DataFrame(columns=["team", "wins", "losses", "ties", "week_updated"])

    start = pd.Timestamp(year=season_year-1, month=8, day=1, tz=tz)
    end   = pd.Timestamp(year=season_year+1, month=7, day=31, tz=tz) + pd.Timedelta(days=1)

    rows = []
    for g in games:
        row = parse_game_row(g)
        if not row or not row["completed"]:
            continue
        kickoff_local = row["kickoff_utc"].tz_convert(tz)
        if not (start <= kickoff_local < end):
            continue

        home, away = row["home"], row["away"]
        hs, as_ = row["home_score"], row["away_score"]
        if hs is None or as_ is None:
            continue

        if hs > as_:
            res = [(home, "W"), (away, "L")]
        elif as_ > hs:
            res = [(away, "W"), (home, "L")]
        else:
            res = [(home, "T"), (away, "T")]

        rows.append({
            "kickoff_local": kickoff_local,
            "week_start": nfl_week_start(kickoff_local),
            "results": res,
        })

    if not rows:
        return pd.DataFrame(columns=["team", "wins", "losses", "ties", "week_updated"])

    df = pd.DataFrame(rows)
    exploded = []
    for _, r in df.iterrows():
        wk = r["week_start"]
        for team, outcome in r["results"]:
            exploded.append({"team": team, "outcome": outcome, "week_start": wk})
    ex = pd.DataFrame(exploded)

    agg = ex.pivot_table(index="team", columns="outcome", values="week_start", aggfunc="count", fill_value=0)
    for c in ["W", "L", "T"]:
        if c not in agg.columns:
            agg[c] = 0
    agg = agg.reset_index().rename(columns={"W": "wins", "L": "losses", "T": "ties"})

    last_week = ex["week_start"].max() if not ex.empty else pd.NaT
    if pd.notna(last_week):
        agg["week_updated"] = pd.to_datetime(last_week).strftime("%Y-%m-%d")
    else:
        agg["week_updated"] = ""

    return agg[["team", "wins", "losses", "ties", "week_updated"]].sort_values("team")

def main():
    parser = argparse.ArgumentParser(description="Met à jour les records NFL dans data/nfl_team_records.csv")
    parser.add_argument("--season", type=int, default=None, help="Année de saison NFL (ex: 2025)")
    parser.add_argument("--dry-run", action="store_true", help="Aperçu sans écrire le CSV")
    parser.add_argument("--api-key", type=str, default=None, help="Clé API The Odds API (prioritaire sur .env)")
    args = parser.parse_args()

    api_key = load_env_key(args.api_key)
    season = args.season or pick_season_for_today(datetime.now(timezone.utc))

    print(f"[INFO] Saison ciblée: {season}")
    print("[INFO] Récupération des scores…")
    games = fetch_completed_scores(api_key)
    print(f"[INFO] {len(games)} matchs bruts reçus.")

    df = compute_records(games, season_year=season)
    if df.empty:
        print("[WARN] Aucun match terminé trouvé.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(df.to_string(index=False))
        print("\n[DRY-RUN] Pas d'écriture.")
    else:
        df.to_csv(RECORDS_CSV, index=False)
        print(f"[OK] Écrit: {RECORDS_CSV} ({len(df)} équipes)")

if __name__ == "__main__":
    main()
