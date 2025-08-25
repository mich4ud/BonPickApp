# scripts/fetch_results_nfl.py
# Récupère des résultats NFL récents via The Odds API /scores
# Écrit data/nfl_results.csv (append) et gère hors-saison (422) sans planter.

from __future__ import annotations
from pathlib import Path
import argparse, os, sys
from typing import List, Dict, Any
import pandas as pd
import requests
from dotenv import load_dotenv

DATA = Path("data")
DATA.mkdir(exist_ok=True)

API_URL_SCORES = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores"

def get_api_key():
    load_dotenv()
    k = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not k:
        raise SystemExit("ODDS_API_KEY manquant. Mets-le dans .env ou passe --api-key")
    return k

def http_get(url: str, params: Dict[str,Any]) -> Any:
    r = requests.get(url, params=params, timeout=30)
    # hors-saison => 422 ; on log et retourne []
    if r.status_code == 422:
        print(f"[GET 422] remaining={r.headers.get('x-requests-remaining')} used={r.headers.get('x-requests-used')} params={params}")
        return []
    r.raise_for_status()
    print(f"[GET {r.status_code}] remaining={r.headers.get('x-requests-remaining')} used={r.headers.get('x-requests-used')} params={params}")
    return r.json()

def normalize_rows(payload: List[Dict[str,Any]]) -> pd.DataFrame:
    rows = []
    for g in payload:
        rows.append({
            "id": g.get("id"),
            "commence_time": g.get("commence_time"),
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "completed": g.get("completed"),
            "home_score": g.get("home_score"),
            "away_score": g.get("away_score"),
        })
    return pd.DataFrame(rows)

def fetch_scores_days(api_key: str, days_from: int) -> pd.DataFrame:
    payload = http_get(API_URL_SCORES, {
        "apiKey": api_key,
        "daysFrom": int(days_from),
        "dateFormat": "iso",
    })
    if not payload:
        return pd.DataFrame()
    return normalize_rows(payload)

def fetch_scores_season(api_key: str, season: int) -> pd.DataFrame:
    # L’API scores n’a pas un vrai filtre saison → on combine des fenêtres si besoin,
    # mais garde simple ici (hors-saison retourne souvent 422)
    return pd.DataFrame()

def save_results(df: pd.DataFrame):
    if df.empty:
        print("[INFO] Aucune donnée à ajouter (hors-saison ou pas de matchs).")
        return
    out = DATA / "nfl_results.csv"
    if out.exists():
        old = pd.read_csv(out)
        merged = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["id"]).reset_index(drop=True)
        merged.to_csv(out, index=False)
        print(f"[OK] MAJ → {out} ({len(merged)} lignes)")
    else:
        df.to_csv(out, index=False)
        print(f"[OK] Créé → {out} ({len(df)} lignes)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="Fenêtre de jours en arrière")
    ap.add_argument("--season", type=int, default=None, help="Optionnel, non utilisé si hors-saison")
    args = ap.parse_args()

    api_key = get_api_key()

    if args.days:
        print(f"[INFO] Fetch by DAYS {args.days}")
        df_raw = fetch_scores_days(api_key, args.days)
    else:
        print(f"[INFO] Fetch by SEASON {args.season}")
        df_raw = fetch_scores_season(api_key, args.season or 2024)

    save_results(df_raw)

if __name__ == "__main__":
    main()
