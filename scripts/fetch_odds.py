# scripts/fetch_odds.py
# Récupère les cotes NFL via The Odds API et écrit data/odds_nfl.json
# Usage:
#   python scripts/fetch_odds.py --league nfl --odds_format decimal --regions us,eu

from __future__ import annotations
from pathlib import Path
import argparse, os, json, sys
import requests
from dotenv import load_dotenv

DATA = Path("data")
DATA.mkdir(exist_ok=True)

def get_api_key():
    load_dotenv()
    k = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not k:
        raise SystemExit("THE_ODDS_API_KEY manquant. Ouvre .env et ajoute ta clé.")
    return k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="nfl")
    ap.add_argument("--odds_format", choices=["decimal","american"], default="decimal")
    ap.add_argument("--regions", default="us")
    args = ap.parse_args()

    api_key = get_api_key()

    sport_key = {
        "nfl": "americanfootball_nfl",
    }.get(args.league.lower(), args.league)

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": args.regions,
        "markets": "h2h,spreads,totals",
        "oddsFormat": args.odds_format,
        "dateFormat": "iso",
    }
    print("[GET]", url)
    print("[params]", params)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    out = DATA / f"odds_{args.league.lower()}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] Écrit →", out)
    print("X-Requests-Remaining:", r.headers.get("x-requests-remaining"))
    print("X-Requests-Used:", r.headers.get("x-requests-used"))

if __name__ == "__main__":
    main()
