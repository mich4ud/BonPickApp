# scripts/test_odds_key.py
from __future__ import annotations
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

def load_api_key(cli_key: str | None = None) -> str:
    if cli_key:
        return cli_key.strip()
    env_root = ROOT / ".env"
    if env_root.exists():
        load_dotenv(env_root, override=True)
    if Path(".env").exists():
        load_dotenv(".env", override=True)
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit("ODDS_API_KEY introuvable. Passe --api-key ou mets-le dans .env")
    return key

def main():
    import argparse
    p = argparse.ArgumentParser(description="Test The Odds API key (NFL odds)")
    p.add_argument("--api-key", type=str, default=None, help="Clé The Odds API (prioritaire sur .env)")
    p.add_argument("--market", type=str, default="h2h", help="h2h | spreads | totals")
    p.add_argument("--format", type=str, default="decimal", help="decimal | american")
    p.add_argument("--regions", type=str, default="us", help="us | eu | uk | au")
    args = p.parse_args()

    key = load_api_key(args.api_key)
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"apiKey": key, "regions": args.regions, "markets": args.market, "oddsFormat": args.format, "dateFormat": "iso"}
    print("[GET]", url)
    print("[params]", params)
    r = requests.get(url, params=params, timeout=30)
    print("HTTP:", r.status_code)
    print("X-Requests-Remaining:", r.headers.get("x-requests-remaining"))
    print("X-Requests-Used:", r.headers.get("x-requests-used"))
    if "application/json" in r.headers.get("content-type",""):
        data = r.json()
        if isinstance(data, list) and data:
            g = data[0]
            print("Sample game:", g.get("home_team"), "vs", g.get("away_team"), "| bookmakers:", len(g.get("bookmakers",[])))
        else:
            print("JSON OK mais liste vide.")
    else:
        print("Réponse non JSON:", r.text[:300])

if __name__ == "__main__":
    main()
