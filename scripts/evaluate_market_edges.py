# scripts/evaluate_market_edges.py
# Edges NFL vs marché (Odds API) pour moneyline, spreads, totals
# Version avec:
# - normalisation des noms d'équipes (robuste),
# - compteurs / logs de debug,
# - options de régions multiples.

from __future__ import annotations
from pathlib import Path
import argparse, os, json, sys, re
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from math import erf, sqrt

DATA = Path("data")
RESULTS_CSV = DATA / "nfl_results.csv"

# Paramètres modèle
ELO_HFA_ELO   = 55.0
ELO_SCALE     = 400.0
ELO_TO_POINTS = 3.0 / ELO_HFA_ELO   # ~0.0545 pt par Elo
SPREAD_SIGMA  = 13.5
TOTAL_SIGMA   = 10.5
TOTAL_WINDOW  = 8
EPS = 1e-12

# ---------- Utils proba ----------
def sigmoid_elo(rdiff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rdiff / ELO_SCALE))

def cdf_normal(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * sqrt(2))
    from math import erf
    return 0.5 * (1.0 + erf(z))

def implied_from_decimal(odds: float) -> float:
    return 1.0 / max(odds, EPS)

def american_to_decimal(american: float) -> Optional[float]:
    a = float(american)
    if a >= 100:
        return 1.0 + a/100.0
    if a <= -100:
        return 1.0 + 100.0/abs(a)
    return None

def devigorize_pair(p1: float, p2: float) -> Tuple[float,float]:
    s = p1 + p2
    if s <= 0:
        return p1, p2
    return p1 / s, p2 / s

# ---------- Normalisation noms d'équipes ----------
_norm_re = re.compile(r"[^A-Z0-9]+")
def norm_team(s: str) -> str:
    """
    Normalise un nom: uppercase, enlève accents/ponctuation/espaces multiples.
    Exemple: 'New York Jets' -> 'NEWYORKJETS'
    """
    if s is None:
        return ""
    s = s.upper()
    # retire accents basiques
    s = (s.replace("É","E").replace("È","E").replace("Ê","E").replace("Ë","E")
            .replace("À","A").replace("Â","A").replace("Ä","A")
            .replace("Ô","O").replace("Ö","O")
            .replace("Î","I").replace("Ï","I")
            .replace("Û","U").replace("Ü","U")
            .replace("Ç","C"))
    s = _norm_re.sub("", s)
    return s

# ---------- Ratings Elo ----------
def load_end_of_season_ratings(season: int) -> Dict[str, float]:
    path = DATA / f"model_moneyline_{season}.csv"
    if not path.exists():
        raise SystemExit(f"{path} introuvable. Lance d’abord build_elo_nfl.py --season {season}")
    df = pd.read_csv(path)
    ratings = {}
    for _, r in df.iterrows():
        h = str(r["home_team"]).strip()
        a = str(r["away_team"]).strip()
        if pd.notna(r.get("home_elo_pre")):
            ratings[norm_team(h)] = float(r["home_elo_pre"])
        if pd.notna(r.get("away_elo_pre")):
            ratings[norm_team(a)] = float(r["away_elo_pre"])
    if not ratings:
        teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
        for t in teams:
            ratings[norm_team(str(t).strip())] = 1500.0
    return ratings

# ---------- Baseline Totals ----------
def load_team_totals_baseline(season: int) -> Dict[str, float]:
    if not RESULTS_CSV.exists():
        return {}
    df = pd.read_csv(RESULTS_CSV)
    if "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    mask = True
    if "commence_time" in df.columns:
        mask = (df["commence_time"].dt.year >= season) & (df["commence_time"].dt.year <= season+1)
    g = df[mask].copy()
    if "home_score" not in g.columns or "away_score" not in g.columns:
        return {}
    g["total_points"] = g["home_score"].fillna(0) + g["away_score"].fillna(0)

    home_long = g[["commence_time","home_team","total_points"]].rename(columns={"home_team":"team"})
    away_long = g[["commence_time","away_team","total_points"]].rename(columns={"away_team":"team"})
    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(["team","commence_time"], na_position="last").reset_index(drop=True)
    long_df["rolling_total"] = (
        long_df.groupby("team")["total_points"]
               .apply(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
               .reset_index(level=0, drop=True)
    )
    last_vals = long_df.dropna(subset=["rolling_total"]).groupby("team")["rolling_total"].last()
    # Normalise les clés
    return {norm_team(str(k)): float(v) for k,v in last_vals.items()}

def total_mu_for_match(team_totals: Dict[str,float], home: str, away: str, league_mean: float) -> float:
    hv = team_totals.get(norm_team(home))
    av = team_totals.get(norm_team(away))
    vals = [v for v in [hv,av] if v is not None]
    if vals:
        return float(np.mean(vals))
    return league_mean

# ---------- Odds API ----------
def fetch_odds_api(api_key: str, regions: str, odds_format: str) -> List[Dict[str,Any]]:
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,                 # ex: "us,eu"
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal" if odds_format=="decimal" else "american",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- Extraction prix/lines ----------
def extract_best_moneyline(game: Dict[str,Any], odds_format: str, home_team: str, away_team: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    nh = norm_team(home_team); na = norm_team(away_team)
    best_home = None; best_away = None; book_home = None; book_away = None
    for bk in game.get("bookmakers", []):
        bname = bk.get("title") or bk.get("key")
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            for out in mk.get("outcomes", []):
                team = out.get("name"); price = out.get("price")
                if price is None or team is None:
                    continue
                if odds_format == "american" and isinstance(price, (int,float)):
                    price = american_to_decimal(price)
                if not isinstance(price, (int,float)):
                    continue
                nt = norm_team(str(team))
                if nt == nh:
                    if (best_home is None) or (price > best_home):
                        best_home = float(price); book_home = bname
                elif nt == na:
                    if (best_away is None) or (price > best_away):
                        best_away = float(price); book_away = bname
    chosen_book = book_home or book_away
    return best_home, best_away, chosen_book

def extract_best_spreads(game: Dict[str,Any], odds_format: str, home_team: str, away_team: str) -> Tuple[Optional[Tuple[float,float,str]], Optional[Tuple[float,float,str]]]:
    nh = norm_team(home_team); na = norm_team(away_team)
    best_home = None  # (line, price, book)
    best_away = None
    for bk in game.get("bookmakers", []):
        bname = bk.get("title") or bk.get("key")
        for mk in bk.get("markets", []):
            if mk.get("key") != "spreads":
                continue
            for out in mk.get("outcomes", []):
                team = out.get("name"); price = out.get("price"); line = out.get("point")
                if price is None or line is None or team is None:
                    continue
                if odds_format == "american" and isinstance(price, (int,float)):
                    price = american_to_decimal(price)
                if not isinstance(price, (int,float)):
                    continue
                nt = norm_team(str(team))
                tup = (float(line), float(price), bname)
                if nt == nh:
                    if (best_home is None) or (price > best_home[1]):
                        best_home = tup
                elif nt == na:
                    if (best_away is None) or (price > best_away[1]):
                        best_away = tup
    return best_home, best_away

def extract_best_totals(game: Dict[str,Any], odds_format: str) -> Tuple[Optional[Tuple[float,float,str]], Optional[Tuple[float,float,str]]]:
    best_over = None; best_under = None
    for bk in game.get("bookmakers", []):
        bname = bk.get("title") or bk.get("key")
        for mk in bk.get("markets", []):
            if mk.get("key") != "totals":
                continue
            for out in mk.get("outcomes", []):
                name = out.get("name"); price = out.get("price"); point = out.get("point")
                if price is None or point is None or name is None:
                    continue
                if odds_format == "american" and isinstance(price, (int,float)):
                    price = american_to_decimal(price)
                if not isinstance(price, (int,float)):
                    continue
                tup = (float(point), float(price), bname)
                nm = str(name).lower()
                if nm.startswith("over"):
                    if (best_over is None) or (price > best_over[1]):
                        best_over = tup
                elif nm.startswith("under"):
                    if (best_under is None) or (price > best_under[1]):
                        best_under = tup
    return best_over, best_under

# ---------- Modèle ----------
def p_home_from_ratings(ratings: Dict[str,float], home: str, away: str) -> Optional[float]:
    rh = ratings.get(norm_team(home)); ra = ratings.get(norm_team(away))
    if rh is None or ra is None:
        return None
    rdiff = (rh + ELO_HFA_ELO) - ra
    return sigmoid_elo(rdiff)

def spread_mu_from_ratings(ratings: Dict[str,float], home: str, away: str) -> Optional[float]:
    rh = ratings.get(norm_team(home)); ra = ratings.get(norm_team(away))
    if rh is None or ra is None:
        return None
    rdiff_hfa = (rh + ELO_HFA_ELO) - ra
    return rdiff_hfa * ELO_TO_POINTS

def prob_home_cover_at_line(spread_mu: float, line_home: float) -> float:
    if line_home < 0:
        thr = abs(line_home)
    elif line_home > 0:
        thr = -abs(line_home)
    else:
        thr = 0.0
    return 1.0 - cdf_normal(thr, spread_mu, SPREAD_SIGMA)

def total_mu_for_match(team_totals: Dict[str,float], home: str, away: str, league_mean: float) -> float:
    hv = team_totals.get(norm_team(home))
    av = team_totals.get(norm_team(away))
    vals = [v for v in [hv,av] if v is not None]
    return float(np.mean(vals)) if vals else league_mean

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Edges NFL vs marché (Odds API) pour moneyline, spreads, totals")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--market", choices=["moneyline","spreads","totals","all"], default="all")
    ap.add_argument("--edge-min", type=float, default=0.03)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--from-api", action="store_true", default=True)
    ap.add_argument("--odds-file", type=str, default=None)
    ap.add_argument("--regions", type=str, default="us")  # accepte "us,eu"
    ap.add_argument("--format", choices=["decimal","american"], default="decimal")
    ap.add_argument("--api-key", type=str, default=None)
    args = ap.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("ODDS_API_KEY")
    if args.from_api and not api_key:
        raise SystemExit("ODDS_API_KEY manquant. Mets-le dans .env ou passe --api-key")

    ratings = load_end_of_season_ratings(args.season)

    # Baseline totals
    league_mean_total = 43.0
    team_total_baseline = {}
    if RESULTS_CSV.exists():
        try:
            df_res = pd.read_csv(RESULTS_CSV)
            if "home_score" in df_res.columns and "away_score" in df_res.columns:
                df_res["total_points"] = df_res["home_score"].fillna(0) + df_res["away_score"].fillna(0)
                if df_res["total_points"].notna().any():
                    league_mean_total = float(df_res["total_points"].dropna().mean())
        except Exception:
            pass
        team_total_baseline = load_team_totals_baseline(args.season)

    # Charge odds
    if args.from_api:
        try:
            games = fetch_odds_api(api_key, args.regions, args.format)
        except requests.HTTPError as e:
            print("[ERREUR API]", e)
            sys.exit(1)
    else:
        if not args.odds_file:
            raise SystemExit("Passe --odds-file si tu n’utilises pas --from-api.")
        p = Path(args.odds_file)
        if not p.exists():
            raise SystemExit(f"Fichier odds introuvable: {p}")
        if p.suffix.lower() == ".json":
            games = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise SystemExit("Fournis un JSON de l’API.")

    # Compteurs debug
    n_games = len(games)
    n_h2h_seen = n_h2h_prices = n_h2h_ratings_miss = 0
    n_sp_seen  = n_sp_prices  = n_sp_ratings_miss  = 0
    n_to_seen  = n_to_prices  = 0

    out_moneyline, out_spreads, out_totals = [], [], []

    for g in games:
        home = g.get("home_team"); away = g.get("away_team"); kickoff = g.get("commence_time")
        if not home or not away:
            continue

        # H2H ?
        has_h2h = any(mk.get("key")=="h2h" for bk in g.get("bookmakers",[]) for mk in bk.get("markets",[]))
        if has_h2h: n_h2h_seen += 1
        # Spreads ?
        has_sp  = any(mk.get("key")=="spreads" for bk in g.get("bookmakers",[]) for mk in bk.get("markets",[]))
        if has_sp: n_sp_seen += 1
        # Totals ?
        has_to  = any(mk.get("key")=="totals" for bk in g.get("bookmakers",[]) for mk in bk.get("markets",[]))
        if has_to: n_to_seen += 1

        # -------- MONEYLINE --------
        if args.market in ("moneyline","all") and has_h2h:
            price_home, price_away, book_ml = extract_best_moneyline(g, args.format, home, away)
            if price_home and price_away:
                n_h2h_prices += 1
                pH_mkt, pA_mkt = devigorize_pair(implied_from_decimal(price_home), implied_from_decimal(price_away))
                p_home = p_home_from_ratings(ratings, home, away)
                if p_home is None:
                    n_h2h_ratings_miss += 1
                else:
                    p_away = 1 - p_home
                    edge_home = p_home - pH_mkt
                    edge_away = p_away - pA_mkt
                    pick_team, pick_edge, pick_odds = (
                        (home, edge_home, price_home) if edge_home >= edge_away else (away, edge_away, price_away)
                    )
                    out_moneyline.append({
                        "kickoff": kickoff, "home_team": home, "away_team": away, "bookmaker": book_ml,
                        "odds_home": round(price_home,3), "odds_away": round(price_away,3),
                        "p_model_home": round(p_home,4), "p_model_away": round(1-p_home,4),
                        "p_mkt_home": round(pH_mkt,4),  "p_mkt_away": round(pA_mkt,4),
                        "edge_home": round(edge_home,4), "edge_away": round(edge_away,4),
                        "pick": pick_team, "edge_pick": round(pick_edge,4), "odds_pick": round(pick_odds,3),
                    })

        # -------- SPREADS --------
        if args.market in ("spreads","all") and has_sp:
            best_home, best_away = extract_best_spreads(g, args.format, home, away)
            if best_home and best_away:
                n_sp_prices += 1
                line_home, price_home_sp, book_sp_h = best_home
                line_away, price_away_sp, book_sp_a = best_away
                pH_sp_mkt, pA_sp_mkt = devigorize_pair(implied_from_decimal(price_home_sp), implied_from_decimal(price_away_sp))
                mu = spread_mu_from_ratings(ratings, home, away)
                if mu is None:
                    n_sp_ratings_miss += 1
                else:
                    p_home_cover = 1.0 - cdf_normal(abs(line_home) if line_home<0 else -abs(line_home), mu, SPREAD_SIGMA)
                    if line_away < 0:
                        thr_away = abs(line_away); p_away_cover = cdf_normal(thr_away, mu, SPREAD_SIGMA)
                    elif line_away > 0:
                        thr_away = -abs(line_away); p_away_cover = cdf_normal(thr_away, mu, SPREAD_SIGMA)
                    else:
                        p_away_cover = cdf_normal(0.0, mu, SPREAD_SIGMA)

                    edge_home = p_home_cover - pH_sp_mkt
                    edge_away = p_away_cover - pA_sp_mkt
                    if edge_home >= edge_away:
                        pick_team, pick_edge, pick_odds, pick_line, pick_book = home, edge_home, price_home_sp, line_home, book_sp_h
                    else:
                        pick_team, pick_edge, pick_odds, pick_line, pick_book = away, edge_away, price_away_sp, line_away, book_sp_a

                    out_spreads.append({
                        "kickoff": kickoff, "home_team": home, "away_team": away, "bookmaker": pick_book,
                        "home_line": round(line_home,1), "away_line": round(line_away,1),
                        "odds_home": round(price_home_sp,3), "odds_away": round(price_away_sp,3),
                        "spread_mu": round(mu,2), "sigma": SPREAD_SIGMA,
                        "p_model_home_cover": round(p_home_cover,4), "p_model_away_cover": round(p_away_cover,4),
                        "p_mkt_home_cover": round(pH_sp_mkt,4), "p_mkt_away_cover": round(pA_sp_mkt,4),
                        "edge_home": round(edge_home,4), "edge_away": round(edge_away,4),
                        "pick": pick_team, "edge_pick": round(pick_edge,4),
                        "line_pick": round(pick_line,1), "odds_pick": round(pick_odds,3),
                    })

        # -------- TOTALS --------
        if args.market in ("totals","all") and has_to:
            best_over, best_under = extract_best_totals(g, args.format)
            if best_over and best_under:
                n_to_prices += 1
                line_over, price_over, book_to = best_over
                line_under, price_under, book_tu = best_under
                p_over_mkt, p_under_mkt = devigorize_pair(implied_from_decimal(price_over), implied_from_decimal(price_under))
                mu = total_mu_for_match(team_total_baseline, home, away, league_mean_total)
                line_eval = float(line_over if abs(line_over - line_under) < 0.01 else (line_over + line_under)/2.0)
                from math import erf, sqrt
                p_over_model  = 1.0 - cdf_normal(line_eval, mu, TOTAL_SIGMA)
                p_under_model = 1.0 - p_over_model
                edge_over  = p_over_model  - p_over_mkt
                edge_under = p_under_model - p_under_mkt
                if edge_over >= edge_under:
                    pick, pick_edge, pick_odds, pick_line, pick_book = "OVER", edge_over, price_over, line_over, book_to
                else:
                    pick, pick_edge, pick_odds, pick_line, pick_book = "UNDER", edge_under, price_under, line_under, book_tu
                out_totals.append({
                    "kickoff": kickoff, "home_team": home, "away_team": away, "bookmaker": pick_book,
                    "total_line_over": round(line_over,1), "total_line_under": round(line_under,1),
                    "odds_over": round(price_over,3), "odds_under": round(price_under,3),
                    "total_mu": round(mu,2), "sigma": TOTAL_SIGMA,
                    "p_model_over": round(p_over_model,4), "p_model_under": round(p_under_model,4),
                    "p_mkt_over": round(p_over_mkt,4), "p_mkt_under": round(p_under_mkt,4),
                    "edge_over": round(edge_over,4), "edge_under": round(edge_under,4),
                    "pick": pick, "edge_pick": round(pick_edge,4),
                    "line_pick": round(pick_line,1), "odds_pick": round(pick_odds,3),
                })

    # ---- Logs debug ----
    print(f"[DEBUG] Jeux API: {n_games}")
    print(f"[DEBUG] h2h vus: {n_h2h_seen} | h2h extraits: {n_h2h_prices} | h2h ratings manquants: {n_h2h_ratings_miss}")
    print(f"[DEBUG] spreads vus: {n_sp_seen} | spreads extraits: {n_sp_prices} | spreads ratings manquants: {n_sp_ratings_miss}")
    print(f"[DEBUG] totals vus: {n_to_seen} | totals extraits: {n_to_prices}")

    # ---- Exports / affichage ----
    def save_and_print(df: pd.DataFrame, name: str):
        if df.empty:
            print(f"\n[{name.upper()}] aucune ligne calculée.")
            return
        out = df[df["edge_pick"] >= args.edge_min].sort_values("edge_pick", ascending=False).reset_index(drop=True)
        if out.empty:
            print(f"\n[{name.upper()}] 0 ligne ≥ edge {args.edge_min:.2f}.")
            return
        path = DATA / f"market_edges_{name}_{args.season}.csv"
        out.to_csv(path, index=False)
        print(f"\n[{name.upper()}]")
        print(out.head(args.top_n).to_string(index=False))
        print(f"[CSV] → {path}  ({len(out)} lignes ≥ edge {args.edge_min:.2f})")

    save_and_print(pd.DataFrame(out_moneyline), "moneyline")
    save_and_print(pd.DataFrame(out_spreads),   "spreads")
    save_and_print(pd.DataFrame(out_totals),    "totals")

if __name__ == "__main__":
    main()
