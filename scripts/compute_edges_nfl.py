#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule les "edges" NFL (moneyline/h2h) à partir de data/nfl_odds.csv (The Odds API),
ajoute l'horaire du match et trie par date/heure (UTC puis affichage local côté dashboard).

Usage :
  python scripts/compute_edges_nfl.py --home_bump 0.02 --min_edge 0.01
Sortie :
  data/nfl_edges_of_day.csv
"""

import argparse
import pandas as pd
import numpy as np


def american_to_prob(odds: int) -> float:
    """Convertit des cotes américaines en probabilité implicite (avec vig)."""
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def no_vig_two_way(p1_hat: float, p2_hat: float):
    """Retire la marge (vig) sur un marché 2-issues en renormalisant les probabilités."""
    z = p1_hat + p2_hat
    if z <= 0:
        return float("nan"), float("nan")
    return p1_hat / z, p2_hat / z


def prob_to_decimal(p: float) -> float:
    """Convertit une probabilité en cote décimale 'juste' (1/p)."""
    return float("nan") if p <= 0 or p >= 1 else 1.0 / p


def market_consensus(df_h2h_event: pd.DataFrame, home_team: str, away_team: str):
    """
    Calcule un consensus de probas 'sans vig' en moyennant par bookmaker
    (uniquement ceux qui proposent les 2 issues home/away).
    Retourne : (p_home_consensus, p_away_consensus, nb_bookmakers_utilisés)
    """
    p_home_list, p_away_list = [], []
    for book, sub in df_h2h_event.groupby("book"):
        h = sub[sub["name"] == home_team]
        a = sub[sub["name"] == away_team]
        if len(h) == 1 and len(a) == 1:
            p_hat_home = american_to_prob(int(h.iloc[0]["price_american"]))
            p_hat_away = american_to_prob(int(a.iloc[0]["price_american"]))
            ph, pa = no_vig_two_way(p_hat_home, p_hat_away)
            if np.isfinite(ph) and np.isfinite(pa):
                p_home_list.append(ph)
                p_away_list.append(pa)
    if not p_home_list:
        return float("nan"), float("nan"), 0
    return float(np.mean(p_home_list)), float(np.mean(p_away_list)), len(p_home_list)


def compute_edges(nfl_odds_csv: str, home_bump: float, min_edge: float) -> pd.DataFrame:
    """
    Lit le CSV d'odds NFL, calcule les edges (modèle - marché sans vig),
    ajoute l'horaire du match, filtre par edge minimal et trie par horaire.
    """
    df = pd.read_csv(nfl_odds_csv)

    # On garde uniquement le marché moneyline (h2h)
    df_h2h = df[df["market"] == "h2h"].copy()

    rows = []
    for event_id, sub in df_h2h.groupby("game_id"):
        home_team = sub["home_team"].iloc[0]
        away_team = sub["away_team"].iloc[0]
        commence_time = sub["commence_time"].iloc[0]  # ISO UTC (ex: 2025-09-07T17:00:00Z)

        p_home_mkt, p_away_mkt, n_books = market_consensus(sub, home_team, away_team)
        if not np.isfinite(p_home_mkt) or not np.isfinite(p_away_mkt):
            continue

        # Mini-modèle pédagogique : +home_bump pour l'équipe à domicile (ex: +0.02 = +2%)
        p_home_model = max(0.0, p_home_mkt + home_bump)
        p_away_model = max(0.0, p_away_mkt - home_bump)

        # Normalisation (sécurité)
        s = p_home_model + p_away_model
        if s == 0:
            continue
        p_home_model /= s
        p_away_model = 1 - p_home_model

        edge_home = p_home_model - p_home_mkt
        edge_away = p_away_model - p_away_mkt

        rows.append(
            {
                "game_id": event_id,
                "commence_time": commence_time,  # on garde l'horaire (UTC)
                "home_team": home_team,
                "away_team": away_team,
                "books_used": n_books,
                "home_market_prob": round(p_home_mkt, 4),
                "away_market_prob": round(p_away_mkt, 4),
                "home_model_prob": round(p_home_model, 4),
                "away_model_prob": round(p_away_model, 4),
                "home_edge": round(edge_home, 4),
                "away_edge": round(edge_away, 4),
                "home_fair_odds_decimal": round(prob_to_decimal(p_home_model), 3),
                "away_fair_odds_decimal": round(prob_to_decimal(p_away_model), 3),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Meilleur edge (home ou away)
    out["best_edge"] = out[["home_edge", "away_edge"]].max(axis=1)

    # Filtre edge minimal et tri chronologique (UTC), puis edge décroissant
    out = out[out["best_edge"] >= min_edge].copy()
    out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
    out = out.sort_values(["commence_time", "best_edge"], ascending=[True, False]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds_csv", default="data/nfl_odds.csv")
    ap.add_argument("--home_bump", type=float, default=0.02, help="Boost domicile (ex: 0.02 = +2%)")
    ap.add_argument("--min_edge", type=float, default=0.01, help="Seuil d'edge minimal (ex: 0.01 = 1%)")
    ap.add_argument("--out_csv", default="data/nfl_edges_of_day.csv")
    args = ap.parse_args()

    df = compute_edges(args.odds_csv, args.home_bump, args.min_edge)
    if df.empty:
        print("Aucun edge détecté (vérifie data/nfl_odds.csv ou baisse --min_edge).")
        return

    df.to_csv(args.out_csv, index=False)
    print(f"✅ Écrit {args.out_csv} ({len(df)} lignes)")
    # Aperçu console (top 10)
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
