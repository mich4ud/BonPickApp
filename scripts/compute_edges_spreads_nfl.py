#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule des "edges" pour NFL — Spreads (market = 'spreads') à partir de data/nfl_odds.csv.

Modèle MVP: on part du consensus marché SANS MARGE (no-vig).
Option: --home_cover_bump permet un petit biais pour l'équipe à domicile à couvrir le spread.

Sortie:
  data/nfl_spreads_edges.csv
"""

import argparse
import pandas as pd
import numpy as np

def american_to_prob(odds: int) -> float:
    return 100.0/(odds+100.0) if odds > 0 else -odds/(-odds+100.0)

def no_vig_two_way(p1_hat: float, p2_hat: float):
    z = p1_hat + p2_hat
    if z <= 0: return float('nan'), float('nan')
    return p1_hat/z, p2_hat/z

def prob_to_decimal(p: float) -> float:
    return float('nan') if p<=0 or p>=1 else 1.0/p

def market_consensus_spread(df_event, home_team, away_team):
    """
    Pour chaque bookmaker, on cherche une cote pour le home_team et une pour le away_team.
    La colonne 'point' indique le spread associé à CE côté (donc home_point et away_point sont opposés).
    Retourne: p_home_cover, p_away_cover, nb_books, home_point_moy, away_point_moy
    """
    p_home_list, p_away_list = [], []
    home_points, away_points = [], []
    for book, sub in df_event.groupby('book'):
        h = sub[sub['name']==home_team]
        a = sub[sub['name']==away_team]
        if len(h)==1 and len(a)==1:
            home_points.append(h.iloc[0]['point'])
            away_points.append(a.iloc[0]['point'])
            ph_hat = american_to_prob(int(h.iloc[0]['price_american']))
            pa_hat = american_to_prob(int(a.iloc[0]['price_american']))
            ph, pa = no_vig_two_way(ph_hat, pa_hat)
            if np.isfinite(ph) and np.isfinite(pa):
                p_home_list.append(ph); p_away_list.append(pa)
    if not p_home_list:
        return float('nan'), float('nan'), 0, float('nan'), float('nan')
    return (float(np.mean(p_home_list)), float(np.mean(p_away_list)), len(p_home_list),
            float(np.nanmean(home_points)), float(np.nanmean(away_points)))

def compute_edges_spreads(odds_csv: str, home_cover_bump: float, min_edge: float) -> pd.DataFrame:
    df = pd.read_csv(odds_csv)
    df_sp = df[df['market']=='spreads'].copy()
    if df_sp.empty:
        return pd.DataFrame()

    rows = []
    for event_id, sub in df_sp.groupby('game_id'):
        home = sub['home_team'].iloc[0]; away = sub['away_team'].iloc[0]
        t = sub['commence_time'].iloc[0]
        p_home_mkt, p_away_mkt, n_books, home_pt, away_pt = market_consensus_spread(sub, home, away)
        if not np.isfinite(p_home_mkt) or not np.isfinite(p_away_mkt):
            continue

        # Modèle MVP: petit biais pour l'équipe à domicile "couvre le spread"
        p_home_model = max(0.0, p_home_mkt + home_cover_bump)
        p_away_model = max(0.0, p_away_mkt - home_cover_bump)
        s = p_home_model + p_away_model
        if s<=0: 
            continue
        p_home_model /= s
        p_away_model = 1 - p_home_model

        edge_home = p_home_model - p_home_mkt
        edge_away = p_away_model - p_away_mkt

        rows.append({
            'game_id': event_id,
            'commence_time': t,
            'home_team': home,
            'away_team': away,
            'books_used': n_books,
            'home_spread': home_pt,
            'away_spread': away_pt,
            'home_market_prob': round(p_home_mkt,4),
            'away_market_prob': round(p_away_mkt,4),
            'home_model_prob': round(p_home_model,4),
            'away_model_prob': round(p_away_model,4),
            'home_edge': round(edge_home,4),
            'away_edge': round(edge_away,4),
            'home_fair_odds_decimal': round(prob_to_decimal(p_home_model),3),
            'away_fair_odds_decimal': round(prob_to_decimal(p_away_model),3),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['best_edge'] = out[['home_edge','away_edge']].max(axis=1)
    out['commence_time'] = pd.to_datetime(out['commence_time'], utc=True, errors='coerce')
    out = out[out['best_edge']>=min_edge].sort_values(['commence_time','best_edge'], ascending=[True, False]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odds_csv', default='data/nfl_odds.csv')
    ap.add_argument('--home_cover_bump', type=float, default=0.00, help="Biais domicile couvre spread (ex: 0.01)")
    ap.add_argument('--min_edge', type=float, default=0.00, help="Seuil d'edge minimal (0 pour tout voir)")
    ap.add_argument('--out_csv', default='data/nfl_spreads_edges.csv')
    args = ap.parse_args()

    df = compute_edges_spreads(args.odds_csv, args.home_cover_bump, args.min_edge)
    if df.empty:
        print("Aucun edge (spreads). Vérifie nfl_odds.csv ou baisse --min_edge.")
        return
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Écrit {args.out_csv} ({len(df)} lignes)")
    print(df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
