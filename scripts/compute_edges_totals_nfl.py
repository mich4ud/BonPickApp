#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule des "edges" pour NFL — Over/Under (market = 'totals') à partir de data/nfl_odds.csv.

Modèle MVP : on part du consensus marché SANS MARGE (no-vig).
Option: --under_bump permet de donner un petit biais vers l'Under (ex: 0.01 = +1 pt de proba),
utile si tu veux être conservateur. Par défaut 0.00 (neutre).

Sortie:
  data/nfl_totals_edges.csv
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

def market_consensus_over_under(df_event):
    """
    Agrège par bookmaker : il faut que le book propose Over et Under (même ligne).
    Retourne: p_over, p_under, nb_books, line (point).
    """
    p_over_list, p_under_list, line_list = [], [], []
    for book, sub in df_event.groupby('book'):
        over = sub[sub['name'].str.lower().str.contains('over', na=False)]
        under = sub[sub['name'].str.lower().str.contains('under', na=False)]
        if len(over)==1 and len(under)==1:
            # ligne (point) parfois légèrement différente d’un book à l’autre → on prend la moyenne
            line_list.append(np.mean([over.iloc[0]['point'], under.iloc[0]['point']]))
            po_hat = american_to_prob(int(over.iloc[0]['price_american']))
            pu_hat = american_to_prob(int(under.iloc[0]['price_american']))
            po, pu = no_vig_two_way(po_hat, pu_hat)
            if np.isfinite(po) and np.isfinite(pu):
                p_over_list.append(po); p_under_list.append(pu)
    if not p_over_list:
        return float('nan'), float('nan'), 0, float('nan')
    return float(np.mean(p_over_list)), float(np.mean(p_under_list)), len(p_over_list), float(np.nanmean(line_list))

def compute_edges_totals(odds_csv: str, under_bump: float, min_edge: float) -> pd.DataFrame:
    df = pd.read_csv(odds_csv)
    df_tot = df[df['market']=='totals'].copy()
    if df_tot.empty: 
        return pd.DataFrame()

    rows = []
    for (event_id), sub in df_tot.groupby('game_id'):
        home = sub['home_team'].iloc[0]; away = sub['away_team'].iloc[0]
        t = sub['commence_time'].iloc[0]
        p_over_mkt, p_under_mkt, n_books, line = market_consensus_over_under(sub)
        if not np.isfinite(p_over_mkt) or not np.isfinite(p_under_mkt):
            continue

        # Modèle MVP: petit biais optionnel vers l'Under
        p_over_model = max(0.0, p_over_mkt - under_bump)
        p_under_model = max(0.0, p_under_mkt + under_bump)
        s = p_over_model + p_under_model
        if s<=0: 
            continue
        p_over_model /= s
        p_under_model = 1 - p_over_model

        edge_over = p_over_model - p_over_mkt
        edge_under = p_under_model - p_under_mkt

        rows.append({
            'game_id': event_id,
            'commence_time': t,
            'home_team': home,
            'away_team': away,
            'books_used': n_books,
            'total_line': round(line, 1) if np.isfinite(line) else np.nan,
            'over_market_prob': round(p_over_mkt,4),
            'under_market_prob': round(p_under_mkt,4),
            'over_model_prob': round(p_over_model,4),
            'under_model_prob': round(p_under_model,4),
            'over_edge': round(edge_over,4),
            'under_edge': round(edge_under,4),
            'over_fair_odds_decimal': round(prob_to_decimal(p_over_model),3),
            'under_fair_odds_decimal': round(prob_to_decimal(p_under_model),3),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['best_edge'] = out[['over_edge','under_edge']].max(axis=1)
    out['commence_time'] = pd.to_datetime(out['commence_time'], utc=True, errors='coerce')
    out = out[out['best_edge']>=min_edge].sort_values(['commence_time','best_edge'], ascending=[True, False]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odds_csv', default='data/nfl_odds.csv')
    ap.add_argument('--under_bump', type=float, default=0.00, help="Biais vers Under (ex: 0.01 = +1pt proba Under)")
    ap.add_argument('--min_edge', type=float, default=0.00, help="Seuil d'edge minimal (0 pour tout voir)")
    ap.add_argument('--out_csv', default='data/nfl_totals_edges.csv')
    args = ap.parse_args()

    df = compute_edges_totals(args.odds_csv, args.under_bump, args.min_edge)
    if df.empty:
        print("Aucun edge (totals). Vérifie nfl_odds.csv ou baisse --min_edge.")
        return
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Écrit {args.out_csv} ({len(df)} lignes)")
    print(df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
