#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applique des ajustements simples (blessures / météo / repos) aux edges NFL.
- Lit les fichiers edges Moneyline / Totals / Spreads
- Lit data/nfl_factors.csv (si présent)
- Écrit des versions *ajustées* avec suffixe _adj.csv
"""

import argparse
import os
import pandas as pd
import numpy as np

DATA_DIR = "data"

def load_factors(path=f"{DATA_DIR}/nfl_factors.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    defaults = dict(inj_sev_home=0.0, inj_sev_away=0.0, wind_mph=0.0, precip=0,
                    rest_home=0, rest_away=0, bye_home=0, bye_away=0)
    for k,v in defaults.items():
        if k not in df.columns: df[k] = v
    return df

def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

# ---------- MONEYLINE ----------
def adjust_moneyline(df_ml: pd.DataFrame, factors: pd.DataFrame,
                     k_inj=0.05, k_rest=0.005) -> pd.DataFrame:
    if df_ml is None or df_ml.empty: 
        return df_ml
    if factors is None or factors.empty:
        return df_ml.copy()

    df = df_ml.merge(factors, on="game_id", how="left")
    df.fillna({'inj_sev_home':0.0,'inj_sev_away':0.0,'rest_home':0,'rest_away':0}, inplace=True)

    out = df.copy()
    home_delta = -(df['inj_sev_home'] * k_inj) + ((df['rest_home'] - df['rest_away']).clip(-3,3) * k_rest)
    away_delta = -(df['inj_sev_away'] * k_inj) + ((df['rest_away'] - df['rest_home']).clip(-3,3) * k_rest)

    out['home_model_prob_adj'] = (df['home_model_prob'] + home_delta).map(clamp01)
    out['away_model_prob_adj'] = (df['away_model_prob'] + away_delta).map(clamp01)
    s = (out['home_model_prob_adj'] + out['away_model_prob_adj']).replace(0, np.nan)
    out['home_model_prob_adj'] = out['home_model_prob_adj'] / s
    out['away_model_prob_adj'] = 1 - out['home_model_prob_adj']

    out['home_edge_adj'] = out['home_model_prob_adj'] - df['home_market_prob']
    out['away_edge_adj'] = out['away_model_prob_adj'] - df['away_market_prob']

    out['home_fair_odds_decimal_adj'] = out['home_model_prob_adj'].apply(lambda p: (1/p) if 0<p<1 else np.nan)
    out['away_fair_odds_decimal_adj'] = out['away_model_prob_adj'].apply(lambda p: (1/p) if 0<p<1 else np.nan)

    return out

# ---------- TOTALS ----------
def adjust_totals(df_tot: pd.DataFrame, factors: pd.DataFrame,
                  k_wind=0.02, k_precip=0.03) -> pd.DataFrame:
    if df_tot is None or df_tot.empty: 
        return df_tot
    if factors is None or factors.empty:
        return df_tot.copy()

    df = df_tot.merge(factors[['game_id','wind_mph','precip']], on="game_id", how="left")
    df.fillna({'wind_mph':0.0,'precip':0}, inplace=True)

    out = df.copy()
    over_delta = - ( (df['wind_mph']/5.0) * k_wind ) - ( (df['precip']>0).astype(float) * k_precip )
    under_delta = -over_delta

    out['over_model_prob_adj'] = (df['over_model_prob'] + over_delta).map(clamp01)
    out['under_model_prob_adj'] = (df['under_model_prob'] + under_delta).map(clamp01)
    s = (out['over_model_prob_adj'] + out['under_model_prob_adj']).replace(0, np.nan)
    out['over_model_prob_adj'] = out['over_model_prob_adj'] / s
    out['under_model_prob_adj'] = 1 - out['over_model_prob_adj']

    out['over_edge_adj'] = out['over_model_prob_adj'] - df['over_market_prob']
    out['under_edge_adj'] = out['under_model_prob_adj'] - df['under_market_prob']

    out['over_fair_odds_decimal_adj'] = out['over_model_prob_adj'].apply(lambda p: (1/p) if 0<p<1 else np.nan)
    out['under_fair_odds_decimal_adj'] = out['under_model_prob_adj'].apply(lambda p: (1/p) if 0<p<1 else np.nan)

    return out

# ---------- SPREADS ----------
def adjust_spreads(df_sp: pd.DataFrame, factors: pd.DataFrame,
                   k_inj=0.04, k_rest=0.004) -> pd.DataFrame:
    if df_sp is None or df_sp.empty: 
        return df_sp
    if factors is None or factors.empty:
        return df_sp.copy()

    df = df_sp.merge(factors[['game_id','inj_sev_home','inj_sev_away','rest_home','rest_away']], on="game_id", how="left")
    df.fillna({'inj_sev_home':0.0,'inj_sev_away':0.0,'rest_home':0,'rest_away':0}, inplace=True)

    out = df.copy()
    home_delta = -(df['inj_sev_home'] * k_inj) + ((df['rest_home'] - df['rest_away']).clip(-3,3) * k_rest)
    away_delta = -(df['inj_sev_away'] * k_inj) + ((df['rest_away'] - df['rest_home']).clip(-3,3) * k_rest)

    out['home_model_prob_adj'] = (df['home_model_prob'] + home_delta).map(clamp01)
    out['away_model_prob_adj'] = (df['away_model_prob'] + away_delta).map(clamp01)
    s = (out['home_model_prob_adj'] + out['away_model_prob_adj']).replace(0, np.nan)
    out['home_model_prob_adj'] = out['home_model_prob_adj'] / s
    out['away_model_prob_adj'] = 1 - out['home_model_prob_adj']

    out['home_edge_adj'] = out['home_model_prob_adj'] - df['home_market_prob']
    out['away_edge_adj'] = out['away_model_prob_adj'] - df['away_market_prob']

    out['home_fair_odds_decimal_adj'] = out['home_model_prob_adj'].apply(lambda p: (1/p) if 0<p<1 else np.nan)
    out['away_fair_odds_decimal_adj'] = out['away_model_prob_adj'].apply(lambda p: (1/p) if 0<p<1 else np.nan)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ml_in', default=f'{DATA_DIR}/nfl_edges_of_day.csv')
    ap.add_argument('--tot_in', default=f'{DATA_DIR}/nfl_totals_edges.csv')
    ap.add_argument('--sp_in',  default=f'{DATA_DIR}/nfl_spreads_edges.csv')
    ap.add_argument('--factors', default=f'{DATA_DIR}/nfl_factors.csv')
    args = ap.parse_args()

    factors = load_factors(args.factors)

    ml = pd.read_csv(args.ml_in) if os.path.exists(args.ml_in) else None
    if ml is not None and not ml.empty:
        ml_adj = adjust_moneyline(ml, factors)
        ml_adj.to_csv(f'{DATA_DIR}/nfl_edges_of_day_adj.csv', index=False)
        print(f'✅ Écrit {DATA_DIR}/nfl_edges_of_day_adj.csv')
    else:
        print("⚠️ Moneyline: fichier absent ou vide.")

    tot = pd.read_csv(args.tot_in) if os.path.exists(args.tot_in) else None
    if tot is not None and not tot.empty:
        tot_adj = adjust_totals(tot, factors)
        tot_adj.to_csv(f'{DATA_DIR}/nfl_totals_edges_adj.csv', index=False)
        print(f'✅ Écrit {DATA_DIR}/nfl_totals_edges_adj.csv')
    else:
        print("⚠️ Totals: fichier absent ou vide.")

    sp = pd.read_csv(args.sp_in) if os.path.exists(args.sp_in) else None
    if sp is not None and not sp.empty:
        sp_adj = adjust_spreads(sp, factors)
        sp_adj.to_csv(f'{DATA_DIR}/nfl_spreads_edges_adj.csv', index=False)
        print(f'✅ Écrit {DATA_DIR}/nfl_spreads_edges_adj.csv')
    else:
        print("⚠️ Spreads: fichier absent ou vide.")

if __name__ == "__main__":
    main()
