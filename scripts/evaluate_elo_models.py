# scripts/evaluate_elo_models.py
# Évalue les modèles Elo (Moneyline / Spreads / Totals) construits pour une saison donnée.
# Sort un résumé (LogLoss pour ML, MAE/RMSE pour Spreads/Totals) et l'écrit dans data/model_eval_<season>.csv

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

DATA = Path("data")

def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pick_scores_cols(df, prefer_suffix_res=True):
    """
    Retourne (col_home, col_away) en choisissant les colonnes de scores disponibles.
    Si prefer_suffix_res=True, on préfère 'home_score_res'/'away_score_res' si elles existent,
    sinon 'home_score'/'away_score' si présentes.
    """
    if prefer_suffix_res and {"home_score_res","away_score_res"}.issubset(df.columns):
        return "home_score_res", "away_score_res"
    # fallback
    if {"home_score","away_score"}.issubset(df.columns):
        return "home_score", "away_score"
    # sinon, essaye variantes communes
    candidates = [
        ("home_score_x","away_score_x"),
        ("home_score_y","away_score_y"),
    ]
    for a,b in candidates:
        if {a,b}.issubset(df.columns):
            return a,b
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Évalue les modèles Elo (ML/Spreads/Totals) sur la saison")
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    # ---------- Charge résultats ----------
    res_path = DATA / "nfl_results.csv"
    if not res_path.exists():
        raise SystemExit(f"{res_path} introuvable.")
    res = pd.read_csv(res_path)

    # Restreint à la saison (large filet)
    if "commence_time" in res.columns:
        res["commence_time"] = pd.to_datetime(res["commence_time"], utc=True, errors="coerce")
        mask = (res["commence_time"].dt.year >= args.season) & (res["commence_time"].dt.year <= args.season + 1)
        res = res[mask | res["commence_time"].isna()].copy()

    # Ne garde que ce qu'il faut pour merger proprement
    keep_res_cols = {"game_id","home_score","away_score"}
    keep_res_cols = [c for c in keep_res_cols if c in res.columns]
    res_slim = res[keep_res_cols].copy()

    summary_rows = []

    # ---------- MONEYLINE ----------
    ml_path = DATA / f"model_moneyline_{args.season}.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        # merge avec suffixes contrôlés, pour ne pas casser les noms
        m = ml.merge(res_slim, on="game_id", how="inner", suffixes=("", "_res"))
        # choisir colonnes de scores (privilégier *_res)
        hs_col, as_col = pick_scores_cols(m, prefer_suffix_res=True)
        if hs_col and as_col and "home_win_prob" in m.columns:
            mm = m.dropna(subset=[hs_col, as_col, "home_win_prob"]).copy()
            labels = (mm[hs_col] > mm[as_col]).astype(int)
            ll = logloss(mm["home_win_prob"], labels)
            summary_rows.append({
                "season": args.season,
                "market": "moneyline",
                "metric_1": "logloss",
                "value_1": round(ll, 4),
                "metric_2": "N",
                "value_2": int(len(mm)),
            })
        else:
            summary_rows.append({
                "season": args.season, "market": "moneyline",
                "metric_1": "logloss", "value_1": None,
                "metric_2": "N", "value_2": 0
            })
    else:
        summary_rows.append({
            "season": args.season, "market": "moneyline",
            "metric_1": "logloss", "value_1": None,
            "metric_2": "N", "value_2": 0
        })

    # ---------- SPREADS ----------
    sp_path = DATA / f"model_spreads_{args.season}.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        s = sp.merge(res_slim, on="game_id", how="inner", suffixes=("", "_res"))
        hs_col, as_col = pick_scores_cols(s, prefer_suffix_res=True)
        if hs_col and as_col and "spread_mu" in s.columns:
            ss = s.dropna(subset=[hs_col, as_col, "spread_mu"]).copy()
            margin = ss[hs_col] - ss[as_col]
            mae_sp = mae(margin, ss["spread_mu"])
            rmse_sp = rmse(margin, ss["spread_mu"])
            summary_rows.append({
                "season": args.season,
                "market": "spreads",
                "metric_1": "mae", "value_1": round(mae_sp, 2),
                "metric_2": "rmse","value_2": round(rmse_sp, 2),
                "N": int(len(ss))
            })
        else:
            summary_rows.append({
                "season": args.season, "market": "spreads",
                "metric_1": "mae", "value_1": None,
                "metric_2": "rmse","value_2": None,
                "N": 0
            })
    else:
        summary_rows.append({
            "season": args.season, "market": "spreads",
            "metric_1": "mae", "value_1": None,
            "metric_2": "rmse","value_2": None,
            "N": 0
        })

    # ---------- TOTALS ----------
    to_path = DATA / f"model_totals_{args.season}.csv"
    if to_path.exists():
        to = pd.read_csv(to_path)
        t = to.merge(res_slim, on="game_id", how="inner", suffixes=("", "_res"))
        hs_col, as_col = pick_scores_cols(t, prefer_suffix_res=True)
        if hs_col and as_col and "total_mu" in t.columns:
            tt = t.dropna(subset=[hs_col, as_col, "total_mu"]).copy()
            total_real = tt[hs_col] + tt[as_col]
            mae_to = mae(total_real, tt["total_mu"])
            rmse_to = rmse(total_real, tt["total_mu"])
            summary_rows.append({
                "season": args.season,
                "market": "totals",
                "metric_1": "mae", "value_1": round(mae_to, 2),
                "metric_2": "rmse","value_2": round(rmse_to, 2),
                "N": int(len(tt))
            })
        else:
            summary_rows.append({
                "season": args.season, "market": "totals",
                "metric_1": "mae", "value_1": None,
                "metric_2": "rmse","value_2": None,
                "N": 0
            })
    else:
        summary_rows.append({
            "season": args.season, "market": "totals",
            "metric_1": "mae", "value_1": None,
            "metric_2": "rmse","value_2": None,
            "N": 0
        })

    # ---------- Sortie ----------
    out = pd.DataFrame(summary_rows)

    # Mise en forme (colonnes cohérentes)
    if "N" not in out.columns:
        out["N"] = None
    out = out[["season","market","metric_1","value_1","metric_2","value_2","N"]]

    out_path = DATA / f"model_eval_{args.season}.csv"
    if out_path.exists():
        prev = pd.read_csv(out_path)
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"[OK] → {out_path}")

if __name__ == "__main__":
    main()
