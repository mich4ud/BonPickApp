# scripts/build_elo_nfl.py
# (version identique à la tienne, juste les prints en ASCII pour Windows)

from __future__ import annotations
from pathlib import Path
import argparse, sys
import pandas as pd
import numpy as np

DATA = Path("data")
DATA.mkdir(exist_ok=True)

# ... (tout ton code d’origine inchangé jusqu’aux prints finaux) ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    args = ap.parse_args()

    # ... ici ta logique habituelle qui produit:
    # ml (DataFrame), sp (DataFrame), to (DataFrame)
    # et les chemins ml_out, sp_out, to_out

    # Exemple minimal pour rappeler la structure :
    # (N’OUBLIE PAS DE GARDER ta vraie logique plus haut)
    ml = pd.read_csv(DATA / f"model_moneyline_{args.season}.csv") if (DATA / f"model_moneyline_{args.season}.csv").exists() else pd.DataFrame()
    sp = pd.read_csv(DATA / f"model_spreads_{args.season}.csv")   if (DATA / f"model_spreads_{args.season}.csv").exists() else pd.DataFrame()
    to = pd.read_csv(DATA / f"model_totals_{args.season}.csv")    if (DATA / f"model_totals_{args.season}.csv").exists() else pd.DataFrame()

    ml_out = DATA / f"model_moneyline_{args.season}.csv"
    sp_out = DATA / f"model_spreads_{args.season}.csv"
    to_out = DATA / f"model_totals_{args.season}.csv"

    # (Dans ta vraie version, tu écris ml/sp/to ici)
    # ml.to_csv(ml_out, index=False)
    # sp.to_csv(sp_out, index=False)
    # to.to_csv(to_out, index=False)

    print(f"[OK] ML -> {ml_out} ({len(ml)} lignes)")
    print(f"[OK] Spreads -> {sp_out} ({len(sp)} lignes)")
    print(f"[OK] Totals -> {to_out} ({len(to)} lignes)")

if __name__ == "__main__":
    main()
