# scripts/update_all.py
# Orchestrateur: odds -> results -> models -> edges -> perf
from __future__ import annotations
from pathlib import Path
import subprocess, sys, os
import argparse

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)

def run_step(title, cmd, fatal=True):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    print(">", PY, *cmd)
    code = subprocess.call([str(PY), *cmd], cwd=str(ROOT))
    if code != 0:
        print("[ECHEC] Code retour:", code)
        if fatal:
            sys.exit(1)
    return code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--edge-min", type=float, default=0.03)
    ap.add_argument("--regions", default="us,eu")
    ap.add_argument("--format", choices=["decimal","american"], default="decimal")
    args = ap.parse_args()

    # 1) Odds (fatal si clé absente)
    run_step("[1/5] Récupération des cotes (Odds API)", [
        "scripts/fetch_odds.py", "--league", "nfl",
        "--odds_format", args.format, "--regions", args.regions
    ], fatal=True)

    # 2) Results (non fatal en hors-saison)
    run_step("[2/5] Importation des résultats récents", [
        "scripts/fetch_results_nfl.py", "--days", str(args.days)
    ], fatal=False)

    # 3) Modèles Elo (fatal si vrai bug)
    run_step("[3/5] Construction des modèles Elo (ML / Spreads / Totals)", [
        "scripts/build_elo_nfl.py", "--season", str(args.season)
    ], fatal=True)

    # 4) Évaluation edges marché (fatal=False : si pas de matchs, on continue)
    run_step("[4/5] Évaluation des edges marché (ML/Spreads/Totals)", [
        "scripts/evaluate_market_edges.py", "--season", str(args.season),
        "--market", "all", "--edge-min", str(args.edge_min),
        "--regions", args.regions, "--format", args.format
    ], fatal=False)

    # 5) (Optionnel) Évaluation modèles Elo vs résultats historiques
    #    Ne rends pas fatal si tu n’as pas encore importé assez de résultats.
    run_step("[5/5] Évaluation Elo (facultative)", [
        "scripts/evaluate_elo_models.py", "--season", str(args.season)
    ], fatal=False)

    print("\n[OK] Pipeline terminé.")

if __name__ == "__main__":
    main()
