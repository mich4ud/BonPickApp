# scripts/import_results_nflfastr.py
# Importe les résultats NFL historiques depuis nflverse/nfldata (open data)
# et enregistre dans data/nfl_results.csv au format attendu par SmartOdds.

from __future__ import annotations
from pathlib import Path
import pandas as pd
import argparse

DATA_DIR = Path("data")
OUT_CSV  = DATA_DIR / "nfl_results.csv"

# Lien RAW du fichier "games.csv" (nflverse/nfldata)
NFLV_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

TEAM_MAP = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","MIA":"Miami Dolphins",
    "MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks",
    "SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders",
}

def season_window(season: int):
    # Fenêtre large: 1 août -> 1 mars (UTC)
    start = pd.Timestamp(f"{season}-08-01", tz="UTC")
    end   = pd.Timestamp(f"{season+1}-03-01", tz="UTC")
    return start, end

def ensure_local_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "commence_time" not in df.columns:
        df["kickoff_local"]=pd.NaT; df["nfl_week_start"]=pd.NaT; df["nfl_week_index"]=pd.NA; return df
    df = df.copy()
    # commence_time est en UTC (tz-aware) → converti Montréal/Toronto
    df["kickoff_local"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce").dt.tz_convert("America/Toronto")
    weekday = df["kickoff_local"].dt.weekday
    delta = (weekday - 3) % 7  # semaine NFL commence le jeudi
    df["nfl_week_start"] = (df["kickoff_local"] - pd.to_timedelta(delta, unit="D")).dt.normalize()
    starts = sorted(df["nfl_week_start"].dropna().unique())
    week_map = {pd.Timestamp(s): i+1 for i, s in enumerate(starts)}
    df["nfl_week_index"] = df["nfl_week_start"].map(week_map)
    return df

def main():
    ap = argparse.ArgumentParser(description="Importe résultats NFL depuis nflverse/nfldata")
    ap.add_argument("--season", type=int, required=True, help="Ex: 2024")
    args = ap.parse_args()

    print("[DL] nfldata games.csv …")
    # Lecture directe du CSV (non compressé)
    games = pd.read_csv(NFLV_GAMES_URL, low_memory=False)

    # Filtrer saison + types de matchs valides
    g = games[(games["season"] == args.season) & (games["game_type"].isin(["REG","POST"]))].copy()

    # Noms longs d’équipes (mapping vers nos labels)
    g["home_team_name"] = g["home_team"].map(TEAM_MAP)
    g["away_team_name"] = g["away_team"].map(TEAM_MAP)

    # Construire commence_time :
    # Colonnes possibles: 'kickoff', 'start_time', 'start_time_utc', sinon fallback sur 'gameday' + 'game_time_eastern'
    commence = None
    kickoff_cols = [c for c in g.columns if c.lower() in ("kickoff","start_time","start_time_utc")]
    if kickoff_cols:
        kcol = kickoff_cols[0]
        # Force en UTC quoi qu'il arrive (si tz-naive → UTC; si tz-aware → converti)
        commence = pd.to_datetime(g[kcol], errors="coerce", utc=True)
    else:
        # Fallback: combine 'gameday' + 'game_time_eastern' si disponibles
        if {"gameday","game_time_eastern"}.issubset(g.columns):
            dt_str = g["gameday"].astype(str).str.strip() + " " + g["game_time_eastern"].astype(str).str.strip()
            # Localise en US/Eastern puis convertit en UTC
            # (on passe par to_datetime sans utc, puis tz_localize)
            tmp = pd.to_datetime(dt_str, errors="coerce")
            # Certains enregistrements peuvent être NaT → tz_localize les valides uniquement
            # On crée une série vide et on remplit
            commence = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns, UTC]")
            mask = tmp.notna()
            # tz_localize() nécessite pytz/zoneinfo; pandas gère "US/Eastern"
            commence.loc[mask] = tmp[mask].dt.tz_localize("US/Eastern", nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
            # Convertir en tz-aware UTC propre
            commence = pd.to_datetime(commence, errors="coerce", utc=True)
        else:
            # En dernier recours: pas de date exploitable
            commence = pd.to_datetime(pd.Series([pd.NaT]*len(g)), errors="coerce", utc=True)

    g["commence_time"] = commence  # tz-aware UTC

    # Sécurité: borne saison (les deux bornes tz-aware UTC)
    start, end = season_window(args.season)
    # Forcer le type UTC pour éviter toute comparaison tz-naive/tz-aware
    g["commence_time"] = pd.to_datetime(g["commence_time"], errors="coerce", utc=True)
    g = g[(g["commence_time"].isna()) | ((g["commence_time"] >= start) & (g["commence_time"] < end))]

    # Sortie normalisée
    out = g.rename(columns={
        "home_team_name": "home_team",
        "away_team_name": "away_team",
    })[["game_id","commence_time","home_team","away_team","home_score","away_score"]].copy()

    # completed + winner
    out["completed"] = True
    def winner(row):
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            return pd.NA
        if row["home_score"] > row["away_score"]: return "home"
        if row["home_score"] < row["away_score"] : return "away"
        return "draw"
    out["winner"] = out.apply(winner, axis=1)

    out = ensure_local_week(out)
    out = out.sort_values("commence_time").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] {len(out)} matchs → {OUT_CSV}")

if __name__ == "__main__":
    main()
