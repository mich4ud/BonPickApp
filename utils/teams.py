from pathlib import Path
from typing import Optional
from .image_utils import load_and_fit_logo

# Mapping noms d'équipes → abréviations (adapte si tes CSV ont d'autres libellés)
TEAM_ABBR = {
    # NFC West
    "Arizona Cardinals": "ARI", "Los Angeles Rams": "LAR", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    # NFC South
    "Atlanta Falcons": "ATL", "Carolina Panthers": "CAR", "New Orleans Saints": "NO", "Tampa Bay Buccaneers": "TB",
    # NFC North
    "Chicago Bears": "CHI", "Detroit Lions": "DET", "Green Bay Packers": "GB", "Minnesota Vikings": "MIN",
    # NFC East
    "Dallas Cowboys": "DAL", "New York Giants": "NYG", "Philadelphia Eagles": "PHI", "Washington Commanders": "WSH",

    # AFC North
    "Baltimore Ravens": "BAL", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Pittsburgh Steelers": "PIT",
    # AFC East
    "Buffalo Bills": "BUF", "Miami Dolphins": "MIA", "New England Patriots": "NE", "New York Jets": "NYJ",
    # AFC South
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Tennessee Titans": "TEN",
    # AFC West
    "Denver Broncos": "DEN", "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
}

# Alias possibles rencontrés dans certaines sources
ALIASES = {
    "LA Rams": "Los Angeles Rams",
    "LA Chargers": "Los Angeles Chargers",
    "Washington Football Team": "Washington Commanders",
    "Jacksonville": "Jacksonville Jaguars",
    "NY Giants": "New York Giants",
    "NY Jets": "New York Jets",
    # ajoute d'autres alias si tu en vois dans tes CSV
}

def normalize_team(name: str) -> str:
    """Retourne un nom 'officiel' d'équipe à partir d'un alias éventuel."""
    if not isinstance(name, str):
        return name
    return ALIASES.get(name, name)

def team_abbr(name: str) -> Optional[str]:
    """Abréviation NFL (ex: 'KC') depuis le nom officiel."""
    n = normalize_team(name)
    return TEAM_ABBR.get(n)

def logo_file_path(team_name: str) -> Optional[Path]:
    """Chemin du logo local d'équipe si présent, sinon None."""
    abbr = team_abbr(team_name)
    if not abbr:
        return None
    p = Path("app/assets/logos/nfl") / f"{abbr}.png"
    return p if p.exists() else None

def get_logo_image(team_name: str, box: int = 72, padding: int = 6):
    """
    Retourne une image PIL 'prête à afficher' (carré box×box, centrée, padding),
    en essayant d'abord le logo d'équipe, sinon en utilisant le fallback NFL.png.
    """
    # 1) Tenter le logo de l'équipe
    p = logo_file_path(team_name)
    if p and p.exists():
        try:
            return load_and_fit_logo(str(p), box=box, padding=padding)
        except Exception:
            pass  # si problème de lecture/format, on tombera sur le fallback

    # 2) Fallback : logo NFL générique (si présent)
    fallback = Path("app/assets/logos/nfl/NFL.png")
    if fallback.exists():
        try:
            return load_and_fit_logo(str(fallback), box=box, padding=padding)
        except Exception:
            return None

    # 3) Rien à afficher (le code appelant pourra montrer du texte)
    return None

