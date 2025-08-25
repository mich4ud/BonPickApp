# utils/odds.py — conversions et helpers de cotes/probas

def american_to_prob(odds: int) -> float:
    """Cote américaine -> probabilité implicite (avec vig)."""
    return 100.0/(odds+100.0) if odds > 0 else -odds/(-odds+100.0)

def no_vig_two_way(p1_hat: float, p2_hat: float):
    """Retire la marge sur un marché à 2 issues (renormalisation)."""
    z = p1_hat + p2_hat
    if z <= 0:
        return float('nan'), float('nan')
    return p1_hat/z, p2_hat/z

def prob_to_decimal(p: float) -> float:
    """Probabilité -> cote décimale 'juste'."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    return 1.0/p

def decimal_to_american(d: float) -> int:
    """Cote décimale -> cote américaine (arrondie)."""
    if d <= 1:
        raise ValueError("decimal odds must be > 1")
    if d >= 2:
        return int(round((d - 1) * 100))
    else:
        return int(round(-100 / (d - 1)))

def american_to_decimal(a: int) -> float:
    """Cote américaine -> cote décimale."""
    return 1 + (a/100.0 if a > 0 else 100.0/(-a))
