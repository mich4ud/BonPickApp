# scripts/check_env.py
import os
from pathlib import Path
from dotenv import load_dotenv

p = Path(r"C:\Users\Alex\Desktop\SmartOdds\.env")
print("EXISTE:", p.exists(), p)

# charge .env
load_dotenv(p, override=True)
val = os.getenv("ODDS_API_KEY")
print("ODDS_API_KEY =", repr(val))
if not val:
    print(">> Problème: ODDS_API_KEY est introuvable. Vérifie le fichier .env.")
else:
    print(">> OK: la clé est bien chargée.")
