@echo off
REM Aller dans le dossier du projet
cd /d C:\Users\Alex\Desktop\SmartOdds

REM Activer l'environnement virtuel
call .venv\Scripts\activate

REM Lancer la mise à jour (fetch + recalcul)
python scripts\update_all.py

REM (Optionnel) Pauser la fenêtre si tu double-cliques
REM pause
