#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges_csv', default='data/nfl_edges_of_day.csv')
    ap.add_argument('--min_edge', type=float, default=0.01)
    ap.add_argument('--min_books', type=int, default=3)
    ap.add_argument('--out_md', default='data/top3_today.md')
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.edges_csv)
    except FileNotFoundError:
        print("Fichier introuvable:", args.edges_csv)
        print("Lance d'abord: python scripts/compute_edges_nfl.py")
        return
    if df.empty:
        print("CSV edges vide. Relance compute_edges_nfl.")
        return

    df['best_edge'] = df[['home_edge','away_edge']].max(axis=1)
    df = df[(df['best_edge']>=args.min_edge) & (df['books_used']>=args.min_books)].copy()
    if df.empty:
        print("Aucun pick ne satisfait les critères (baisse --min_edge ou --min_books).")
        return

    def pick_side(row):
        if row['home_edge'] >= row['away_edge']:
            return ('home', row['home_team'], row['home_model_prob'], row['home_market_prob'], row['home_edge'], row['home_fair_odds_decimal'])
        else:
            return ('away', row['away_team'], row['away_model_prob'], row['away_market_prob'], row['away_edge'], row['away_fair_odds_decimal'])

    picks = []
    for _, r in df.sort_values('best_edge', ascending=False).head(3).iterrows():
        side, team, p_model, p_mkt, edge, fair = pick_side(r)
        picks.append({
            'match': f"{r['away_team']} @ {r['home_team']}",
            'bet': f"{team} ({side})",
            'books_used': int(r['books_used']),
            'model_prob': float(p_model), 'market_prob': float(p_mkt),
            'edge': float(edge), 'fair_odds': float(fair)
        })

    lines = ["# SmartOdds — Top 3 du jour (NFL)", ""]
    for i, p in enumerate(picks, 1):
        lines += [
            f"## #{i} — {p['bet']}",
            f"Match : **{p['match']}** (books utilisés : {p['books_used']})",
            f"Proba modèle : **{p['model_prob']:.1%}**  |  Proba marché (no-vig) : {p['market_prob']:.1%}",
            f"Edge estimé : **{p['edge']:.1%}**  |  Cote 'juste' (décimale) : {p['fair_odds']}",
            "---"
        ]
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ Top 3 écrit dans {args.out_md}")
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
