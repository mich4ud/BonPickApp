import sys
from pathlib import Path
from io import BytesIO

# ---------- PATH ROOT ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- IMPORTS ----------
import streamlit as st
import pandas as pd

from utils.odds import decimal_to_american
from utils.teams import get_logo_image
from utils.css_bridge import inject_css_file

# ---------- PAGE / CSS ----------
st.set_page_config(page_title="BonPick", layout="wide")
inject_css_file("app/custom.css")

# ---------- CONSTANTES ----------
DATA_DIR = Path("data")
NFL_ML_EDGES_CSV      = DATA_DIR / "nfl_edges_of_day.csv"
NFL_TOTALS_EDGES_CSV  = DATA_DIR / "nfl_totals_edges.csv"
NFL_SPREADS_EDGES_CSV = DATA_DIR / "nfl_spreads_edges.csv"
METRICS_CSV           = DATA_DIR / "metrics_history.csv"

# Chemins logos (locaux)
BRAND_LOGO_PATH = Path("app/assets/logos/brand/bonpick.png")
NFL_LOGO_PATH   = Path("app/assets/logos/nfl/NFL.png")

LEAGUES = ["NFL","NHL","NBA"]
MARKETS = ["Moneyline","Over/Under","Spreads"]

# ---------- STATE / ROUTER ----------
qs = st.query_params
if "view" not in st.session_state:
    st.session_state.view = qs.get("view","home")   # HOME par défaut
if "league" not in st.session_state:
    st.session_state.league = qs.get("league","NFL")
if "market" not in st.session_state:
    st.session_state.market = qs.get("market","Moneyline")
if "odds_fmt" not in st.session_state:
    st.session_state.odds_fmt = "Décimal"

def set_qp(**kwargs):
    for k,v in kwargs.items():
        st.query_params[k] = v

# ---------- HELPERS ----------
def format_cote(x):
    if pd.isna(x):
        return "N/A"
    if st.session_state.odds_fmt == "Décimal":
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)
    a = decimal_to_american(float(x))
    return f"{'+' if a>0 else ''}{a}"

def fmt_pct(p):
    return "N/A" if pd.isna(p) else f"{float(p)*100:.1f}%"

FR_MONTHS = {1:"JANV.",2:"FÉVR.",3:"MARS",4:"AVR.",5:"MAI",6:"JUIN",7:"JUIL.",8:"AOÛT",9:"SEPT.",10:"OCT.",11:"NOV.",12:"DÉC."}
def split_date_time(ts):
    if pd.isna(ts):
        return ("N/A","N/A")
    dt = pd.to_datetime(ts)
    return (f"{dt.day} {FR_MONTHS.get(int(dt.month), dt.strftime('%b').upper())}", dt.strftime("%H:%M")+" ET")

def ensure_local_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "commence_time" not in df.columns:
        df = df.copy()
        df["kickoff_local"]=pd.NaT
        df["nfl_week_start"]=pd.NaT
        df["nfl_week_index"]=pd.NA
        return df
    df=df.copy()
    df["commence_time"]=pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["kickoff_local"]=df["commence_time"].dt.tz_convert("America/Toronto")
    weekday=df["kickoff_local"].dt.weekday
    delta=(weekday-3)%7  # jeudi->mercredi
    df["nfl_week_start"]=(df["kickoff_local"]-pd.to_timedelta(delta, unit="D")).dt.normalize()
    starts=sorted(df["nfl_week_start"].dropna().unique())
    week_map={pd.Timestamp(s):i+1 for i,s in enumerate(starts)}
    df["nfl_week_index"]=df["nfl_week_start"].map(week_map)
    return df

def week_options_default(df: pd.DataFrame):
    if df.empty or "nfl_week_index" not in df.columns or "nfl_week_start" not in df.columns:
        return [], 0
    wk=df[["nfl_week_index","nfl_week_start"]].dropna().drop_duplicates().sort_values("nfl_week_start")
    if wk.empty:
        return [], 0
    now=pd.Timestamp.now(tz="America/Toronto").normalize()
    sel=None
    for _,row in wk.iterrows():
        s=pd.to_datetime(row["nfl_week_start"]); e=s+pd.Timedelta(days=7)
        if s<=now<e:
            sel=int(row["nfl_week_index"]); break
    if sel is None:
        sel=int(wk["nfl_week_index"].iloc[0])
    options=wk["nfl_week_index"].astype(int).tolist()
    idx=options.index(sel) if sel in options else 0
    return options, idx

def pick_week_for_home(dfs: list[pd.DataFrame]) -> int | None:
    parts=[]
    for d in dfs:
        if d is not None and not d.empty and "nfl_week_index" in d.columns and "nfl_week_start" in d.columns:
            parts.append(d[["nfl_week_index","nfl_week_start"]].dropna())
    if not parts:
        return None
    allwk = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values("nfl_week_start")
    if allwk.empty:
        return None
    opts, idx = week_options_default(allwk)
    if not opts:
        return None
    return opts[idx]

def edge_class(edge):
    if edge is None or pd.isna(edge):
        return ""
    if edge>0.05:
        return "good"
    if edge<0:
        return "bad"
    return "info"

def img_to_data_uri(img):
    if img is None:
        return None
    buf=BytesIO(); img.save(buf, format="PNG")
    import base64 as b64
    return "data:image/png;base64,"+b64.b64encode(buf.getvalue()).decode("ascii")

def file_to_data_uri(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            import base64 as b64
            b = f.read()
            return "data:image/png;base64," + b64.b64encode(b).decode("ascii")
    except Exception:
        return None

def chip_value(value, kind="info"):
    return f"<div class='chip {kind}'><span class='v'>{value}</span></div>"

def odds_table_html(p_model, p_mkt, edge, fair, kind):
    return f"""
<div class="odds-table {kind}">
  <table>
    <colgroup><col style="width:25%"><col style="width:25%"><col style="width:25%"><col style="width:25%"></colgroup>
    <thead><tr><th>PROB. MODÈLE</th><th>PROB. MARCHÉ</th><th>EDGE</th><th>COTE</th></tr></thead>
    <tbody><tr><td>{p_model}</td><td>{p_mkt}</td><td>{edge}</td><td>{fair}</td></tr></tbody>
  </table>
</div>
"""

def selection_pill(inner_html):
    return f"<div class='selection-pill'><div class='pill'><span class='val'>{inner_html}</span></div></div>"

def header_html(away, home, date_str=None, time_str=None, logo_size=64):
    def _img(team):
        uri = img_to_data_uri(get_logo_image(team, box=64, padding=6))
        return f"<img src='{uri}' width='{logo_size}' height='{logo_size}'/>" if uri else ""
    center = "<span class='at-pill'>@</span>"
    if date_str and time_str:
        center = f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
  <div class="chips center compact nowrap" style="margin-top:0;">{chip_value(date_str,'info')}{chip_value(time_str,'info')}</div>
  <span class="at-pill">@</span>
</div>"""
    return f"""
<div class="card-head">
  <div style="text-align:center">{_img(away)}<div class="team-name smallcap">{away}</div></div>
  {center}
  <div style="text-align:center">{_img(home)}<div class="team-name smallcap">{home}</div></div>
</div>"""

def _fmt_spread(val):
    if pd.isna(val): return "N/A"
    try: return f"{float(val):+.1f}"
    except Exception: return str(val)

# ---------- EXPLICATION SIMPLE (tooltip) ----------
def explain_moneyline(r) -> str:
    he = float(r.get("home_edge", 0) or 0)
    ae = float(r.get("away_edge", 0) or 0)
    home_better = he >= ae
    team   = r["home_team"] if home_better else r["away_team"]
    p_mod  = float(r["home_model_prob"] if home_better else r["away_model_prob"])
    p_mkt  = float(r["home_market_prob"] if home_better else r["away_market_prob"])
    edge   = float(max(he, ae))
    return (f"NOTRE MODÈLE VOIT {team} À {p_mod*100:.0f}% (MARCHÉ {p_mkt*100:.0f}%). "
            f"ÉCART INTÉRESSANT (EDGE {edge*100:.1f}%).")

def explain_spreads(r) -> str:
    he = float(r.get("home_edge", 0) or 0)
    ae = float(r.get("away_edge", 0) or 0)
    home_better = he >= ae
    team  = r["home_team"] if home_better else r["away_team"]
    line  = _fmt_spread(r["home_spread"] if home_better else r["away_spread"])
    p_mod = float(r["home_model_prob"] if home_better else r["away_model_prob"])
    p_mkt = float(r["home_market_prob"] if home_better else r["away_market_prob"])
    edge  = float(max(he, ae))
    return (f"{team} COUVRE {line} ~ {p_mod*100:.0f}% DU TEMPS (MARCHÉ {p_mkt*100:.0f}%). "
            f"EDGE {edge*100:.1f}%.")

def explain_totals(r) -> str:
    oe = float(r.get("over_edge", 0) or 0)
    ue = float(r.get("under_edge", 0) or 0)
    over_better = oe >= ue
    pick  = "OVER" if over_better else "UNDER"
    p_mod = float(r["over_model_prob"] if over_better else r["under_model_prob"])
    p_mkt = float(r["over_market_prob"] if over_better else r["under_market_prob"])
    edge  = float(max(oe, ue))
    total_line = r.get("total_line", None)
    line_txt = f" (LIGNE {total_line})" if pd.notna(total_line) else ""
    return (f"{pick}{line_txt} — MODÈLE {p_mod*100:.0f}% VS MARCHÉ {p_mkt*100:.0f}%. "
            f"EDGE {edge*100:.1f}%.")

def help_pill_html(text: str) -> str:
    safe = (str(text)
            .replace("&","&amp;")
            .replace("<","&lt;")
            .replace(">","&gt;")
            .replace('"',"&quot;")
            .replace("'","&#39;"))
    return f"""
<div class="help-pill" tabindex="0">
  ?
  <div class="help-tip">{safe}</div>
</div>
"""

# ---------- RUBAN TOP PICKS (ligues) ----------
def top_pick_ribbon():
    # Ruban centré, discret ; styling finalisé côté CSS (voir snippets en bas)
    return "<div class='top-pick-ribbon'>TOP PICKS</div>"

# ---------- CARTES MATCH ----------
def card_moneyline_html(r, rank=None):
    he = float(r.get("home_edge", 0) or 0)
    ae = float(r.get("away_edge", 0) or 0)
    home_better = he >= ae
    selection   = r["home_team"] if home_better else r["away_team"]
    p_model     = r["home_model_prob"] if home_better else r["away_model_prob"]
    p_mkt       = r["home_market_prob"] if home_better else r["away_market_prob"]
    fair        = r["home_fair_odds_decimal"] if home_better else r["away_fair_odds_decimal"]
    edge        = max(he, ae)
    date_str, time_str = split_date_time(r["kickoff_local"])
    header = header_html(r["away_team"], r["home_team"], date_str, time_str)
    table  = odds_table_html(fmt_pct(p_model), fmt_pct(p_mkt), fmt_pct(edge), format_cote(fair), edge_class(edge))
    sel    = selection_pill(f"<span class='team-pill'>{selection}</span> MONEYLINE")
    badge  = top_pick_ribbon() if (rank in (1,2,3)) else ""   # << remplace les 1/2/3 par “TOP PICKS”
    helpb  = help_pill_html(explain_moneyline(r))
    return f"<div class='card'>{badge}{header}{table}{sel}{helpb}</div>"

def card_spreads_html(r, rank=None):
    he = float(r.get("home_edge", 0) or 0)
    ae = float(r.get("away_edge", 0) or 0)
    home_better = he >= ae
    team        = r["home_team"] if home_better else r["away_team"]
    line        = _fmt_spread(r["home_spread"] if home_better else r["away_spread"])
    p_model     = r["home_model_prob"] if home_better else r["away_model_prob"]
    p_mkt       = r["home_market_prob"] if home_better else r["away_market_prob"]
    fair        = r["home_fair_odds_decimal"] if home_better else r["away_fair_odds_decimal"]
    edge        = max(he, ae)
    date_str, time_str = split_date_time(r["kickoff_local"])
    header = header_html(r["away_team"], r["home_team"], date_str, time_str)
    table  = odds_table_html(fmt_pct(p_model), fmt_pct(p_mkt), fmt_pct(edge), format_cote(fair), edge_class(edge))
    sel    = selection_pill(f"<span class='team-pill'>{team}</span> {line}")
    badge  = top_pick_ribbon() if (rank in (1,2,3)) else ""
    helpb  = help_pill_html(explain_spreads(r))
    return f"<div class='card'>{badge}{header}{table}{sel}{helpb}</div>"

def card_totals_html(r, rank=None):
    oe = float(r.get("over_edge", 0) or 0)
    ue = float(r.get("under_edge", 0) or 0)
    over_better = oe >= ue
    pick        = "OVER" if over_better else "UNDER"
    p_model     = r["over_model_prob"] if over_better else r["under_model_prob"]
    p_mkt       = r["over_market_prob"] if over_better else r["under_market_prob"]
    fair        = r["over_fair_odds_decimal"] if over_better else r["under_fair_odds_decimal"]
    edge        = max(oe, ue)
    total_line  = r.get("total_line", None)
    val_line    = total_line if pd.notna(total_line) else "N/A"
    date_str, time_str = split_date_time(r["kickoff_local"])
    header = header_html(r["away_team"], r["home_team"], date_str, time_str)
    table  = odds_table_html(fmt_pct(p_model), fmt_pct(p_mkt), fmt_pct(edge), format_cote(fair), edge_class(edge))
    sel    = selection_pill(f"<span class='team-pill'>{pick}</span> {val_line}")
    badge  = top_pick_ribbon() if (rank in (1,2,3)) else ""
    helpb  = help_pill_html(explain_totals(r))
    return f"<div class='card'>{badge}{header}{table}{sel}{helpb}</div>"

def render_grid_html(cards):
    if not cards: return
    grid = "<div class='cards-grid'>" + "".join(f"<div>{c}</div>" for c in cards) + "</div>"
    st.markdown(grid, unsafe_allow_html=True)

# ---------- NAVBAR ----------
def render_navbar():
    home_active = " active" if st.session_state.view=="home" else ""
    home_link = f"<a class='nav-link nav-home{home_active}' href='?view=home'>HOME</a>"

    def nfl_dropdown_html():
        if NFL_LOGO_PATH.exists():
            nfl_uri = file_to_data_uri(NFL_LOGO_PATH)
            logo_html = f"<img src='{nfl_uri}' class='nav-nfl-logo' alt='NFL'/>"
        else:
            logo_html = "<span class='nav-link'>NFL</span>"

        def m_active(m):
            return " active" if (st.session_state.league=="NFL" and st.session_state.market==m and st.session_state.view=="app") else ""

        return f"""
<div class="dropdown">
  <input id="dd-nfl" type="checkbox" />
  <label class="nfl-btn" for="dd-nfl">{logo_html}<span class="caret"></span></label>
  <div class="menu">
    <a class="sub-pill{m_active('Moneyline')}" href='?view=app&league=NFL&market=Moneyline'>MONEYLINE</a>
    <a class="sub-pill{m_active('Over/Under')}" href='?view=app&league=NFL&market=Over/Under'>OVER/UNDER</a>
    <a class="sub-pill{m_active('Spreads')}" href='?view=app&league=NFL&market=Spreads'>SPREADS</a>
  </div>
</div>
"""

    perf_active = " active" if st.session_state.view=="performance" else ""
    perf_link = f"<a class='nav-link{perf_active}' href='?view=performance'>PERFORMANCE</a>"

    st.markdown(
        f"""
<div class="navbar">
  <div class="nav-left">
    {home_link}
    {nfl_dropdown_html()}
  </div>
  <div class="nav-right">
    {perf_link}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

# ---------- EN-TÊTE (logo BonPick centré) ----------
def render_page_hero_top():
    if BRAND_LOGO_PATH.exists():
        uri = file_to_data_uri(BRAND_LOGO_PATH)
        st.markdown(f"<div class='page-hero'><img src='{uri}' alt='BonPick' /></div>", unsafe_allow_html=True)

# ---------- BARRE D’OUTILS (SEMAINE + FORMAT) ----------
def render_tools_bar(df_for_week: pd.DataFrame | None, week_key: str):
    st.markdown("<div class='toolsbar'>", unsafe_allow_html=True)
    c = st.columns([1,1,1], gap="small")
    wk_val = None
    with c[1]:
        if df_for_week is not None and not df_for_week.empty:
            options, idx = week_options_default(df_for_week)
            if options:
                wk_val = st.selectbox("Semaine NFL", options=options, index=idx,
                                      key=week_key, label_visibility="collapsed",
                                      format_func=lambda w: f"Semaine {w}")
    with c[2]:
        current_idx = 0 if st.session_state.odds_fmt == "Décimal" else 1
        val = st.selectbox("FORMAT COTE", ["Décimal","Américain"], index=current_idx,
                           key=f"fmt_{week_key}", label_visibility="collapsed")
        st.session_state.odds_fmt = val
    st.markdown("</div>", unsafe_allow_html=True)
    return wk_val

# ---------- PAGE PERFORMANCE (placeholder simple) ----------
def render_performance():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    if METRICS_CSV.exists():
        M = pd.read_csv(METRICS_CSV)
        st.subheader("APERÇU DES MÉTRIQUES")
        st.dataframe(M, use_container_width=True, hide_index=True)
    else:
        st.info("PAS ENCORE DE MÉTRIQUES — LANCE TES SCRIPTS D'ÉVALUATION.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- SCORE & RANKING ----------
def _score_row_binary(p_mod, p_mkt, edge):
    return 0.6*edge + 0.3*(p_mod - p_mkt) + 0.1*p_mod

def compute_score_moneyline(df):
    def row_score(r):
        he = float(r.get("home_edge", 0) or 0)
        ae = float(r.get("away_edge", 0) or 0)
        home_better = he >= ae
        p_mod = float(r["home_model_prob"] if home_better else r["away_model_prob"])
        p_mkt = float(r["home_market_prob"] if home_better else r["away_market_prob"])
        edge  = he if home_better else ae
        return _score_row_binary(p_mod, p_mkt, edge)
    df = df.copy()
    df["score"] = df.apply(row_score, axis=1)
    return df.sort_values("score", ascending=False)

def compute_score_spreads(df):
    def row_score(r):
        he = float(r.get("home_edge", 0) or 0)
        ae = float(r.get("away_edge", 0) or 0)
        home_better = he >= ae
        p_mod = float(r["home_model_prob"] if home_better else r["away_model_prob"])
        p_mkt = float(r["home_market_prob"] if home_better else r["away_market_prob"])
        edge  = he if home_better else ae
        return _score_row_binary(p_mod, p_mkt, edge)
    df = df.copy()
    df["score"] = df.apply(row_score, axis=1)
    return df.sort_values("score", ascending=False)

def compute_score_totals(df):
    def row_score(r):
        oe = float(r.get("over_edge", 0) or 0)
        ue = float(r.get("under_edge", 0) or 0)
        over_better = oe >= ue
        p_mod = float(r["over_model_prob"] if over_better else r["under_model_prob"])
        p_mkt = float(r["over_market_prob"] if over_better else r["under_market_prob"])
        edge  = oe if over_better else ue
        return _score_row_binary(p_mod, p_mkt, edge)
    df = df.copy()
    df["score"] = df.apply(row_score, axis=1)
    return df.sort_values("score", ascending=False)

# ---------- PODIUM (HOME) ----------
def build_podium_entries() -> list[dict]:
    """Retourne une liste de 3 entrées top (tous marchés confondus), sur la même semaine."""
    dfs_ml = ensure_local_week(pd.read_csv(NFL_ML_EDGES_CSV)) if NFL_ML_EDGES_CSV.exists() else pd.DataFrame()
    dfs_sp = ensure_local_week(pd.read_csv(NFL_SPREADS_EDGES_CSV)) if NFL_SPREADS_EDGES_CSV.exists() else pd.DataFrame()
    dfs_to = ensure_local_week(pd.read_csv(NFL_TOTALS_EDGES_CSV)) if NFL_TOTALS_EDGES_CSV.exists() else pd.DataFrame()

    wk = pick_week_for_home([dfs_ml, dfs_sp, dfs_to])
    if wk is None:
        return []

    tops = []

    if not dfs_ml.empty:
        ml = dfs_ml[dfs_ml["nfl_week_index"]==wk].copy()
        if not ml.empty:
            ml = compute_score_moneyline(ml).head(5)
            for _, r in ml.iterrows():
                html = card_moneyline_html(r, rank=None)  # pas de ruban sur HOME
                tops.append({"score": float(r["score"]), "html": html})

    if not dfs_sp.empty:
        sp = dfs_sp[dfs_sp["nfl_week_index"]==wk].copy()
        if not sp.empty:
            sp = compute_score_spreads(sp).head(5)
            for _, r in sp.iterrows():
                html = card_spreads_html(r, rank=None)
                tops.append({"score": float(r["score"]), "html": html})

    if not dfs_to.empty:
        to = dfs_to[dfs_to["nfl_week_index"]==wk].copy()
        if not to.empty:
            to = compute_score_totals(to).head(5)
            for _, r in to.iterrows():
                html = card_totals_html(r, rank=None)
                tops.append({"score": float(r["score"]), "html": html})

    if not tops:
        return []
    tops = sorted(tops, key=lambda d: d["score"], reverse=True)[:3]
    return tops

def render_home_podium():
    tops = build_podium_entries()
    if not tops:
        st.info("PAS DE DONNÉES POUR CONSTRUIRE LE PODIUM CETTE SEMAINE.")
        return

    if len(tops)==1:
        ordered = [(1, tops[0])]
    elif len(tops)==2:
        ordered = [(2, tops[1]), (1, tops[0])]
    else:
        ordered = [(2, tops[1]), (1, tops[0]), (3, tops[2])]

    st.markdown("""
<style>
.podium-wrap{display:grid;grid-template-columns:repeat(3, minmax(260px, 1fr));gap:22px;align-items:end;padding:8px 40px 24px;}
.podium-col{display:flex;flex-direction:column;align-items:center;gap:10px;}
.podium-col.col-1{transform:translateY(-8px);}
.podium-num{border-radius:50%;width:44px;height:44px;display:flex;align-items:center;justify-content:center;
            font-weight:800;font-size:16px;box-shadow:0 2px 6px rgba(0,0,0,.35);}
.podium-card{width:100%;max-width:420px;}
.podium-note{margin:8px auto 14px auto;max-width:750px;text-align:jsutify;}
.podium-note .card{padding:16px 18px;}
.podium-note-title{font-weight:900;margin-bottom:8px;letter-spacing:1px;}
.podium-note-text{opacity:.95;line-height:1.6;}
</style>
""", unsafe_allow_html=True)

    st.markdown(
        """
<div class='podium-note'>
  <div class='card'>
    <div class='podium-note-title'>MEILLEURS CHOIX DE LA SEMAINE</div>
    <div class='podium-note-text'>
      NOUS COMBINONS <b>PROBABILITÉS DU MODÈLE</b>, <b>ÉCART VS MARCHÉ (EDGE)</b> ET <b>ROBUSTESSE GLOBALE</b>
      POUR DÉGAGER LES 3 PARIS LES PLUS PROMETTEURS, TOUTES LIGUES ET MARCHÉS CONFONDUS.<br/>
      <small>ASTUCE&nbsp;: UTILISE CE PODIUM COMME POINT DE DÉPART, VÉRIFIE LES COTES EN DIRECT,
      ET RESTE DISCIPLINÉ DANS TA GESTION DE BANKROLL. 
      AUCUNE GARANTIE&nbsp;: LE SPORT RESTE IMPRÉVISIBLE.</small>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    cols_html = []
    for rank, item in ordered:
        num_class = f"num-{rank}"
        col_class = f"col-{rank}"
        col_html = f"""
<div class="podium-col {col_class}">
  <div class="podium-num {num_class}">{rank}</div>
  <div class="podium-card">{item['html']}</div>
</div>
"""
        cols_html.append(col_html)

    st.markdown(f"<div class='podium-wrap'>{''.join(cols_html)}</div>", unsafe_allow_html=True)

# ---------- PAGES ----------
def render_home():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    render_home_podium()
    st.markdown("</div>", unsafe_allow_html=True)

def render_nfl(market_name: str):
    if market_name == "Moneyline" and NFL_ML_EDGES_CSV.exists():
        df = ensure_local_week(pd.read_csv(NFL_ML_EDGES_CSV))
        wk = render_tools_bar(df, "wk_ml")
        if wk:
            show = df[df["nfl_week_index"]==wk].copy()
            if not show.empty:
                show = compute_score_moneyline(show)
                cards = []
                for i, (_, r) in enumerate(show.iterrows(), start=1):
                    rank = i if i<=3 else None
                    cards.append(card_moneyline_html(r, rank=rank))
                render_grid_html(cards)
            else:
                st.info("AUCUN MATCH POUR CETTE SEMAINE.")
    elif market_name == "Over/Under" and NFL_TOTALS_EDGES_CSV.exists():
        df = ensure_local_week(pd.read_csv(NFL_TOTALS_EDGES_CSV))
        wk = render_tools_bar(df, "wk_tot")
        if wk:
            show = df[df["nfl_week_index"]==wk].copy()
            if not show.empty:
                show = compute_score_totals(show)
                cards = []
                for i, (_, r) in enumerate(show.iterrows(), start=1):
                    rank = i if i<=3 else None
                    cards.append(card_totals_html(r, rank=rank))
                render_grid_html(cards)
            else:
                st.info("AUCUN MATCH POUR CETTE SEMAINE.")
    elif market_name == "Spreads" and NFL_SPREADS_EDGES_CSV.exists():
        df = ensure_local_week(pd.read_csv(NFL_SPREADS_EDGES_CSV))
        wk = render_tools_bar(df, "wk_sp")
        if wk:
            show = df[df["nfl_week_index"]==wk].copy()
            if not show.empty:
                show = compute_score_spreads(show)
                cards = []
                for i, (_, r) in enumerate(show.iterrows(), start=1):
                    rank = i if i<=3 else None
                    cards.append(card_spreads_html(r, rank=rank))
                render_grid_html(cards)
            else:
                st.info("AUCUN MATCH POUR CETTE SEMAINE.")
    else:
        render_tools_bar(pd.DataFrame(), "wk_none")
        st.info("AUCUNE DONNÉE POUR CE MARCHÉ.")

def render_app():
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    league, market = st.session_state.league, st.session_state.market
    if league == "NFL":
        render_nfl(market)
    else:
        render_tools_bar(pd.DataFrame(), "wk_other")
        st.info("LIGUE PAS ENCORE BRANCHÉE.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- MAIN ----------
def render_navbar():
    home_active = " active" if st.session_state.view=="home" else ""
    home_link = f"<a class='nav-link nav-home{home_active}' href='?view=home'>HOME</a>"

    def nfl_dropdown_html():
        if NFL_LOGO_PATH.exists():
            nfl_uri = file_to_data_uri(NFL_LOGO_PATH)
            logo_html = f"<img src='{nfl_uri}' class='nav-nfl-logo' alt='NFL'/>"
        else:
            logo_html = "<span class='nav-link'>NFL</span>"

        def m_active(m):
            return " active" if (st.session_state.league=="NFL" and st.session_state.market==m and st.session_state.view=="app") else ""

        return f"""
<div class="dropdown">
  <input id="dd-nfl" type="checkbox" />
  <label class="nfl-btn" for="dd-nfl">{logo_html}<span class="caret"></span></label>
  <div class="menu">
    <a class="sub-pill{m_active('Moneyline')}" href='?view=app&league=NFL&market=Moneyline'>MONEYLINE</a>
    <a class="sub-pill{m_active('Over/Under')}" href='?view=app&league=NFL&market=Over/Under'>OVER/UNDER</a>
    <a class="sub-pill{m_active('Spreads')}" href='?view=app&league=NFL&market=Spreads'>SPREADS</a>
  </div>
</div>
"""

    perf_active = " active" if st.session_state.view=="performance" else ""
    perf_link = f"<a class='nav-link{perf_active}' href='?view=performance'>PERFORMANCE</a>"

    st.markdown(
        f"""
<div class="navbar">
  <div class="nav-left">
    {home_link}
    {nfl_dropdown_html()}
  </div>
  <div class="nav-right">
    {perf_link}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

def main():
    # Logo BonPick en haut (entête centré)
    render_page_hero_top()
    # Navbar
    render_navbar()
    # Router
    if st.session_state.view == "home":
        render_home()
    elif st.session_state.view == "performance":
        render_performance()
    else:
        render_app()

if __name__ == "__main__":
    main()
