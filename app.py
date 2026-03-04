"""
app.py — IIT Pokerbots Log Analysis · Streamlit Dashboard
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io, tempfile, zipfile, json
from pathlib import Path
from datetime import datetime

from parser import LogParser
from metrics import MetricsEngine, hand_strength_bucket
from leak_detection import LeakDetector
from comparison import ComparisonEngine

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pokerbots Analyzer",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0e1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2d3561; }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1f35;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #ccd6f6 !important; font-size: 22px !important; }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* Section headers */
    .section-header {
        font-size: 18px; font-weight: 700; color: #64ffda;
        border-left: 4px solid #64ffda;
        padding-left: 12px; margin: 24px 0 16px 0;
    }

    /* Leak cards */
    .leak-high   { border-left: 4px solid #ff6b6b; background: #1e1318; padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
    .leak-medium { border-left: 4px solid #ffd93d; background: #1e1b13; padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
    .leak-low    { border-left: 4px solid #6bcb77; background: #131e15; padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
    .leak-title  { font-weight: 700; font-size: 15px; margin-bottom: 6px; }
    .leak-desc   { color: #8892b0; font-size: 13px; margin-bottom: 6px; }
    .leak-rec    { color: #64ffda; font-size: 13px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #1a2035 !important; color: #64ffda !important; }

    /* Upload area */
    [data-testid="stFileUploader"] { border: 2px dashed #2d3561; border-radius: 10px; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #2d3561; border-radius: 8px; }

    /* Hand history cards */
    .hand-card {
        background: #161b27; border: 1px solid #2d3561; border-radius: 8px;
        padding: 14px; margin: 8px 0; font-family: monospace; font-size: 13px;
    }
    .hand-win  { border-left: 4px solid #6bcb77; }
    .hand-loss { border-left: 4px solid #ff6b6b; }
    
    /* Badge */
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; margin: 0 4px;
    }
    .badge-green  { background: #0d2b1f; color: #6bcb77; border: 1px solid #6bcb77; }
    .badge-red    { background: #2b0d0d; color: #ff6b6b; border: 1px solid #ff6b6b; }
    .badge-yellow { background: #2b2408; color: #ffd93d; border: 1px solid #ffd93d; }
    .badge-blue   { background: #0d1b2b; color: #74b9ff; border: 1px solid #74b9ff; }
    
    h1, h2, h3 { color: #ccd6f6 !important; }
    p, li { color: #8892b0; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────
# xaxis/yaxis are NOT inside PLOTLY_THEME to avoid duplicate-keyword crashes
# when individual charts also pass yaxis_title= explicitly. Use _apply_theme().
PLOTLY_THEME = dict(
    paper_bgcolor='#0e1117',
    plot_bgcolor='#161b27',
    font=dict(color='#8892b0', family='monospace'),
)
_AXES = dict(gridcolor='#1e2640', linecolor='#2d3561')

def _apply_theme(fig, **kwargs):
    """Apply dark theme + axis grid, then any extra layout kwargs."""
    fig.update_layout(**PLOTLY_THEME, **kwargs)
    fig.update_xaxes(**_AXES)
    fig.update_yaxes(**_AXES)
    return fig

GREEN  = '#6bcb77'
RED    = '#ff6b6b'
YELLOW = '#ffd93d'
BLUE   = '#74b9ff'
CYAN   = '#64ffda'
PURPLE = '#c792ea'


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def parse_and_analyse(file_bytes: bytes, filename: str, bot_name: str):
    # Preserve original extension so gzip detection works correctly
    suffix = Path(filename).suffix or '.glog'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='wb') as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        parser = LogParser(tmp_path)
        parser.parse()
        rounds_df, actions_df = parser.to_dataframes()
        if rounds_df.empty:
            return None, None, None, None, None, None, None
        if not bot_name:
            bot_name = parser.bot_name
        engine = MetricsEngine(rounds_df, actions_df, bot_name)
        metrics = engine.all_metrics()
        detector = LeakDetector(metrics, engine.rounds_df, actions_df, bot_name)
        leaks = detector.run()
        return engine.rounds_df, actions_df, metrics, leaks, bot_name, parser.opponent_name, parser.final_bankrolls
    finally:
        os.unlink(tmp_path)

def color_val(v, invert=False):
    if v > 0:
        return GREEN if not invert else RED
    elif v < 0:
        return RED if not invert else GREEN
    return YELLOW

def delta_arrow(v):
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def badge(text, color='blue'):
    return f'<span class="badge badge-{color}">{text}</span>'


# ─── Chart functions ──────────────────────────────────────────────────────────

def chart_bankroll(rounds_df):
    pnl = rounds_df['bot_payoff'].cumsum()
    roll50 = pnl.rolling(50, min_periods=1).mean()

    fig = go.Figure()
    pos_pnl = pnl.where(pnl >= 0, 0)
    neg_pnl = pnl.where(pnl < 0, 0)
    fig.add_trace(go.Scatter(x=rounds_df['round_num'], y=pos_pnl,
        fill='tozeroy', fillcolor='rgba(107,203,119,0.12)', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=rounds_df['round_num'], y=neg_pnl,
        fill='tozeroy', fillcolor='rgba(255,107,107,0.12)', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=rounds_df['round_num'], y=pnl,
        line=dict(color=BLUE, width=1.8), name='Cumulative P&L'))
    fig.add_trace(go.Scatter(x=rounds_df['round_num'], y=roll50,
        line=dict(color=YELLOW, width=1.5, dash='dot'), name='50-round avg'))
    fig.add_hline(y=0, line_color='#2d3561', line_width=1)
    _apply_theme(fig, height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        title=dict(text='Cumulative P&L', font=dict(color=CYAN, size=14)),
        xaxis_title='Round', yaxis_title='Chips')
    return fig


def chart_round_pnl(rounds_df):
    colors = [GREEN if v > 0 else RED for v in rounds_df['bot_payoff']]
    fig = go.Figure(go.Bar(
        x=rounds_df['round_num'], y=rounds_df['bot_payoff'],
        marker_color=colors, marker_line_width=0))
    fig.add_hline(y=0, line_color='#2d3561')
    _apply_theme(fig, height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='Per-Round P&L', font=dict(color=CYAN, size=14)),
        xaxis_title='Round', yaxis_title='Chips')
    return fig


def chart_auction_dist(rounds_df, bot_name):
    bot_bids = rounds_df['bot_bid'].dropna()
    opp_bids = rounds_df['opp_bid'].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bot_bids, nbinsx=40,
        name=bot_name, marker_color=BLUE, opacity=0.75))
    if len(opp_bids):
        fig.add_trace(go.Histogram(x=opp_bids, nbinsx=40,
            name='Opponent', marker_color=PURPLE, opacity=0.6))
    if len(bot_bids):
        fig.add_vline(x=bot_bids.mean(), line_color=YELLOW,
            line_dash='dot', annotation_text=f"Mean {bot_bids.mean():.0f}",
            annotation_font_color=YELLOW)
    _apply_theme(fig, height=320,
        barmode='overlay', margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='Auction Bid Distribution', font=dict(color=CYAN, size=14)),
        xaxis_title='Bid (chips)', yaxis_title='Count',
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    return fig


def chart_auction_ev(rounds_df, bot_name):
    won = rounds_df[rounds_df['auction_winner'] == bot_name]['bot_payoff'].mean()
    lost = rounds_df[(rounds_df['auction_winner'].notna()) &
                     (rounds_df['auction_winner'] != bot_name)]['bot_payoff'].mean()
    no_auc = rounds_df[rounds_df['auction_winner'].isna()]['bot_payoff'].mean()

    labels = ['Won Auction', 'Lost Auction', 'No Auction']
    values = [won if not pd.isna(won) else 0,
              lost if not pd.isna(lost) else 0,
              no_auc if not pd.isna(no_auc) else 0]
    colors = [GREEN if v >= 0 else RED for v in values]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
        text=[f'{v:+.1f}' for v in values], textposition='outside',
        textfont=dict(color='white'), marker_line_width=0))
    fig.add_hline(y=0, line_color='#2d3561')
    _apply_theme(fig, height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='Avg Profit by Auction Outcome', font=dict(color=CYAN, size=14)),
        yaxis_title='Avg Chips/Round')
    return fig


def chart_bid_vs_strength(rounds_df):
    df = rounds_df[rounds_df['bot_bid'].notna() &
                   (rounds_df['hand_bucket'] != 'Unknown')].copy()
    if df.empty:
        return None
    order = ['Weak', 'Medium', 'Strong', 'Premium']
    pal = {'Weak': RED, 'Medium': YELLOW, 'Strong': BLUE, 'Premium': GREEN}
    fig = go.Figure()
    for bucket in order:
        sub = df[df['hand_bucket'] == bucket]['bot_bid']
        if sub.empty:
            continue
        fig.add_trace(go.Box(y=sub, name=bucket, marker_color=pal.get(bucket, CYAN),
            boxmean=True, line=dict(color=pal.get(bucket, CYAN))))
    _apply_theme(fig, height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='Bid Amount vs Hand Strength', font=dict(color=CYAN, size=14)),
        yaxis_title='Bid (chips)')
    return fig


def chart_action_frequency(actions_df, bot_name):
    bot_acts = actions_df[actions_df['actor'] == bot_name]
    streets = ['preflop', 'flop', 'turn', 'river']
    pal = {'fold': RED, 'call': BLUE, 'check': YELLOW, 'raise': GREEN}
    fig = make_subplots(rows=1, cols=4,
        subplot_titles=[s.capitalize() for s in streets],
        shared_yaxes=False)

    for col, street in enumerate(streets, 1):
        sub = bot_acts[bot_acts['street'] == street]
        vc = sub['action'].value_counts()
        for action, cnt in vc.items():
            fig.add_trace(go.Bar(name=action, x=[action], y=[cnt],
                marker_color=pal.get(action, PURPLE),
                showlegend=(col == 1),
                text=[str(cnt)], textposition='outside',
                textfont=dict(color='white', size=11)),
                row=1, col=col)

    _apply_theme(fig, height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text='Action Frequency by Street', font=dict(color=CYAN, size=14)),
        barmode='group',
        legend=dict(orientation='h', y=1.15, bgcolor='rgba(0,0,0,0)'))
    for i in range(1, 5):
        fig.update_xaxes(showticklabels=False, row=1, col=i)
    return fig


def chart_profit_by_street(profit_by_street):
    streets = list(profit_by_street.keys())
    totals = [profit_by_street[s]['total_payoff'] for s in streets]
    avgs = [profit_by_street[s]['avg_payoff'] for s in streets]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Total Profit by Decision Street', 'Avg Profit per Round'])
    fig.add_trace(go.Bar(x=streets, y=totals,
        marker_color=[GREEN if v >= 0 else RED for v in totals],
        text=[f'{v:+,}' for v in totals], textposition='outside',
        textfont=dict(color='white'), marker_line_width=0), row=1, col=1)
    fig.add_trace(go.Bar(x=streets, y=avgs,
        marker_color=[GREEN if v >= 0 else RED for v in avgs],
        text=[f'{v:+.1f}' for v in avgs], textposition='outside',
        textfont=dict(color='white'), marker_line_width=0), row=1, col=2)
    fig.add_hline(y=0, line_color='#2d3561', row=1, col=1)
    fig.add_hline(y=0, line_color='#2d3561', row=1, col=2)
    _apply_theme(fig, height=320,
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def chart_fold_heatmap(rounds_df, actions_df, bot_name):
    df = rounds_df.copy()
    fold_rnds = set(actions_df[(actions_df['actor'] == bot_name) &
                                (actions_df['action'] == 'fold')]['round_num'])
    df['bot_folded'] = df['round_num'].isin(fold_rnds)
    buckets = ['Weak', 'Medium', 'Strong', 'Premium']
    positions = ['SB', 'BB']
    matrix = []
    for bucket in buckets:
        row = []
        for pos in positions:
            sub = df[(df['hand_bucket'] == bucket) & (df['bot_position'] == pos)]
            row.append(sub['bot_folded'].mean() * 100 if len(sub) else None)
        matrix.append(row)

    z_text = [[f"{v:.1f}%" if v is not None else "N/A" for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=positions, y=buckets,
        colorscale='RdYlGn_r', zmin=0, zmax=100,
        text=z_text, texttemplate='%{text}',
        textfont=dict(size=13, color='white'),
        # FIX: use nested title dict instead of deprecated titlefont
        colorbar=dict(
            title=dict(text='Fold %', font=dict(color='#8892b0')),
            tickfont=dict(color='#8892b0'),
        )))
    _apply_theme(fig, height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text='Fold Rate by Hand Strength & Position', font=dict(color=CYAN, size=14)),
        xaxis_title='Position', yaxis_title='Hand Strength')
    return fig


def chart_opponent_breakdown(rounds_df):
    df = rounds_df.groupby('opponent')['bot_payoff'].agg(['sum', 'mean', 'count']).reset_index()
    if df.empty:
        return None
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Total Profit vs Opponent', 'Avg Profit/Round vs Opponent'])
    fig.add_trace(go.Bar(x=df['opponent'], y=df['sum'],
        marker_color=[GREEN if v >= 0 else RED for v in df['sum']],
        text=[f'{v:+,}' for v in df['sum']], textposition='outside',
        textfont=dict(color='white'), marker_line_width=0), row=1, col=1)
    fig.add_trace(go.Bar(x=df['opponent'], y=df['mean'],
        marker_color=[GREEN if v >= 0 else RED for v in df['mean']],
        text=[f'{v:+.1f}' for v in df['mean']], textposition='outside',
        textfont=dict(color='white'), marker_line_width=0), row=1, col=2)
    _apply_theme(fig, height=320,
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def chart_rolling_winrate(rounds_df, window=50):
    won = (rounds_df['bot_payoff'] > 0).astype(int)
    roll = won.rolling(window, min_periods=1).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rounds_df['round_num'], y=roll,
        line=dict(color=CYAN, width=1.8), name=f'{window}-round win rate'))
    fig.add_hline(y=50, line_color=YELLOW, line_dash='dot',
        annotation_text='50%', annotation_font_color=YELLOW)
    _apply_theme(fig, height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f'Rolling {window}-Round Win Rate', font=dict(color=CYAN, size=14)),
        xaxis_title='Round', yaxis_title='Win Rate (%)')
    fig.update_yaxes(range=[0, 100])
    return fig


def chart_street_pnl_scatter(rounds_df):
    df = rounds_df.copy()
    won = df[df['bot_won_auction'] == True]
    lost = df[df['bot_won_auction'] == False]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=won['round_num'], y=won['bot_payoff'],
        mode='markers', name='Won Auction',
        marker=dict(color=GREEN, size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=lost['round_num'], y=lost['bot_payoff'],
        mode='markers', name='Lost Auction',
        marker=dict(color=PURPLE, size=5, opacity=0.6)))
    fig.add_hline(y=0, line_color='#2d3561')
    _apply_theme(fig, height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='Round P&L by Auction Outcome', font=dict(color=CYAN, size=14)),
        xaxis_title='Round', yaxis_title='Chips',
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    return fig


def chart_position_breakdown(rounds_df):
    df = rounds_df.groupby('bot_position').agg(
        total=('bot_payoff', 'sum'),
        avg=('bot_payoff', 'mean'),
        count=('bot_payoff', 'count')).reset_index()
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['Total P&L by Position', 'Avg P&L by Position'])
    fig.add_trace(go.Bar(x=df['bot_position'], y=df['total'],
        marker_color=[GREEN if v >= 0 else RED for v in df['total']],
        text=[f'{v:+,}' for v in df['total']], textposition='outside',
        textfont=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Bar(x=df['bot_position'], y=df['avg'],
        marker_color=[GREEN if v >= 0 else RED for v in df['avg']],
        text=[f'{v:+.1f}' for v in df['avg']], textposition='outside',
        textfont=dict(color='white')), row=1, col=2)
    _apply_theme(fig, height=300,
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def chart_version_delta(comp_df):
    delta_cols = [c for c in comp_df.columns if c.startswith('Δ')]
    if not delta_cols:
        return None
    col = delta_cols[0]
    sub = comp_df[col].dropna().sort_values()
    colors = [GREEN if v >= 0 else RED for v in sub.values]
    fig = go.Figure(go.Bar(y=sub.index, x=sub.values, orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f'{v:+.3f}' for v in sub.values], textposition='outside',
        textfont=dict(color='white')))
    fig.add_vline(x=0, line_color='#2d3561')
    _apply_theme(fig, height=max(350, len(sub) * 28),
        margin=dict(l=10, r=60, t=40, b=10),
        title=dict(text=f'Version Delta: {col}', font=dict(color=CYAN, size=14)),
        xaxis_title='Change')
    return fig


# ─── Hand Browser ─────────────────────────────────────────────────────────────

def render_hand_browser(rounds_df, actions_df, bot_name):
    section("🔍 Hand Browser")
    st.caption("Filter and inspect individual hands")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_rnd = int(rounds_df['round_num'].min())
        max_rnd = int(rounds_df['round_num'].max())
        rnd_range = st.slider("Round Range", min_rnd, max_rnd, (min_rnd, max_rnd))
    with col2:
        result_filter = st.selectbox("Result", ["All", "Wins", "Losses", "Big Wins (>200)", "Big Losses (<-200)"])
    with col3:
        street_filter = st.selectbox("Ended at", ["All", "preflop", "flop", "turn", "river", "showdown"])
    with col4:
        auction_filter = st.selectbox("Auction", ["All", "Won Auction", "Lost Auction", "No Auction"])

    df = rounds_df[(rounds_df['round_num'] >= rnd_range[0]) &
                   (rounds_df['round_num'] <= rnd_range[1])].copy()

    if result_filter == "Wins":
        df = df[df['bot_payoff'] > 0]
    elif result_filter == "Losses":
        df = df[df['bot_payoff'] < 0]
    elif result_filter == "Big Wins (>200)":
        df = df[df['bot_payoff'] > 200]
    elif result_filter == "Big Losses (<-200)":
        df = df[df['bot_payoff'] < -200]

    if auction_filter == "Won Auction":
        df = df[df['bot_won_auction'] == True]
    elif auction_filter == "Lost Auction":
        df = df[df['bot_won_auction'] == False]
    elif auction_filter == "No Auction":
        df = df[df['auction_winner'].isna()]

    if street_filter != "All" and 'decision_street' in df.columns:
        df = df[df['decision_street'] == street_filter]

    st.markdown(f"**{len(df)} hands** matching filters")

    show_n = st.slider("Show top N hands (sorted by |payoff|)", 5, min(50, len(df)), 10)
    df_show = df.reindex(df['bot_payoff'].abs().sort_values(ascending=False).index).head(show_n)

    for _, row in df_show.iterrows():
        rnd = int(row['round_num'])
        payoff = int(row['bot_payoff'])
        card_class = "hand-win" if payoff >= 0 else "hand-loss"
        payoff_str = f'<span style="color:{GREEN if payoff>=0 else RED};font-weight:700">{payoff:+,} chips</span>'

        rnd_acts = actions_df[actions_df['round_num'] == rnd]
        act_str = ""
        for _, a in rnd_acts.iterrows():
            clr = {'fold': RED, 'call': BLUE, 'check': YELLOW, 'raise': GREEN}.get(a['action'], CYAN)
            amt = f"({a['amount']})" if pd.notna(a['amount']) and a['amount'] else ""
            act_str += f'<span style="color:{clr}">[{a["street"][:2].upper()}] {a["actor"]}: {a["action"]} {amt}</span>  '

        hole = row.get('bot_hole', '')
        revealed = row.get('revealed_card', '')
        auction_won = '🏆 Won auction' if row.get('bot_won_auction') else '❌ Lost auction' if pd.notna(row.get('auction_winner')) else ''

        st.markdown(f'''
<div class="hand-card {card_class}">
  <b>Round #{rnd}</b> &nbsp;|&nbsp; {payoff_str} &nbsp;|&nbsp; 
  <span style="color:#8892b0">Hole: {hole}</span> &nbsp;|&nbsp;
  <span style="color:{CYAN}">{auction_won}</span>
  {f'&nbsp;→ revealed [{revealed}]' if revealed else ''}
  <div style="margin-top:8px;font-size:12px;color:#8892b0">{act_str}</div>
</div>''', unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
if 'active_session' not in st.session_state:
    st.session_state.active_session = None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🃏 Pokerbots Analyzer")
    st.markdown("---")

    st.markdown("### 📂 Upload Log Files")
    uploaded_files = st.file_uploader(
        "Drop log files (.glog, .fz, .log, .log.gh, etc.)",
        type=None,
        accept_multiple_files=True,
        help="Upload any IIT Pokerbots game log file — any extension is accepted"
    )
    bot_name_input = st.text_input(
        "Bot name (auto-detect if blank)",
        placeholder="e.g. phoenix_1"
    )

    if uploaded_files:
        for uf in uploaded_files:
            label = uf.name
            if label not in st.session_state.sessions:
                with st.spinner(f"Analysing {label}…"):
                    result = parse_and_analyse(uf.read(), uf.name,
                                               bot_name_input.strip() or None)
                    rdf, adf, m, l, bn, opp, fb = result
                    if rdf is not None:
                        st.session_state.sessions[label] = {
                            'rounds_df': rdf, 'actions_df': adf,
                            'metrics': m, 'leaks': l,
                            'bot_name': bn, 'opponent_name': opp,
                            'final_bankrolls': fb, 'filename': label
                        }
                        st.success(f"✅ {label}")
                    else:
                        st.error(f"❌ Could not parse {label}")

    if st.session_state.sessions:
        st.markdown("---")
        st.markdown("### 📋 Sessions")
        session_labels = list(st.session_state.sessions.keys())
        selected = st.radio("Active session", session_labels, index=0)
        st.session_state.active_session = selected

        if len(session_labels) > 1:
            st.markdown("---")
            st.markdown("### 🔄 Compare Versions")
            v1 = st.selectbox("Version A", session_labels, index=0)
            v2 = st.selectbox("Version B", session_labels,
                               index=min(1, len(session_labels)-1))
            run_compare = st.button("⚡ Run Comparison", width='stretch')
        else:
            run_compare = False
            v1, v2 = None, None

        st.markdown("---")
        if st.button("🗑 Clear All", width='stretch'):
            st.session_state.sessions = {}
            st.session_state.active_session = None
            st.rerun()
    else:
        run_compare = False
        v1, v2 = None, None


# ─── Main area ────────────────────────────────────────────────────────────────

if not st.session_state.sessions:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:72px">🃏</div>
        <h1 style="color:#64ffda;font-size:36px;margin-bottom:8px">IIT Pokerbots Analyzer</h1>
        <p style="color:#8892b0;font-size:17px;max-width:560px;margin:0 auto 32px">
            Drop your <code>.glog</code> files in the sidebar to get instant strategic analysis, 
            leak detection, and interactive charts for your Sneak Peek Hold'em bot.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📊", "Full Metrics", "VPIP, PFR, C-bet, auction stats, barrel frequencies"),
        (c2, "🚨", "Leak Detection", "Auto-identifies 15+ strategic weaknesses with fixes"),
        (c3, "📈", "Interactive Charts", "Plotly charts for bankroll, bids, heatmaps & more"),
        (c4, "🔄", "Version Compare", "Side-by-side delta comparison of two bot versions"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #2d3561;border-radius:10px;padding:20px;text-align:center">
                <div style="font-size:32px">{icon}</div>
                <div style="color:#ccd6f6;font-weight:700;margin:8px 0">{title}</div>
                <div style="color:#8892b0;font-size:13px">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📄 Supported Log Format")
    st.code("""Round #1, phoenix_1 (0), NPC48 (0)
phoenix_1 posts blind: 10
NPC48 posts blind: 20
phoenix_1 received [Ac Kd]
NPC48 received [9s Qs]
phoenix_1 raises to 40
NPC48 calls
Flop [Ad Qc 3d], phoenix_1 (40), NPC48 (40)
phoenix_1 bids 75
NPC48 bids 104
NPC48 won the auction and was revealed [9s]
...
phoenix_1 awarded 115
Final, phoenix_1 (-24464), NPC48 (24464)""", language='text')
    st.stop()


# ─── Active session data ──────────────────────────────────────────────────────

sess = st.session_state.sessions[st.session_state.active_session]
rounds_df  = sess['rounds_df']
actions_df = sess['actions_df']
metrics    = sess['metrics']
leaks      = sess['leaks']
bot_name   = sess['bot_name']

pf   = metrics.get('preflop', {})
auc  = metrics.get('auction', {})
flop = metrics.get('flop', {})
turn = metrics.get('turn', {})
rvr  = metrics.get('river', {})


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <span style="font-size:36px">🃏</span>
    <div>
        <h1 style="margin:0;color:#ccd6f6">Pokerbots Analyzer</h1>
        <p style="margin:0;color:#8892b0;font-size:13px">
            Bot: <b style="color:#64ffda">{bot_name}</b> &nbsp;|&nbsp;
            File: <b style="color:#74b9ff">{sess['filename']}</b> &nbsp;|&nbsp;
            {len(rounds_df):,} rounds analysed
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

if leaks:
    highs  = sum(1 for l in leaks if l.severity == 'HIGH')
    meds   = sum(1 for l in leaks if l.severity == 'MEDIUM')
    lows   = sum(1 for l in leaks if l.severity == 'LOW')
    st.markdown(
        f'🚨 Leaks detected: '
        f'{badge(f"{highs} HIGH", "red")} '
        f'{badge(f"{meds} MEDIUM", "yellow")} '
        f'{badge(f"{lows} LOW", "green")}',
        unsafe_allow_html=True)
else:
    st.markdown('✅ No significant leaks detected', unsafe_allow_html=True)

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 Overview",
    "🎯 Auction",
    "🌊 Post-Flop",
    "🔍 Hand Browser",
    "🚨 Leaks",
    "🔄 Compare",
    "📋 Data Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    total = metrics.get('total_payoff', 0)
    avg   = metrics.get('avg_payoff_per_round', 0)
    wr    = metrics.get('win_rate', 0)
    with k1:
        st.metric("Total P&L", f"{total:+,}", delta=f"{'Profit' if total>=0 else 'Loss'}")
    with k2:
        st.metric("Avg/Round", f"{avg:+.1f}", delta=f"{'Edge' if avg>=0 else 'Deficit'}")
    with k3:
        st.metric("Win Rate", f"{wr:.1%}")
    with k4:
        st.metric("VPIP", f"{pf.get('VPIP',0):.1%}")
    with k5:
        st.metric("PFR", f"{pf.get('PFR',0):.1%}")
    with k6:
        st.metric("Rounds", f"{len(rounds_df):,}")

    st.markdown("")
    st.plotly_chart(chart_bankroll(rounds_df), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_round_pnl(rounds_df), width='stretch')
    with col2:
        window = st.slider("Win rate window", 20, 200, 50, step=10, key='wr_window')
        st.plotly_chart(chart_rolling_winrate(rounds_df, window), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        section("📍 Position Breakdown")
        st.plotly_chart(chart_position_breakdown(rounds_df), width='stretch')
    with col2:
        section("🗺 Profit by Decision Street")
        pbs = metrics.get('profit_by_street', {})
        if pbs:
            st.plotly_chart(chart_profit_by_street(pbs), width='stretch')

    section("🃏 Preflop Metrics")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1: st.metric("VPIP", f"{pf.get('VPIP',0):.1%}", help="Voluntarily Put money In Pot")
    with p2: st.metric("PFR", f"{pf.get('PFR',0):.1%}", help="Pre-Flop Raise frequency")
    with p3: st.metric("3-Bet %", f"{pf.get('3bet_pct',0):.1%}")
    with p4: st.metric("Fold-to-Raise", f"{pf.get('fold_to_raise_pct',0):.1%}")
    with p5: st.metric("Avg Open Size", f"{pf.get('avg_open_size',0):.0f} chips")

    pf_profit = pf.get('profit_by_pf_action', {})
    if pf_profit:
        st.markdown("**Avg Payoff by First Preflop Action:**")
        cols = st.columns(len(pf_profit))
        for col, (act, val) in zip(cols, pf_profit.items()):
            with col:
                st.metric(act.capitalize(), f"{val:+.1f}")

    section("📊 Action Frequency by Street")
    st.plotly_chart(chart_action_frequency(actions_df, bot_name), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AUCTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("🎯 Auction Performance")

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1: st.metric("Win Rate", f"{auc.get('auction_win_rate',0):.1%}")
    with a2: st.metric("Avg Bid", f"{auc.get('avg_bid',0):.0f}")
    with a3: st.metric("Median Bid", f"{auc.get('median_bid',0):.0f}")
    with a4: st.metric("Bid Variance", f"{auc.get('bid_variance',0):.0f}")
    with a5: st.metric("Overbid Rate", f"{auc.get('overbid_rate',0):.1%}")
    with a6: st.metric("Close-Loss Rate", f"{auc.get('close_loss_rate',0):.1%}",
        help="% of rounds where you lost auction by ≤5 chips")

    a7, a8, a9 = st.columns(3)
    with a7:
        ev_win = auc.get('avg_profit_when_winning_auction', 0)
        st.metric("EV When Winning", f"{ev_win:+.1f}",
            delta="Positive" if ev_win >= 0 else "Negative")
    with a8:
        ev_lose = auc.get('avg_profit_when_losing_auction', 0)
        st.metric("EV When Losing", f"{ev_lose:+.1f}")
    with a9:
        st.metric("Bid/Stack Ratio", f"{auc.get('bid_to_stack_ratio',0):.1%}")

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_auction_dist(rounds_df, bot_name), width='stretch')
    with col2:
        st.plotly_chart(chart_auction_ev(rounds_df, bot_name), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        fig = chart_bid_vs_strength(rounds_df)
        if fig:
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No hand strength data available")
    with col2:
        st.plotly_chart(chart_street_pnl_scatter(rounds_df), width='stretch')

    section("💡 Bid by Hand Strength")
    bid_strength = auc.get('avg_bid_by_hand_strength', {})
    if bid_strength:
        cols = st.columns(len(bid_strength))
        for col, (bucket, val) in zip(cols, bid_strength.items()):
            with col:
                st.metric(bucket, f"{val:.0f} chips")

    section("📈 Bid Timeline")
    df_bids = rounds_df[rounds_df['bot_bid'].notna()].copy()
    if not df_bids.empty:
        roll_bid = df_bids['bot_bid'].rolling(30, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bids['round_num'], y=df_bids['bot_bid'],
            mode='markers', marker=dict(color=BLUE, size=4, opacity=0.4),
            name='Individual bids'))
        fig.add_trace(go.Scatter(x=df_bids['round_num'], y=roll_bid,
            line=dict(color=CYAN, width=2), name='30-round avg'))
        _apply_theme(fig, height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text='Bid Amount Over Time', font=dict(color=CYAN, size=14)),
            xaxis_title='Round', yaxis_title='Bid (chips)',
            legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POST-FLOP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section("🌊 Post-Flop Metrics")

    f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
    with f1: st.metric("C-Bet %", f"{flop.get('cbet_pct',0):.1%}")
    with f2: st.metric("Fold-to-CBet", f"{flop.get('fold_to_cbet_pct',0):.1%}")
    with f3: st.metric("Double Barrel", f"{turn.get('double_barrel_pct',0):.1%}")
    with f4: st.metric("Triple Barrel", f"{rvr.get('triple_barrel_pct',0):.1%}")
    with f5: st.metric("River Agg", f"{rvr.get('river_aggression_pct',0):.1%}")
    with f6: st.metric("River Fold", f"{rvr.get('river_fold_pct',0):.1%}")
    with f7: st.metric("Call Success", f"{rvr.get('river_call_success_pct',0):.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_fold_heatmap(rounds_df, actions_df, bot_name),
            width='stretch')
    with col2:
        section("🎯 Barrel EV Breakdown")
        barrel_data = {
            '1 Barrel (flop only)': turn.get('avg_profit_2barrel', 0),
            '2 Barrels (flop+turn)': turn.get('avg_profit_2barrel', 0),
            '3 Barrels (all streets)': rvr.get('avg_profit_3barrel', 0),
        }
        labels = list(barrel_data.keys())
        vals   = list(barrel_data.values())
        fig = go.Figure(go.Bar(x=labels, y=vals,
            marker_color=[GREEN if v >= 0 else RED for v in vals],
            text=[f'{v:+.1f}' for v in vals], textposition='outside',
            textfont=dict(color='white'), marker_line_width=0))
        fig.add_hline(y=0, line_color='#2d3561')
        _apply_theme(fig, height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text='Avg Profit by Barrel Count', font=dict(color=CYAN, size=14)))
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        section("📊 Street-level Action Mix")
        bot_acts = actions_df[actions_df['actor'] == bot_name]
        street_mix = bot_acts.groupby(['street', 'action']).size().reset_index(name='count')
        fig = px.sunburst(street_mix, path=['street', 'action'], values='count',
            color='action',
            color_discrete_map={'fold': RED, 'call': BLUE, 'check': YELLOW, 'raise': GREEN})
        _apply_theme(fig, height=340,
            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with col2:
        section("💰 Opponent Breakdown")
        st.plotly_chart(chart_opponent_breakdown(rounds_df), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HAND BROWSER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    render_hand_browser(rounds_df, actions_df, bot_name)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LEAKS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section(f"🚨 Leak Report — {len(leaks)} issues found")

    if not leaks:
        st.success("✅ No significant strategic leaks detected in this log!")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            sev_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for l in leaks:
                sev_counts[l.severity] += 1
            fig = go.Figure(go.Pie(
                labels=list(sev_counts.keys()),
                values=list(sev_counts.values()),
                hole=0.6,
                marker=dict(colors=[RED, YELLOW, GREEN]),
                textfont=dict(color='white')))
            _apply_theme(fig, height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')))
            st.plotly_chart(fig, width='stretch')

        with col2:
            cat_counts = {}
            for l in leaks:
                cat_counts[l.category] = cat_counts.get(l.category, 0) + 1
            fig = go.Figure(go.Bar(
                x=list(cat_counts.keys()), y=list(cat_counts.values()),
                marker_color=PURPLE, marker_line_width=0,
                text=list(cat_counts.values()), textposition='outside',
                textfont=dict(color='white')))
            _apply_theme(fig, height=260,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text='Leaks by Category', font=dict(color=CYAN, size=13)),
                yaxis_title='Count')
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        sev_filter = st.multiselect("Show severities", ['HIGH', 'MEDIUM', 'LOW'],
                                     default=['HIGH', 'MEDIUM', 'LOW'])

        for leak in leaks:
            if leak.severity not in sev_filter:
                continue
            css_class = f"leak-{leak.severity.lower()}"
            icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[leak.severity]
            sev_color = {'HIGH': RED, 'MEDIUM': YELLOW, 'LOW': GREEN}[leak.severity]
            st.markdown(f"""
<div class="{css_class}">
  <div class="leak-title">
    {icon} <span style="color:{sev_color}">[{leak.severity}]</span> 
    {leak.category} — {leak.name}
  </div>
  <div class="leak-desc">📊 {leak.description}</div>
  <div class="leak-desc">🔍 Evidence: {leak.evidence}</div>
  <div class="leak-rec">💡 Fix: {leak.recommendation}</div>
</div>""", unsafe_allow_html=True)

    if leaks:
        st.markdown("---")
        leak_text = "\n\n".join(
            f"[{l.severity}] {l.category} — {l.name}\n"
            f"Description: {l.description}\n"
            f"Evidence: {l.evidence}\n"
            f"Recommendation: {l.recommendation}"
            for l in leaks
        )
        st.download_button("📥 Download Leak Report (.txt)",
            leak_text.encode(), f"leaks_{bot_name}.txt", "text/plain")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section("🔄 Bot Version Comparison")

    sessions = st.session_state.sessions
    if len(sessions) < 2:
        st.info("📂 Upload **at least 2 log files** in the sidebar to compare versions.")
        st.markdown("""
        **How to use:**
        1. Upload your v1 log as one file
        2. Upload your v2 log as another file  
        3. Select both in the dropdowns in the sidebar
        4. Click **Run Comparison**
        """)
    else:
        all_labels = list(sessions.keys())
        col1, col2 = st.columns(2)
        with col1:
            va_label = st.selectbox("📦 Version A", all_labels, index=0, key='va')
        with col2:
            vb_label = st.selectbox("📦 Version B", all_labels,
                                     index=min(1, len(all_labels)-1), key='vb')

        if st.button("⚡ Run Comparison", width='stretch', type='primary') or run_compare:
            sa = sessions[va_label]
            sb = sessions[vb_label]

            versions = {va_label: sa['metrics'], vb_label: sb['metrics']}
            comp = ComparisonEngine()
            df_comp = comp.compare_versions(versions)

            st.markdown("### 📊 Side-by-Side Metrics")
            delta_cols = [c for c in df_comp.columns if c.startswith('Δ')]

            def highlight_delta(val):
                if pd.isna(val): return ''
                if isinstance(val, (int, float)):
                    if val > 0.001:  return f'color: {GREEN}'
                    if val < -0.001: return f'color: {RED}'
                return f'color: {YELLOW}'

            styled = df_comp.style.applymap(highlight_delta, subset=delta_cols)
            st.dataframe(styled, width='stretch')

            if delta_cols:
                delta_col = delta_cols[0]
                sub = df_comp[delta_col].dropna().sort_values()
                fig = go.Figure(go.Bar(
                    y=sub.index, x=sub.values, orientation='h',
                    marker_color=[GREEN if v >= 0 else RED for v in sub.values],
                    marker_line_width=0,
                    text=[f'{v:+.3f}' for v in sub.values], textposition='outside',
                    textfont=dict(color='white')))
                fig.add_vline(x=0, line_color='#2d3561')
                _apply_theme(fig,
                    height=max(350, len(sub) * 30),
                    margin=dict(l=10, r=80, t=40, b=10),
                    title=dict(text=f'Metric Delta: {va_label} → {vb_label}',
                               font=dict(color=CYAN, size=14)),
                    xaxis_title='Change')
                st.plotly_chart(fig, width='stretch')

            st.markdown("### 📈 Bankroll Comparison")
            fig = go.Figure()
            for label, s, color in [(va_label, sa, BLUE), (vb_label, sb, GREEN)]:
                pnl = s['rounds_df']['bot_payoff'].cumsum()
                fig.add_trace(go.Scatter(
                    x=s['rounds_df']['round_num'], y=pnl,
                    line=dict(color=color, width=2), name=label))
            fig.add_hline(y=0, line_color='#2d3561', line_width=1)
            _apply_theme(fig, height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text='Cumulative P&L Comparison', font=dict(color=CYAN, size=14)),
                legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, width='stretch')

            csv_buf = io.StringIO()
            df_comp.to_csv(csv_buf)
            st.download_button("📥 Download Comparison CSV",
                csv_buf.getvalue().encode(), "version_comparison.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section("📋 Raw Data Explorer")

    exp_tab1, exp_tab2, exp_tab3 = st.tabs(["Rounds", "Actions", "Export"])

    with exp_tab1:
        st.caption("Full rounds DataFrame — sortable, filterable")
        all_cols = rounds_df.columns.tolist()
        default_cols = ['round_num', 'bot_position', 'bot_hole', 'bot_bid', 'opp_bid',
                        'auction_winner', 'revealed_card', 'bot_payoff', 'hand_bucket',
                        'folded', 'went_to_showdown']
        show_cols = st.multiselect("Columns", all_cols,
            default=[c for c in default_cols if c in all_cols])
        df_display = rounds_df[show_cols] if show_cols else rounds_df
        st.dataframe(df_display, width='stretch', height=450)

    with exp_tab2:
        st.caption("Full actions DataFrame")
        street_sel = st.multiselect("Street filter", ['preflop','flop','turn','river'],
            default=['preflop','flop','turn','river'])
        df_acts_show = actions_df[actions_df['street'].isin(street_sel)]
        st.dataframe(df_acts_show, width='stretch', height=450)

    with exp_tab3:
        st.markdown("### 📥 Export Options")
        c1, c2, c3 = st.columns(3)
        with c1:
            csv_rounds = rounds_df.to_csv(index=False).encode()
            st.download_button("📊 Rounds CSV", csv_rounds,
                f"rounds_{bot_name}.csv", "text/csv", width='stretch')
        with c2:
            csv_actions = actions_df.to_csv(index=False).encode()
            st.download_button("🃏 Actions CSV", csv_actions,
                f"actions_{bot_name}.csv", "text/csv", width='stretch')
        with c3:
            metrics_json = json.dumps(
                {k: (v if not isinstance(v, dict) or all(isinstance(vv, (int,float,str,type(None)))
                     for vv in v.values()) else v)
                 for k, v in metrics.items()
                 if k not in ('opponent_breakdown',)},
                default=str, indent=2)
            st.download_button("📈 Metrics JSON", metrics_json.encode(),
                f"metrics_{bot_name}.json", "application/json", width='stretch')

        st.markdown("---")
        st.markdown("### 📄 Full Metrics Summary")
        st.json(metrics)
