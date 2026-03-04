
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─── Style ────────────────────────────────────────────────────────────────────
DARK_BG = '#1a1a2e'
CARD_BG = '#16213e'
ACCENT  = '#0f3460'
GREEN   = '#00b894'
RED     = '#e17055'
YELLOW  = '#fdcb6e'
BLUE    = '#74b9ff'
PURPLE  = '#a29bfe'

def _style():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': CARD_BG,
        'axes.edgecolor': '#555',
        'axes.labelcolor': 'white',
        'xtick.color': '#aaa',
        'ytick.color': '#aaa',
        'text.color': 'white',
        'grid.color': '#2d3561',
        'grid.alpha': 0.5,
        'font.family': 'DejaVu Sans',
    })

_style()


class Visualizer:

    def __init__(self, rounds_df: pd.DataFrame, actions_df: pd.DataFrame,
                 bot_name: str, output_dir: str = 'output'):
        self.rounds_df = rounds_df
        self.actions_df = actions_df
        self.bot_name = bot_name
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.bot_actions = actions_df[actions_df['actor'] == bot_name]

    def save(self, fig, name: str):
        path = self.out / name
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        return str(path)

    # ── 1. Cumulative Bankroll ────────────────────────────────────────────────

    def plot_bankroll(self) -> str:
        df = self.rounds_df
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(DARK_BG)

        pnl = df['bot_payoff'].cumsum()
        rounds = df['round_num']

        color = np.where(pnl >= 0, GREEN, RED)
        ax.fill_between(rounds, 0, pnl, where=(pnl >= 0), color=GREEN, alpha=0.25, interpolate=True)
        ax.fill_between(rounds, 0, pnl, where=(pnl < 0), color=RED, alpha=0.25, interpolate=True)
        ax.plot(rounds, pnl, color=BLUE, lw=1.5, zorder=3)
        ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5)

        # Rolling 50-round avg
        roll = pnl.rolling(50, min_periods=1).mean()
        ax.plot(rounds, roll, color=YELLOW, lw=1.5, ls='--', alpha=0.8, label='50-round rolling avg')

        final = int(pnl.iloc[-1]) if len(pnl) else 0
        color_final = GREEN if final >= 0 else RED
        ax.set_title(f'📈 Cumulative P&L — {self.bot_name}   (Final: {final:+,} chips)',
                     fontsize=14, pad=12)
        ax.set_xlabel('Round')
        ax.set_ylabel('Chips')
        ax.legend(loc='upper left')
        ax.grid(True)
        return self.save(fig, 'bankroll.png')

    # ── 2. Profit by Street ────────────────────────────────────────────────────

    def plot_profit_by_street(self, profit_by_street: dict) -> str:
        streets = list(profit_by_street.keys())
        totals = [profit_by_street[s]['total_payoff'] for s in streets]
        counts = [profit_by_street[s]['rounds'] for s in streets]
        avgs = [profit_by_street[s]['avg_payoff'] for s in streets]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
            ax.set_facecolor(CARD_BG)

        bar_colors = [GREEN if t >= 0 else RED for t in totals]
        axes[0].bar(streets, totals, color=bar_colors, edgecolor='#333', linewidth=0.5)
        axes[0].set_title('Total Profit by Decision Street', fontsize=12)
        axes[0].set_ylabel('Total Chips')
        axes[0].axhline(0, color='white', lw=0.7, ls='--', alpha=0.5)
        for i, (v, c) in enumerate(zip(totals, counts)):
            axes[0].text(i, v + (20 if v >= 0 else -40), f'n={c}', ha='center',
                         fontsize=8, color='white')

        avg_colors = [GREEN if a >= 0 else RED for a in avgs]
        axes[1].bar(streets, avgs, color=avg_colors, edgecolor='#333', linewidth=0.5)
        axes[1].set_title('Average Profit per Round by Decision Street', fontsize=12)
        axes[1].set_ylabel('Avg Chips')
        axes[1].axhline(0, color='white', lw=0.7, ls='--', alpha=0.5)

        fig.suptitle('Profit by Street', fontsize=14, y=1.01)
        return self.save(fig, 'profit_by_street.png')

    # ── 3. Auction Bid Distribution ────────────────────────────────────────────

    def plot_auction_distribution(self) -> str:
        df = self.rounds_df
        bot_bids = df['bot_bid'].dropna()
        opp_bids = df['opp_bid'].dropna()

        if len(bot_bids) == 0:
            return ''

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax in axes:
            ax.set_facecolor(CARD_BG)

        bins = 40
        axes[0].hist(bot_bids, bins=bins, color=BLUE, alpha=0.75, label=self.bot_name, edgecolor='#333')
        if len(opp_bids):
            axes[0].hist(opp_bids, bins=bins, color=PURPLE, alpha=0.55, label='Opponent', edgecolor='#333')
        axes[0].axvline(bot_bids.mean(), color=YELLOW, ls='--', lw=1.5, label=f'Mean={bot_bids.mean():.0f}')
        axes[0].set_title('Auction Bid Distribution', fontsize=12)
        axes[0].set_xlabel('Bid Amount (chips)')
        axes[0].legend()
        axes[0].grid(True)

        # Win/Loss comparison
        won = df[df['auction_winner'] == self.bot_name]['bot_bid'].dropna()
        lost = df[(df['auction_winner'].notna()) & (df['auction_winner'] != self.bot_name)]['bot_bid'].dropna()
        axes[1].hist(won, bins=bins, color=GREEN, alpha=0.75, label='Won Auction', edgecolor='#333')
        if len(lost):
            axes[1].hist(lost, bins=bins, color=RED, alpha=0.65, label='Lost Auction', edgecolor='#333')
        axes[1].set_title('Bid Amount: Win vs Loss', fontsize=12)
        axes[1].set_xlabel('Bid Amount (chips)')
        axes[1].legend()
        axes[1].grid(True)

        return self.save(fig, 'auction_distribution.png')

    # ── 4. Bid vs Hand Strength ────────────────────────────────────────────────

    def plot_bid_vs_strength(self) -> str:
        df = self.rounds_df[self.rounds_df['bot_bid'].notna() & (self.rounds_df['hand_bucket'] != 'Unknown')]
        if len(df) == 0:
            return ''

        fig, ax = plt.subplots(figsize=(10, 5))
        order = ['Weak', 'Medium', 'Strong', 'Premium']
        present = [o for o in order if o in df['hand_bucket'].unique()]
        palette = {'Weak': RED, 'Medium': YELLOW, 'Strong': BLUE, 'Premium': GREEN}

        for i, bucket in enumerate(present):
            sub = df[df['hand_bucket'] == bucket]['bot_bid']
            ax.boxplot(sub, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=palette[bucket], alpha=0.7),
                       medianprops=dict(color='white', lw=2),
                       whiskerprops=dict(color='#aaa'),
                       capprops=dict(color='#aaa'),
                       flierprops=dict(marker='o', color=palette[bucket], alpha=0.3, markersize=3))

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        ax.set_title('Auction Bid vs Preflop Hand Strength', fontsize=13)
        ax.set_ylabel('Bid Amount (chips)')
        ax.grid(True, axis='y')
        return self.save(fig, 'bid_vs_strength.png')

    # ── 5. Heatmap: Hand Strength vs Fold Rate ─────────────────────────────────

    def plot_fold_heatmap(self) -> str:
        df = self.rounds_df.copy()
        if 'hand_bucket' not in df.columns:
            return ''

        bot_acts = self.bot_actions.copy()
        fold_rnds = set(bot_acts[bot_acts['action'] == 'fold']['round_num'])
        df['bot_folded'] = df['round_num'].isin(fold_rnds)

        positions = ['SB', 'BB']
        buckets = ['Weak', 'Medium', 'Strong', 'Premium']

        matrix = pd.DataFrame(index=buckets, columns=positions, dtype=float)
        for bucket in buckets:
            for pos in positions:
                sub = df[(df['hand_bucket'] == bucket) & (df['bot_position'] == pos)]
                if len(sub):
                    matrix.loc[bucket, pos] = sub['bot_folded'].mean()
                else:
                    matrix.loc[bucket, pos] = np.nan

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(matrix.astype(float), annot=True, fmt='.1%', cmap='RdYlGn_r',
                    ax=ax, linewidths=0.5, cbar_kws={'label': 'Fold Rate'},
                    annot_kws={'fontsize': 11})
        ax.set_title('Fold Rate by Hand Strength & Position', fontsize=12)
        ax.set_ylabel('Hand Strength')
        ax.set_xlabel('Position')
        return self.save(fig, 'fold_heatmap.png')

    # ── 6. Action Frequency by Street ────────────────────────────────────────

    def plot_action_frequency(self) -> str:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
        streets = ['preflop', 'flop', 'turn', 'river']
        colors = [BLUE, GREEN, YELLOW, RED]

        for ax, street, color in zip(axes, streets, colors):
            ax.set_facecolor(CARD_BG)
            sub = self.bot_actions[self.bot_actions['street'] == street]
            if len(sub) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(street.capitalize())
                continue
            vc = sub['action'].value_counts()
            ax.bar(vc.index, vc.values, color=color, alpha=0.8, edgecolor='#333')
            ax.set_title(street.capitalize(), fontsize=12)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=20)
            ax.grid(True, axis='y')
            for j, (idx, val) in enumerate(vc.items()):
                ax.text(j, val + 0.5, str(val), ha='center', fontsize=8, color='white')

        fig.suptitle(f'Action Frequency by Street — {self.bot_name}', fontsize=14)
        plt.tight_layout()
        return self.save(fig, 'action_frequency.png')

    # ── 7. Profit by Opponent ───────────────────────────────────────────────

    def plot_opponent_breakdown(self) -> str:
        df = self.rounds_df.groupby('opponent')['bot_payoff'].agg(['sum', 'mean', 'count']).reset_index()
        if len(df) == 0:
            return ''

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
            ax.set_facecolor(CARD_BG)

        colors = [GREEN if v >= 0 else RED for v in df['sum']]
        axes[0].barh(df['opponent'], df['sum'], color=colors, edgecolor='#333')
        axes[0].set_title('Total Profit vs Each Opponent', fontsize=12)
        axes[0].set_xlabel('Total Chips')
        axes[0].axvline(0, color='white', lw=0.8, ls='--')

        avg_colors = [GREEN if v >= 0 else RED for v in df['mean']]
        axes[1].barh(df['opponent'], df['mean'], color=avg_colors, edgecolor='#333')
        axes[1].set_title('Avg Profit Per Round vs Each Opponent', fontsize=12)
        axes[1].set_xlabel('Avg Chips')
        axes[1].axvline(0, color='white', lw=0.8, ls='--')

        return self.save(fig, 'opponent_breakdown.png')

    # ── 8. Auction EV Summary ────────────────────────────────────────────────

    def plot_auction_ev(self) -> str:
        df = self.rounds_df
        won_mask = df['auction_winner'] == self.bot_name
        lost_mask = (df['auction_winner'].notna()) & (df['auction_winner'] != self.bot_name)

        won_pnl = df[won_mask]['bot_payoff'].mean()
        lost_pnl = df[lost_mask]['bot_payoff'].mean()
        no_auction_pnl = df[df['auction_winner'].isna()]['bot_payoff'].mean()

        labels = ['Won Auction', 'Lost Auction', 'No Auction']
        values = [won_pnl if not pd.isna(won_pnl) else 0,
                  lost_pnl if not pd.isna(lost_pnl) else 0,
                  no_auction_pnl if not pd.isna(no_auction_pnl) else 0]
        bar_colors = [GREEN if v >= 0 else RED for v in values]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=bar_colors, edgecolor='#444', linewidth=0.7, width=0.5)
        ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + (3 if v >= 0 else -6),
                    f'{v:+.1f}', ha='center', va='bottom', fontsize=11, color='white')
        ax.set_title('Average Profit by Auction Outcome', fontsize=13)
        ax.set_ylabel('Avg Chips per Round')
        ax.grid(True, axis='y')
        return self.save(fig, 'auction_ev.png')

    # ── 9. Round-by-Round Win/Loss Bar ─────────────────────────────────────

    def plot_round_pnl(self) -> str:
        df = self.rounds_df
        fig, ax = plt.subplots(figsize=(16, 4))
        colors = [GREEN if v > 0 else RED for v in df['bot_payoff']]
        ax.bar(df['round_num'], df['bot_payoff'], color=colors, width=1.0, linewidth=0)
        ax.axhline(0, color='white', lw=0.7)
        ax.set_title(f'Per-Round P&L — {self.bot_name}', fontsize=13)
        ax.set_xlabel('Round')
        ax.set_ylabel('Chips')
        ax.grid(True, axis='y')
        return self.save(fig, 'round_pnl.png')

    # ── 10. Metrics Dashboard ─────────────────────────────────────────────────

    def plot_metrics_dashboard(self, metrics: dict) -> str:
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor(DARK_BG)
        fig.suptitle(f'📊 Strategic Metrics Dashboard — {self.bot_name}',
                     fontsize=15, color='white', y=0.98)

        pf  = metrics.get('preflop', {})
        auc = metrics.get('auction', {})
        flop = metrics.get('flop', {})
        turn = metrics.get('turn', {})
        river = metrics.get('river', {})

        metric_groups = [
            ('Preflop', [(f"VPIP\n{pf.get('VPIP', 0):.1%}", pf.get('VPIP', 0), 0.25, 0.65),
                          (f"PFR\n{pf.get('PFR', 0):.1%}", pf.get('PFR', 0), 0.15, 0.5),
                          (f"3bet\n{pf.get('3bet_pct', 0):.1%}", pf.get('3bet_pct', 0), 0.05, 0.15)]),
            ('Auction', [(f"Win%\n{auc.get('auction_win_rate', 0):.1%}", auc.get('auction_win_rate', 0), 0.4, 0.6),
                          (f"AvgBid\n{auc.get('avg_bid', 0):.0f}", None, None, None),
                          (f"EV Win\n{auc.get('avg_profit_when_winning_auction', 0):+.0f}", None, None, None)]),
            ('Post-Flop', [(f"Cbet\n{flop.get('cbet_pct', 0):.1%}", flop.get('cbet_pct', 0), 0.40, 0.90),
                           (f"2xBarrel\n{turn.get('double_barrel_pct', 0):.1%}", turn.get('double_barrel_pct', 0), 0.15, 0.55),
                           (f"3xBarrel\n{river.get('triple_barrel_pct', 0):.1%}", river.get('triple_barrel_pct', 0), 0.05, 0.35)]),
        ]

        axes = []
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)
        for col, (group_name, items) in enumerate(metric_groups):
            sub_gs = gs[col].subgridspec(1, 3, wspace=0.15)
            group_ax = fig.add_subplot(gs[col])
            group_ax.set_facecolor(ACCENT)
            group_ax.set_xticks([])
            group_ax.set_yticks([])
            for spine in group_ax.spines.values():
                spine.set_edgecolor('#555')
            group_ax.set_title(group_name, fontsize=11, pad=8, color='white')

            for i, (label, val, lo, hi) in enumerate(items):
                inner_ax = fig.add_axes([0.05 + col*0.33 + i*0.095, 0.15, 0.08, 0.65])
                inner_ax.set_facecolor(CARD_BG)
                inner_ax.set_xticks([])
                inner_ax.set_yticks([])
                for spine in inner_ax.spines.values():
                    spine.set_edgecolor('#333')

                if val is not None and lo is not None:
                    norm_val = max(0, min(1, (val - lo) / max(hi - lo, 0.01)))
                    bar_color = GREEN if (lo + hi) / 2 <= val <= hi else (YELLOW if val >= lo else RED)
                    inner_ax.bar([0], [norm_val], color=bar_color, alpha=0.8, width=0.8)
                    inner_ax.set_ylim(0, 1)

                parts = label.split('\n')
                inner_ax.text(0, 1.08, parts[0], ha='center', va='bottom',
                              fontsize=8, color='#aaa', transform=inner_ax.transAxes)
                inner_ax.text(0, 0.5, parts[1] if len(parts) > 1 else '', ha='center', va='center',
                              fontsize=10, color='white', fontweight='bold',
                              transform=inner_ax.transAxes)

        return self.save(fig, 'metrics_dashboard.png')

    # ── Generate All ─────────────────────────────────────────────────────────

    def generate_all(self, metrics: dict) -> dict:
        paths = {}
        paths['bankroll'] = self.plot_bankroll()
        paths['round_pnl'] = self.plot_round_pnl()
        paths['action_frequency'] = self.plot_action_frequency()
        paths['auction_distribution'] = self.plot_auction_distribution()
        paths['bid_vs_strength'] = self.plot_bid_vs_strength()
        paths['auction_ev'] = self.plot_auction_ev()
        paths['fold_heatmap'] = self.plot_fold_heatmap()
        paths['opponent_breakdown'] = self.plot_opponent_breakdown()
        paths['profit_by_street'] = self.plot_profit_by_street(
            metrics.get('profit_by_street', {}))
        return paths
