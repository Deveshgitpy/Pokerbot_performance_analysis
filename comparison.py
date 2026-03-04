

import pandas as pd
import numpy as np
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


DARK_BG = '#1a1a2e'
CARD_BG = '#16213e'
GREEN = '#00b894'
RED = '#e17055'
YELLOW = '#fdcb6e'
BLUE = '#74b9ff'
PURPLE = '#a29bfe'


class ComparisonEngine:
    """
    Compare multiple bot versions or multiple opponents.
    Pass in a dict of {label: metrics_dict} for version comparison,
    or use compare_opponents for opponent-level breakdown.
    """

    def __init__(self, output_dir: str = 'output'):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ── Bot Version Comparison ────────────────────────────────────────────────

    def compare_versions(self, versions: Dict[str, dict]) -> pd.DataFrame:
        """
        versions: {'v6': metrics_dict_v6, 'v8': metrics_dict_v8, ...}
        Returns a DataFrame with key metrics side by side and delta columns.
        """
        rows = {}
        key_metrics = [
            ('VPIP',              lambda m: m['preflop']['VPIP']),
            ('PFR',               lambda m: m['preflop']['PFR']),
            ('3bet%',             lambda m: m['preflop']['3bet_pct']),
            ('Fold-to-Raise%',    lambda m: m['preflop']['fold_to_raise_pct']),
            ('Auction Win%',      lambda m: m['auction']['auction_win_rate']),
            ('Avg Bid',           lambda m: m['auction']['avg_bid']),
            ('Auction EV (win)',  lambda m: m['auction']['avg_profit_when_winning_auction']),
            ('Auction EV (lose)', lambda m: m['auction']['avg_profit_when_losing_auction']),
            ('Overbid Rate',      lambda m: m['auction']['overbid_rate']),
            ('Cbet%',             lambda m: m['flop']['cbet_pct']),
            ('Double Barrel%',    lambda m: m['turn']['double_barrel_pct']),
            ('Triple Barrel%',    lambda m: m['river']['triple_barrel_pct']),
            ('River Fold%',       lambda m: m['river']['river_fold_pct']),
            ('River Agg%',        lambda m: m['river']['river_aggression_pct']),
            ('Win Rate',          lambda m: m['win_rate']),
            ('Avg Payoff/Round',  lambda m: m['avg_payoff_per_round']),
            ('Total Payoff',      lambda m: m['total_payoff']),
        ]

        for metric_name, extractor in key_metrics:
            row = {}
            for version, mdata in versions.items():
                try:
                    row[version] = extractor(mdata)
                except (KeyError, TypeError):
                    row[version] = None
            rows[metric_name] = row

        df = pd.DataFrame(rows).T
        labels = list(versions.keys())

        # Add deltas between consecutive versions
        for i in range(1, len(labels)):
            prev, curr = labels[i-1], labels[i]
            delta_col = f'Δ {prev}→{curr}'
            try:
                df[delta_col] = df[curr] - df[prev]
            except Exception:
                pass

        return df.round(4)

    def plot_version_comparison(self, versions: Dict[str, dict], save=True) -> str:
        df = self.compare_versions(versions)
        labels = list(versions.keys())

        # Only keep non-delta columns for radar + bar
        metric_cols = [c for c in df.columns if not c.startswith('Δ')]
        # Select key visual metrics
        visual_keys = ['VPIP', 'PFR', 'Auction Win%', 'Cbet%', 'Double Barrel%', 'Triple Barrel%',
                       'River Agg%', 'Win Rate']
        visual_df = df.loc[[k for k in visual_keys if k in df.index], metric_cols]

        n = len(visual_df)
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 6), sharey=True)
        if len(labels) == 1:
            axes = [axes]

        palette = [BLUE, GREEN, YELLOW, PURPLE, RED]
        for ax, (label, color) in zip(axes, zip(labels, palette)):
            ax.set_facecolor(CARD_BG)
            vals = visual_df[label].fillna(0).values
            ax.barh(visual_df.index, vals, color=color, alpha=0.8, edgecolor='#333')
            ax.set_title(label, fontsize=12, color='white')
            ax.set_xlabel('Rate / Value')
            ax.axvline(0, color='white', lw=0.5)
            ax.grid(True, axis='x')

        fig.suptitle('Bot Version Comparison', fontsize=14, color='white')
        fig.patch.set_facecolor(DARK_BG)
        plt.tight_layout()

        path = self.out / 'version_comparison.png'
        if save:
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        return str(path)

    # ── Opponent Comparison ───────────────────────────────────────────────────

    def opponent_table(self, versions: Dict[str, dict]) -> pd.DataFrame:
        """
        Build per-opponent EV table from multiple log files.
        versions: {'log1.glog': metrics_from_log1, ...}
        """
        rows = []
        for logname, m in versions.items():
            opp_bd = m.get('opponent_breakdown', {})
            for opp, stats in opp_bd.items():
                rows.append({
                    'log': logname,
                    'opponent': opp,
                    'avg_payoff': stats.get('avg_payoff', 0),
                    'total_payoff': stats.get('total_payoff', 0),
                    'rounds': stats.get('rounds', 0),
                })
        return pd.DataFrame(rows)

    def plot_delta_report(self, versions: Dict[str, dict]) -> str:
        """Visual delta chart — highlight improvements and regressions."""
        df = self.compare_versions(versions)
        delta_cols = [c for c in df.columns if c.startswith('Δ')]
        if not delta_cols:
            return ''

        fig, axes = plt.subplots(1, len(delta_cols), figsize=(8 * len(delta_cols), 7))
        if len(delta_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, delta_cols):
            ax.set_facecolor(CARD_BG)
            sub = df[col].dropna().sort_values()
            colors = [GREEN if v >= 0 else RED for v in sub.values]
            ax.barh(sub.index, sub.values, color=colors, edgecolor='#333', alpha=0.85)
            ax.axvline(0, color='white', lw=1, ls='--')
            ax.set_title(col, fontsize=12)
            ax.grid(True, axis='x')
            for i, (idx, val) in enumerate(sub.items()):
                ax.text(val + 0.001 if val >= 0 else val - 0.001,
                        i, f'{val:+.3f}', va='center',
                        ha='left' if val >= 0 else 'right', fontsize=8, color='white')

        fig.patch.set_facecolor(DARK_BG)
        fig.suptitle('Version Delta Report', fontsize=14, color='white')
        plt.tight_layout()
        path = self.out / 'delta_report.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        return str(path)
