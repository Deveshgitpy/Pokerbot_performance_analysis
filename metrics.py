
import pandas as pd
import numpy as np
from dataclasses import dataclass


CARD_RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                 '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def hand_strength_bucket(hole_str: str) -> str:
    """
    Crude preflop hand-strength bucket: Premium / Strong / Medium / Weak / Unknown.
    """
    if not hole_str or pd.isna(hole_str):
        return 'Unknown'
    cards = hole_str.strip().split()
    if len(cards) != 2:
        return 'Unknown'
    try:
        r1 = CARD_RANK_MAP[cards[0][0]]
        r2 = CARD_RANK_MAP[cards[1][0]]
        suited = cards[0][1] == cards[1][1]
        hi, lo = max(r1, r2), min(r1, r2)
        # Premium: AA KK QQ AKs
        if (hi == lo and hi >= 12) or (hi == 14 and lo == 13 and suited):
            return 'Premium'
        # Strong: JJ TJ+ AK AQ AJs KQs
        if (hi == lo and hi >= 10) or (hi == 14 and lo >= 11) or (hi == 13 and lo == 12):
            return 'Strong'
        # Medium: 66-99 suited connectors broadway
        if (hi == lo and hi >= 6) or (suited and hi - lo <= 2 and hi >= 8):
            return 'Medium'
        return 'Weak'
    except (KeyError, IndexError):
        return 'Unknown'


class MetricsEngine:
    """
    Computes all strategic metrics from rounds_df and actions_df.
    Target bot name is set at init.
    """

    def __init__(self, rounds_df: pd.DataFrame, actions_df: pd.DataFrame, bot_name: str):
        self.rounds_df = rounds_df.copy()
        self.actions_df = actions_df.copy()
        self.bot_name = bot_name
        self._enrich()

    # ── Enrichment ────────────────────────────────────────────────────────────

    def _enrich(self):
        df = self.rounds_df
        # Identify which column is the bot
        df['bot_is_p0'] = df['player_0'] == self.bot_name

        df['bot_payoff'] = np.where(df['bot_is_p0'], df['payoff_p0'], df['payoff_p1'])
        df['opp_payoff'] = np.where(df['bot_is_p0'], df['payoff_p1'], df['payoff_p0'])
        df['bot_position'] = np.where(df['bot_is_p0'], df['pos_p0'], df['pos_p1'])
        df['bot_hole'] = np.where(df['bot_is_p0'], df['hole_p0'], df['hole_p1'])
        df['bot_bid'] = np.where(df['bot_is_p0'], df['bid_p0'], df['bid_p1'])
        df['opp_bid'] = np.where(df['bot_is_p0'], df['bid_p1'], df['bid_p0'])
        df['bot_won_auction'] = df['auction_winner'] == self.bot_name
        df['hand_bucket'] = df['bot_hole'].apply(hand_strength_bucket)
        df['cumulative_pnl'] = df['bot_payoff'].cumsum()
        df['opponent'] = np.where(df['bot_is_p0'], df['player_1'], df['player_0'])

        # Bot-specific action subsets
        self.bot_actions = self.actions_df[self.actions_df['actor'] == self.bot_name].copy()

    # ── Preflop ───────────────────────────────────────────────────────────────

    def preflop_metrics(self) -> dict:
        pf = self.bot_actions[self.bot_actions['street'] == 'preflop']
        total_rounds = len(self.rounds_df)

        # VPIP – Voluntarily Put money In Pot (call or raise, not forced blind)
        vpip_rounds = set(pf[pf['action'].isin(['call', 'raise'])]['round_num'])
        vpip = len(vpip_rounds) / total_rounds if total_rounds else 0

        # PFR – Pre-Flop Raise
        pfr_rounds = set(pf[pf['action'] == 'raise']['round_num'])
        pfr = len(pfr_rounds) / total_rounds if total_rounds else 0

        # 3-bet: bot raises AFTER an opponent raise preflop
        threebets = 0
        for rnd in self.rounds_df['round_num']:
            rnd_pf = self.actions_df[(self.actions_df['round_num'] == rnd) &
                                      (self.actions_df['street'] == 'preflop')]
            rnd_list = rnd_pf.to_dict('records')
            saw_opp_raise = False
            for act in rnd_list:
                if act['actor'] != self.bot_name and act['action'] == 'raise':
                    saw_opp_raise = True
                if act['actor'] == self.bot_name and act['action'] == 'raise' and saw_opp_raise:
                    threebets += 1
                    break
        threebet_pct = threebets / total_rounds if total_rounds else 0

        # Fold to raise
        fold_to_raise = 0
        for rnd in self.rounds_df['round_num']:
            rnd_pf = self.actions_df[(self.actions_df['round_num'] == rnd) &
                                      (self.actions_df['street'] == 'preflop')]
            rnd_list = rnd_pf.to_dict('records')
            saw_opp_raise = False
            for act in rnd_list:
                if act['actor'] != self.bot_name and act['action'] == 'raise':
                    saw_opp_raise = True
                if act['actor'] == self.bot_name and act['action'] == 'fold' and saw_opp_raise:
                    fold_to_raise += 1
                    break
        fold_to_raise_pct = fold_to_raise / max(threebets + fold_to_raise, 1)

        avg_open = pf[pf['action'] == 'raise']['amount'].mean()
        profit_by_pf_action = self._profit_by_pf_action()

        return {
            'VPIP': round(vpip, 4),
            'PFR': round(pfr, 4),
            '3bet_pct': round(threebet_pct, 4),
            'fold_to_raise_pct': round(fold_to_raise_pct, 4),
            'avg_open_size': round(avg_open, 2) if not pd.isna(avg_open) else 0,
            'profit_by_pf_action': profit_by_pf_action,
        }

    def _profit_by_pf_action(self) -> dict:
        """Average payoff grouped by bot's first preflop action."""
        first_actions = (self.bot_actions[self.bot_actions['street'] == 'preflop']
                         .groupby('round_num').first().reset_index()[['round_num', 'action']])
        merged = first_actions.merge(
            self.rounds_df[['round_num', 'bot_payoff']], on='round_num')
        return merged.groupby('action')['bot_payoff'].mean().round(2).to_dict()

    # ── Auction ───────────────────────────────────────────────────────────────

    def auction_metrics(self) -> dict:
        df = self.rounds_df
        bids = df['bot_bid'].dropna()

        won_mask = df['bot_won_auction']
        lost_mask = ~won_mask & df['auction_winner'].notna()

        avg_bid = bids.mean()
        median_bid = bids.median()
        max_bid = bids.max()
        bid_var = bids.var()
        win_rate = won_mask.mean()

        profit_win = df[won_mask]['bot_payoff'].mean()
        profit_loss = df[lost_mask]['bot_payoff'].mean()

        # Bid-to-stack ratio (stack ≈ 5000 minus already committed blinds, approx)
        bid_stack_ratio = (bids / 5000).mean()

        # Overbid detection: bot bid > opp bid by >= 100 (paying too much)
        overbid_rounds = df[(df['bot_bid'].notna()) & (df['opp_bid'].notna()) &
                             (df['bot_bid'] - df['opp_bid'] >= 100)]
        overbid_rate = len(overbid_rounds) / max(len(df[df['bot_bid'].notna()]), 1)

        # Close loss: lost by <= 5 chips
        close_loss = df[(df['opp_bid'].notna()) & (df['bot_bid'].notna()) &
                         (df['opp_bid'] > df['bot_bid']) &
                         (df['opp_bid'] - df['bot_bid'] <= 5)]
        close_loss_rate = len(close_loss) / max(len(df), 1)

        bid_by_strength = (df.groupby('hand_bucket')['bot_bid']
                           .mean().round(2).to_dict())

        return {
            'avg_bid': round(avg_bid, 2) if not pd.isna(avg_bid) else 0,
            'median_bid': round(median_bid, 2) if not pd.isna(median_bid) else 0,
            'max_bid': round(max_bid, 2) if not pd.isna(max_bid) else 0,
            'bid_variance': round(bid_var, 2) if not pd.isna(bid_var) else 0,
            'auction_win_rate': round(win_rate, 4),
            'avg_profit_when_winning_auction': round(profit_win, 2) if not pd.isna(profit_win) else 0,
            'avg_profit_when_losing_auction': round(profit_loss, 2) if not pd.isna(profit_loss) else 0,
            'bid_to_stack_ratio': round(bid_stack_ratio, 4) if not pd.isna(bid_stack_ratio) else 0,
            'overbid_rate': round(overbid_rate, 4),
            'close_loss_rate': round(close_loss_rate, 4),
            'avg_bid_by_hand_strength': bid_by_strength,
        }

    # ── Post-flop helpers ─────────────────────────────────────────────────────

    def _cbet_metrics(self) -> dict:
        """C-bet: bot raised preflop and bets flop first."""
        pf_raisers = set(self.bot_actions[(self.bot_actions['street'] == 'preflop') &
                                           (self.bot_actions['action'] == 'raise')]['round_num'])
        flop_bets_bot = set(self.bot_actions[(self.bot_actions['street'] == 'flop') &
                                              (self.bot_actions['action'] == 'raise')]['round_num'])
        # opponent folded to cbet
        opp_folds_flop = set(
            self.actions_df[(self.actions_df['street'] == 'flop') &
                             (self.actions_df['actor'] != self.bot_name) &
                             (self.actions_df['action'] == 'fold')]['round_num'])

        cbet_opps = pf_raisers
        cbets = pf_raisers & flop_bets_bot
        cbet_pct = len(cbets) / max(len(cbet_opps), 1)
        fold_to_cbet = len(cbets & opp_folds_flop) / max(len(cbets), 1)

        return {'cbet_pct': round(cbet_pct, 4),
                'fold_to_cbet_pct': round(fold_to_cbet, 4)}

    def _barrel_metrics(self) -> dict:
        """Double barrel = bot bet flop AND turn. Triple = + river."""
        flop_bets = set(self.bot_actions[(self.bot_actions['street'] == 'flop') &
                                          (self.bot_actions['action'] == 'raise')]['round_num'])
        turn_bets = set(self.bot_actions[(self.bot_actions['street'] == 'turn') &
                                          (self.bot_actions['action'] == 'raise')]['round_num'])
        river_bets = set(self.bot_actions[(self.bot_actions['street'] == 'river') &
                                           (self.bot_actions['action'] == 'raise')]['round_num'])
        double = flop_bets & turn_bets
        triple = double & river_bets

        double_pct = len(double) / max(len(flop_bets), 1)
        triple_pct = len(triple) / max(len(double), 1)

        profit_1b = self.rounds_df[self.rounds_df['round_num'].isin(flop_bets - turn_bets)]['bot_payoff'].mean()
        profit_2b = self.rounds_df[self.rounds_df['round_num'].isin(double - triple)]['bot_payoff'].mean()
        profit_3b = self.rounds_df[self.rounds_df['round_num'].isin(triple)]['bot_payoff'].mean()

        return {
            'double_barrel_pct': round(double_pct, 4),
            'triple_barrel_pct': round(triple_pct, 4),
            'avg_profit_1barrel': round(profit_1b, 2) if not pd.isna(profit_1b) else 0,
            'avg_profit_2barrel': round(profit_2b, 2) if not pd.isna(profit_2b) else 0,
            'avg_profit_3barrel': round(profit_3b, 2) if not pd.isna(profit_3b) else 0,
        }

    # ── Flop ──────────────────────────────────────────────────────────────────

    def flop_metrics(self) -> dict:
        return self._cbet_metrics()

    # ── Turn ──────────────────────────────────────────────────────────────────

    def turn_metrics(self) -> dict:
        bm = self._barrel_metrics()
        turn_folds = set(self.bot_actions[(self.bot_actions['street'] == 'turn') &
                                           (self.bot_actions['action'] == 'fold')]['round_num'])
        turn_total = len(self.rounds_df[self.rounds_df['flop'].str.len() > 0])
        return {
            'double_barrel_pct': bm['double_barrel_pct'],
            'turn_fold_pct': round(len(turn_folds) / max(turn_total, 1), 4),
            'avg_profit_2barrel': bm['avg_profit_2barrel'],
        }

    # ── River ─────────────────────────────────────────────────────────────────

    def river_metrics(self) -> dict:
        river_acts = self.bot_actions[self.bot_actions['street'] == 'river']
        river_rounds = self.rounds_df[self.rounds_df['river'].str.len() > 0]

        aggression = river_acts[river_acts['action'] == 'raise']['round_num'].nunique()
        river_agg_pct = aggression / max(len(river_rounds), 1)

        river_folds = river_acts[river_acts['action'] == 'fold']['round_num'].nunique()
        river_fold_pct = river_folds / max(len(river_rounds), 1)

        river_calls = river_acts[river_acts['action'] == 'call']['round_num'].nunique()
        # Call success: bot called river and won
        call_rounds = set(river_acts[river_acts['action'] == 'call']['round_num'])
        won_call = self.rounds_df[(self.rounds_df['round_num'].isin(call_rounds)) &
                                   (self.rounds_df['bot_payoff'] > 0)]['round_num'].nunique()
        call_success = won_call / max(river_calls, 1)

        bm = self._barrel_metrics()

        return {
            'river_aggression_pct': round(river_agg_pct, 4),
            'river_fold_pct': round(river_fold_pct, 4),
            'river_call_success_pct': round(call_success, 4),
            'triple_barrel_pct': bm['triple_barrel_pct'],
            'avg_profit_3barrel': bm['avg_profit_3barrel'],
        }

    # ── Profit by street ──────────────────────────────────────────────────────

    def profit_by_street(self) -> dict:
        """Estimate which street the pot was won/lost by examining fold patterns."""
        def street_where_fold(rnd_num):
            for street in ['preflop', 'flop', 'turn', 'river']:
                acts = self.actions_df[(self.actions_df['round_num'] == rnd_num) &
                                        (self.actions_df['street'] == street) &
                                        (self.actions_df['action'] == 'fold')]
                if not acts.empty:
                    return street
            return 'showdown'

        self.rounds_df['decision_street'] = self.rounds_df['round_num'].apply(street_where_fold)
        return (self.rounds_df.groupby('decision_street')['bot_payoff']
                .agg(['mean', 'sum', 'count'])
                .round(2)
                .rename(columns={'mean': 'avg_payoff', 'sum': 'total_payoff', 'count': 'rounds'})
                .to_dict('index'))

    # ── Opponent breakdown ────────────────────────────────────────────────────

    def opponent_breakdown(self) -> pd.DataFrame:
        return (self.rounds_df.groupby('opponent')['bot_payoff']
                .agg(['mean', 'sum', 'count'])
                .rename(columns={'mean': 'avg_payoff', 'sum': 'total_payoff', 'count': 'rounds'})
                .round(2))

    # ── Summary ───────────────────────────────────────────────────────────────

    def all_metrics(self) -> dict:
        return {
            'preflop': self.preflop_metrics(),
            'auction': self.auction_metrics(),
            'flop': self.flop_metrics(),
            'turn': self.turn_metrics(),
            'river': self.river_metrics(),
            'profit_by_street': self.profit_by_street(),
            'opponent_breakdown': self.opponent_breakdown().to_dict('index'),
            'total_payoff': int(self.rounds_df['bot_payoff'].sum()),
            'win_rate': round((self.rounds_df['bot_payoff'] > 0).mean(), 4),
            'avg_payoff_per_round': round(self.rounds_df['bot_payoff'].mean(), 2),
        }
