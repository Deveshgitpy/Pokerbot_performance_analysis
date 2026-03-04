
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class RoundData:
    round_num: int
    player_names: list
    positions: dict          # {name: 'SB' or 'BB'}
    hole_cards: dict         # {name: [card, card]}
    preflop_actions: list    # [(name, action, amount)]
    flop_cards: list
    flop_actions: list
    auction_bids: dict       # {name: amount}
    auction_winner: Optional[str]
    revealed_card: Optional[str]
    turn_cards: list
    turn_actions: list
    river_cards: list
    river_actions: list
    showdown: dict           # {name: cards}
    payoffs: dict            # {name: amount}
    pot_sizes: dict          # pot at each street
    folded_player: Optional[str] = None


# ─── Core Parser ─────────────────────────────────────────────────────────────

class LogParser:
    """
    Parses IIT Pokerbots .glog or .fz files into structured data.
    Handles both verbose and small_log (compressed) formats.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.raw_lines = []
        self.rounds: list[RoundData] = []
        self.player_names = []
        self.final_bankrolls = {}
        self.bot_name = None          # the bot we're analysing (player 0 in first round)
        self.opponent_name = None

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> list[RoundData]:
        # Auto-detect gzip (.log.gh and similar compressed logs)
        import gzip, struct
        with open(self.filepath, 'rb') as fb:
            magic = fb.read(2)
        if magic == b'\x1f\x8b':
            # gzip compressed — decompress first
            with gzip.open(self.filepath, 'rt', encoding='utf-8', errors='replace') as f:
                self.raw_lines = [l.rstrip('\n') for l in f.readlines()]
        else:
            # plain text — try utf-8, fall back to latin-1
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.raw_lines = [l.rstrip('\n') for l in f.readlines()]
            except UnicodeDecodeError:
                with open(self.filepath, 'r', encoding='latin-1') as f:
                    self.raw_lines = [l.rstrip('\n') for l in f.readlines()]

        self._split_and_parse_rounds()
        return self.rounds

    def to_dataframes(self):
        """Returns (rounds_df, actions_df) for downstream analysis."""
        rounds_rows = []
        actions_rows = []

        for r in self.rounds:
            # aggregate round-level row
            p0, p1 = r.player_names
            rounds_rows.append({
                'round_num': r.round_num,
                'player_0': p0,
                'player_1': p1,
                'pos_p0': r.positions.get(p0, ''),
                'pos_p1': r.positions.get(p1, ''),
                'hole_p0': ' '.join(r.hole_cards.get(p0, [])),
                'hole_p1': ' '.join(r.hole_cards.get(p1, [])),
                'flop': ' '.join(r.flop_cards),
                'turn': ' '.join(r.turn_cards),
                'river': ' '.join(r.river_cards),
                'bid_p0': r.auction_bids.get(p0),
                'bid_p1': r.auction_bids.get(p1),
                'auction_winner': r.auction_winner,
                'revealed_card': r.revealed_card,
                'payoff_p0': r.payoffs.get(p0, 0),
                'payoff_p1': r.payoffs.get(p1, 0),
                'folded': r.folded_player,
                'went_to_showdown': bool(r.showdown),
            })

            # action-level rows
            for street, actions in [
                ('preflop', r.preflop_actions),
                ('flop',    r.flop_actions),
                ('turn',    r.turn_actions),
                ('river',   r.river_actions),
            ]:
                for actor, action_type, amount in actions:
                    actions_rows.append({
                        'round_num': r.round_num,
                        'street': street,
                        'actor': actor,
                        'action': action_type,
                        'amount': amount,
                    })

        rounds_df = pd.DataFrame(rounds_rows)
        actions_df = pd.DataFrame(actions_rows) if actions_rows else pd.DataFrame(
            columns=['round_num', 'street', 'actor', 'action', 'amount'])

        return rounds_df, actions_df

    # ── Internal ──────────────────────────────────────────────────────────────

    def _split_and_parse_rounds(self):
        """Split log into per-round blocks and parse each."""
        round_starts = []
        for i, line in enumerate(self.raw_lines):
            if re.match(r'^Round #\d+', line):
                round_starts.append(i)
            elif line.startswith('Final'):
                self._parse_final(line)

        for idx, start in enumerate(round_starts):
            end = round_starts[idx + 1] if idx + 1 < len(round_starts) else len(self.raw_lines)
            block = self.raw_lines[start:end]
            rd = self._parse_round_block(block)
            if rd:
                self.rounds.append(rd)

        if self.rounds:
            first = self.rounds[0]
            # The SB in round 1 is player 0 (position alternates each round)
            self.player_names = first.player_names
            self.bot_name = first.player_names[0]
            self.opponent_name = first.player_names[1]

    def _parse_round_block(self, lines: list) -> Optional[RoundData]:
        """Parse a single round block."""
        if not lines:
            return None

        # Header: Round #1, NPC48 (0), phoenix_1 (0)
        header_match = re.match(r'^Round #(\d+),\s*(\S+)\s*\([^)]+\),\s*(\S+)\s*\([^)]+\)', lines[0])
        if not header_match:
            return None

        round_num = int(header_match.group(1))
        p0 = header_match.group(2)
        p1 = header_match.group(3)
        player_names = [p0, p1]

        rd = RoundData(
            round_num=round_num,
            player_names=player_names,
            positions={p0: 'SB', p1: 'BB'},   # p0 posts SB in every round block
            hole_cards={},
            preflop_actions=[],
            flop_cards=[],
            flop_actions=[],
            auction_bids={},
            auction_winner=None,
            revealed_card=None,
            turn_cards=[],
            turn_actions=[],
            river_cards=[],
            river_actions=[],
            showdown={},
            payoffs={},
            pot_sizes={},
        )

        current_street = 'preflop'

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # ── Hole cards ──
            m = re.match(r'^(\S+)\s+received\s+\[([^\]]+)\]', line)
            if m:
                rd.hole_cards[m.group(1)] = m.group(2).split()
                continue

            # Small log hole cards: "name: [Ac Kd]"
            m = re.match(r'^(\S+):\s+\[([^\]]+)\]', line)
            if m and m.group(1) in player_names:
                rd.hole_cards[m.group(1)] = m.group(2).split()
                continue

            # ── Street transitions ──
            m = re.match(r'^(Flop|Turn|River)\s+\[([^\]]+)\]', line)
            if m:
                street_name = m.group(1).lower()
                cards = m.group(2).split()
                if street_name == 'flop':
                    rd.flop_cards = cards
                    current_street = 'flop'
                elif street_name == 'turn':
                    rd.turn_cards = cards[-1:]   # only the new card
                    current_street = 'turn'
                elif street_name == 'river':
                    rd.river_cards = cards[-1:]
                    current_street = 'river'
                continue

            # ── Auction bids ──
            m = re.match(r'^(\S+)\s+bids\s+(\d+)', line)
            if m:
                rd.auction_bids[m.group(1)] = int(m.group(2))
                continue

            # Small-log bid: "name A75"
            m = re.match(r'^(\S+)\s+A(\d+)$', line)
            if m and m.group(1) in player_names:
                rd.auction_bids[m.group(1)] = int(m.group(2))
                continue

            # ── Auction result ──
            m = re.match(r'^(\S+)\s+won the auction and was revealed\s+\[([^\]]+)\]', line)
            if m:
                rd.auction_winner = m.group(1)
                rd.revealed_card = m.group(2).strip()
                continue

            # ── Fold ──
            m = re.match(r'^(\S+)\s+folds', line)
            if m:
                actor = m.group(1)
                rd.folded_player = actor
                self._add_action(rd, current_street, actor, 'fold', None)
                continue

            # Small-log fold: "name F"
            m = re.match(r'^(\S+)\s+F$', line)
            if m and m.group(1) in player_names:
                actor = m.group(1)
                rd.folded_player = actor
                self._add_action(rd, current_street, actor, 'fold', None)
                continue

            # ── Call ──
            m = re.match(r'^(\S+)\s+calls', line)
            if m:
                self._add_action(rd, current_street, m.group(1), 'call', None)
                continue
            m = re.match(r'^(\S+)\s+C$', line)
            if m and m.group(1) in player_names:
                self._add_action(rd, current_street, m.group(1), 'call', None)
                continue

            # ── Check ──
            m = re.match(r'^(\S+)\s+checks', line)
            if m:
                self._add_action(rd, current_street, m.group(1), 'check', None)
                continue
            m = re.match(r'^(\S+)\s+K$', line)
            if m and m.group(1) in player_names:
                self._add_action(rd, current_street, m.group(1), 'check', None)
                continue

            # ── Raise/Bet ──
            m = re.match(r'^(\S+)\s+(?:raises to|bets)\s+(\d+)', line)
            if m:
                self._add_action(rd, current_street, m.group(1), 'raise', int(m.group(2)))
                continue
            m = re.match(r'^(\S+)\s+R(\d+)$', line)
            if m and m.group(1) in player_names:
                self._add_action(rd, current_street, m.group(1), 'raise', int(m.group(2)))
                continue

            # ── Showdown ──
            m = re.match(r'^(\S+)\s+shows\s+\[([^\]]+)\]', line)
            if m:
                rd.showdown[m.group(1)] = m.group(2).split()
                continue

            # ── Payoffs ──
            m = re.match(r'^(\S+)\s+awarded\s+([+-]?\d+)', line)
            if m:
                rd.payoffs[m.group(1)] = int(m.group(2))
                continue
            # Small-log payoff: "name: +120"
            m = re.match(r'^(\S+):\s+([+-]?\d+)$', line)
            if m and m.group(1) in player_names:
                rd.payoffs[m.group(1)] = int(m.group(2))
                continue

        # Derive second player's payoff from first if needed
        if len(rd.payoffs) == 1:
            known_name, known_val = list(rd.payoffs.items())[0]
            other = p1 if known_name == p0 else p0
            rd.payoffs[other] = -known_val

        return rd

    def _add_action(self, rd: RoundData, street: str, actor: str, action: str, amount):
        target = {
            'preflop': rd.preflop_actions,
            'flop':    rd.flop_actions,
            'turn':    rd.turn_actions,
            'river':   rd.river_actions,
        }.get(street, rd.preflop_actions)
        target.append((actor, action, amount))

    def _parse_final(self, line: str):
        """Parse final bankroll line: Final, NPC48 (-24464), phoenix_1 (24464)"""
        for m in re.finditer(r'(\S+)\s+\(([+-]?\d+)\)', line):
            self.final_bankrolls[m.group(1)] = int(m.group(2))
