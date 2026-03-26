"""
Evaluation: game runner + ablation agents + ELO estimation.

Ablation ladder (run in order after supervised pretraining):

  Step 1 — RandomAgent vs RandomAgent
             Floor. Confirms the runner works.

  Step 2 — GreedyAgent vs RandomAgent
             Does the trained policy beat random? Should be ~95%+.
             If not, supervised pretraining failed.

  Step 3 — MCTSAgent(value=zero) vs GreedyAgent
             Does search with uniform value help over greedy policy?
             Usually hurts slightly — establishes the search baseline.

  Step 4 — MCTSAgent(value=material) vs GreedyAgent
             Does hardcoded material value add over greedy?
             Expected: yes. Sets the material baseline.

  Step 5 — MCTSAgent(value=learned) vs MCTSAgent(value=material)
             Does the trained value head add over material?
             This is the critical gate. If no: stop. If yes: proceed to self-play.

Usage
-----
    python3 evaluate.py --model models/best.pt --games 100 --step 5
    python3 evaluate.py --model models/best.pt --games 200 --all-steps
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Callable, Optional

import chess
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import PetraNet
from mcts import MCTS
from config import device


# ---------------------------------------------------------------------------
# Material value function (no neural network)
# ---------------------------------------------------------------------------

_PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

def material_value(board: chess.Board) -> float:
    """
    Hardcoded material balance from the perspective of the side to move.
    Returns tanh(balance / 10) so values stay in (-1, 1).
    """
    balance = 0
    for pt, val in _PIECE_VALUE.items():
        balance += val * len(board.pieces(pt, chess.WHITE))
        balance -= val * len(board.pieces(pt, chess.BLACK))
    if board.turn == chess.BLACK:
        balance = -balance
    return math.tanh(balance / 10.0)


def zero_value(board: chess.Board) -> float:
    """Always returns 0 — search guided by policy only."""
    return 0.0


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Agent:
    def select_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class RandomAgent(Agent):
    """Picks a random legal move. The floor."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))

    @property
    def name(self):
        return "Random"


class GreedyAgent(Agent):
    """Top-1 from the policy head. No search."""

    def __init__(self, model: PetraNet):
        self._model = model

    def select_move(self, board: chess.Board) -> chess.Move:
        probs = self._model.policy(board, device)
        return max(probs, key=probs.get)

    @property
    def name(self):
        return "Greedy(policy)"


class MCTSAgent(Agent):
    """
    MCTS with configurable value function.

    value: "learned"  — PetraNet value head (default)
           "material" — hardcoded piece counts
           "zero"     — uniform value (search guided by policy only)

    temperature_moves: number of half-moves at the start of each game where
        temperature=1 is used (sample from visit distribution rather than argmax).
        This creates varied openings so repeated games diverge and produce
        independent results. Set to 0 for deterministic play.
    """

    def __init__(self, model: PetraNet, n_simulations: int = 200,
                 value: str = "learned", temperature_moves: int = 10):
        assert value in ("learned", "material", "zero")
        if value == "material":
            value_fn = material_value
        elif value == "zero":
            value_fn = zero_value
        else:
            value_fn = None   # MCTS defaults to model.value()

        self._mcts      = MCTS(model, device, value_fn=value_fn)
        self._n         = n_simulations
        self._val       = value
        self._temp_moves = temperature_moves

    def select_move(self, board: chess.Board) -> chess.Move:
        # Use temperature=1 for opening moves so repeated games diverge.
        # After _temp_moves half-moves, switch to greedy (temperature=0).
        temp = 1.0 if len(board.move_stack) < self._temp_moves else 0.0
        move, _ = self._mcts.search(
            board, n_simulations=self._n, temperature=temp, add_noise=False
        )
        return move

    @property
    def name(self):
        return f"MCTS(n={self._n}, value={self._val})"


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def play_game(white: Agent, black: Agent, max_moves: int = 300) -> str:
    """
    Play one game. Returns "1-0", "0-1", or "1/2-1/2".
    Enforces a move limit to prevent infinite games.
    """
    board = chess.Board()
    agents = {chess.WHITE: white, chess.BLACK: black}

    for _ in range(max_moves):
        if board.is_game_over():
            break
        move = agents[board.turn].select_move(board)
        board.push(move)

    outcome = board.outcome()
    if outcome is None:
        return "1/2-1/2"   # move limit reached
    if outcome.winner == chess.WHITE:
        return "1-0"
    if outcome.winner == chess.BLACK:
        return "0-1"
    return "1/2-1/2"


def run_match(agent_a: Agent, agent_b: Agent,
              n_games: int = 100,
              verbose: bool = True) -> dict:
    """
    Play n_games between agent_a and agent_b, alternating colours every game.
    Returns a results dict with win/draw/loss counts and ELO estimates.
    """
    if n_games % 2 != 0:
        n_games += 1   # ensure equal colours

    wins = draws = losses = 0
    t0 = time.time()

    for i in range(n_games):
        # Alternate: even games A=White, odd games A=Black
        if i % 2 == 0:
            white, black = agent_a, agent_b
        else:
            white, black = agent_b, agent_a

        result = play_game(white, black)

        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0" and white is agent_a) or \
             (result == "0-1" and black is agent_a):
            wins += 1
        else:
            losses += 1

        if verbose and (i + 1) % max(1, n_games // 10) == 0:
            total = wins + draws + losses
            wr = (wins + 0.5 * draws) / total
            elapsed = time.time() - t0
            print(f"  [{i+1:>4}/{n_games}]  "
                  f"W={wins} D={draws} L={losses}  "
                  f"wr={wr:.3f}  ({elapsed:.0f}s)")

    return _summarise(agent_a.name, agent_b.name, wins, draws, losses)


def _summarise(name_a: str, name_b: str,
               wins: int, draws: int, losses: int) -> dict:
    total = wins + draws + losses
    score = wins + 0.5 * draws
    wr    = score / total

    # ELO difference: D = -400 * log10(1/wr - 1)
    # Clamp to avoid log(0)
    wr_clamped = max(0.001, min(0.999, wr))
    elo_diff   = -400 * math.log10(1 / wr_clamped - 1)

    # Wilson 95% confidence interval on win rate
    z = 1.96
    lo = (wr + z*z/(2*total) - z*math.sqrt(wr*(1-wr)/total + z*z/(4*total*total))) \
         / (1 + z*z/total)
    hi = (wr + z*z/(2*total) + z*math.sqrt(wr*(1-wr)/total + z*z/(4*total*total))) \
         / (1 + z*z/total)

    result = {
        "agent_a": name_a, "agent_b": name_b,
        "wins": wins, "draws": draws, "losses": losses, "total": total,
        "win_rate": wr,
        "elo_diff": elo_diff,
        "ci_lo": lo, "ci_hi": hi,
    }

    print(f"\n{'='*55}")
    print(f"  {name_a}")
    print(f"    vs")
    print(f"  {name_b}")
    print(f"{'='*55}")
    print(f"  Games : {total}  (W={wins} D={draws} L={losses})")
    print(f"  Score : {score:.1f}/{total}  ({wr*100:.1f}%)")
    print(f"  ELO Δ : {elo_diff:+.0f}  (95% CI: [{lo*100:.1f}%, {hi*100:.1f}%])")
    if abs(elo_diff) < 50 and total < 200:
        print(f"  NOTE  : ELO diff < 50 — run more games for a reliable estimate")
    print(f"{'='*55}\n")

    return result


# ---------------------------------------------------------------------------
# Ablation ladder
# ---------------------------------------------------------------------------

ABLATION_STEPS = {
    1: ("Random floor",          "RandomAgent vs RandomAgent"),
    2: ("Policy check",          "Greedy vs Random"),
    3: ("Search + zero value",   "MCTS(zero) vs Greedy"),
    4: ("Material value",        "MCTS(material) vs Greedy"),
    5: ("Learned value (gate)",  "MCTS(learned) vs MCTS(material)"),
}

def run_ablation(model: Optional[PetraNet], n_games: int = 100,
                 steps: list = None, n_sim: int = 200,
                 temperature_moves: int = 10):
    """
    Run the full ablation ladder or a subset of steps.
    model may be None for step 1 only.
    """
    steps = steps or list(ABLATION_STEPS.keys())
    results = {}

    for step in steps:
        desc, matchup = ABLATION_STEPS[step]
        print(f"\n--- Step {step}: {desc} ({matchup}) ---")

        if step == 1:
            a = RandomAgent(seed=0)
            b = RandomAgent(seed=1)
        elif step == 2:
            a = GreedyAgent(model)
            b = RandomAgent(seed=0)
        elif step == 3:
            a = MCTSAgent(model, n_simulations=n_sim, value="zero",
                          temperature_moves=temperature_moves)
            b = GreedyAgent(model)
        elif step == 4:
            a = MCTSAgent(model, n_simulations=n_sim, value="material",
                          temperature_moves=temperature_moves)
            b = GreedyAgent(model)
        elif step == 5:
            a = MCTSAgent(model, n_simulations=n_sim, value="learned",
                          temperature_moves=temperature_moves)
            b = MCTSAgent(model, n_simulations=n_sim, value="material",
                          temperature_moves=temperature_moves)

        results[step] = run_match(a, b, n_games=n_games)

    _print_ablation_summary(results)
    return results


def _print_ablation_summary(results: dict):
    print("\n" + "="*55)
    print("ABLATION SUMMARY")
    print("="*55)
    for step, r in sorted(results.items()):
        desc = ABLATION_STEPS[step][0]
        verdict = "PASS" if r["elo_diff"] > 0 else "FAIL"
        print(f"  Step {step} [{verdict}]  {desc:25s}  "
              f"wr={r['win_rate']*100:.1f}%  ELO Δ={r['elo_diff']:+.0f}")
    print("="*55)

    # Gate check: step 5 is the critical one
    if 5 in results:
        gate = results[5]
        if gate["win_rate"] > 0.55:
            print("\nGATE PASSED — learned value beats material. Proceed to self-play.")
        else:
            print("\nGATE FAILED — learned value does not beat material.")
            print("Do not proceed to self-play. Review training data and model.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default=None, help="Path to model .pt file")
    ap.add_argument("--games",      type=int, default=100)
    ap.add_argument("--step",       type=int, default=None,
                    help="Run a single ablation step (1-5)")
    ap.add_argument("--all-steps",  action="store_true",
                    help="Run all ablation steps in order")
    ap.add_argument("--n-sim",      type=int, default=200,
                    help="MCTS simulations per move")
    ap.add_argument("--temp-moves", type=int, default=10,
                    help="Half-moves at start of each game to use temperature=1 "
                         "(prevents deterministic game repetition; default: 10)")
    args = ap.parse_args()

    model = None
    if args.model:
        model = PetraNet().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device,
                                         weights_only=True))
        model.eval()
        print(f"Loaded model from {args.model}")
    elif args.step != 1:
        print("--model required for steps 2-5")
        sys.exit(1)

    steps = list(ABLATION_STEPS.keys()) if args.all_steps else \
            [args.step] if args.step else [5]

    run_ablation(model, n_games=args.games, steps=steps, n_sim=args.n_sim,
                 temperature_moves=args.temp_moves)


if __name__ == "__main__":
    main()
