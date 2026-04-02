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

  Step 6 — MCTSAgent(value=geometry) vs MCTSAgent(value=material)
             Does geometry-only evaluation (no value head) add over material?
             Tests whether the bottleneck encodes directional signal independently
             of the value head. Useful for diagnosing the passenger problem.

Usage
-----
    python3 evaluate.py --model models/best.pt --games 100 --step 5
    python3 evaluate.py --model models/best.pt --games 200 --all-steps
    python3 evaluate.py --model models/best.pt --games 100 --step 5 --workers 4
    python3 evaluate.py --model models/best.pt --games 100 --step 6
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Callable, Optional

import chess
import chess.pgn
import io
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from mcts import MCTS
from board import board_to_tensor
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
# Geometry value function
# ---------------------------------------------------------------------------

_GEOMETRY_ANCHORS_WIN = [
    chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"),
    chess.Board("4k3/8/8/2Q5/8/8/8/4K3 w - - 0 1"),
    chess.Board("8/8/8/8/3k4/8/8/3K2Q1 w - - 0 1"),
    chess.Board("8/8/8/8/8/2k5/8/3K2Q1 w - - 0 1"),
]
_GEOMETRY_ANCHORS_LOSS = [
    chess.Board("4K3/8/8/8/8/8/8/4k2q w - - 0 1"),
    chess.Board("4K3/8/8/2q5/8/8/8/4k3 w - - 0 1"),
    chess.Board("8/8/8/8/3K4/8/8/3k2q1 w - - 0 1"),
    chess.Board("8/8/8/8/8/2K5/8/3k2q1 w - - 0 1"),
]


def _geo_vec(model: PetraNet, board: chess.Board) -> np.ndarray:
    dev = next(model.parameters()).device
    t = board_to_tensor(board).unsqueeze(0).float().to(dev)
    return model.geometry(t).cpu().numpy()[0]


def make_geometry_value_fn(model: PetraNet) -> Callable:
    """
    Build a value function that uses the geometry projection onto the
    win/loss axis derived from KQ vs K anchor positions.

    Returns a callable: board -> float in roughly (-1, +1).
    The geometry drives the evaluation — the value head is not used.
    """
    c_win  = np.mean([_geo_vec(model, b) for b in _GEOMETRY_ANCHORS_WIN],  axis=0)
    c_loss = np.mean([_geo_vec(model, b) for b in _GEOMETRY_ANCHORS_LOSS], axis=0)
    axis   = c_win - c_loss
    axis   = axis / (np.linalg.norm(axis) + 1e-8)

    def geometry_value(board: chess.Board) -> float:
        g = _geo_vec(model, board)
        proj = float(np.dot(g, axis))
        if board.turn == chess.BLACK:
            proj = -proj
        return math.tanh(proj)

    return geometry_value


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Agent:
    def select_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def cfg(self) -> dict:
        raise NotImplementedError


class RandomAgent(Agent):
    """Picks a random legal move. The floor."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)
        self._seed = seed

    def select_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))

    @property
    def name(self):
        return "Random"

    @property
    def cfg(self):
        return {"type": "random", "seed": self._seed}


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

    @property
    def cfg(self):
        return {"type": "greedy"}


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
        assert value in ("learned", "material", "zero", "geometry")
        if value == "material":
            value_fn = material_value
        elif value == "zero":
            value_fn = zero_value
        elif value == "geometry":
            value_fn = make_geometry_value_fn(model)
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

    @property
    def cfg(self):
        return {"type": "mcts", "value": self._val, "n_sim": self._n,
                "temp_moves": self._temp_moves}


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def play_game(white: Agent, black: Agent, max_moves: int = 300):
    """
    Play one game. Returns (result, board) where result is "1-0", "0-1",
    or "1/2-1/2" and board contains the full move history.
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
        result = "1/2-1/2"   # move limit reached
    elif outcome.winner == chess.WHITE:
        result = "1-0"
    elif outcome.winner == chess.BLACK:
        result = "0-1"
    else:
        result = "1/2-1/2"
    return result, board


# ---------------------------------------------------------------------------
# Parallel game worker (module-level for picklability)
# ---------------------------------------------------------------------------

def _game_worker(args):
    """
    Play one game in a subprocess. Each worker loads its own model copy.
    Returns "win", "loss", or "draw" from agent_a's perspective.
    """
    game_idx, model_path, baseline_model_path, agent_a_cfg, agent_b_cfg, max_moves, record_pgn = args

    def _load(path):
        if not path:
            return None
        m = PetraNet()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        return m

    model_a = _load(model_path)
    model_b = _load(baseline_model_path) if baseline_model_path else model_a

    def _make(cfg, model):
        t = cfg["type"]
        if t == "random":
            return RandomAgent(seed=cfg.get("seed"))
        if t == "greedy":
            return GreedyAgent(model)
        if t == "mcts":
            return MCTSAgent(model, n_simulations=cfg["n_sim"],
                             value=cfg["value"],
                             temperature_moves=cfg["temp_moves"])
        raise ValueError(f"Unknown agent type: {t}")

    a, b = _make(agent_a_cfg, model_a), _make(agent_b_cfg, model_b)
    white, black = (a, b) if game_idx % 2 == 0 else (b, a)
    result, board = play_game(white, black, max_moves=max_moves)

    if result == "1/2-1/2":
        outcome = "draw"
    else:
        a_is_white = (game_idx % 2 == 0)
        outcome = "win" if (result == "1-0") == a_is_white else "loss"

    board_outcome = board.outcome()
    is_checkmate = (board_outcome is not None and
                    board_outcome.termination == chess.Termination.CHECKMATE)
    white_mated = is_checkmate and board_outcome.winner == chess.WHITE  # white delivered mate

    pgn_str = None
    if record_pgn:
        game = chess.pgn.Game()
        game.headers["White"] = white.name
        game.headers["Black"] = black.name
        game.headers["Result"] = result
        board_outcome = board.outcome()
        if board_outcome is not None:
            game.headers["Termination"] = board_outcome.termination.name
            if board_outcome.termination == chess.Termination.CHECKMATE:
                game.headers["Mated"] = "Black" if board_outcome.winner == chess.WHITE else "White"
        else:
            game.headers["Termination"] = "move_limit"
        game.headers["PlyCount"] = str(len(board.move_stack))
        node = game
        tmp = chess.Board()
        for move in board.move_stack:
            node = node.add_variation(move)
            tmp.push(move)
        buf = io.StringIO()
        print(game, file=buf)
        pgn_str = buf.getvalue()

    return outcome, pgn_str, is_checkmate, white_mated


def run_match(agent_a: Agent, agent_b: Agent,
              n_games: int = 100,
              verbose: bool = True,
              model_path: str = None,
              baseline_model_path: str = None,
              workers: int = 1,
              max_moves: int = 300,
              pgn_out: str = None) -> dict:
    """
    Play n_games between agent_a and agent_b, alternating colours every game.
    Returns a results dict with win/draw/loss counts and ELO estimates.

    workers > 1: games are played in parallel subprocesses, each loading
    their own model copy from model_path.
    """
    if n_games % 2 != 0:
        n_games += 1   # ensure equal colours

    wins = draws = losses = 0
    white_mates = black_mates = 0
    t0 = time.time()
    report_every = max(1, n_games // 10)

    record_pgn = pgn_out is not None
    args_list = [
        (i, model_path, baseline_model_path, agent_a.cfg, agent_b.cfg, max_moves, record_pgn)
        for i in range(n_games)
    ]

    pgn_file = open(pgn_out, "w") if pgn_out else None

    def _record(outcome, pgn_str, is_checkmate, white_mated, i):
        nonlocal wins, draws, losses, white_mates, black_mates
        if outcome == "draw":
            draws += 1
        elif outcome == "win":
            wins += 1
        else:
            losses += 1
        if is_checkmate:
            if white_mated:
                white_mates += 1
            else:
                black_mates += 1
        if pgn_file and pgn_str:
            pgn_file.write(pgn_str + "\n")
            pgn_file.flush()
        if verbose and (i + 1) % report_every == 0:
            total = wins + draws + losses
            wr = (wins + 0.5 * draws) / total
            print(f"  [{i+1:>4}/{n_games}]  "
                  f"W={wins} D={draws} L={losses}  "
                  f"wr={wr:.3f}  "
                  f"(wm={white_mates} bm={black_mates})  "
                  f"({time.time()-t0:.0f}s)")

    try:
        if workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for i, (outcome, pgn_str, is_checkmate, white_mated) in enumerate(
                        pool.imap(_game_worker, args_list)):
                    _record(outcome, pgn_str, is_checkmate, white_mated, i)
        else:
            for i, args in enumerate(args_list):
                outcome, pgn_str, is_checkmate, white_mated = _game_worker(args)
                _record(outcome, pgn_str, is_checkmate, white_mated, i)
    finally:
        if pgn_file:
            pgn_file.close()

    return _summarise(agent_a.name, agent_b.name, wins, draws, losses,
                      white_mates, black_mates)


def _summarise(name_a: str, name_b: str,
               wins: int, draws: int, losses: int,
               white_mates: int = 0, black_mates: int = 0) -> dict:
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
        "white_mates": white_mates, "black_mates": black_mates,
    }

    print(f"\n{'='*55}")
    print(f"  {name_a}")
    print(f"    vs")
    print(f"  {name_b}")
    print(f"{'='*55}")
    print(f"  Games : {total}  (W={wins} D={draws} L={losses})")
    print(f"  Score : {score:.1f}/{total}  ({wr*100:.1f}%)")
    print(f"  ELO Δ : {elo_diff:+.0f}  (95% CI: [{lo*100:.1f}%, {hi*100:.1f}%])")
    print(f"  Mates : white={white_mates} black={black_mates} total={white_mates+black_mates}")
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
    6: ("Geometry value",        "MCTS(geometry) vs MCTS(material)"),
}

def run_ablation(model: Optional[PetraNet], n_games: int = 100,
                 steps: list = None, n_sim: int = 200,
                 temperature_moves: int = 10,
                 model_path: str = None,
                 baseline_model: Optional[PetraNet] = None,
                 baseline_model_path: str = None,
                 workers: int = 1,
                 pgn_out: str = None):
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
            if baseline_model is not None:
                b = MCTSAgent(baseline_model, n_simulations=n_sim, value="learned",
                              temperature_moves=temperature_moves)
            else:
                b = MCTSAgent(model, n_simulations=n_sim, value="material",
                              temperature_moves=temperature_moves)
        elif step == 6:
            a = MCTSAgent(model, n_simulations=n_sim, value="geometry",
                          temperature_moves=temperature_moves)
            b = MCTSAgent(model, n_simulations=n_sim, value="material",
                          temperature_moves=temperature_moves)

        results[step] = run_match(a, b, n_games=n_games,
                                  model_path=model_path,
                                  baseline_model_path=baseline_model_path,
                                  workers=workers,
                                  pgn_out=pgn_out)

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
    if 6 in results:
        geo_result = results[6]
        if geo_result["win_rate"] > 0.52:
            print("\nGEOMETRY SIGNAL — geometry evaluation beats material.")
            print("Bottleneck encodes directional signal independently of the value head.")
        else:
            print("\nGEOMETRY FLAT — geometry evaluation does not beat material.")
            print("Passenger problem likely: geometry is not directionally structured.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default=None, help="Path to model .pt file")
    ap.add_argument("--games",      type=int, default=100)
    ap.add_argument("--step",       type=int, default=None,
                    help="Run a single ablation step (1-6)")
    ap.add_argument("--all-steps",  action="store_true",
                    help="Run all ablation steps in order")
    ap.add_argument("--n-sim",      type=int, default=200,
                    help="MCTS simulations per move")
    ap.add_argument("--temp-moves", type=int, default=10,
                    help="Half-moves at start of each game to use temperature=1 "
                         "(prevents deterministic game repetition; default: 10)")
    ap.add_argument("--workers",        type=int, default=1,
                    help="Parallel worker processes for game playing (default: 1)")
    ap.add_argument("--baseline-model", default=None,
                    help="Path to baseline model .pt for head-to-head (step 5 only). "
                         "If omitted, step 5 uses the material baseline.")
    ap.add_argument("--pgn-out",        default=None,
                    help="Path to write PGN file with all games (optional).")
    args = ap.parse_args()

    model = None
    if args.model:
        model = PetraNet().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device,
                                         weights_only=True))
        model.eval()
        print(f"Loaded model from {args.model}")
    elif args.step != 1:
        print("--model required for steps 2-6")
        sys.exit(1)

    baseline_model = None
    if args.baseline_model:
        baseline_model = PetraNet().to(device)
        baseline_model.load_state_dict(torch.load(args.baseline_model,
                                                   map_location=device,
                                                   weights_only=True))
        baseline_model.eval()
        print(f"Loaded baseline model from {args.baseline_model}")

    steps = list(ABLATION_STEPS.keys()) if args.all_steps else \
            [args.step] if args.step else [5]

    run_ablation(model, n_games=args.games, steps=steps, n_sim=args.n_sim,
                 temperature_moves=args.temp_moves,
                 model_path=args.model,
                 baseline_model=baseline_model,
                 baseline_model_path=args.baseline_model,
                 workers=args.workers,
                 pgn_out=args.pgn_out)


if __name__ == "__main__":
    main()
