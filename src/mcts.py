"""
MCTS — AlphaZero-style search using PetraNet for policy and value.

No external dependencies. No Maia. No numpy pipeline.
PetraNet provides both move priors and position evaluation.

Value convention
----------------
+1 = the side to move wins. Backup negates at every level — the parent's
perspective is always opposite to the child's. This means Q(s, a) at any
node is always from the perspective of the side to move at that node.

Draw contempt
-------------
Draws score DRAW_VALUE = -0.1 from the side to move's perspective.
This makes repetition-seeking strictly worse than playing for a win,
breaking the collapse behaviour seen in Phase 0.

Self-play interface
-------------------
search() returns (best_move, visit_distribution).
visit_distribution is the policy target for training: {move: N_i / N_total}.
Temperature controls how best_move is sampled from the distribution.
"""

import math
import numpy as np
import chess
import torch

from board import move_to_index

DRAW_VALUE = -0.1
C_PUCT     = 1.5


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = (
        "board", "move", "move_idx", "parent",
        "children", "move_list",
        "child_priors", "child_W", "child_N",
        "is_expanded", "terminal_value",
    )

    def __init__(self, board: chess.Board, move=None, move_idx=0, parent=None):
        self.board          = board
        self.move           = move
        self.move_idx       = move_idx    # index in parent.move_list — O(1) backup
        self.parent         = parent
        self.children       = {}          # chess.Move → _Node
        self.move_list      = []
        self.child_priors   = None        # (n,) float32 — set on expand
        self.child_W        = None        # (n,) float64 — total value per child
        self.child_N        = None        # (n,) float64 — visit count per child
        self.is_expanded    = False
        self.terminal_value = None        # set once, reused on revisit

    def Q(self, idx: int) -> float:
        n = self.child_N[idx]
        return self.child_W[idx] / n if n > 0 else 0.0

    def U(self, idx: int) -> float:
        return (C_PUCT * self.child_priors[idx]
                * math.sqrt(self.child_N.sum() + 1)
                / (1 + self.child_N[idx]))

    def best_child_idx(self) -> int:
        n = len(self.move_list)
        scores = np.fromiter(
            (self.Q(i) + self.U(i) for i in range(n)),
            dtype=np.float64, count=n,
        )
        return int(np.argmax(scores))

    def select_leaf(self) -> "_Node":
        node = self
        while node.is_expanded and node.terminal_value is None:
            idx  = node.best_child_idx()
            move = node.move_list[idx]
            if move not in node.children:
                b = node.board.copy()
                b.push(move)
                node.children[move] = _Node(b, move=move, move_idx=idx, parent=node)
            node = node.children[move]
        return node

    def expand(self, priors: np.ndarray, move_list: list):
        self.move_list    = move_list
        self.child_priors = priors.copy()
        self.child_W      = np.zeros(len(move_list), dtype=np.float64)
        self.child_N      = np.zeros(len(move_list), dtype=np.float64)
        self.is_expanded  = True


class _DummyRoot:
    """Fake parent for the root — avoids special-casing root in backup."""
    parent    = None
    move_list = [None]

    def __init__(self):
        self.child_N = np.zeros(1, dtype=np.float64)
        self.child_W = np.zeros(1, dtype=np.float64)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """
    Parameters
    ----------
    model      : PetraNet
    device     : torch.device
    value_fn   : optional callable (chess.Board) -> float
                 If provided, used instead of model.value().
                 Enables ablation: swap in material-only or zero value
                 without touching the model or the search logic.
    c_puct     : exploration constant (default 1.5)
    dir_alpha  : Dirichlet concentration for root noise (default 0.3)
    dir_frac   : fraction of Dirichlet noise mixed into root priors (default 0.25)
    """

    def __init__(self, model, device,
                 value_fn=None,
                 c_puct: float = 1.5,
                 dir_alpha: float = 0.3,
                 dir_frac: float = 0.25):
        self.model     = model
        self.device    = device
        self.value_fn  = value_fn or (lambda b: model.value(b, device))
        self.dir_alpha = dir_alpha
        self.dir_frac  = dir_frac

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, board: chess.Board,
               n_simulations: int = 200,
               temperature: float = 1.0,
               add_noise: bool = True):
        """
        Run MCTS from board. Returns (best_move, visit_distribution).

        Parameters
        ----------
        board         : position to search from (not modified)
        n_simulations : number of MCTS simulations
        temperature   : 1.0 = sample proportional to visit counts (self-play opening)
                        0.0 = greedy — pick highest visit count (evaluation / endgame)
        add_noise     : True during self-play, False during evaluation games

        Returns
        -------
        best_move         : chess.Move
        visit_distribution: {chess.Move: float} — policy target for training
        """
        # Mate-in-1 precheck: play forced mates immediately
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move, {move: 1.0}
            board.pop()

        dummy = _DummyRoot()
        root  = _Node(board.copy(), move=None, move_idx=0, parent=dummy)
        self._expand(root, add_noise=add_noise)

        for _ in range(n_simulations):
            leaf = root.select_leaf()

            if leaf.terminal_value is not None:
                self._backup(leaf, leaf.terminal_value)
                continue

            if leaf.board.is_game_over():
                value = self._terminal_value(leaf.board)
                leaf.terminal_value = value
                self._backup(leaf, value)
                continue

            self._expand(leaf, add_noise=False)
            value = self._evaluate(leaf.board)
            self._backup(leaf, value)

        return self._select_move(root, temperature), self._visit_distribution(root)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _expand(self, node: _Node, add_noise: bool):
        if node.is_expanded:
            return
        move_list = list(node.board.legal_moves)
        if not move_list:
            return

        priors_dict = self.model.policy(node.board, self.device)
        priors = np.array(
            [priors_dict.get(m, 0.0) for m in move_list], dtype=np.float32
        )
        s = priors.sum()
        priors = priors / s if s > 1e-8 else np.full(len(move_list), 1.0 / len(move_list))

        if add_noise:
            noise  = np.random.dirichlet(np.full(len(move_list), self.dir_alpha))
            priors = (1 - self.dir_frac) * priors + self.dir_frac * noise

        node.expand(priors, move_list)

    def _evaluate(self, board: chess.Board) -> float:
        """Scalar value from the perspective of the side to move."""
        return self.value_fn(board)

    def _terminal_value(self, board: chess.Board) -> float:
        """Score a finished game from the perspective of the side to move."""
        if board.is_checkmate():
            return -1.0       # side to move has been checkmated
        return DRAW_VALUE     # stalemate, repetition, 50-move, insufficient material

    def _backup(self, node: _Node, value: float):
        """
        Propagate value up the tree.

        value is from the perspective of node.board.turn (the side to move
        at the evaluated leaf). At each step up, we negate — the parent's
        perspective is always opposite to the child's.
        """
        while node.parent is not None:
            value = -value
            node.parent.child_W[node.move_idx] += value
            node.parent.child_N[node.move_idx] += 1
            node = node.parent

    def _visit_distribution(self, root: _Node) -> dict:
        """Normalised visit counts — policy training target."""
        total = root.child_N.sum()
        if total == 0:
            return {}
        return {m: root.child_N[i] / total for i, m in enumerate(root.move_list)}

    def _select_move(self, root: _Node, temperature: float) -> chess.Move:
        if temperature == 0 or not root.move_list:
            return root.move_list[int(np.argmax(root.child_N))]
        counts = root.child_N ** (1.0 / temperature)
        probs  = counts / counts.sum()
        return root.move_list[np.random.choice(len(root.move_list), p=probs)]
