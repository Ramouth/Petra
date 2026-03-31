"""
PetraNet — CNN backbone with per-piece geometry bottleneck.

Architecture
------------
Input:       (B, 14, 8, 8)
ConvBlock:   14 → 64 channels, 3×3
ResBlocks:   64 channels × N_BLOCKS (default 4), each with skip connection

Per-piece bottleneck:
  - CNN produces a (64, 8, 8) feature map
  - For each occupied square, extract the 64-dim feature vector at that square
  - Sum all piece feature vectors → (64,) board geometry vector
  - Project: Linear(64 → 128) + Tanh

The board geometry is the sum of what each piece contributes from where it
stands — analogous to summing force vectors. The CNN learns what each piece
at each square contributes. Nothing is hand-engineered: queen vectors will
be stronger than pawn vectors because the training outcomes demand it, not
because we told the model. Hidden geometry can emerge freely.

Value head:  Linear(128 → 64) + Tanh + Linear(64 → 1) + Tanh
Policy head: Linear(128 → 4096)  [logits over all 64×64 from/to pairs]

Value convention: +1 = current side to move wins.
Policy: logits masked to legal moves before softmax at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from board import board_to_tensor, move_to_index


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return F.relu(r + x)


class PetraNet(nn.Module):
    """
    CNN backbone + per-piece geometry bottleneck + value and policy heads.
    """

    def __init__(self, n_blocks: int = 4, channels: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        self.input_block = ConvBlock(14, channels)
        self.res_blocks  = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        # Per-piece geometry projection
        # Input: sum of CNN feature vectors at occupied squares (channels-dim)
        # Output: bottleneck_dim geometry space
        self.geo_proj = nn.Sequential(
            nn.Linear(channels, bottleneck_dim),
            nn.Tanh(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(bottleneck_dim, 64 * 64)

    def _piece_geometry(self, feat: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """
        Sum CNN feature vectors at all occupied squares, project to geometry space.

        feat:  (B, channels, 8, 8) — CNN output
        board: (B, 14, 8, 8) float — original input tensor

        Planes 0–11 are piece planes (0–5 white, 6–11 black), value 1 where
        a piece stands. Summing across piece planes gives a (B, 1, 8, 8) mask
        that is 1 at every occupied square and 0 elsewhere.

        The sum is unweighted — the CNN learns what each piece at each square
        contributes. No piece-type information is injected here.
        """
        piece_mask = board[:, :12, :, :].sum(dim=1, keepdim=True)  # (B, 1, 8, 8)
        g = (feat * piece_mask).sum(dim=(-2, -1))                   # (B, channels)
        return self.geo_proj(g)                                      # (B, bottleneck_dim)

    def forward(self, board: torch.Tensor):
        """
        board: (B, 14, 8, 8)
        Returns: value (B, 1), policy_logits (B, 4096)
        """
        feat = self.input_block(board)
        feat = self.res_blocks(feat)
        g    = self._piece_geometry(feat, board)
        return self.value_head(g), self.policy_head(g)

    def geometry(self, board: torch.Tensor) -> torch.Tensor:
        """
        Return the 128-dim geometry vector for a batch of board tensors.
        No gradient — use for probing and nearest-neighbour queries.
        """
        with torch.no_grad():
            feat = self.input_block(board)
            feat = self.res_blocks(feat)
            return self._piece_geometry(feat, board)

    @torch.no_grad()
    def policy(self, board: chess.Board, device: torch.device) -> dict:
        """
        Return a probability distribution over legal moves for a single board.
        Returns: {chess.Move: float}
        """
        self.eval()
        t = board_to_tensor(board).unsqueeze(0).float().to(device)
        _, logits = self.forward(t)
        logits = logits.squeeze(0)

        mask = torch.full((4096,), float("-inf"), device=device)
        for move in board.legal_moves:
            mask[move_to_index(move)] = logits[move_to_index(move)]

        probs = torch.softmax(mask, dim=0)
        return {move: probs[move_to_index(move)].item() for move in board.legal_moves}

    @torch.no_grad()
    def value(self, board: chess.Board, device: torch.device) -> float:
        """
        Return the scalar value for a single board.
        +1 = current side to move wins.
        """
        self.eval()
        t = board_to_tensor(board).unsqueeze(0).float().to(device)
        v, _ = self.forward(t)
        return v.item()
