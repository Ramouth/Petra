"""
PetraNet — CNN backbone with explicit geometry bottleneck.

Architecture
------------
Input:       (B, 14, 8, 8)
ConvBlock:   14 → 64 channels, 3×3
ResBlocks:   64 channels × N_BLOCKS (default 4), each with skip connection
Flatten:     64 × 8 × 8 = 4096
Bottleneck:  Linear(4096 → 128) + ReLU   ← geometry space

Value head:  Linear(128 → 64) + ReLU + Linear(64 → 1) + Tanh
Policy head: Linear(128 → 4096)  [logits over all 64×64 from/to pairs]

The bottleneck is the only path from board to decisions. Everything Petra
knows about a position must be compressed through 128 dimensions. The
geometry space is not engineered — it emerges from training.

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
    CNN backbone + 128-dim geometry bottleneck + value and policy heads.
    """

    def __init__(self, n_blocks: int = 4, channels: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        self.input_block = ConvBlock(14, channels)
        self.res_blocks  = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, bottleneck_dim),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(bottleneck_dim, 64 * 64)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 14, 8, 8)
        Returns: value (B, 1), policy_logits (B, 4096)
        """
        x = self.input_block(x)
        x = self.res_blocks(x)
        g = self.bottleneck(x)
        return self.value_head(g), self.policy_head(g)

    def geometry(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 128-dim geometry vector for a batch of board tensors.
        No gradient — use for probing and nearest-neighbour queries.
        """
        with torch.no_grad():
            x = self.input_block(x)
            x = self.res_blocks(x)
            return self.bottleneck(x)

    @torch.no_grad()
    def policy(self, board: chess.Board, device: torch.device) -> dict:
        """
        Return a probability distribution over legal moves for a single board.
        Returns: {chess.Move: float}
        """
        self.eval()
        t = board_to_tensor(board).unsqueeze(0).to(device)
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
        t = board_to_tensor(board).unsqueeze(0).to(device)
        v, _ = self.forward(t)
        return v.item()
