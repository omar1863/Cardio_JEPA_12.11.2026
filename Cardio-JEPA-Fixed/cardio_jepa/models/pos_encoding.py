"""Sin-cos positional embeddings.

Vendored from Kim's ECG-JEPA (https://github.com/sehunfromdaegu/ECG_JEPA,
`pos_encoding.py`), which in turn follows the MAE / ST-MEM implementations.
Reused verbatim so that our reproduction of the masked-infilling baseline is
faithful to the published model.
"""

from __future__ import annotations

import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """embed_dim: output dimension per position; pos: (M,) -> (M, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size_h: int, grid_size_w: int,
                            cls_token: bool = False) -> np.ndarray:
    """Return (grid_h*grid_w, embed_dim) laid out lead-major, time-minor."""
    grid_h = np.arange(grid_size_h, dtype=float)
    grid_w = np.arange(grid_size_w, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """1-D sin-cos embedding, used by the Level-2 (rhythm) transformer."""
    return get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(length, dtype=float))
