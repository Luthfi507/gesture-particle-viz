from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from utils import normalize as _normalize_vec

Sampler = Callable[[int, np.random.Generator], np.ndarray]


@dataclass
class Part:
    name: str
    sampler: Sampler
    color: tuple
    weight: float = 1.0
    info: str = ""
    direction: Optional[tuple] = None
    spread: float = 1.0
    stage: float = 0.0


@dataclass
class Model:
    name: str
    category: str
    points: np.ndarray
    colors: np.ndarray
    part_ids: np.ndarray
    parts: list = field(default_factory=list)
    part_centers: np.ndarray = None
    part_dirs: np.ndarray = None
    beat_bpm: Optional[float] = None


def assemble(name, category, parts_defs, n, seed=0, beat_bpm=None):
    rng = np.random.default_rng(seed)
    weights = np.array([p.weight for p in parts_defs], dtype=np.float64)
    counts = np.floor(weights / weights.sum() * n).astype(int)
    counts[0] += n - counts.sum()
    counts = np.maximum(counts, 0)
    if counts.sum() < n:
        counts[0] += n - counts.sum()

    all_pts, all_cols, all_ids = [], [], []
    for i, (part, k) in enumerate(zip(parts_defs, counts)):
        k = int(k)
        if k <= 0:
            continue
        p = part.sampler(k, rng)
        shade = 0.7 + 0.6 * rng.random((k, 1))
        all_pts.append(p)
        all_cols.append(np.asarray(part.color, dtype=np.float32) * shade)
        all_ids.append(np.full(k, i, dtype=np.int32))

    if not all_pts:
        return Model(name, category, np.zeros((0, 3), dtype=np.float32),
                     np.zeros((0, 3), dtype=np.float32),
                     np.zeros(0, dtype=np.int32), parts_defs, None, None, beat_bpm)

    points = np.concatenate(all_pts).astype(np.float32)
    colors = np.clip(np.concatenate(all_cols), 0.0, 1.0).astype(np.float32)
    part_ids = np.concatenate(all_ids)

    lo, hi = points.min(axis=0), points.max(axis=0)
    points -= (lo + hi) / 2
    scale = 1.15 / np.percentile(np.linalg.norm(points, axis=1), 98) if len(points) > 0 else 1.0
    points *= scale

    centers = np.zeros((len(parts_defs), 3), dtype=np.float32)
    dirs = np.zeros((len(parts_defs), 3), dtype=np.float32)
    for i, part in enumerate(parts_defs):
        mask = part_ids == i
        if mask.any():
            centers[i] = points[mask].mean(axis=0)
        raw = np.asarray(part.direction, dtype=np.float32) if part.direction else centers[i]
        nrm = np.linalg.norm(raw)
        if nrm > 1e-12:
            dirs[i] = raw / nrm * part.spread
        else:
            dirs[i] = raw * part.spread

    return Model(name, category, points, colors, part_ids, parts_defs, centers, dirs, beat_bpm)