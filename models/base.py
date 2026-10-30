from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from utils import normalize

Sampler = Callable[[int, np.random.Generator], np.ndarray]


@dataclass
class Part:
    name: str
    sampler: Sampler
    color: tuple
    weight: float = 1.0
    info: str = ""
    direction: Optional[tuple] = None   # arah saat meledak, kalau kosong: menjauh dari pusat
    spread: float = 1.0


@dataclass
class Model:
    name: str
    category: str                # wonder, anatomy, biology, engine
    points: np.ndarray           # (n, 3)
    colors: np.ndarray           # (n, 3) RGB 0..1
    part_ids: np.ndarray         # (n,)
    parts: list = field(default_factory=list)
    part_centers: np.ndarray = None
    part_dirs: np.ndarray = None
    beat_bpm: Optional[float] = None


def assemble(name, category, parts, n, seed=0, beat_bpm=None) -> Model:
    rng = np.random.default_rng(seed)
    weights = np.array([p.weight for p in parts], dtype=np.float64)
    counts = np.floor(weights / weights.sum() * n).astype(int)
    counts[0] += n - counts.sum()

    pts, cols, ids = [], [], []
    for i, (part, k) in enumerate(zip(parts, counts)):
        p = part.sampler(int(k), rng)
        shade = 0.7 + 0.6 * rng.random((int(k), 1))
        pts.append(p)
        cols.append(np.asarray(part.color, dtype=np.float32) * shade)
        ids.append(np.full(int(k), i, dtype=np.int32))

    points = np.concatenate(pts).astype(np.float32)
    colors = np.clip(np.concatenate(cols), 0.0, 1.0).astype(np.float32)
    part_ids = np.concatenate(ids)

    # pusatkan dan skalakan supaya semua model muat di layar dengan ukuran sama
    lo, hi = points.min(axis=0), points.max(axis=0)
    points -= (lo + hi) / 2
    points *= 1.15 / np.percentile(np.linalg.norm(points, axis=1), 98)

    centers = np.stack([points[part_ids == i].mean(axis=0) for i in range(len(parts))])
    dirs = np.zeros_like(centers)
    for i, part in enumerate(parts):
        raw = np.asarray(part.direction, dtype=np.float32) if part.direction else centers[i]
        dirs[i] = normalize(raw) * part.spread

    return Model(name, category, points, colors, part_ids, parts, centers, dirs, beat_bpm)
