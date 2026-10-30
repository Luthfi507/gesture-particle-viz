import numpy as np


def dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize(v, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def ease(current: float, target: float, rate: float, dt: float) -> float:
    return current + (target - current) * (1.0 - float(np.exp(-rate * dt)))


def rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return rx @ ry
