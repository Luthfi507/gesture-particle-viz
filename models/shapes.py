import numpy as np


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def sphere(k, rng, r=1.0, center=(0, 0, 0)):
    v = rng.normal(size=(k, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return _f32(v * r + np.asarray(center))


def box(k, rng, size, center=(0, 0, 0)):
    sx, sy, sz = size
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], dtype=np.float64)
    face = rng.choice(6, size=k, p=areas / areas.sum())
    p = rng.uniform(-0.5, 0.5, size=(k, 3)) * np.array([sx, sy, sz])
    axis = face // 2
    sign = np.where(face % 2 == 0, -0.5, 0.5)
    for a, s in enumerate(size):
        m = axis == a
        p[m, a] = sign[m] * s
    return _f32(p + np.asarray(center))


def cylinder(k, rng, radius, height, center=(0, 0, 0), axis=1, caps=True):
    side_area = 2 * np.pi * radius * height
    cap_area = 2 * np.pi * radius ** 2 if caps else 0.0
    on_cap = rng.random(k) < cap_area / (side_area + cap_area)
    ang = rng.uniform(0, 2 * np.pi, k)
    rr = np.where(on_cap, radius * np.sqrt(rng.random(k)), radius)
    along = np.where(
        on_cap,
        np.where(rng.random(k) < 0.5, -0.5, 0.5) * height,
        rng.uniform(-0.5, 0.5, k) * height,
    )
    p = np.stack([rr * np.cos(ang), along, rr * np.sin(ang)], axis=1)
    if axis == 0:
        p = p[:, [1, 0, 2]]
    elif axis == 2:
        p = p[:, [0, 2, 1]]
    return _f32(p + np.asarray(center))


def tube(k, rng, path, radius):
    t = rng.random(k)
    return _f32(path(t) + rng.normal(0, radius, size=(k, 3)))


def segment(k, rng, a, b, jitter=0.01):
    t = rng.random((k, 1))
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return _f32(a + (b - a) * t + rng.normal(0, jitter, size=(k, 3)))
