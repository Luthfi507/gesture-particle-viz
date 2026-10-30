import numpy as np

from .base import Part, assemble
from .shapes import box, cylinder, segment, sphere, tube

TAU = 2 * np.pi


# ---------------------------------------------------------------- wonders
EIFFEL_H = 3.0
EIFFEL_W = 0.6
CORNERS = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)


def _eiffel_profile(h):
    return 0.08 + 0.92 * (1 - h) ** 2.0


def _eiffel_legs(k, rng):
    h = rng.random(k)
    corner = CORNERS[rng.integers(0, 4, k)]
    prof = _eiffel_profile(h)
    xz = corner * (EIFFEL_W * prof)[:, None]
    p = np.stack([xz[:, 0], h * EIFFEL_H, xz[:, 1]], axis=1)
    return (p + rng.normal(0, 1, (k, 3)) * (0.008 + 0.02 * prof)[:, None]).astype(np.float32)


def _eiffel_lattice(k, rng):
    h = rng.random(k) * 0.88
    flat = rng.random(k) < 0.25
    h = np.where(flat, np.round(h * 18) / 18, h)
    u = np.abs(2 * ((h * 9) % 1.0) - 1)
    u = np.where(rng.random(k) < 0.5, u, 1 - u)
    u = np.where(flat, rng.random(k), u)
    side = rng.integers(0, 4, k)
    a, b = CORNERS[side], CORNERS[(side + 1) % 4]
    xz = (a + (b - a) * u[:, None]) * (EIFFEL_W * _eiffel_profile(h))[:, None]
    p = np.stack([xz[:, 0], h * EIFFEL_H, xz[:, 1]], axis=1)
    return (p + rng.normal(0, 0.007, (k, 3))).astype(np.float32)


def _eiffel_platforms(k, rng):
    levels = np.array([0.28, 0.58, 0.86])
    pick = rng.integers(0, 3, k)
    h = levels[pick]
    half = EIFFEL_W * _eiffel_profile(h) * 1.1
    x = rng.uniform(-1, 1, k) * half
    z = rng.uniform(-1, 1, k) * half
    edge = rng.random(k) < 0.7
    side = rng.integers(0, 4, k)
    x = np.where(edge & (side == 0), half, np.where(edge & (side == 1), -half, x))
    z = np.where(edge & (side == 2), half, np.where(edge & (side == 3), -half, z))
    p = np.stack([x, h * EIFFEL_H, z], axis=1)
    return (p + rng.normal(0, 0.008, (k, 3))).astype(np.float32)


def eiffel_tower(n: int):
    parts = [
        Part("Legs", _eiffel_legs, (1.0, 0.72, 0.35), 3.0, "Four curved legs carry the whole load"),
        Part("Lattice", _eiffel_lattice, (1.0, 0.55, 0.2), 4.0, "Wrought-iron cross bracing"),
        Part("Platforms", _eiffel_platforms, (1.0, 0.85, 0.5), 1.5, "Three observation levels"),
        Part(
            "Spire",
            lambda k, rng: segment(k, rng, (0, EIFFEL_H * 0.9, 0), (0, EIFFEL_H + 0.35, 0), 0.006),
            (1.0, 1.0, 0.9), 0.4, "Antenna at the very top",
        ),
    ]
    return assemble("Eiffel Tower", "wonder", parts, n, seed=1)


def _pyramid_courses(k, rng):
    height, base = 2.0, 1.6
    y = height * (1 - np.sqrt(rng.random(k)))
    snap = rng.random(k) < 0.6
    y = np.where(snap, np.round(y / height * 22) / 22 * height, y)
    w = base * (1 - y / height)
    u = rng.uniform(-1, 1, k) * w
    side = rng.integers(0, 4, k)
    x = np.where(side == 0, w, np.where(side == 1, -w, u))
    z = np.where(side == 2, w, np.where(side == 3, -w, u))
    p = np.stack([x, y, z], axis=1)
    return (p + rng.normal(0, 0.006, (k, 3))).astype(np.float32)


def great_pyramid(n: int):
    parts = [
        Part("Casing", _pyramid_courses, (1.0, 0.8, 0.4), 6.0, "Stepped limestone courses"),
        Part(
            "Capstone",
            lambda k, rng: sphere(k, rng, 0.07, (0, 2.02, 0)),
            (1.0, 0.95, 0.7), 0.3, "Gilded pyramidion",
        ),
        Part(
            "King's Chamber",
            lambda k, rng: box(k, rng, (0.4, 0.22, 0.25), (0, 0.8, 0)),
            (1.0, 0.45, 0.25), 0.8, "Granite burial chamber",
        ),
        Part(
            "Grand Gallery",
            lambda k, rng: tube(k, rng, lambda t: np.stack([t * 0 , 0.3 + 0.5 * t, -0.6 + 0.6 * t], axis=1), 0.03),
            (0.9, 0.6, 0.3), 0.8, "Sloping corbelled passage",
        ),
    ]
    return assemble("Great Pyramid", "wonder", parts, n, seed=2)


def _burj_body(k, rng):
    y = rng.random(k)
    step = np.floor(y * 7) / 7
    r = 0.35 * (1 - 0.7 * step)
    lobe = rng.integers(0, 4, k)
    ang = lobe * TAU / 3 + np.pi / 6
    cx = np.where(lobe == 3, 0, 0.55 * r * np.cos(ang))
    cz = np.where(lobe == 3, 0, 0.55 * r * np.sin(ang))
    rl = np.where(lobe == 3, 0.5 * r, 0.55 * r)
    a = rng.uniform(0, TAU, k)
    p = np.stack([cx + rl * np.cos(a), y * 2.6, cz + rl * np.sin(a)], axis=1)
    return (p + rng.normal(0, 0.006, (k, 3))).astype(np.float32)


def burj_khalifa(n: int):
    parts = [
        Part("Tower", _burj_body, (0.55, 0.8, 1.0), 6.0, "Three-lobed Y-shaped body with setbacks"),
        Part(
            "Spire",
            lambda k, rng: segment(k, rng, (0, 2.6, 0), (0, 3.5, 0), 0.008),
            (1.0, 1.0, 1.0), 0.6, "Telescoping steel spire",
        ),
    ]
    return assemble("Burj Khalifa", "wonder", parts, n, seed=3)


# ---------------------------------------------------------------- anatomy
def _heart_shell(k, rng, keep=None):
    out = np.empty((0, 3))
    while len(out) < k:
        p = rng.uniform((-1.3, -0.8, -1.2), (1.3, 0.8, 1.4), size=(k * 30, 3))
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        a = x ** 2 + 2.25 * y ** 2 + z ** 2 - 1
        f = a ** 3 - x ** 2 * z ** 3 - 0.1125 * y ** 2 * z ** 3
        m = (f < 0) & (f > -0.03)
        if keep is not None:
            m &= keep(x, z)
        out = np.concatenate([out, p[m]])
    q = out[:k]
    return np.stack([q[:, 0], q[:, 2], q[:, 1]], axis=1).astype(np.float32)


def _heart_region(keep):
    return lambda k, rng: _heart_shell(k, rng, keep)


def heart(n: int):
    aorta = lambda t: np.stack([0.2 + 0.4 * t, 0.65 + 0.7 * np.sin(np.pi * t), 0 * t], axis=1)
    pulmonary = lambda t: np.stack([-0.05 - 0.5 * t, 0.7 + 0.4 * t, 0.25 - 0.15 * t], axis=1)
    cava = lambda t: np.stack([-0.45 + 0 * t, 0.5 + 0.85 * t, -0.1 + 0 * t], axis=1)
    parts = [
        Part("Left ventricle", _heart_region(lambda x, z: (z < 0.1) & (x >= -0.05)),
             (1.0, 0.15, 0.2), 3.0, "Pumps oxygen-rich blood to the body"),
        Part("Right ventricle", _heart_region(lambda x, z: (z < 0.1) & (x < -0.05)),
             (0.85, 0.25, 0.45), 2.2, "Pumps blood to the lungs"),
        Part("Left atrium", _heart_region(lambda x, z: (z >= 0.1) & (x >= 0)),
             (1.0, 0.35, 0.3), 1.6, "Receives blood from the lungs"),
        Part("Right atrium", _heart_region(lambda x, z: (z >= 0.1) & (x < 0)),
             (0.9, 0.4, 0.55), 1.6, "Receives blood from the body"),
        Part("Aorta", lambda k, rng: tube(k, rng, aorta, 0.06),
             (1.0, 0.6, 0.2), 0.7, "Largest artery, carries blood out to the body", (0.6, 1.0, 0.0)),
        Part("Pulmonary artery", lambda k, rng: tube(k, rng, pulmonary, 0.055),
             (0.4, 0.6, 1.0), 0.6, "Carries blood to the lungs", (-1.0, 0.8, 0.4)),
        Part("Vena cava", lambda k, rng: tube(k, rng, cava, 0.055),
             (0.3, 0.5, 1.0), 0.6, "Returns blood from the body", (-1.0, 0.4, -0.3)),
    ]
    return assemble("Heart", "anatomy", parts, n, seed=4, beat_bpm=72.0)


# ---------------------------------------------------------------- biology
DNA_TURNS = 3.0
DNA_RADIUS = 0.45
DNA_RUNGS = 60


def _helix(offset):
    def sample(k, rng):
        t = rng.random(k)
        a = TAU * DNA_TURNS * t + offset
        p = np.stack([DNA_RADIUS * np.cos(a), (t - 0.5) * 3.0, DNA_RADIUS * np.sin(a)], axis=1)
        return (p + rng.normal(0, 0.014, (k, 3))).astype(np.float32)
    return sample


def _rungs(parity):
    def sample(k, rng):
        m = rng.integers(0, DNA_RUNGS // 2, k) * 2 + parity
        t = m / DNA_RUNGS
        a = TAU * DNA_TURNS * t
        pa = np.stack([DNA_RADIUS * np.cos(a), (t - 0.5) * 3.0, DNA_RADIUS * np.sin(a)], axis=1)
        pb = np.stack([-DNA_RADIUS * np.cos(a), (t - 0.5) * 3.0, -DNA_RADIUS * np.sin(a)], axis=1)
        u = rng.random((k, 1))
        return (pa + (pb - pa) * u + rng.normal(0, 0.008, (k, 3))).astype(np.float32)
    return sample


def dna_helix(n: int):
    parts = [
        Part("Strand A", _helix(0.0), (0.3, 0.8, 1.0), 3.0, "Sugar-phosphate backbone", (1, 0, 0)),
        Part("Strand B", _helix(np.pi), (0.5, 0.4, 1.0), 3.0, "Complementary backbone", (-1, 0, 0)),
        Part("A-T pairs", _rungs(0), (1.0, 0.4, 0.4), 1.6, "Adenine pairs with thymine", (0, 1, 0), 0.3),
        Part("G-C pairs", _rungs(1), (0.4, 1.0, 0.5), 1.6, "Guanine pairs with cytosine", (0, -1, 0), 0.3),
    ]
    return assemble("DNA Double Helix", "biology", parts, n, seed=5)


# ---------------------------------------------------------------- engines
CYL_X = (-0.8, -0.27, 0.27, 0.8)
PISTON_Y = (0.15, -0.1, -0.1, 0.15)


def _pick(samplers):
    def sample(k, rng):
        which = rng.integers(0, len(samplers), k)
        out = np.zeros((k, 3), dtype=np.float32)
        for i, s in enumerate(samplers):
            m = which == i
            if m.any():
                out[m] = s(int(m.sum()), rng)
        return out
    return sample


def inline_four(n: int):
    pistons = _pick([
        (lambda x, y: lambda k, rng: cylinder(k, rng, 0.27, 0.2, (x, y, 0)))(x, y)
        for x, y in zip(CYL_X, PISTON_Y)
    ])
    rods = _pick([
        (lambda x, y: lambda k, rng: segment(k, rng, (x, y, 0), (x, -0.5, 0), 0.012))(x, y)
        for x, y in zip(CYL_X, PISTON_Y)
    ])
    weights = _pick([
        (lambda x: lambda k, rng: cylinder(k, rng, 0.22, 0.06, (x, -0.5, 0), axis=0))(x)
        for x in CYL_X
    ])
    crank = lambda k, rng: np.concatenate([
        cylinder(k - k // 3, rng, 0.1, 2.4, (0, -0.5, 0), axis=0),
        weights(k // 3, rng),
    ])
    parts = [
        Part("Engine block", lambda k, rng: box(k, rng, (2.2, 0.9, 0.9), (0, 0, 0)),
             (0.6, 0.65, 0.75), 3.0, "Cast iron body holding the cylinders", (0, 0, 0), 0.0),
        Part("Cylinder head", lambda k, rng: box(k, rng, (2.2, 0.3, 0.9), (0, 0.65, 0)),
             (0.8, 0.8, 0.9), 1.5, "Holds valves and spark plugs", (0, 1, 0)),
        Part("Pistons", pistons, (1.0, 0.6, 0.2), 1.5, "Compressed gas pushes them down", (0, 0.6, 0)),
        Part("Connecting rods", rods, (0.9, 0.9, 0.9), 0.8, "Link pistons to the crankshaft", (0, -0.2, 0)),
        Part("Crankshaft", crank, (1.0, 0.35, 0.3), 1.2, "Turns up-and-down motion into rotation", (0, -0.9, 0)),
        Part("Oil pan", lambda k, rng: box(k, rng, (2.2, 0.25, 0.8), (0, -0.85, 0)),
             (0.35, 0.5, 0.7), 1.0, "Oil reservoir at the bottom", (0, -1, 0)),
        Part("Intake manifold", lambda k, rng: cylinder(k, rng, 0.1, 2.0, (0, 1.0, 0.55), axis=0),
             (0.3, 0.9, 1.0), 1.0, "Feeds air to each cylinder", (0, 0.6, 0.9)),
        Part("Exhaust manifold", lambda k, rng: cylinder(k, rng, 0.1, 2.0, (0, 0.7, -0.55), axis=0),
             (1.0, 0.3, 0.15), 1.0, "Collects hot exhaust gas", (0, 0.2, -0.9)),
    ]
    return assemble("Inline-4 Engine", "engine", parts, n, seed=6)
