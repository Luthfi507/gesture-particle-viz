import numpy as np

from utils import ease, normalize

SPREAD = 0.65
PULL = np.array([0.0, 0.0, 0.9], dtype=np.float32)


def lub_dub(phase: float) -> float:
    def bump(center, width):
        d = (phase - center + 0.5) % 1.0 - 0.5
        return float(np.exp(-(d / width) ** 2))
    return bump(0.0, 0.05) + 0.6 * bump(0.28, 0.06)


class ParticleEngine:
    def __init__(self, count: int = 60000, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.n = count
        self.pos = (rng.normal(size=(count, 3)) * 0.05).astype(np.float32)
        self.vel = np.zeros((count, 3), dtype=np.float32)
        self.stiff = (0.6 + 0.8 * rng.random(count)).astype(np.float32)

        self.orbit_r = (0.4 + 1.1 * rng.random(count) ** 0.7).astype(np.float32)
        self.orbit_h = rng.normal(0, 0.55, count).astype(np.float32)
        self.orbit_phase = rng.uniform(0, 2 * np.pi, count).astype(np.float32)
        self.orbit_speed = (0.6 + 1.4 / (0.4 + self.orbit_r)).astype(np.float32)

        self.visibility = 0.0
        self.time = 0.0
        self.model = None
        self.dirs = None
        self.prev_state = "idle"

    def set_model(self, model):
        self.model = model
        self.dirs = model.part_dirs[model.part_ids]

    def beat_scale(self) -> float:
        if self.model is None or not self.model.beat_bpm:
            return 1.0
        phase = (self.time * self.model.beat_bpm / 60.0) % 1.0
        return 1.0 + 0.06 * lub_dub(phase)

    def orbit(self) -> np.ndarray:
        a = self.orbit_phase + self.orbit_speed * self.time
        return np.stack(
            [self.orbit_r * np.cos(a), self.orbit_h, self.orbit_r * np.sin(a)], axis=1
        )

    def formed_target(self, explode: float, pulled) -> np.ndarray:
        target = self.model.points * self.beat_scale() + self.dirs * (explode * SPREAD)
        if pulled is not None:
            target[self.model.part_ids == pulled] += PULL
        return target

    def part_centers(self, explode: float, pulled) -> np.ndarray:
        centers = self.model.part_centers * self.beat_scale()
        centers = centers + self.model.part_dirs * (explode * SPREAD)
        if pulled is not None:
            centers[pulled] += PULL
        return centers

    def colors(self, selected) -> np.ndarray:
        colors = self.model.colors
        if selected is None:
            return colors
        colors = colors.copy()
        mask = self.model.part_ids == selected
        colors[mask] = colors[mask] * 0.35 + 0.65
        return colors

    def step(self, dt: float, state: str, explode: float = 0.0, pulled=None):
        dt = min(dt, 1 / 30)
        self.time += dt

        goal = 0.0 if state == "idle" else 1.0
        self.visibility = ease(self.visibility, goal, 4.0 if goal else 3.0, dt)

        if self.prev_state == "formed" and state == "idle":
            self.vel += normalize(self.pos) * 2.5
        self.prev_state = state

        if state == "formed" and self.model is not None:
            target = self.formed_target(explode, pulled)
        else:
            target = self.orbit()

        k = 14.0 * self.stiff[:, None]
        self.vel += (k * (target - self.pos) - 5.5 * self.vel) * dt
        self.pos += self.vel * dt
