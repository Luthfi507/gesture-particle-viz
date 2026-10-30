from dataclasses import dataclass

import numpy as np

from utils import (
    Hand,
    dist,
    clamp,
    WRIST,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)

FINGER_TIPS = {
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}

EXTEND_RATIO = 1.5   # jarak ujung jari ke pergelangan, dibagi ukuran telapak
THUMB_RATIO = 0.6
PINCH_RATIO = 0.22
OPEN_MIN = 0.8
OPEN_MAX = 1.8


def palm_size(lm: np.ndarray) -> float:
    return dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6


def finger_states(lm: np.ndarray) -> dict:
    palm = palm_size(lm)
    states = {
        name: dist(lm[tip], lm[WRIST]) / palm > EXTEND_RATIO
        for name, tip in FINGER_TIPS.items()
    }
    states["thumb"] = dist(lm[THUMB_TIP], lm[INDEX_MCP]) / palm > THUMB_RATIO
    return states


def classify_pose(lm: np.ndarray) -> str:
    palm = palm_size(lm)
    states = finger_states(lm)
    count = sum(states[name] for name in FINGER_TIPS)

    if dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm < PINCH_RATIO:
        return "pinch"
    if count == 0:
        return "fist"
    if count == 4:
        return "open"
    if states["index"] and not states["middle"] and count == 1:
        return "point"
    if states["index"] and states["middle"] and count == 2:
        return "peace"
    return "none"


def openness(lm: np.ndarray) -> float:
    palm = palm_size(lm)
    ratios = [dist(lm[tip], lm[WRIST]) / palm for tip in FINGER_TIPS.values()]
    value = (float(np.mean(ratios)) - OPEN_MIN) / (OPEN_MAX - OPEN_MIN)
    return clamp(value, 0.0, 1.0)


def roll(lm: np.ndarray) -> float:
    v = lm[MIDDLE_MCP] - lm[WRIST]
    return float(np.arctan2(v[0], -v[1]))


class SnapDetector:
    # jentikan = jempol dan jari tengah menempel sebentar, lalu lepas dengan cepat
    def __init__(
        self,
        close: float = 0.30,
        release: float = 0.70,
        min_hold: float = 0.05,
        max_release: float = 0.20,
        cooldown: float = 0.6,
        min_reach: float = 1.1,
    ):
        self.min_reach = min_reach
        self.close = close
        self.release = release
        self.min_hold = min_hold
        self.max_release = max_release
        self.cooldown = cooldown
        self.pinched_since = None
        self.last_close = 0.0
        self.last_snap = -10.0

    def update(self, lm: np.ndarray, now: float) -> bool:
        palm = palm_size(lm)
        d = dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / palm
        # di dalam kepalan jempol juga menempel ke jari tengah, jadi jari harus masih terjulur
        reach = dist(lm[MIDDLE_TIP], lm[WRIST]) / palm

        if d < self.close and reach >= self.min_reach:
            if self.pinched_since is None:
                self.pinched_since = now
            self.last_close = now
            return False

        if self.pinched_since is None:
            return False

        if d > self.release:
            held = self.last_close - self.pinched_since
            quick = now - self.last_close <= self.max_release
            ready = now - self.last_snap >= self.cooldown
            self.pinched_since = None
            if quick and held >= self.min_hold and ready:
                self.last_snap = now
                return True
            return False

        if now - self.last_close > self.max_release:
            self.pinched_since = None
        return False


class Debouncer:
    def __init__(self, hold: float = 0.15, initial: str = "none"):
        self.hold = hold
        self.stable = initial
        self.candidate = initial
        self.since = 0.0

    def update(self, value: str, now: float) -> bool:
        if value != self.candidate:
            self.candidate = value
            self.since = now
        if self.candidate != self.stable and now - self.since >= self.hold:
            self.stable = self.candidate
            return True
        return False


@dataclass
class Intent:
    snap: bool = False
    pose: str = "none"
    changed: bool = False
    openness: float = 0.0
    yaw: float = None
    pitch: float = None
    pointer: tuple = None
    zoom: float = None


class GestureController:
    def __init__(self, hold: float = 0.15):
        self.snaps = [SnapDetector(), SnapDetector()]
        self.debounce = Debouncer(hold)
        self.zoom_base = None

    def update(self, hands: list, now: float) -> Intent:
        intent = Intent()

        for detector, hand in zip(self.snaps, hands):
            if detector.update(hand.lm, now):
                intent.snap = True

        raw = classify_pose(hands[0].lm) if hands else "none"
        intent.changed = self.debounce.update(raw, now)
        intent.pose = self.debounce.stable

        if hands:
            main: Hand = hands[0]
            intent.openness = openness(main.lm)
            intent.yaw = roll(main.lm)
            intent.pitch = (0.5 - float(main.norm[WRIST][1])) * 1.2
            if intent.pose == "point":
                intent.pointer = tuple(main.norm[INDEX_TIP])

        if len(hands) == 2:
            gap = dist(hands[0].norm[WRIST], hands[1].norm[WRIST])
            if self.zoom_base is None:
                self.zoom_base = max(gap, 1e-3)
            intent.zoom = gap / self.zoom_base
        else:
            self.zoom_base = None

        return intent
