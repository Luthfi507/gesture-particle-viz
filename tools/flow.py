import numpy as np

from utils import clamp, ease, dist

IDLE = "idle"
SWIRL = "swirl"
FORMED = "formed"

PICK_RADIUS = 0.15
DWELL = 0.35


class Flow:
    def __init__(self, models: dict):
        self.models = models   # name -> Model yang sudah dibangun, dipakai bersama antar sesi
        self.names = list(self.models.keys())
        self.index = 0
        self.state = IDLE

        self.explode = 0.0
        self.explode_target = 0.0
        self.zoom = 1.0
        self.zoom_target = 1.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.auto_rotate = False
        self.hand_rotate = True
        self.labels = False

        self.selected = None
        self.pulled = None
        self._dwell_part = None
        self._dwell_time = 0.0
        self._zoom_anchor = None

    @property
    def model(self):
        return self.models[self.names[self.index]]

    @property
    def name(self) -> str:
        return self.names[self.index]

    def snap(self):
        if self.state == IDLE:
            self.state = SWIRL
        else:
            self.state = IDLE
            self.deselect()
            self.explode_target = 0.0

    def form(self):
        if self.state == SWIRL:
            self.state = FORMED

    def next_model(self, step: int = 1):
        self.index = (self.index + step) % len(self.names)
        self.deselect()
        self.explode_target = 0.0

    def deselect(self):
        self.selected = None
        self.pulled = None
        self._dwell_part = None
        self._dwell_time = 0.0

    def toggle_pull(self):
        if self.selected is None:
            return
        self.pulled = None if self.pulled == self.selected else self.selected

    def nudge_explode(self, amount: float):
        self.explode_target = clamp(self.explode_target + amount, 0.0, 1.0)

    def nudge_zoom(self, factor: float):
        self.zoom_target = clamp(self.zoom_target * factor, 0.5, 3.0)

    def pick(self, pointer, part_screen, dt: float):
        # part_screen: (P, 2) posisi tiap bagian di layar, dinormalisasi 0..1
        if pointer is None or part_screen is None:
            self._dwell_part = None
            self._dwell_time = 0.0
            return
        gaps = [dist(pointer, xy) for xy in part_screen]
        best = int(np.argmin(gaps))
        if gaps[best] > PICK_RADIUS:
            self._dwell_part = None
            self._dwell_time = 0.0
            return
        if best != self._dwell_part:
            self._dwell_part = best
            self._dwell_time = 0.0
        self._dwell_time += dt
        if self._dwell_time >= DWELL:
            self.selected = best

    def apply(self, intent, dt: float, part_screen=None):
        if intent.snap:
            self.snap()

        if intent.changed:
            if intent.pose == "open" and self.state == FORMED:
                if self.model.category == "wonder":
                    self.next_model()
            elif intent.pose == "peace" and self.state != IDLE:
                self.next_model()
            elif intent.pose == "pinch" and self.state == FORMED:
                self.toggle_pull()

        # fist bersifat level-triggered: tangan yang sudah mengepal sebelum jentikan tetap membentuk model
        if intent.pose == "fist":
            self.form()

        formed = self.state == FORMED
        if formed and self.model.category != "wonder":
            if intent.pose == "open":
                self.explode_target = intent.openness
            elif intent.pose == "fist":
                self.explode_target = 0.0

        if formed:
            self.pick(intent.pointer, part_screen, dt)

        if intent.zoom is not None:
            if self._zoom_anchor is None:
                self._zoom_anchor = self.zoom_target
            self.zoom_target = clamp(self._zoom_anchor * intent.zoom, 0.5, 3.0)
        else:
            self._zoom_anchor = None

        if formed and self.hand_rotate and intent.yaw is not None and intent.pose != "point":
            self.yaw = ease(self.yaw, intent.yaw * 1.5, 6.0, dt)
            self.pitch = ease(self.pitch, intent.pitch, 6.0, dt)
        elif self.auto_rotate:
            self.yaw += 0.4 * dt

        self.explode = ease(self.explode, self.explode_target, 6.0, dt)
        self.zoom = ease(self.zoom, self.zoom_target, 8.0, dt)

    def command(self, msg: dict):
        # perintah dari keyboard/mouse di browser
        cmd = msg.get("cmd")
        if cmd == "snap":
            self.snap()
        elif cmd == "form":
            self.form()
        elif cmd == "open":
            if self.model.category == "wonder":
                self.next_model()
            else:
                self.explode_target = 0.0 if self.explode_target > 0.5 else 1.0
        elif cmd == "next":
            self.next_model(1)
        elif cmd == "prev":
            self.next_model(-1)
        elif cmd == "explode":
            self.nudge_explode(clamp(float(msg.get("amount", 0.1)), -1.0, 1.0))
        elif cmd == "zoom":
            self.nudge_zoom(clamp(float(msg.get("factor", 1.1)), 0.5, 2.0))
        elif cmd == "zoom_reset":
            self.zoom_target = 1.0
        elif cmd == "labels":
            self.labels = not self.labels
        elif cmd == "rotate":
            self.auto_rotate = not self.auto_rotate
        elif cmd == "hand_rotate":
            self.hand_rotate = not self.hand_rotate
        elif cmd == "pull":
            self.toggle_pull()
        elif cmd == "cycle":
            total = len(self.model.parts)
            self.selected = 0 if self.selected is None else (self.selected + 1) % total
        elif cmd == "deselect":
            self.deselect()
        elif cmd == "select":
            index = int(msg.get("index", -1))
            if 0 <= index < len(self.model.parts):
                self.selected = index
        elif cmd == "drag":
            self.yaw += clamp(float(msg.get("dx", 0.0)), -1.0, 1.0)
            self.pitch = clamp(self.pitch + clamp(float(msg.get("dy", 0.0)), -1.0, 1.0), -1.4, 1.4)
