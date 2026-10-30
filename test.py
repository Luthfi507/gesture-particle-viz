import asyncio
import json
import struct
import unittest

import cv2

import numpy as np

from models import MODELS
from tools import (
    Debouncer,
    Flow,
    GestureController,
    Intent,
    SnapDetector,
    classify_pose,
    openness,
    roll,
    FORMED,
    IDLE,
    SWIRL,
)
from aiohttp.test_utils import TestClient, TestServer

from server import MSG_FRAME, MSG_MODEL, QUANT, create_app
from engines.camera import project
from tools import Flow
from utils import Hand

MCP = {"index": (-0.35, 0.95), "middle": (0.0, 1.0), "ring": (0.33, 0.93), "pinky": (0.62, 0.8)}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(index=False, middle=False, ring=False, pinky=False, thumb=False, pinch=False,
              thumb_to_middle=False, scale=100.0, origin=(300.0, 400.0), tilt=0.0) -> Hand:
    pts = np.zeros((21, 3))
    pts[0] = (0, 0, 0)
    pts[1:5] = [(-0.35, 0.3, 0), (-0.6, 0.55, 0), (-0.8, 0.8, 0),
                (-1.05, 1.0, 0) if thumb else (-0.1, 0.6, 0)]
    for name, up in zip(MCP, (index, middle, ring, pinky)):
        mx, my = MCP[name]
        b = BASE[name]
        if up:
            offsets = [0, 0.45, 0.75, 1.0]
        else:
            offsets = [0, 0.30, 0.15, -0.4]
        for j, off in enumerate(offsets):
            pts[b + j] = (mx, my + off, 0)
    if pinch:
        pts[4] = pts[8] + (-0.03, 0, 0)
    if thumb_to_middle:
        pts[4] = pts[12] + (-0.03, 0, 0)

    c, s = np.cos(tilt), np.sin(tilt)
    xy = pts[:, :2] @ np.array([[c, -s], [s, c]]).T
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = origin[0] + xy[:, 0] * scale
    lm[:, 1] = origin[1] - xy[:, 1] * scale   # y gambar mengarah ke bawah
    norm = lm[:, :2] / np.array([640.0, 480.0])
    return Hand(lm, norm.astype(np.float32))


FIST = dict()
OPEN = dict(index=True, middle=True, ring=True, pinky=True, thumb=True)
POINT = dict(index=True)
PEACE = dict(index=True, middle=True)


class PoseTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_pose(make_hand(**FIST).lm), "fist")
        self.assertEqual(classify_pose(make_hand(**OPEN).lm), "open")
        self.assertEqual(classify_pose(make_hand(**POINT).lm), "point")
        self.assertEqual(classify_pose(make_hand(**PEACE).lm), "peace")
        self.assertEqual(classify_pose(make_hand(middle=True, ring=True, pinky=True, pinch=True).lm), "pinch")

    def test_scale_invariant(self):
        for scale in (40.0, 100.0, 220.0):
            self.assertEqual(classify_pose(make_hand(scale=scale, **OPEN).lm), "open")
            self.assertEqual(classify_pose(make_hand(scale=scale, **FIST).lm), "fist")

    def test_openness(self):
        self.assertLess(openness(make_hand(**FIST).lm), 0.2)
        self.assertGreater(openness(make_hand(**OPEN).lm), 0.9)

    def test_roll(self):
        self.assertAlmostEqual(roll(make_hand(**OPEN).lm), 0.0, places=3)
        self.assertAlmostEqual(roll(make_hand(tilt=-0.5, **OPEN).lm), 0.5, places=2)


class SnapTests(unittest.TestCase):
    def run_sequence(self, gaps, step=1 / 30):
        det, fired, now = SnapDetector(), [], 0.0
        for g in gaps:
            hand = make_hand(index=True, middle=True, ring=True, pinky=True, thumb=g)
            if g == "close":
                hand = make_hand(index=True, middle=True, ring=True, pinky=True, thumb_to_middle=True)
            now += step
            fired.append(det.update(hand.lm, now))
        return fired

    def test_snap_fires_once(self):
        fired = self.run_sequence([True] * 5 + ["close"] * 4 + [True] * 5)
        self.assertEqual(sum(fired), 1)

    def test_slow_release_is_not_a_snap(self):
        det, now, fired = SnapDetector(), 0.0, []
        closed = make_hand(index=True, middle=True, ring=True, pinky=True, thumb_to_middle=True)
        away = make_hand(index=True, middle=True, ring=True, pinky=True, thumb=True)
        for hand, frames in ((away, 3), (closed, 4), (away, 1)):
            for _ in range(frames):
                now += 1 / 30
                fired.append(det.update(hand.lm, now))
        # tahan terlalu lama di zona tengah sebelum lepas
        mid = make_hand(index=True, middle=True, ring=True, pinky=True, thumb_to_middle=True)
        mid.lm[4] = mid.lm[12] + np.array([-45.0, 0, 0])
        det2, now, fired = SnapDetector(), 0.0, []
        for hand, frames in ((closed, 4), (mid, 10), (away, 2)):
            for _ in range(frames):
                now += 1 / 30
                fired.append(det2.update(hand.lm, now))
        self.assertEqual(sum(fired), 0)

    def test_opening_a_fist_is_not_a_snap(self):
        det, now, fired = SnapDetector(), 0.0, []
        for hand, frames in ((make_hand(**FIST), 8), (make_hand(**OPEN), 4), (make_hand(**FIST), 8)):
            for _ in range(frames):
                now += 1 / 30
                fired.append(det.update(hand.lm, now))
        self.assertEqual(sum(fired), 0)

    def test_cooldown(self):
        seq = ([True] * 3 + ["close"] * 3 + [True] * 2) * 2
        self.assertEqual(sum(self.run_sequence(seq)), 1)


class DebounceTests(unittest.TestCase):
    def test_needs_hold(self):
        d = Debouncer(0.15)
        self.assertFalse(d.update("fist", 0.00))
        self.assertFalse(d.update("fist", 0.10))
        self.assertTrue(d.update("fist", 0.16))
        self.assertEqual(d.stable, "fist")

    def test_flicker_ignored(self):
        d = Debouncer(0.15)
        for t, v in enumerate(["fist", "open", "fist", "open", "fist"]):
            self.assertFalse(d.update(v, t * 0.05))


class ControllerTests(unittest.TestCase):
    def test_two_hand_zoom(self):
        ctrl, now = GestureController(), 0.0
        near = [make_hand(origin=(200, 400), **OPEN), make_hand(origin=(300, 400), **OPEN)]
        far = [make_hand(origin=(100, 400), **OPEN), make_hand(origin=(400, 400), **OPEN)]
        ctrl.update(near, now)
        self.assertAlmostEqual(ctrl.update(near, now + 0.1).zoom, 1.0, places=3)
        self.assertGreater(ctrl.update(far, now + 0.2).zoom, 2.5)
        self.assertIsNone(ctrl.update(near[:1], now + 0.3).zoom)


class ModelTests(unittest.TestCase):
    def test_models(self):
        for name, build in MODELS.items():
            a, b = build(3000), build(3000)
            self.assertEqual(a.points.shape, (3000, 3), name)
            self.assertTrue(np.isfinite(a.points).all(), name)
            np.testing.assert_array_equal(a.points, b.points)
            self.assertEqual(a.part_dirs.shape, (len(a.parts), 3))
            self.assertEqual(len(np.unique(a.part_ids)), len(a.parts), name)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.flow = Flow({name: build(2000) for name, build in MODELS.items()})

    def pose(self, name, **kw):
        self.flow.apply(Intent(pose=name, changed=True, **kw), 0.033)

    def test_summon_form_dissolve(self):
        self.assertEqual(self.flow.state, IDLE)
        self.pose("fist")
        self.assertEqual(self.flow.state, IDLE)
        self.flow.apply(Intent(snap=True), 0.033)
        self.assertEqual(self.flow.state, SWIRL)
        self.flow.apply(Intent(pose="open"), 0.033)
        self.assertEqual(self.flow.state, SWIRL)
        self.pose("fist")
        self.assertEqual(self.flow.state, FORMED)
        self.flow.apply(Intent(snap=True), 0.033)
        self.assertEqual(self.flow.state, IDLE)

    def test_open_morphs_wonder_but_explodes_heart(self):
        self.flow.apply(Intent(snap=True), 0.033)
        self.pose("fist")
        self.assertEqual(self.flow.model.category, "wonder")
        first = self.flow.name
        self.pose("open", openness=1.0)
        self.assertNotEqual(self.flow.name, first)

        self.flow.index = self.flow.names.index("heart")
        for _ in range(60):
            self.flow.apply(Intent(pose="open", openness=0.8), 0.033)
        self.assertGreater(self.flow.explode, 0.7)
        for _ in range(60):
            self.flow.apply(Intent(pose="fist", openness=0.0), 0.033)
        self.assertLess(self.flow.explode, 0.05)

    def test_point_select_and_pinch_pull(self):
        self.flow.apply(Intent(snap=True), 0.033)
        self.pose("fist")
        self.flow.index = self.flow.names.index("engine")
        screen = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]] + [[0.1, 0.9]] * 5)
        for _ in range(20):
            self.flow.apply(Intent(pose="point", pointer=(0.5, 0.5)), 0.033, screen)
        self.assertEqual(self.flow.selected, 1)
        self.pose("pinch")
        self.assertEqual(self.flow.pulled, 1)
        self.pose("pinch")
        self.assertIsNone(self.flow.pulled)



class FakeTracker:
    # menggantikan MediaPipe: mengembalikan tangan yang diatur lewat atribut hands
    hands = []

    def detect(self, frame):
        assert frame.ndim == 3
        return list(FakeTracker.hands)


def jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
    return buf.tobytes()


def parse_frame(data: bytes):
    kind, length = struct.unpack_from("<BI", data, 0)
    assert kind == MSG_FRAME
    state = json.loads(data[5:5 + length])
    assert (5 + length) % 2 == 0
    pos = np.frombuffer(data, dtype="<i2", offset=5 + length)
    return state, pos


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.count = 5000
        self.client = TestClient(TestServer(create_app(self.count, FakeTracker)))
        await self.client.start_server()
        FakeTracker.hands = []

    async def asyncTearDown(self):
        await self.client.close()

    async def latest_frame(self, ws, wait=0.25):
        # kosongkan antrean dan ambil frame terbaru
        state = None
        end = asyncio.get_event_loop().time() + wait
        while asyncio.get_event_loop().time() < end:
            try:
                msg = await ws.receive(timeout=0.2)
            except asyncio.TimeoutError:
                break
            if msg.type == 2 and msg.data[0] == MSG_FRAME:
                state = parse_frame(msg.data)
        return state

    async def test_protocol_and_commands(self):
        ws = await self.client.ws_connect("/ws")
        init = json.loads((await ws.receive(timeout=20)).data)
        self.assertEqual(init["type"], "init")
        self.assertEqual(init["count"], self.count)
        self.assertEqual(len(init["models"]), len(MODELS))

        kinds = set()
        for _ in range(4):
            kinds.add((await ws.receive(timeout=5)).data[0])
        self.assertEqual(kinds, {MSG_MODEL, MSG_FRAME})

        state, pos = await self.latest_frame(ws)
        self.assertEqual(state["state"], "idle")
        self.assertEqual(pos.size, self.count * 3)
        self.assertLessEqual(np.abs(pos).max(), 32767)

        await ws.send_str(json.dumps({"cmd": "snap"}))
        state, _ = await self.latest_frame(ws)
        self.assertEqual(state["state"], "swirl")
        await ws.send_str(json.dumps({"cmd": "form"}))
        await ws.send_str(json.dumps({"cmd": "next"}))
        state, _ = await self.latest_frame(ws, 0.4)
        self.assertEqual(state["state"], "formed")
        self.assertEqual(state["model"], 1)
        self.assertEqual(len(state["centers"]), len(init["models"][1]["parts"]))

        # perintah sampah tidak boleh menjatuhkan sesi
        await ws.send_str("bukan json")
        await ws.send_str(json.dumps({"cmd": "select", "index": "x"}))
        await ws.send_str(json.dumps({"cmd": "explode", "amount": 99}))
        state, _ = await self.latest_frame(ws)
        self.assertIsNotNone(state)
        await ws.close()

    async def test_camera_frames_drive_gestures(self):
        ws = await self.client.ws_connect("/ws")
        await ws.receive(timeout=20)
        await ws.send_str(json.dumps({"cmd": "snap"}))

        FakeTracker.hands = [make_hand(**FIST)]
        for _ in range(14):
            await ws.send_bytes(jpeg())
            await asyncio.sleep(0.05)
        state, _ = await self.latest_frame(ws)
        self.assertEqual(state["state"], "formed")
        self.assertEqual(state["pose"], "fist")
        self.assertEqual(len(state["hands"]), 1)
        self.assertEqual(len(state["hands"][0]), 21)

        await ws.send_str(json.dumps({"cmd": "next"}))
        await ws.send_str(json.dumps({"cmd": "next"}))
        await ws.send_str(json.dumps({"cmd": "next"}))   # index 3 = heart
        FakeTracker.hands = [make_hand(**OPEN)]
        for _ in range(30):
            await ws.send_bytes(jpeg())
            await asyncio.sleep(0.05)
        state, _ = await self.latest_frame(ws)
        self.assertEqual(state["model"], 3)
        self.assertEqual(state["pose"], "open")
        self.assertGreater(state["explode"], 0.5)

        # kamera berhenti mengirim: tangan dianggap hilang
        await asyncio.sleep(0.9)
        state, _ = await self.latest_frame(ws)
        self.assertEqual(state["hands"], [])
        await ws.close()

    async def test_viewport_and_projection_match(self):
        ws = await self.client.ws_connect("/ws")
        await ws.receive(timeout=20)
        await ws.send_str(json.dumps({"type": "viewport", "w": 800, "h": 600}))
        await ws.send_str(json.dumps({"cmd": "snap"}))
        await ws.send_str(json.dumps({"cmd": "form"}))
        state, pos = await self.latest_frame(ws, 0.5)
        centers = np.array(state["centers"], dtype=np.float32)
        x, y, _ = project(centers, state["yaw"], state["pitch"], state["zoom"], 800, 600)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
        self.assertTrue(((x > 0) & (x < 800) & (y > 0) & (y < 600)).all())
        await ws.close()


class QuantTests(unittest.TestCase):
    def test_int16_precision(self):
        values = np.linspace(-3.9, 3.9, 1001, dtype=np.float32)
        restored = np.clip(values * QUANT, -32767, 32767).astype("<i2") / 32767 * 4.0
        self.assertLess(np.abs(restored - values).max(), 2e-4)



if __name__ == "__main__":
    unittest.main()
