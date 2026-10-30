import argparse
import asyncio
import json
import struct
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from engines import CAM, FOCAL, ParticleEngine, project
from models import MODELS
from tools import FORMED, Flow, GestureController, Intent

COUNT = web.AppKey("count", int)
MODELS_CACHE = web.AppKey("models", dict)
TRACKER = web.AppKey("tracker_factory", object)

STATIC = Path(__file__).resolve().parent / "static"

FPS = 30
QUANT = 32767 / 4.0          # posisi dikirim sebagai int16 dalam rentang -4..4
MSG_MODEL = 1
MSG_FRAME = 2
STALE_VISION = 0.5           # detik tanpa frame kamera sebelum tangan dianggap hilang


def build_models(count: int) -> dict:
    return {name: build(count) for name, build in MODELS.items()}


def model_message(index: int, model) -> bytes:
    colors = (np.clip(model.colors, 0, 1) * 255).astype(np.uint8)
    return bytes([MSG_MODEL, index]) + colors.tobytes() + model.part_ids.astype(np.uint8).tobytes()


class Session:
    def __init__(self, ws, models: dict, count: int, tracker_factory):
        self.ws = ws
        self.count = count
        self.tracker_factory = tracker_factory
        self.tracker = None

        self.flow = Flow(models)
        self.engine = ParticleEngine(count)
        self.engine.set_model(self.flow.model)
        self.controller = GestureController()

        self.size = (1280, 720)
        self.intent = Intent()
        self.fresh = False
        self.hands = []
        self.latest = None
        self.frame_ready = asyncio.Event()
        self.last_vision = 0.0
        self.sent_model = None
        self.model_cache = {}

    def init_message(self) -> str:
        return json.dumps({
            "type": "init",
            "count": self.count,
            "cam": CAM,
            "focal": FOCAL,
            "models": [
                {
                    "name": m.name,
                    "category": m.category,
                    "parts": [{"name": p.name, "info": p.info} for p in m.parts],
                }
                for m in self.flow.models.values()
            ],
        })

    def on_text(self, raw: str):
        try:
            msg = json.loads(raw)
            if msg.get("type") == "viewport":
                w, h = int(msg["w"]), int(msg["h"])
                self.size = (max(100, min(w, 8000)), max(100, min(h, 8000)))
            elif "cmd" in msg:
                self.flow.command(msg)
        except (ValueError, KeyError, TypeError):
            pass

    def push_intent(self, new: Intent):
        # event sesaat (snap, ganti pose) tidak boleh hilang kalau dua frame kamera datang sebelum satu tick
        if self.fresh:
            new.snap = new.snap or self.intent.snap
            if self.intent.pose == new.pose:
                new.changed = new.changed or self.intent.changed
        self.intent, self.fresh = new, True

    def take_intent(self, now: float) -> Intent:
        if self.last_vision and now - self.last_vision > STALE_VISION:
            self.hands, self.last_vision = [], 0.0
            self.push_intent(self.controller.update([], now))
        if self.fresh:
            self.fresh = False
            return self.intent
        return replace(self.intent, snap=False, changed=False)

    def part_screen(self):
        if self.flow.state != FORMED:
            return None
        centers = self.engine.part_centers(self.flow.explode, self.flow.pulled)
        w, h = self.size
        x, y, _ = project(centers, self.flow.yaw, self.flow.pitch, self.flow.zoom, w, h)
        return np.stack([x / w, y / h], axis=1)

    def detect(self, data: bytes):
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else self.tracker.detect(img)

    async def vision_loop(self):
        loop = asyncio.get_running_loop()
        self.tracker = await loop.run_in_executor(None, self.tracker_factory)
        while True:
            await self.frame_ready.wait()
            self.frame_ready.clear()
            data, self.latest = self.latest, None   # hanya frame terbaru, yang lama dibuang
            if data is None:
                continue
            hands = await loop.run_in_executor(None, self.detect, data)
            if hands is None:
                continue
            now = time.monotonic()
            self.hands, self.last_vision = hands, now
            self.push_intent(self.controller.update(hands, now))

    def step(self, dt: float, now: float):
        flow, engine = self.flow, self.engine
        flow.apply(self.take_intent(now), dt, self.part_screen())
        if engine.model is not flow.model:
            engine.set_model(flow.model)
        engine.step(dt, flow.state, flow.explode, flow.pulled)

    async def send_state(self):
        flow, engine = self.flow, self.engine

        if self.sent_model != flow.index:
            if flow.index not in self.model_cache:
                self.model_cache[flow.index] = model_message(flow.index, flow.model)
            await self.ws.send_bytes(self.model_cache[flow.index])
            self.sent_model = flow.index

        formed = flow.state == FORMED
        state = {
            "state": flow.state,
            "model": flow.index,
            "pose": self.intent.pose,
            "explode": round(flow.explode, 3),
            "yaw": round(flow.yaw, 4),
            "pitch": round(flow.pitch, 4),
            "zoom": round(flow.zoom, 4),
            "vis": round(engine.visibility, 3),
            "selected": flow.selected,
            "pulled": flow.pulled,
            "labels": flow.labels,
            "centers": engine.part_centers(flow.explode, flow.pulled).round(3).tolist() if formed else None,
            "hands": [h.norm.round(3).tolist() for h in self.hands],
        }
        head = json.dumps(state, separators=(",", ":")).encode()
        if (5 + len(head)) % 2:
            head += b" "   # Int16Array di browser butuh offset genap
        body = np.clip(engine.pos * QUANT, -32767, 32767).astype("<i2").tobytes()
        await self.ws.send_bytes(struct.pack("<BI", MSG_FRAME, len(head)) + head + body)

    async def sim_loop(self):
        period = 1.0 / FPS
        last = time.monotonic()
        due = last
        while True:
            due += period
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            now = time.monotonic()
            dt, last = now - last, now
            self.step(dt, now)
            await self.send_state()

    async def run(self):
        await self.ws.send_str(self.init_message())
        tasks = [asyncio.create_task(self.vision_loop()), asyncio.create_task(self.sim_loop())]
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.BINARY:
                    self.latest = msg.data
                    self.frame_ready.set()
                elif msg.type == WSMsgType.TEXT:
                    self.on_text(msg.data)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def index(request):
    return web.FileResponse(STATIC / "index.html")


async def socket(request):
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)

    try:
        count = int(request.query.get("count", request.app[COUNT]))
    except ValueError:
        count = request.app[COUNT]
    count = max(5000, min(count, 120000))

    cache = request.app[MODELS_CACHE]
    if count not in cache:
        cache[count] = await asyncio.get_running_loop().run_in_executor(None, build_models, count)

    await Session(ws, cache[count], count, request.app[TRACKER]).run()
    return ws


def create_app(count: int = 50000, tracker_factory=None) -> web.Application:
    if tracker_factory is None:
        from tools.hands import HandTracker

        tracker_factory = HandTracker

    app = web.Application()
    app[COUNT] = count
    app[MODELS_CACHE] = {}
    app[TRACKER] = tracker_factory
    app.router.add_get("/", index)
    app.router.add_get("/ws", socket)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--count", type=int, default=50000)
    args = parser.parse_args()

    from tools.hands import ensure_model

    ensure_model()
    app = create_app(args.count)
    app[MODELS_CACHE][args.count] = build_models(args.count)
    print(f"open http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)
