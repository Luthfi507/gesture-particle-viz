import { state } from "./state.js";
import { $ } from "./config.js";
import { send } from "./websocket.js";
import { startCamera, stopCamera } from "./camera.js";
import { project } from "./projection.js";

const KEYS = {
  " ": { cmd: "snap" },
  f: { cmd: "form" },
  o: { cmd: "open" },
  v: { cmd: "next" },
  ArrowLeft: { cmd: "prev" },
  a: { cmd: "prev" },
  ArrowRight: { cmd: "next" },
  d: { cmd: "next" },
  ArrowUp: { cmd: "explode", amount: 0.1 },
  ArrowDown: { cmd: "explode", amount: -0.1 },
  "+": { cmd: "zoom", factor: 1.1 },
  "=": { cmd: "zoom", factor: 1.1 },
  "-": { cmd: "zoom", factor: 1 / 1.1 },
  "0": { cmd: "zoom_reset" },
  l: { cmd: "labels" },
  r: { cmd: "rotate" },
  g: { cmd: "hand_rotate" },
  p: { cmd: "pull" },
  Tab: { cmd: "cycle" },
  Escape: { cmd: "deselect" },
};

export function initInput(canvas) {
  window.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (key === "c") {
      if (state.camOn) stopCamera();
      else startCamera().catch(() => {});
      return;
    }
    const cmd = KEYS[key];
    if (cmd) {
      e.preventDefault();
      send(cmd);
    }
  });

  window.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (e.ctrlKey) send({ cmd: "zoom", factor: e.deltaY < 0 ? 1.08 : 1 / 1.08 });
    else send({ cmd: "explode", amount: e.deltaY < 0 ? 0.05 : -0.05 });
  }, { passive: false });

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    state.dragging = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY };
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!state.dragging) return;
    send({ cmd: "drag", dx: (e.clientX - state.dragging.x) * 0.01, dy: (e.clientY - state.dragging.y) * 0.01 });
    state.dragging.x = e.clientX;
    state.dragging.y = e.clientY;
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!state.dragging) return;
    const moved = Math.hypot(e.clientX - state.dragging.sx, e.clientY - state.dragging.sy);
    state.dragging = null;
    if (moved > 4 || !state.frame || !state.frame.centers) return;
    let best = -1, bestDist = 60;
    state.frame.centers.forEach((c, i) => {
      const [px, py] = project(c);
      const d = Math.hypot(px - e.clientX, py - e.clientY);
      if (d < bestDist) { best = i; bestDist = d; }
    });
    send(best >= 0 ? { cmd: "select", index: best } : { cmd: "deselect" });
  });
}