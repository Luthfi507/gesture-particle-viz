import { state } from "./state.js";
import { $, COUNT_PARAM } from "./config.js";
import { setupBuffers, uploadColors } from "./gl.js";

const decoder = new TextDecoder();

export function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const qs = COUNT_PARAM ? `?count=${encodeURIComponent(COUNT_PARAM)}` : "";
  const ws = new WebSocket(`${proto}://${location.host}/ws${qs}`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      state.meta = JSON.parse(ev.data);
      setupBuffers(state.meta.count);
      $("conn").textContent = "connected";
      $("withCam").disabled = false;
      ws.send(JSON.stringify({ type: "viewport", w: state.W, h: state.H }));
      return;
    }
    const buf = ev.data;
    const dv = new DataView(buf);
    const type = dv.getUint8(0);
    if (type === 1) {
      const n = state.meta.count;
      uploadColors(new Uint8Array(buf, 2, n * 3), new Uint8Array(buf, 2 + n * 3, n));
      state.colorsReady = true;
    } else if (type === 2) {
      const len = dv.getUint32(1, true);
      state.frame = JSON.parse(decoder.decode(new Uint8Array(buf, 5, len)));
      state.positions = new Int16Array(buf, 5 + len, state.meta.count * 3);
    }
  };

  ws.onclose = () => {
    $("conn").textContent = "disconnected, reload the page";
    $("start").hidden = false;
  };
}

export function send(msg) {
  const ws = state.ws;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg));
}