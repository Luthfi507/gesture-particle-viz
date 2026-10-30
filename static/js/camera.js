import { state } from "./state.js";
import { $, SEND_MS } from "./config.js";

let capCanvas, cctx;
let video;

function pump() {
  if (!state.camOn) return;
  setTimeout(pump, SEND_MS);
  const ws = state.ws;
  if (state.busy || !ws || ws.readyState !== 1 || ws.bufferedAmount > 262144 || video.readyState < 2) return;
  state.busy = true;
  cctx.save();
  cctx.translate(capCanvas.width, 0);
  cctx.scale(-1, 1);
  cctx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
  cctx.restore();
  capCanvas.toBlob(async (blob) => {
    try {
      if (blob && state.ws && state.ws.readyState === 1) {
        state.ws.send(await blob.arrayBuffer());
      }
    } finally {
      state.busy = false;
    }
  }, "image/jpeg", 0.65);
}

export async function startCamera() {
  video = $("video");
  capCanvas = $("cap");
  cctx = capCanvas.getContext("2d", { willReadFrequently: false });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  capCanvas.width = 320;
  capCanvas.height = Math.round((320 * video.videoHeight) / video.videoWidth) || 240;

  const wrap = $("camwrap");
  wrap.hidden = false;

  const skel = $("skel");
  skel.width = capCanvas.width;
  skel.height = capCanvas.height;

  state.camOn = true;
  pump();
}

export function stopCamera() {
  state.camOn = false;
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  $("camwrap").hidden = true;
}