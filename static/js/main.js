import { state } from "./state.js";
import { $ } from "./config.js";
import { initGL, updateViewport, drawParticles } from "./gl.js";
import { initHud, drawHud } from "./hud.js";
import { connect, send } from "./websocket.js";
import { startCamera } from "./camera.js";
import { initInput } from "./input.js";

const glCanvas = $("gl");
const hudCanvas = $("hud");
const skelCanvas = $("skel");

function resize() {
  state.DPR = Math.min(window.devicePixelRatio || 1, 2);
  state.W = window.innerWidth;
  state.H = window.innerHeight;
  hudCanvas.width = glCanvas.width = Math.round(state.W * state.DPR);
  hudCanvas.height = glCanvas.height = Math.round(state.H * state.DPR);
  updateViewport();
  send({ type: "viewport", w: state.W, h: state.H });
}

function loop() {
  drawParticles();
  drawHud();
  requestAnimationFrame(loop);
}

function init() {
  try {
    initGL(glCanvas);
  } catch (err) {
    $("conn").textContent = err.message;
    return;
  }

  initHud(hudCanvas, skelCanvas);
  initInput(glCanvas);

  window.addEventListener("resize", resize);
  resize();

  $("withCam").disabled = true;
  $("withCam").onclick = async () => {
    try {
      await startCamera();
      $("start").hidden = true;
    } catch (err) {
      $("conn").textContent = "camera unavailable: " + err.message;
    }
  };
  $("noCam").onclick = () => {
    $("start").hidden = true;
  };

  connect();
  requestAnimationFrame(loop);
}

init();