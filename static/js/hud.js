import { state } from "./state.js";
import { $, HAND_CONNECTIONS } from "./config.js";
import { project } from "./projection.js";

let hctx, skCtx;

export function initHud(hudCanvas, skelCanvas) {
  hctx = hudCanvas.getContext("2d");
  skCtx = skelCanvas.getContext("2d");
}

export function drawHud() {
  hctx.setTransform(state.DPR, 0, 0, state.DPR, 0, 0);
  hctx.clearRect(0, 0, state.W, state.H);
  if (!state.frame || !state.meta) return;
  const model = state.meta.models[state.frame.model];

  if (state.frame.labels && state.frame.centers) {
    hctx.font = "12px system-ui, sans-serif";
    state.frame.centers.forEach((c, i) => {
      const [px, py] = project(c);
      const name = model.parts[i].name;
      const tx = px + (px < state.W / 2 ? 40 : -40 - hctx.measureText(name).width);
      hctx.strokeStyle = hctx.fillStyle = i === state.frame.selected ? "#fff" : "#bdbdbd";
      hctx.lineWidth = 1;
      hctx.beginPath();
      hctx.moveTo(px, py);
      hctx.lineTo(tx, py - 18);
      hctx.stroke();
      hctx.fillText(name, tx, py - 22);
    });
  }

  $("status").textContent =
    `${model.name} [${model.category}]   state=${state.frame.state}   pose=${state.frame.pose}   explode=${state.frame.explode.toFixed(2)}`;

  const card = $("card");
  if (state.frame.state === "formed" && state.frame.selected !== null && model.parts[state.frame.selected]) {
    const part = model.parts[state.frame.selected];
    card.innerHTML = "";
    const b = document.createElement("b");
    b.textContent = part.name;
    const s = document.createElement("span");
    s.textContent = part.info;
    card.append(b, s);
  } else {
    card.textContent = "";
  }

  const sk = skCtx.canvas;
  const capW = sk.width;
  const capH = sk.height;
  skCtx.clearRect(0, 0, capW, capH);
  if (!state.frame.hands) return;

  skCtx.strokeStyle = "#3f3";
  skCtx.lineWidth = 2;
  for (const hand of state.frame.hands) {
    for (const [a, b] of HAND_CONNECTIONS) {
      skCtx.beginPath();
      skCtx.moveTo(hand[a][0] * capW, hand[a][1] * capH);
      skCtx.lineTo(hand[b][0] * capW, hand[b][1] * capH);
      skCtx.stroke();
    }
  }
}