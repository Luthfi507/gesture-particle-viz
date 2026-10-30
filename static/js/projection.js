import { state } from "./state.js";

export function project(v) {
  const f = state.frame;
  const cy = Math.cos(f.yaw), sy = Math.sin(f.yaw);
  const cp = Math.cos(f.pitch), sp = Math.sin(f.pitch);
  const ax = cy * v[0] + sy * v[2];
  const ay = v[1];
  const az = -sy * v[0] + cy * v[2];
  const x = ax * f.zoom;
  const y = (cp * ay - sp * az) * f.zoom;
  const z = (sp * ay + cp * az) * f.zoom;
  const depth = Math.max(state.meta.cam - z, 0.2);
  const s = state.meta.focal * Math.min(state.W, state.H) / depth;
  return [state.W / 2 + x * s, state.H / 2 - y * s];
}