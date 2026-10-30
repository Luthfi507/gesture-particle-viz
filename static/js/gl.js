import { state } from "./state.js";
import { GAIN, TRAIL } from "./config.js";
import { POINT_VS, POINT_FS, FADE_VS, FADE_FS } from "./shaders.js";

let gl, pointProg, fadeProg, U = {}, vao, posBuf, colBuf, partBuf, count = 0;

export function getGL() {
  return gl;
}

export function getCount() {
  return count;
}

function compileProgram(vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

export function initGL(canvas) {
  gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false, alpha: false });
  if (!gl) throw new Error("WebGL2 is not available");
}

export function setupBuffers(n) {
  count = n;
  pointProg = compileProgram(POINT_VS, POINT_FS);
  fadeProg = compileProgram(FADE_VS, FADE_FS);
  for (const name of ["uYaw", "uPitch", "uZoom", "uCam", "uFocal", "uSelected", "uVis", "uBase", "uGain", "uDpr", "uRes"]) {
    U[name] = gl.getUniformLocation(pointProg, name);
  }
  U.uFade = gl.getUniformLocation(fadeProg, "uFade");

  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, n * 3 * 2, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.SHORT, true, 0, 0);

  colBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.bufferData(gl.ARRAY_BUFFER, n * 3, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);

  partBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, partBuf);
  gl.bufferData(gl.ARRAY_BUFFER, n, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 1, gl.UNSIGNED_BYTE, false, 0, 0);
}

export function uploadColors(colData, partData) {
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, colData);
  gl.bindBuffer(gl.ARRAY_BUFFER, partBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, partData);
}

export function uploadPositions(buf) {
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, buf);
}

export function drawParticles() {
  if (!fadeProg) return;
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.useProgram(fadeProg);
  gl.uniform1f(U.uFade, TRAIL);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  if (!state.frame || !state.colorsReady || !state.meta) return;
  gl.bindVertexArray(vao);
  if (state.positions) {
    uploadPositions(state.positions);
    state.positions = null;
  }
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(pointProg);
  const f = state.frame;
  gl.uniform1f(U.uYaw, f.yaw);
  gl.uniform1f(U.uPitch, f.pitch);
  gl.uniform1f(U.uZoom, f.zoom);
  gl.uniform1f(U.uCam, state.meta.cam);
  gl.uniform1f(U.uFocal, state.meta.focal);
  gl.uniform1f(U.uSelected, f.state === "formed" && f.selected !== null ? f.selected : -1);
  gl.uniform1f(U.uVis, f.vis);
  gl.uniform1f(U.uBase, 3.4);
  gl.uniform1f(U.uGain, GAIN);
  gl.uniform1f(U.uDpr, state.DPR);
  gl.uniform2f(U.uRes, state.W, state.H);
  gl.drawArrays(gl.POINTS, 0, count);
}

export function updateViewport() {
  gl.viewport(0, 0, state.W * state.DPR, state.H * state.DPR);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
}