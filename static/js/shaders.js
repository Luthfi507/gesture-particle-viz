export const POINT_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aCol;
layout(location=2) in float aPart;
uniform float uYaw, uPitch, uZoom, uCam, uFocal, uSelected, uVis, uBase, uGain, uDpr;
uniform vec2 uRes;
out vec3 vCol;
out float vA;
void main() {
  vec3 v = aPos * 4.0;
  float cy = cos(uYaw), sy = sin(uYaw), cp = cos(uPitch), sp = sin(uPitch);
  vec3 a = vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
  v = vec3(a.x, cp * a.y - sp * a.z, sp * a.y + cp * a.z) * uZoom;
  float depth = max(uCam - v.z, 0.2);
  float s = uFocal * min(uRes.x, uRes.y) / depth;
  gl_Position = vec4(v.x * s / (uRes.x * 0.5), v.y * s / (uRes.y * 0.5), 0.0, 1.0);
  gl_PointSize = max(2.0, uBase * (uCam / depth)) * uDpr;
  float sel = 1.0 - step(0.5, abs(aPart - uSelected));
  vCol = mix(aCol, vec3(1.0), 0.65 * sel);
  float k = uCam / depth;
  vA = k * k * uVis * uGain;
}`;

export const POINT_FS = `#version 300 es
precision mediump float;
in vec3 vCol;
in float vA;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float a = exp(-dot(d, d) * 12.0);
  o = vec4(vCol * vA * a, 1.0);
}`;

export const FADE_VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const FADE_FS = `#version 300 es
precision mediump float;
uniform float uFade;
out vec4 o;
void main() { o = vec4(0.0, 0.0, 0.0, uFade); }`;