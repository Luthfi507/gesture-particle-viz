# WonderSnap (browser + Python backend)

Same idea as [AkbarSheikh-debug/wondersnap](https://github.com/AkbarSheikh-debug/wondersnap), split differently:

| Where | What runs there |
| ----- | --------------- |
| **Browser** | Webcam capture, WebGL2 point rendering, labels, keyboard and mouse |
| **Python** | MediaPipe hand tracking, gesture recognition, state machine, model generation, particle physics |

This is a reimplementation from the original README, not a line-by-line port.

## Run

```
conda create -n wondersnap python=3.12
conda activate wondersnap
pip install -r requirements.txt
python server.py
```

Open <http://localhost:5173> in Chrome or Edge and click **Start with camera**
(or **Continue without camera** to use the keyboard).
The hand model (`assets/hand_landmarker.task`) is downloaded on first start.

Options: `python server.py --count 30000 --port 8080 --host 127.0.0.1`.
Page options: `?count=30000` (particles), `?gain=0.1` (brightness), `?trail=0.5` (trail fade).

The browser only allows camera access on `localhost` or `https`. Binding `--host 0.0.0.0`
to use it from another device needs https in front (a reverse proxy, for example), and exposes
your camera frames to the network, so keep the default unless you know why you need it.

## How the two halves talk (WebSocket `/ws`)

```
browser -> server   binary  JPEG camera frame (mirrored, ~320 px wide, ~30 per second)
                    text    {"cmd": "snap" | "form" | "open" | "next" | "explode" | ...}
                    text    {"type": "viewport", "w": 1280, "h": 720}

server  -> browser  text    init: particle count, camera constants, model and part names
                    binary  0x01 model data: colors (uint8 RGB) + part id per particle, sent when the model changes
                    binary  0x02 frame (30 per second): JSON state + int16 particle positions
```

Positions are quantised to int16 in the range -4..4 and uploaded straight to the GPU as normalised
`SHORT` attributes, so the browser does no per-particle work on the CPU. At 50k particles a frame is
about 300 KB, which is fine on localhost.

The server keeps only the newest camera frame (older ones are dropped), and runs vision and physics
in separate loops, so a slow detector never stalls the animation.

## Gestures

| Gesture | What it does |
| ------- | ------------ |
| Snap (thumb + middle finger) | Summon the particles, or dissolve the model |
| Fist | Form the model |
| Open hand | Wonders morph to the next one. Anatomy, biology and engines explode, and how wide you open your hand sets how far |
| Twist / raise hand | Turn and tilt the model |
| Point (hold ~0.35 s) | Select the nearest part, it glows and a card explains it |
| Pinch | Pull the selected part out, pinch again to put it back |
| Two hands | Move apart / together to zoom |
| Peace | Next model |

## Keyboard and mouse

| Key | Action |
| --- | ------ |
| `Space` | Snap |
| `F` / `O` / `V` | Form / open (or toggle explode) / next model |
| `<` `>` or `A` `D` | Previous / next model |
| `^` `v` or wheel | Explode amount |
| `+` `-` `0` or ctrl + wheel | Zoom |
| `L` | Part labels |
| `Tab` / `P` / `Esc` | Cycle part / pull part / deselect |
| `R` / `G` | Auto-rotate / hand rotation |
| `C` | Camera on/off |
| Drag / click | Rotate / select a part |

## Structure

```
server.py           aiohttp app: Session (vision loop, simulation loop, protocol)
static/index.html   WebGL2 renderer, camera capture, input
engines/            ParticleEngine (physics), camera.project (must match the vertex shader)
models/             MODELS dict, shape samplers, procedural model builders
tools/              gestures (poses, snap, debounce), flow (state machine, commands), hands (MediaPipe)
utils/              vector math, Hand dataclass
test.py             unit tests + WebSocket end-to-end tests with a fake tracker (python test.py)
```

## Ported vs not ported

Ported: hand tracking, pose classifiers, snap detector, debouncer, state machine, two-hand zoom,
point-to-pick, pinch-to-pull, explode, heartbeat, part labels, mouse drag/click, 6 of the 33 models
(Eiffel Tower, Great Pyramid, Burj Khalifa, Heart, DNA, Inline-4 engine).

Not ported: the other 27 models, quiz mode, voice commands, cut-away, video recording, demo mode,
gesture sensitivity setting, and the Playwright suite.