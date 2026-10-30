import numpy as np
from .base import Part as HiPart, assemble
from .shapes import (Part, line, polyline, tri, quad, box_surface, box_edges, box_filled,
                     revolve, cylinder_s, ring, disc, tube_shape, param_surface,
                     torus_shape, helix_shape, gear_shape, ellipsoid, capsule, blob, dots,
                     transform, orient, _pick, flatten,
                     rot_x, rot_z, deg, TAU, sphere, vessel, path_tube, bone,
                     linspace, range_n, _spline, _norm)


_norm = _norm


def _plumb(pts, r, density=1):
    parts = [tube_shape(pts[i], pts[i + 1], r, r, density) for i in range(len(pts) - 1)]
    w = sum(p.weight for p in parts)
    return Part(w, _pick(parts))


def _plines(pts, jitter=0, weight=1, closed=False):
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    segs = n - 1
    diffs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    if closed:
        wrap = np.linalg.norm(pts[0] - pts[-1])
        lengths = np.append(lengths, wrap)
        segs = n

    def sample(k, rng):
        seg = rng.choice(segs, size=k, p=lengths / lengths.sum())
        t = rng.random(k)
        a = pts[seg]
        b = pts[(seg + 1) % n]
        p = a + (b - a) * t[:, None]
        return (p + rng.normal(0, max(jitter, 1e-9), size=(k, 3))).astype(np.float32)
    return Part(weight * lengths.sum(), sample)


def _add(a, b):
    return np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)


def _tube_path(pts, r, density=1):
    pts_arr = np.asarray(pts, dtype=np.float64)
    parts = [tube_shape(pts_arr[i], pts_arr[i + 1], r, r, density) for i in range(len(pts_arr) - 1)]
    w = sum(p.weight for p in parts)
    return Part(w, _pick(parts))


def _spline_tube(ctrl, r0, r1, density=1, per=8):
    pts = _spline(ctrl, per)
    parts = []
    for i in range(len(pts) - 1):
        ra = r0 + (r1 - r0) * i / (len(pts) - 1)
        rb = r0 + (r1 - r0) * (i + 1) / (len(pts) - 1)
        parts.append(tube_shape(pts[i], pts[i + 1], ra, rb, density))
    w = sum(p.weight for p in parts)
    return Part(w, _pick(parts))


def _sub(a, b):
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def _mul(a, s):
    return np.asarray(a, dtype=np.float64) * s


# ═══════════════════════════════════════════════════════════════════════════
#  1  Sports Car (cm scale, 440 x 185 x 130, wheelbase 260)
# ═══════════════════════════════════════════════════════════════════════════

def sports_car(n):
    wb = 260
    track = 156
    wheel_r = 32

    # ── Body lower ────────────────────────────────────────────────────
    body_low_fn = lambda u, v: [u, 38 + 15 * np.sin(np.pi * v / 185) * (1 - abs(u / 220) ** 3), v]
    body_low = HiPart("Body lower", param_surface(body_low_fn, -220, 220, -85, 85, density=4).sample,
                      (1.0, 0.2, 0.15), 5, "Red sculpted lower body panels", (0, -100, 0), 0.8)

    # ── Body upper ────────────────────────────────────────────────────
    body_upper_fn = lambda u, v: [u, 65 + 30 * np.exp(-(u / 160) ** 2) * (1 - abs(v / 80) ** 2), v]
    body_upper = HiPart("Body upper", param_surface(body_upper_fn, -215, 215, -78, 78, density=3).sample,
                        (1.0, 0.25, 0.15), 4, "Roof and greenhouse panels", (0, 100, 0), 0.7)

    # ── Hood ──────────────────────────────────────────────────────────
    hood_fn = lambda u, v: [u, 56 + 6 * np.sin(np.pi * (u + 60) / 120), v]
    hood = HiPart("Hood", param_surface(hood_fn, -200, -50, -73, 73, density=2).sample,
                  (1.0, 0.22, 0.12), 2, "Front bonnet panel", (0, 120, 0), 0.6)

    # ── Rear deck ─────────────────────────────────────────────────────
    rear_fn = lambda u, v: [u, 52 + 8 * np.sin(np.pi * (u - 60) / 80), v]
    rear_deck = HiPart("Rear deck", param_surface(rear_fn, 40, 215, -72, 72, density=1.5).sample,
                       (1.0, 0.28, 0.18), 1.5, "Rear trunk lid", (0, 80, 0), 0.6)

    # ── Windshield ────────────────────────────────────────────────────
    wind_fn = lambda u, v: [-40 + u * 70 / 100, 56 + u * 30 / 100, v]
    windshield = HiPart("Windshield", param_surface(wind_fn, 0, 100, -65, 65, density=1.5).sample,
                        (0.2, 0.5, 0.8), 1.2, "Raked front glass", (0, 140, 0), 0.5)

    # ── Rear window ───────────────────────────────────────────────────
    rear_win_fn = lambda u, v: [90 + u * 50 / 100, 56 + (100 - u) * 24 / 100, v]
    rear_win = HiPart("Rear window", param_surface(rear_win_fn, 0, 100, -58, 58, density=1).sample,
                      (0.15, 0.4, 0.7), 0.8, "Fastback rear glass", (0, 60, 0), 0.5)

    # ── Headlights ────────────────────────────────────────────────────
    hl_parts = []
    for side in [-1, 1]:
        hz = side * 60
        hl_parts.append(ellipsoid([-210, 45, hz], [10, 6, 4], 0.5))
        hl_parts.append(disc([-210, 45, hz], 0, 10, None, 0.3))
    headlights = HiPart("Headlights", _pick(hl_parts), (1.0, 0.9, 0.8), 1,
                        "Projector headlamps", (0, 200, 0), 0.5)

    # ── Tail lights ───────────────────────────────────────────────────
    tl_parts = []
    for side in [-1, 1]:
        tz = side * 60
        tl_parts.append(box_surface([210, 50, tz], [12, 10, 18]))
        tl_parts.append(line([205, 50, tz - 9], [205, 50, tz + 9], weight=0.05))
    taillights = HiPart("Tail lights", _pick(tl_parts), (1.0, 0.15, 0.1), 0.8,
                        "LED tail light clusters", (0, -80, 0), 0.5)

    # ── Grille ────────────────────────────────────────────────────────
    grille_parts = []
    grille_parts.append(box_surface([-220, 42, 0], [8, 20, 60]))
    for i in range(8):
        gz = -28 + i * 8
        grille_parts.append(line([-224, 36, gz], [-224, 48, gz], weight=0.04))
    grille = HiPart("Grille", _pick(grille_parts), (0.15, 0.15, 0.15), 0.6,
                    "Front grille with chrome slats", (0, 160, 0), 0.4)

    # ── Engine ────────────────────────────────────────────────────────
    eng_parts = []
    eng_parts.append(box_surface([-100, 55, 0], [60, 40, 50]))
    eng_parts.append(box_edges([-100, 55, 0], [60, 40, 50], weight=0.15))
    eng_parts.append(box_filled([-80, 65, 0], [30, 10, 15], weight=0.15))
    eng_parts.append(tube_shape([-80, 65, 12], [-80, 65, 30], 5, 5, 0.2))
    engine2 = HiPart("Engine", _pick(eng_parts), (0.7, 0.7, 0.75), 2,
                     "V8 engine block", (0, 180, 0), 0.5)

    # ── Gearbox ───────────────────────────────────────────────────────
    gb_parts = []
    gb_parts.append(box_surface([-40, 40, 0], [50, 35, 35]))
    gb_parts.append(box_edges([-40, 40, 0], [50, 35, 35], weight=0.1))
    gearbox = HiPart("Gearbox", _pick(gb_parts), (0.6, 0.6, 0.65), 1.5,
                     "Transmission housing", (0, 60, 0), 0.5)

    # ── Driveshaft ────────────────────────────────────────────────────
    drive_parts = []
    drive_parts.append(tube_shape([0, 30, 0], [120, 25, 0], 4, 4, 0.3))
    driveshaft = HiPart("Driveshaft", _pick(drive_parts), (0.4, 0.4, 0.4), 0.5,
                        "Central driveshaft to rear differential", (0, -50, 0), 0.5)

    # ── Exhaust ───────────────────────────────────────────────────────
    exh_parts = []
    for side in [-1, 1]:
        sz = side * 18
        exh_parts.append(_plumb([[0, 25, sz], [60, 20, sz], [130, 15, sz], [200, 10, sz]], 3, 0.3))
        exh_parts.append(torus_shape([215, 10, sz], 4, 2, None, 0.1))
    exhaust_car = HiPart("Exhaust", _pick(exh_parts), (0.5, 0.5, 0.55), 0.8,
                         "Stainless twin exhaust system", (0, -30, 0), 0.5)

    # ── Wheels ────────────────────────────────────────────────────────
    wheel_parts = []
    for ax in [-120, 140]:
        for side in [-1, 1]:
            wz = side * track / 2
            wy = 15
            wheel_parts.append(revolve([wy - 4, wy + 4], [wheel_r - 2, wheel_r - 2],
                                       density=0.5, center=[ax, 0, wz], a0=-np.pi / 2, a1=np.pi / 2))
            wheel_parts.append(revolve([wy - 4, wy + 4], [wheel_r, wheel_r],
                                       density=0.6, center=[ax, 0, wz],
                                       a0=np.pi / 2, a1=3 * np.pi / 2))
            for ri in range(3):
                r_r = wheel_r - 2 - ri * 8
                wheel_parts.append(ring([ax, wy, wz], r_r))
            for si in range(5):
                a = si * TAU / 5
                sx = ax + 6 * np.cos(a)
                sz = wz + 6 * np.sin(a)
                wheel_parts.append(line([ax, wy, wz], [sx, wy, sz], jitter=0.15, weight=0.04))
            wheel_parts.append(disc([ax, wy, wz], 4, 14, None, 0.3))
    wheels = HiPart("Wheels", _pick(wheel_parts), (0.15, 0.15, 0.15), 3,
                    "Alloy wheels with 5-spoke design", (0, -120, 0), 0.6)

    # ── Disc brakes ───────────────────────────────────────────────────
    brake_parts = []
    for ax in [-120, 140]:
        for side in [-1, 1]:
            wz = side * track / 2
            wy = 15
            brake_parts.append(disc([ax, wy, wz], 8, 24, None, 0.4))
            for i in range(8):
                a = i * TAU / 8
                dx = ax + 12 * np.cos(a)
                dz = wz + 12 * np.sin(a)
                brake_parts.append(line([ax, wy, wz], [dx, wy, dz], jitter=0.1, weight=0.02))
    brakes = HiPart("Disc brakes", _pick(brake_parts), (0.5, 0.5, 0.5), 1,
                    "Ventilated disc brakes on all corners", (0, -40, 0), 0.5)

    # ── Suspension ────────────────────────────────────────────────────
    susp_parts = []
    for ax in [-120, 140]:
        for side in [-1, 1]:
            wz = side * track / 2
            susp_parts.append(helix_shape([ax, 38, wz], 6, 20, 6, None, 0.3))
            susp_parts.append(line([ax, 30, wz], [ax, 20, wz], weight=0.05))
            susp_parts.append(line([ax - 15, 30, wz], [ax + 15, 30, wz], jitter=0.15, weight=0.04))
            susp_parts.append(line([ax, 30, wz], [ax, 30, wz + side * 30], jitter=0.15, weight=0.04))
    suspension = HiPart("Suspension", _pick(susp_parts), (0.4, 0.45, 0.5), 1.2,
                        "Coil-over double wishbone suspension", (0, 30, 0), 0.5)

    # ── Interior: seats ───────────────────────────────────────────────
    seat_parts = []
    for side in [-1, 1]:
        sz = side * 28
        for fwd in [True, False]:
            sx = 20 if fwd else 80
            seat_parts.append(blob([sx, 45, sz], [22, 8, 20], 0.5))
            seat_parts.append(blob([sx + 5, 58, sz], [20, 14, 18], 0.4))
    seats = HiPart("Seats", _pick(seat_parts), (0.2, 0.2, 0.22), 1.5,
                   "Leather bucket seats", (0, 90, 0), 0.5)

    # ── Dashboard ─────────────────────────────────────────────────────
    dash_parts = []
    dash_parts.append(box_surface([-50, 58, 0], [40, 12, 110]))
    for i in range(4):
        gx = -50
        gz = -35 + i * 23
        dash_parts.append(ring([gx, 62, gz], 8))
        dash_parts.append(line([gx, 62, gz - 8], [gx, 62, gz + 8], weight=0.03))
    dashboard = HiPart("Dashboard", _pick(dash_parts), (0.15, 0.15, 0.17), 1,
                       "Alcantara-wrapped dash with gauges", (0, 130, 0), 0.5)

    # ── Steering wheel ────────────────────────────────────────────────
    sw_parts = []
    sw_parts.append(torus_shape([-50, 62, 20], 16, 2, None, 0.3))
    sw_parts.append(tube_shape([-50, 62, 20], [-50, 68, 20], 2, 1.5, 0.1))
    sw_parts.append(disc([-50, 62, 20], 0, 6, None, 0.1))
    sw = HiPart("Steering wheel", _pick(sw_parts), (0.1, 0.1, 0.1), 0.4,
               "Alcantara steering wheel", (0, 110, 0), 0.4)

    # ── Side mirrors ──────────────────────────────────────────────────
    mirror_parts = []
    for side in [-1, 1]:
        mz = side * 75
        mirror_parts.append(ellipsoid([-60, 85, mz], [8, 4, 5], 0.3))
        mirror_parts.append(line([-60, 80, mz], [-60, 66, mz], weight=0.04))
    mirrors = HiPart("Side mirrors", _pick(mirror_parts), (1.0, 0.25, 0.15), 0.4,
                     "Aero side mirrors", (0, 150, 0), 0.6)

    # ── Spoiler ───────────────────────────────────────────────────────
    spoiler_parts = []
    spoiler_parts.append(quad([195, 68, -40], [205, 75, -40],
                              [205, 75, 40], [195, 68, 40]))
    for side in [-1, 1]:
        sz = side * 40
        spoiler_parts.append(_spline_tube([[195, 68, sz], [198, 60, sz], [200, 68, sz]], 1.5, 1, 0.2))
    spoiler = HiPart("Spoiler", _pick(spoiler_parts), (1.0, 0.2, 0.12), 0.5,
                     "Decklid spoiler", (0, 100, 0), 0.5)

    # ── Doors ─────────────────────────────────────────────────────────
    door_parts = []
    for side in [-1, 1]:
        dz = side * 75
        door_parts.append(line([-70, 40, dz], [50, 40, dz], weight=0.06))
        door_parts.append(line([-70, 40, dz], [-70, 75, dz], weight=0.04))
        door_parts.append(line([50, 40, dz], [50, 75, dz], weight=0.04))
        door_parts.append(line([-70, 75, dz], [50, 75, dz], weight=0.06))
        door_parts.append(line([-20, 75, dz], [-20, 40, dz], weight=0.04))
    doors = HiPart("Doors", _pick(door_parts), (0.8, 0.1, 0.08), 0.7,
                   "Scissor door cut lines", (0, 50, 0), 0.6)

    # ── Front splitter ────────────────────────────────────────────────
    split_parts = []
    split_parts.append(quad([-220, 20, -85], [-180, 20, -85],
                            [-180, 20, 85], [-220, 20, 85]))
    splitter = HiPart("Front splitter", _pick(split_parts), (0.1, 0.1, 0.1), 0.3,
                      "Carbon fiber front splitter", (0, 0, 0), 0.5)

    parts = [body_low, body_upper, hood, rear_deck, windshield, rear_win, headlights, taillights,
             grille, engine2, gearbox, driveshaft, exhaust_car, wheels, brakes, suspension,
             seats, dashboard, sw, mirrors, spoiler, doors, splitter]
    return assemble("Sports Car", "vehicle", parts, n, seed=30, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  2  Motorcycle (cm scale, V-twin, chain drive)
# ═══════════════════════════════════════════════════════════════════════════

def motorcycle(n):
    wb = 145
    wheel_r = 32

    # ── Frame ─────────────────────────────────────────────────────────
    frame_pts = [[-40, 15, 0], [-20, 55, 0], [10, 70, 0], [40, 55, 0], [60, 15, 0], [30, -10, 0]]
    frame_parts = []
    frame_parts.append(_spline_tube(frame_pts, 3, 2.5, 0.5))
    frame_parts.append(line([-40, 15, 0], [-40, 15, 12], weight=0.03))
    frame_parts.append(line([-40, 15, 0], [-40, 15, -12], weight=0.03))
    frame_parts.append(line([60, 15, 0], [60, 15, 12], weight=0.03))
    frame_parts.append(line([60, 15, 0], [60, 15, -12], weight=0.03))
    frame_parts.append(_tube_path([[-20, 55, 0], [-30, 40, 0], [-40, 15, 0]], 3, 0.4))
    frame_parts.append(_tube_path([[40, 55, 0], [50, 30, 0], [60, 15, 0]], 3, 0.4))
    for i in range(6):
        a = i * TAU / 6
        tx = -40 + 18 * np.cos(a)
        ty = 15 + 18 * np.sin(a)
        frame_parts.append(line([-40, 15, 0], [tx, ty, 0], weight=0.03))
    frame_moto = HiPart("Frame", _pick(frame_parts), (0.7, 0.15, 0.12), 2,
                        "Trellis steel tube frame", (0, 0, 0), 0.3)

    # ── Fork ──────────────────────────────────────────────────────────
    fork_parts = []
    for side in [-1, 1]:
        fz = side * 10
        fork_parts.append(tube_shape([-60, 60, fz], [-80, 15, fz], 5, 5, 0.4))
        fork_parts.append(tube_shape([-80, 15, fz], [-90, -10, fz], 4.5, 3.5, 0.3))
        fork_parts.append(ring([-80, 40, fz], 6))
    fork_parts.append(_spline_tube([[-90, 60, 0], [-60, 70, 0], [-40, 60, 0]], 2, 1.5, 0.1))
    fork = HiPart("Fork", _pick(fork_parts), (0.65, 0.65, 0.68), 1.2,
                  "USD telescopic fork assembly", (0, -50, 0), 0.5)

    # ── Handlebar ─────────────────────────────────────────────────────
    bar_parts = []
    bar_parts.append(_spline_tube([[-40, 60, -20], [-30, 63, -25], [-20, 66, -20]], 2, 1.5, 0.2))
    bar_parts.append(_spline_tube([[-40, 60, 20], [-30, 63, 25], [-20, 66, 20]], 2, 1.5, 0.2))
    bar_parts.append(_tube_path([[-20, 66, -20], [-10, 67, 0], [-20, 66, 20]], 2, 0.2))
    bar = HiPart("Handlebar", _pick(bar_parts), (0.15, 0.15, 0.15), 0.5,
                 "Clip-on handlebars", (0, 60, 0), 0.4)

    # ── Engine ────────────────────────────────────────────────────────
    moto_eng_parts = []
    moto_eng_parts.append(blob([10, 40, 0], [35, 28, 20], 0.8))
    moto_eng_parts.append(blob([5, 25, 12], [18, 15, 10], 0.4))
    moto_eng_parts.append(blob([5, 25, -12], [18, 15, 10], 0.4))
    moto_eng_parts.append(ring([25, 38, 0], 12))
    moto_eng_parts.append(line([25, 38, 0], [25, 38, 3], weight=0.02))
    moto_eng_parts.append(line([25, 38, 0], [25, 38, -3], weight=0.02))
    moto_eng_parts.append(box_filled([10, 55, 0], [20, 8, 10], weight=0.2))
    for side in [-1, 1]:
        sz = side * 14
        moto_eng_parts.append(_plumb([[10, 55, sz], [0, 45, sz], [-10, 30, sz]],
                                   3, 0.15))
    engine_moto = HiPart("Engine", _pick(moto_eng_parts), (0.65, 0.65, 0.7), 2,
                         "90-degree V-twin engine", (0, -30, 0), 0.6)

    # ── Exhaust ───────────────────────────────────────────────────────
    moto_exh_parts = []
    for side in [-1, 1]:
        sz = side * 16
        moto_exh_parts.append(_plumb([[5, 25, sz], [20, 12, sz], [70, 8, sz]], 3.5, 0.3))
        moto_exh_parts.append(torus_shape([72, 8, sz], 4, 2, None, 0.1))
    moto_exh = HiPart("Exhaust", _pick(moto_exh_parts), (0.55, 0.55, 0.58), 0.8,
                      "Dual under-seat exhaust pipes", (0, 20, 0), 0.5)

    # ── Chain ─────────────────────────────────────────────────────────
    chain_parts = []
    chain_pts_upper = [[5, 11, -8], [30, 11, -8], [60, 16, -8]]
    chain_pts_lower = [[5, 11, -8], [30, 8, -8], [60, 12, -8]]
    for pts in [chain_pts_upper, chain_pts_lower]:
        for i in range(len(pts) - 1):
            for j in range(15):
                t = j / 15
                p = [pts[i][k] + (pts[i + 1][k] - pts[i][k]) * t for k in range(3)]
                chain_parts.append(line(p, [p[0] + 0.5, p[1], p[2]], jitter=0.1, weight=0.02))
    chain = HiPart("Chain", _pick(chain_parts), (0.3, 0.3, 0.3), 0.5,
                   "520 O-ring drive chain", (0, 0, 0), 0.7)

    # ── Sprockets ─────────────────────────────────────────────────────
    sprocket_parts = []
    sprocket_parts.append(gear_shape([5, 13, -8], 10, 16, None, 4, 3, None, 0.3))
    sprocket_parts.append(gear_shape([62, 16, -8], 15, 32, None, 4, 3, None, 0.4))
    sprockets = HiPart("Sprockets", _pick(sprocket_parts), (0.3, 0.3, 0.3), 0.6,
                       "Front and rear sprockets with chain", (0, -20, 0), 0.5)

    # ── Front wheel ───────────────────────────────────────────────────
    fw_parts = []
    fw_parts.append(revolve([-10, 10], [wheel_r - 1, wheel_r - 1], density=0.4,
                            center=[-90, 15, 0], a0=-np.pi / 2, a1=np.pi / 2))
    fw_parts.append(revolve([-10, 10], [wheel_r + 1, wheel_r + 1], density=0.5,
                            center=[-90, 15, 0], a0=np.pi / 2, a1=3 * np.pi / 2))
    for ri in range(2):
        r_r = wheel_r - 2 - ri * 12
        fw_parts.append(ring([-90, 0, 0], r_r))
    for si in range(6):
        a = si * TAU / 6
        fx = -90 + 6 * np.cos(a)
        fz = 6 * np.sin(a)
        fw_parts.append(line([-90, 0, 0], [fx, 0, fz], jitter=0.1, weight=0.03))
    fw_parts.append(disc([-90, 0, 0], 4, 12, None, 0.2))
    fwheel = HiPart("Front wheel", _pick(fw_parts), (0.15, 0.15, 0.15), 1,
                    "Cast alloy front wheel", (0, -80, 0), 0.5)

    # ── Rear wheel ────────────────────────────────────────────────────
    rw_parts = []
    rw_parts.append(revolve([-12, 12], [wheel_r - 1, wheel_r - 1], density=0.4,
                            center=[60, 15, 0], a0=-np.pi / 2, a1=np.pi / 2))
    rw_parts.append(revolve([-12, 12], [wheel_r, wheel_r], density=0.5,
                            center=[60, 15, 0], a0=np.pi / 2, a1=3 * np.pi / 2))
    for ri in range(2):
        r_r = wheel_r - 2 - ri * 12
        rw_parts.append(ring([60, 0, 0], r_r))
    for si in range(6):
        a = si * TAU / 6
        rx = 60 + 6 * np.cos(a)
        rz = 6 * np.sin(a)
        rw_parts.append(line([60, 0, 0], [rx, 0, rz], jitter=0.1, weight=0.03))
    rw_parts.append(disc([60, 15, 0], 4, 14, None, 0.3))
    rwheel = HiPart("Rear wheel", _pick(rw_parts), (0.15, 0.15, 0.15), 1,
                    "Cast alloy rear wheel", (0, 80, 0), 0.5)

    # ── Swingarm ──────────────────────────────────────────────────────
    swing_parts = []
    for side in [-1, 1]:
        sz = side * 8
        swing_parts.append(_spline_tube([[10, 30, sz], [35, 25, sz], [60, 18, sz]], 2.5, 2, 0.3))
    swingarm = HiPart("Swingarm", _pick(swing_parts), (0.55, 0.55, 0.58), 0.8,
                      "Aluminum single-sided swingarm", (0, 30, 0), 0.5)

    # ── Shock ─────────────────────────────────────────────────────────
    shock_parts = []
    shock_parts.append(helix_shape([30, 35, 0], 4.5, 18, 8, None, 0.2))
    shock_parts.append(tube_shape([30, 42, 0], [30, 55, 0], 3, 2.5, 0.15))
    shock_parts.append(tube_shape([30, 28, 0], [30, 18, 0], 3, 2.5, 0.15))
    shock = HiPart("Shock", _pick(shock_parts), (0.9, 0.2, 0.15), 0.5,
                   "Monoshock rear suspension", (0, 50, 0), 0.4)

    # ── Seat ──────────────────────────────────────────────────────────
    seat_parts = []
    seat_parts.append(blob([20, 60, 0], [30, 8, 15], 0.4))
    seat_bike = HiPart("Seat", _pick(seat_parts), (0.1, 0.1, 0.1), 0.5,
                       "Rider seat", (0, 70, 0), 0.5)

    # ── Fuel tank ─────────────────────────────────────────────────────
    tank_parts = []
    tank_parts.append(blob([-10, 58, 0], [25, 12, 16], 0.6))
    tank_parts.append(line([-20, 60, 0], [5, 60, 0], jitter=0.2, weight=0.03))
    tank = HiPart("Fuel tank", _pick(tank_parts), (0.9, 0.15, 0.05), 0.6,
                  "Sculpted fuel tank", (0, 50, 0), 0.5)

    # ── Brake rotors ──────────────────────────────────────────────────
    mbrake_parts = []
    for ax in [-90, 60]:
        for side in [-1, 1]:
            bz = side * 6
            mbrake_parts.append(disc([ax, 15, bz], 7, 24, None, 0.3))
            for i in range(6):
                a = i * TAU / 6
                dx = ax + 12 * np.cos(a)
                dz = bz + 12 * np.sin(a)
                mbrake_parts.append(line([ax, 15, bz], [dx, 15, dz], jitter=0.1, weight=0.02))
    mbrakes = HiPart("Brake rotors", _pick(mbrake_parts), (0.5, 0.5, 0.5), 0.6,
                     "Floating disc brakes front and rear", (0, 0, 0), 0.5)

    parts = [frame_moto, fork, bar, engine_moto, moto_exh, chain, sprockets,
             fwheel, rwheel, swingarm, shock, seat_bike, tank, mbrakes]
    return assemble("Motorcycle", "vehicle", parts, n, seed=31, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  3  Airliner (m scale)
# ═══════════════════════════════════════════════════════════════════════════

def airliner(n):
    fuse_len = 38
    fuse_r = 2.1
    UPPER = lambda a: np.pi / 2 <= a <= 3 * np.pi / 2
    LOWER = lambda a: -np.pi / 2 <= a <= np.pi / 2

    def _revX(xs, rs, density=1, a0=0, a1=TAU, keep=None):
        rev = revolve(xs, rs, density=density, a0=a0, a1=a1, keep=keep)
        return orient(rev, [1, 0, 0], [0, 0, 0])[0]

    def _ringX(x, r, density=1, a0=0, a1=TAU):
        rng = ring([0, 0, 0], r, density=density, a0=a0, a1=a1)
        return orient(rng, [1, 0, 0], [x, 0, 0])[0]

    # ── Fuselage (upper) ──────────────────────────────────────────────
    fuse_up_parts = []
    fuse_up_parts.append(_revX([0, fuse_len], [fuse_r, fuse_r], density=1.2, a0=np.pi / 2, a1=3 * np.pi / 2))
    for i in range(12):
        xi = i * fuse_len / 12
        fuse_up_parts.append(_ringX(xi, fuse_r, 0.3, a0=np.pi / 2, a1=3 * np.pi / 2))
    fuse_u = HiPart("Fuselage (upper)", _pick(fuse_up_parts), (0.6, 0.8, 1.0), 3,
                    "Aluminum fuselage upper shell", (0, -100, 0), 0.4)

    # ── Fuselage (lower) ──────────────────────────────────────────────
    fuse_lo_parts = []
    fuse_lo_parts.append(_revX([0, fuse_len], [fuse_r, fuse_r], density=1.2, a0=-np.pi / 2, a1=np.pi / 2))
    for i in range(12):
        xi = i * fuse_len / 12
        fuse_lo_parts.append(_ringX(xi, fuse_r, 0.3, a0=-np.pi / 2, a1=np.pi / 2))
    fuse_l = HiPart("Fuselage (lower)", _pick(fuse_lo_parts), (0.55, 0.75, 0.95), 3,
                    "", (0, 100, 0), 0.4)

    # ── Cabin boxes ───────────────────────────────────────────────────
    cabin_parts = []
    for ri in range(20):
        xi = 3 + ri * 1.6
        for side in [-1, 1]:
            cz = side * 1.3
            cabin_parts.append(box_surface([xi, cz, 0], [1.2, 0.8, 0.3]))
            win_fn = lambda u, v: [xi, cz + side * 0.3 * (1 - abs(v) / 0.2), v]
            cabin_parts.append(param_surface(win_fn, -0.6, 0.6, -0.15, 0.15, density=0.1))
    cabin = HiPart("Cabin boxes", _pick(cabin_parts), (0.3, 0.6, 0.85), 1.5,
                   "Passenger cabin with window rows", (0, 0, 0), 0.7)

    # ── Wings ─────────────────────────────────────────────────────────
    wing_fn = lambda u, v: [14 + u * 0.3, v * 12, 2 * np.exp(-((u - 6) / 4) ** 2) * (1 - abs(v / 12) ** 4) - 0.2]
    wing_parts = []
    wing_parts.append(param_surface(wing_fn, 0, 12, -1, 1, density=3))
    wing_parts.append(line([14, -12, 0], [18, 12, 0], weight=0.08))
    wing_parts.append(line([14, -12, 0], [14, 12, 0], weight=0.1))
    wing = HiPart("Wings", _pick(wing_parts), (0.65, 0.85, 1.0), 2.5,
                  "Swept main wings with aerodynamic profile", (0, 30, 0), 0.6)

    # ── Flaps ─────────────────────────────────────────────────────────
    flap_fn_R = lambda u, v: [22 + u * 3, v * 8, 0.5 * np.sin(np.pi * u / 3) * (1 - abs(v / 8) ** 2)]
    flap_parts = []
    flap_parts.append(param_surface(flap_fn_R, 0, 1, -1, 1, density=0.8))
    sup_fn = lambda u, v: [22 + u * 3, v * 8, 0.5 * np.sin(np.pi * u / 3) * (1 - abs(v / 8) ** 2)]
    flap_parts.append(param_surface(sup_fn, 0, 1, -1, 1, density=0.8))
    flaps = HiPart("Flaps", _pick(flap_parts), (0.6, 0.75, 0.9), 1,
                   "Trailing edge flaps and slats", (0, 20, 0), 0.6)

    # ── Turbofan engines (2) ──────────────────────────────────────────
    eng_plane_parts = []
    for side in [-1, 1]:
        ey = side * 6.5
        eng_plane_parts.extend(orient(revolve([-1, 1], [1.2, 1.2], density=0.5),
                                      [1, 0, 0], [13, ey, 0]))
        eng_fan = orient(revolve([-0.1, 0.1], [0.3, 1.15], density=0.4),
                         [1, 0, 0], [11.5, ey, 0])
        eng_plane_parts.extend(eng_fan)
        eng_plane_parts.extend(orient(ring([0, 0, 0], 1.2), [1, 0, 0], [12, ey, 0]))
        eng_plane_parts.extend(orient(ring([0, 0, 0], 1.2), [1, 0, 0], [14, ey, 0]))
        for i in range(22):
            a = i * TAU / 22
            ex = 11.5
            ey2 = ey + 1.15 * np.cos(a)
            ez = 1.15 * np.sin(a)
            eng_plane_parts.append(line([ex, ey2, ez], [12.5, ey2 * 0.5, ez * 0.5], weight=0.02))
    engines_plane = HiPart("Turbofan engines", _pick(eng_plane_parts), (0.5, 0.7, 0.85), 2,
                           "Twin high-bypass turbofan engines", (0, 50, 0), 0.5)

    # ── Tail fin ───────────────────────────────────────────────────────
    tail_fn = lambda u, v: [32 + u * 5, 0, v * 4]
    tail_parts = []
    tail_parts.append(param_surface(tail_fn, 0, 1, 0, 1, density=1.2))
    tail = HiPart("Tail fin", _pick(tail_parts), (0.5, 0.7, 1.0), 1,
                  "Vertical stabilizer with airline livery", (0, 0, 0), 0.5)

    # ── Stabilizers ───────────────────────────────────────────────────
    stab_fn = lambda u, v: [30 + u * 3, v * 4.5, 0.3 * np.sin(np.pi * u / 3) * (1 - abs(v / 4.5) ** 3)]
    stab_parts = []
    stab_parts.append(param_surface(stab_fn, 0, 1, -1, 1, density=1.5))
    stabs = HiPart("Stabilizers", _pick(stab_parts), (0.6, 0.8, 1.0), 1,
                   "Horizontal stabilizers with elevators", (0, 15, 0), 0.6)

    # ── Nose cone ─────────────────────────────────────────────────────
    nose_parts = []
    nose_rev = revolve([-2, 0], [0, fuse_r], density=0.4)
    nose_parts.extend(orient(nose_rev, [1, 0, 0], [0, 0, 0]))
    nose = HiPart("Nose cone", _pick(nose_parts), (0.65, 0.85, 1.0), 0.6,
                  "Radome nose cone", (0, -150, 0), 0.4)

    # ── Cockpit windows ───────────────────────────────────────────────
    cockpit_fn = lambda u, v: [2 + u * 3, 0, v * 1.8]
    cockpit_parts = []
    cockpit_parts.append(param_surface(cockpit_fn, 0, 1, -1, 1, density=0.5))
    cockpit = HiPart("Cockpit windows", _pick(cockpit_parts), (0.2, 0.5, 0.8), 0.5,
                     "Flight deck cockpit windows", (0, -120, 0), 0.4)

    # ── Landing gear ──────────────────────────────────────────────────
    gear_parts = []
    for side in [-1, 1]:
        gz = side * 2
        gear_parts.append(line([10, 0, gz], [10, -3, gz], weight=0.1))
        gear_parts.append(line([18, 0, gz], [18, -3, gz], weight=0.1))
        gear_parts.append(line([10, -3, gz], [18, -3, gz], weight=0.06))
        for axle in [14]:
            gear_parts.append(cylinder_s([axle, -3.2, gz], 0.6, 0.3, axis=2, cap=True))
            gear_parts.append(line([axle, -3.2, gz - 0.6], [axle, -3.2, gz + 0.6], weight=0.04))
    nose_gear = HiPart("Landing gear", _pick(gear_parts), (0.3, 0.35, 0.4), 0.8,
                       "Tricycle retractable landing gear", (0, -60, 0), 0.5)

    # ── APU exhaust ───────────────────────────────────────────────────
    apu_parts = []
    apu_parts.append(_revX([36, 37], [0.4, 0.15], density=0.2))
    apu = HiPart("APU exhaust", _pick(apu_parts), (0.4, 0.5, 0.6), 0.2,
                 "Auxiliary power unit exhaust", (0, 100, 0), 0.3)

    parts = [fuse_u, fuse_l, cabin, wing, flaps, engines_plane, tail, stabs,
             nose, cockpit, nose_gear, apu]
    return assemble("Airliner", "vehicle", parts, n, seed=32, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  4  Saturn V Rocket (m scale)
# ═══════════════════════════════════════════════════════════════════════════

def saturn_v(n):
    stage1_h = 42
    stage2_h = 25
    stage3_h = 18
    stage1_r = 5
    stage2_r = 5
    stage3_r = 3.3
    lma_r = 2.2

    def _revY(ys, rs, density=1, a0=0, a1=TAU):
        return revolve(ys, rs, density=density, a0=a0, a1=a1)

    # ── S-IC First stage ──────────────────────────────────────────────
    s1_parts = []
    s1_parts.append(_revY([0, stage1_h], [stage1_r, stage1_r], density=1.5))
    s1_parts.append(_revY([0, stage1_h], [3.9, 3.9], density=0.6))
    s1_black_ys = []
    s1_black_rs = []
    for i in range(8):
        y0 = i * stage1_h / 8
        y1 = (i + 1) * stage1_h / 8
        if i % 2 == 0:
            s1_parts.append(_revY([y0, y1], [stage1_r + 0.1, stage1_r + 0.1], density=0.2))
    for i in range(10):
        yi = i * stage1_h / 10
        s1_parts.append(ring([0, yi, 0], stage1_r))
    s1 = HiPart("S-IC First stage", _pick(s1_parts), (0.95, 0.95, 1.0), 3,
                "5x F-1 engines, kerosene/LOX 33 MN thrust", (0, -200, 0), 0.3)

    # ── F-1 engines ───────────────────────────────────────────────────
    f1_parts = []
    for i in range(4):
        a = i * TAU / 4 + np.pi / 8
        ex = 2.5 * np.cos(a)
        ez = 2.5 * np.sin(a)
        f1_parts.append(_revY([-4, 0], [1.8, 0.6], density=0.4, a0=a - 0.3, a1=a + 0.3))
        f1_parts.append(torus_shape([0, -1.5, 0], 1.8, 0.15, None, 0.2))
    f1_parts.append(_revY([-4, 0], [1.8, 0.6], density=0.4, a0=0, a1=TAU))
    f1_parts.append(_revY([0, 3], [0.6, 2.8], density=0.3))
    f1_engines = HiPart("F-1 engines", _pick(f1_parts), (0.6, 0.65, 0.7), 2,
                        "Five F-1 kerosene engines with nozzle extensions", (0, -300, 0), 0.5)

    # ── Interstage ring ───────────────────────────────────────────────
    inter_parts = []
    inter_parts.append(_revY([stage1_h - 2, stage1_h + 2], [stage1_r, stage1_r], density=0.5))
    inter_parts.append(ring([0, stage1_h, 0], stage1_r))
    inter_parts.append(ring([0, stage1_h, 0], stage2_r))
    interstage = HiPart("Interstage ring", _pick(inter_parts), (0.9, 0.9, 0.95), 0.8,
                        "Separation ring with ullage motors", (0, -50, 0), 0.4)

    # ── S-II Second stage ─────────────────────────────────────────────
    s2_parts = []
    s2_parts.append(_revY([stage1_h, stage1_h + stage2_h], [stage2_r, stage2_r], density=1.2))
    s2_parts.append(_revY([stage1_h, stage1_h + stage2_h], [3.9, 3.9], density=0.5))
    for i in range(8):
        yi = stage1_h + i * stage2_h / 8
        s2_parts.append(ring([0, yi, 0], stage2_r))
    s2 = HiPart("S-II Second stage", _pick(s2_parts), (0.95, 0.95, 1.0), 2,
                "5x J-2 engines, liquid hydrogen/LOX", (0, -150, 0), 0.4)

    # ── J-2 engines ───────────────────────────────────────────────────
    j2_parts = []
    for i in range(4):
        a = i * TAU / 4 + np.pi / 8
        j2_parts.append(_revY([stage1_h - 3, stage1_h], [1.5, 0.5], density=0.3,
                              a0=a - 0.3, a1=a + 0.3))
    j2_parts.append(_revY([stage1_h - 3, stage1_h], [1.5, 0.5], density=0.3, a0=0, a1=TAU))
    j2_parts.append(_revY([stage1_h, stage1_h + 2], [0.5, 2.2], density=0.25))
    j2_engines = HiPart("J-2 engines", _pick(j2_parts), (0.7, 0.7, 0.75), 1.5,
                        "Five J-2 hydrogen upper stage engines", (0, -200, 0), 0.5)

    # ── S-IVB Third stage ─────────────────────────────────────────────
    s3_parts = []
    s3_top = stage1_h + stage2_h + stage3_h
    s3_parts.append(_revY([stage1_h + stage2_h, s3_top], [stage3_r, stage3_r], density=1.0))
    for i in range(6):
        yi = stage1_h + stage2_h + i * stage3_h / 6
        s3_parts.append(ring([0, yi, 0], stage3_r))
    s3 = HiPart("S-IVB Third stage", _pick(s3_parts), (0.95, 0.95, 1.0), 1.5,
                "Single J-2 engine, restartable for TLI burn", (0, 0, 0), 0.4)

    # ── Instrument unit ───────────────────────────────────────────────
    iu_parts = []
    iu_parts.append(_revY([s3_top, s3_top + 1.5], [stage3_r, 3.3], density=0.4))
    iu_parts.append(ring([0, s3_top + 0.5, 0], stage3_r))
    iu = HiPart("Instrument unit", _pick(iu_parts), (0.85, 0.85, 0.9), 0.5,
                "Guidance computer and telemetry ring", (0, 0, 0), 0.4)

    # ── Lunar module adapter ──────────────────────────────────────────
    lma_parts = []
    lma_top = s3_top + 6
    lma_parts.append(_revY([s3_top + 1.5, lma_top], [3.3, 3.3], density=0.8, a0=np.pi / 2, a1=3 * np.pi / 2))
    lma_parts.append(_revY([s3_top + 1.5, lma_top], [3.3, 3.3], density=0.8, a0=-np.pi / 2, a1=np.pi / 2))
    for i in range(3):
        yi = s3_top + 1.5 + i * 2
        lma_parts.append(ring([0, yi, 0], 3.3))
    lma = HiPart("Lunar module adapter", _pick(lma_parts), (0.9, 0.9, 0.95), 0.8,
                 "Houses the lunar module during ascent", (0, 0, 0), 0.5)

    # ── Lunar module ──────────────────────────────────────────────────
    lm_parts = []
    lm_parts.append(box_surface([0, lma_top + 0.5, 0], [3, 3, 3]))
    lm_parts.append(box_edges([0, lma_top + 0.5, 0], [3, 3, 3], weight=0.15))
    for side in [-1, 1]:
        for fwd in [-1, 1]:
            lg_y = lma_top - 3
            lg_r = 3
            lm_parts.append(line([side * 1.2, lma_top, fwd * 1.2], [side * 2, lg_y, fwd * 2], weight=0.05))
            lm_parts.append(disc([side * 2, lg_y, fwd * 2], 0.3, 0.8, None, 0.15))
    lm = HiPart("Lunar module", _pick(lm_parts), (0.85, 0.8, 0.4), 1.5,
                "Apollo lunar module with descent stage and legs", (0, 0, 0), 0.6)

    # ── Command / Service module ───────────────────────────────────────
    csm_parts = []
    csm_base = lma_top + 4
    csm_parts.append(_revY([csm_base, csm_base + 7], [2, 2], density=0.8))
    csm_cone = revolve([csm_base + 7, csm_base + 10], [2, 0.6], density=0.5)
    csm_parts.append(csm_cone)
    for i in range(5):
        yi = csm_base + i * 1.5
        csm_parts.append(ring([0, yi, 0], 2))
    csm = HiPart("Command/Service module", _pick(csm_parts), (0.9, 0.9, 0.95), 1.5,
                 "Apollo CSM with heat shield and crew compartment", (0, 0, 0), 0.5)

    # ── Escape tower ──────────────────────────────────────────────────
    tower_parts = []
    tower_base = csm_base + 10
    tower_parts.append(tube_shape([0, tower_base, 0], [0, tower_base + 8, 0], 0.15, 0.15, 0.3))
    tower_parts.append(_revY([tower_base + 8, tower_base + 11], [0.15, 0.8], density=0.3))
    for i in range(4):
        a = i * TAU / 4
        tx = 0.8 * np.cos(a)
        tz = 0.8 * np.sin(a)
        tower_parts.append(line([0, tower_base + 3, 0], [tx, tower_base + 6, tz], weight=0.04))
    tower = HiPart("Escape tower", _pick(tower_parts), (1.0, 0.25, 0.15), 0.7,
                   "Launch escape system with solid rocket motors", (0, 100, 0), 0.4)

    # ── Fins ──────────────────────────────────────────────────────────
    fin_parts = []
    for i in range(4):
        a = i * TAU / 4
        fx = stage1_r * np.cos(a)
        fz = stage1_r * np.sin(a)
        fin_pts = [[fx * 0.8, 2, fz * 0.8], [fx * 1.5, 8, fz * 1.5], [fx * 1.8, 0, fz * 1.8]]
        fin_parts.append(tri(fin_pts[0], fin_pts[1], fin_pts[2]))
    fins = HiPart("Fins", _pick(fin_parts), (0.85, 0.85, 0.9), 0.6,
                  "Four aerodynamic stabilizer fins", (0, 0, 0), 0.6)

    # ── USA lettering (first stage) ───────────────────────────────────
    usa_pts = []
    for i in range(30):
        usa_pts.append([-1.5 + i * 0.1, 10, stage1_r + 0.05])
    usa = HiPart("USA lettering", _pick([_plines(usa_pts, jitter=0.2, weight=0.1)]),
                 (1.0, 0.1, 0.1), 0.3, "Painted USA on first stage", (0, 0, 0), 0.5)

    # ── Umbilical connections ─────────────────────────────────────────
    umb_parts = []
    umb_parts.append(line([stage3_r, stage1_h + stage2_h + 10, 0],
                          [stage3_r + 2, stage1_h + stage2_h + 5, 0], weight=0.05))
    umb_parts.append(line([stage3_r, stage1_h + stage2_h + 12, 0],
                          [stage3_r + 1.5, stage1_h + stage2_h + 3, 0], weight=0.04))
    umb = HiPart("Umbilical connections", _pick(umb_parts), (0.5, 0.5, 0.55), 0.3,
                 "External cabling and umbilical ports", (0, 0, 0), 0.7)

    parts = [s1, f1_engines, interstage, s2, j2_engines, s3, iu, lma, lm, csm, tower,
             fins, usa, umb]
    return assemble("Saturn V Rocket", "vehicle", parts, n, seed=33, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  5  Mechanical Wristwatch (40mm scale)
# ═══════════════════════════════════════════════════════════════════════════

def wristwatch(n):
    case_r = 20
    case_h = 4

    # ── Case ─────────────────────────────────────────────────────────
    case_parts = []
    case_rev = revolve([-case_h / 2, case_h / 2], [case_r, case_r], density=0.8)
    case_parts.extend(orient(case_rev, [0, 0, 1], [0, 0, 0]))
    for i in range(4):
        yi = -case_h / 2 + i * case_h / 4
        case_parts.append(ring([0, 0, yi], case_r))
    case_watch = HiPart("Case", _pick(case_parts), (0.85, 0.85, 0.85), 2,
                        "Stainless steel 316L case with polished finish", (0, 0, 100), 0.3)

    # ── Crystal ───────────────────────────────────────────────────────
    crystal_parts = []
    crystal_parts.append(disc([0, 0, case_h / 2 + 1], 0, case_r - 1.5, None, 0.5))
    crystal_parts.append(ring([0, 0, case_h / 2 + 1], case_r - 1.5))
    crystal = HiPart("Crystal", _pick(crystal_parts), (0.5, 0.7, 0.85), 0.8,
                     "Sapphire crystal with AR coating", (0, 0, 120), 0.2)

    # ── Bezel ─────────────────────────────────────────────────────────
    bezel_parts = []
    bezel_rev = revolve([case_h / 2, case_h / 2 + 2], [case_r, case_r + 1], density=0.6)
    bezel_parts.extend(orient(bezel_rev, [0, 0, 1], [0, 0, 0]))
    bezel_parts.append(ring([0, 0, case_h / 2 + 1], case_r + 1))
    for i in range(60):
        a = i * TAU / 60
        bx = (case_r + 1.5) * np.cos(a)
        bz = (case_r + 1.5) * np.sin(a)
        bezel_parts.append(line([bx, case_h / 2 + 1, bz], [bx, case_h / 2 + 3, bz], weight=0.02))
    bezel = HiPart("Bezel", _pick(bezel_parts), (0.9, 0.85, 0.8), 1.2,
                   "Fluted 18k gold bezel", (0, 0, 140), 0.3)

    # ── Dial ──────────────────────────────────────────────────────────
    dial_parts = []
    dial_parts.append(disc([0, 0, case_h / 2 - 0.5], 0, case_r - 3, None, 0.8))
    for i in range(12):
        a = i * TAU / 12
        hx = (case_r - 6) * np.cos(a)
        hz = (case_r - 6) * np.sin(a)
        dial_parts.append(line([hx * 0.9, 0, hz * 0.9], [hx, 0, hz], jitter=0.1, weight=0.02))
    for i in range(60):
        a = i * TAU / 60
        mx = (case_r - 5) * np.cos(a)
        mz = (case_r - 5) * np.sin(a)
        if i % 5 != 0:
            dial_parts.append(line([mx * 0.95, 0, mz * 0.95], [mx, 0, mz], weight=0.008))
    dial_s = HiPart("Dial", _pick(dial_parts), (0.15, 0.2, 0.35), 1,
                    "Sunburst blue dial with applied indices", (0, 0, -100), 0.3)

    # ── Hour hand ─────────────────────────────────────────────────────
    hh_parts = []
    hh_parts.append(line([0, 0, 0], [9, 0, 2], jitter=0.1, weight=0.08))
    hh_parts.append(line([0, 0, 0], [9, 0, -2], jitter=0.1, weight=0.08))
    hh_parts.append(line([9, 0, 2], [9, 0, -2], weight=0.03))
    hh_fn = lambda u, v: [u * 10, v * 3, 0.5 * max(0, 1 - abs(v / 3) - abs(u - 0.5) / 2)]
    hh_parts.append(param_surface(hh_fn, 0.1, 1, -1, 1, density=0.2))
    hour_hand = HiPart("Hour hand", _pick(hh_parts), (0.15, 0.18, 0.2), 0.5,
                       "Blued steel hour hand", (0, 0, -150), 0.4)

    # ── Minute hand ───────────────────────────────────────────────────
    mh_parts = []
    mh_parts.append(line([0, 0, 0], [13, 0, 1.5], jitter=0.1, weight=0.08))
    mh_parts.append(line([0, 0, 0], [13, 0, -1.5], jitter=0.1, weight=0.08))
    mh_fn = lambda u, v: [u * 13, v * 2, 0.4 * max(0, 1 - abs(v / 2) - abs(u - 0.5) / 2)]
    mh_parts.append(param_surface(mh_fn, 0.1, 1, -1, 1, density=0.2))
    minute_hand = HiPart("Minute hand", _pick(mh_parts), (0.2, 0.22, 0.25), 0.5,
                         "Blued steel minute hand", (0, 0, -120), 0.4)

    # ── Seconds hand ──────────────────────────────────────────────────
    sh_parts = []
    sh_parts.append(line([0, 0, 0], [15, 0, 0], jitter=0.08, weight=0.04))
    sh_parts.append(line([0, 0, 0], [-4, 0, 0], jitter=0.08, weight=0.02))
    sh_parts.append(disc([-3, 0, 0], 1, 3, None, 0.1))
    sh_parts.append(disc([15, 0, 0], 0.5, 1.5, None, 0.05))
    seconds_hand = HiPart("Seconds hand", _pick(sh_parts), (0.9, 0.15, 0.1), 0.3,
                          "Thin red seconds hand with counterweight", (0, 0, -100), 0.5)

    # ── Gear train ────────────────────────────────────────────────────
    gear_parts = []
    gear_positions = [(0, 0, -3), (5, 4, -3), (-4, 3, -3), (7, -3, -3), (-3, -5, -3)]
    sizes = [(6, 24), (5, 20), (4, 16), (3, 12), (2, 10)]
    for (gx, gy, gz), (or2, teeth) in zip(gear_positions, sizes):
        gear_parts.append(gear_shape([gx, gy, gz], or2 * 0.8, teeth, None, 2, 1.2, None, 0.2))
        gear_parts.append(disc([gx, gy, gz], 0, 2, None, 0.1))
    gear_train = HiPart("Gear train", _pick(gear_parts), (0.9, 0.85, 0.45), 1.5,
                        "Brass going train: center, third, seconds wheels with pinions", (0, 0, -80), 0.5)

    # ── Balance wheel ─────────────────────────────────────────────────
    bal_parts = []
    bal_parts.append(ring([-8, 6, -3], 5))
    bal_parts.append(ring([-8, 6, -3], 3))
    for i in range(3):
        a = i * TAU / 3
        bx2 = -8 + 4 * np.cos(a)
        by = 6 + 4 * np.sin(a)
        bal_parts.append(line([-8, 6, -3], [bx2, by, -3], weight=0.04))
    bal = HiPart("Balance wheel", _pick(bal_parts), (0.85, 0.8, 0.35), 1,
                 "Glucydur balance with regulating screws", (0, 0, 50), 0.5)

    # ── Mainspring barrel ─────────────────────────────────────────────
    ms_parts = []
    ms_rev = revolve([-2, 2], [7, 7], density=0.4)
    ms_parts.extend(orient(ms_rev, [0, 0, 1], [6, -5, -3]))
    ms_parts.append(ring([6, -5, -1], 7))
    for i in range(5):
        a = i * TAU / 5
        sx = 6 + 6 * np.cos(a)
        sy = -5 + 6 * np.sin(a)
        ms_parts.append(line([6, -5, 0], [sx, sy, 0], weight=0.02))
    mainspring = HiPart("Mainspring barrel", _pick(ms_parts), (0.85, 0.8, 0.5), 0.8,
                        "Mainspring barrel with 72-hour power reserve", (0, 0, -60), 0.5)

    # ── Bridges ───────────────────────────────────────────────────────
    bridge_parts = []
    for i, (bx2, by2) in enumerate([(0, 0), (5, 4), (-4, 3), (-3, -5)]):
        bridge_parts.append(capsule([bx2 - 4, by2 - 3, -3], [bx2 + 4, by2 + 3, -3], 1, 0.1))
        bridge_parts.append(line([bx2 - 3, by2 - 2, -3], [bx2 + 3, by2 + 2, -3], weight=0.03))
        bridge_parts.append(line([bx2 - 3, by2 + 2, -3], [bx2 + 3, by2 - 2, -3], weight=0.03))
    bridges = HiPart("Bridges", _pick(bridge_parts), (0.82, 0.78, 0.4), 0.8,
                     "Brass bridges with Geneva stripes", (0, 0, -40), 0.5)

    # ── Jewels ────────────────────────────────────────────────────────
    jewel_positions = [(0, 0, -3), (5, 4, -3), (-4, 3, -3), (7, -3, -3),
                       (-3, -5, -3), (-8, 6, -3), (6, -5, -3)]
    jewel_parts = []
    for jx, jy, jz in jewel_positions:
        jewel_parts.append(disc([jx, jy, jz - 0.1], 0.5, 1.8, None, 0.05))
        jewel_parts.append(ring([jx, jy, jz - 0.1], 1.8))
    jewels = HiPart("Jewels", _pick(jewel_parts), (0.8, 0.1, 0.2), 0.4,
                    "17 synthetic ruby jewels reducing friction at pivots", (0, 0, 60), 0.6)

    # ── Crown ────────────────────────────────────────────────────────
    crown_parts = []
    crown_rev = revolve([-1, 1], [2.5, 2.5], density=0.3)
    crown_parts.extend(orient(crown_rev, [1, 0, 0], [case_r + 3, 0, 0]))
    for i in range(30):
        a = i * TAU / 30
        cx = case_r + 3 + 2.5 * np.cos(a)
        cz = 2.5 * np.sin(a)
        crown_parts.append(line([case_r + 2, 0, 0], [cx, 0, cz], weight=0.01))
    crown = HiPart("Crown", _pick(crown_parts), (0.85, 0.85, 0.85), 0.5,
                   "Screw-down winding crown", (0, 80, 0), 0.5)

    # ── Rotor ─────────────────────────────────────────────────────────
    rotor_parts = []
    rotor_parts.append(disc([0, 0, -case_h / 2 - 1.5], 3, 15, None, 0.4))
    rotor_parts.append(line([0, 0, -case_h / 2 - 1.5], [12, 0, -case_h / 2 - 1.5], weight=0.06))
    rotor_parts.append(disc([12, 0, -case_h / 2 - 1.5], 1, 5, None, 0.2))
    rotor_parts.append(ring([0, 0, -case_h / 2 - 1.5], 15))
    rotor = HiPart("Rotor", _pick(rotor_parts), (0.88, 0.82, 0.35), 0.6,
                   "Skeletonized gold winding rotor", (0, 0, -80), 0.5)

    parts = [case_watch, crystal, bezel, dial_s, hour_hand, minute_hand, seconds_hand,
             gear_train, bal, mainspring, bridges, jewels, crown, rotor]
    return assemble("Mechanical Watch", "machine", parts, n, seed=34, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  6  EV Battery Pack (cm scale)
# ═══════════════════════════════════════════════════════════════════════════

def battery_pack(n):
    tray_w = 160
    tray_d = 115
    tray_h = 16
    cell_r = 1.05
    cell_h = 6.5
    mod_w = 36
    mod_d = 18
    mod_h = 8

    # ── Battery tray ──────────────────────────────────────────────────
    tray_parts = []
    tray_parts.append(box_surface([0, 0, 0], [tray_w, tray_h, tray_d]))
    tray_parts.append(box_edges([0, 0, 0], [tray_w, tray_h, tray_d], weight=0.3))
    for i in range(4):
        xi = -tray_w / 2 + 5 + i * (tray_w - 10) / 3
        tray_parts.append(line([xi, -tray_h / 2, -tray_d / 2],
                               [xi, -tray_h / 2, tray_d / 2], weight=0.05))
    tray = HiPart("Battery tray", _pick(tray_parts), (0.25, 0.3, 0.35), 3,
                  "Aluminum structural battery tray with crossmembers", (0, -80, 0), 0.3)

    # ── Cell modules (8 modules × 35 cells each) ──────────────────────
    module_parts = []
    for mi in range(8):
        col = mi % 4
        row2 = mi // 4
        mx = -tray_w / 2 + 20 + col * 38
        my = row2 * 26 - 25
        mz = 0
        module_parts.append(box_surface([mx, my, mz], [mod_w, mod_h, mod_d]))
        module_parts.append(box_edges([mx, my, mz], [mod_w, mod_h, mod_d], weight=0.1))
        for ci in range(35):
            ccol = ci % 7
            crow = ci // 7
            cx = mx - mod_w / 2 + 3 + ccol * 5
            cy = my
            cz = mz - mod_d / 2 + 3 + crow * 5
            module_parts.append(cylinder_s([cx, cy, cz], cell_r, cell_h, axis=1, cap=True))
            module_parts.append(ring([cx, cy, cz + cell_h / 2], cell_r))
            module_parts.append(ring([cx, cy, cz - cell_h / 2], cell_r))
    modules = HiPart("Cell modules", _pick(module_parts), (0.4, 0.5, 0.6), 4,
                     "8 modules with 35 cylindrical cells each (18650 format)", (0, 0, 0), 0.4)

    # ── Busbars ───────────────────────────────────────────────────────
    bus_parts = []
    for mi in range(8):
        col = mi % 4
        row2 = mi // 4
        mx = -tray_w / 2 + 20 + col * 38
        my = row2 * 26 - 25
        mz = 0
        for layer in [-1, 1]:
            bz = mz + layer * (mod_d / 2 + 0.5)
            bus_parts.append(quad([mx - mod_w / 2, my, bz], [mx + mod_w / 2, my, bz],
                                  [mx + mod_w / 2, my + 0.3, bz], [mx - mod_w / 2, my + 0.3, bz]))
    for mi in range(4):
        col = mi
        mx = -tray_w / 2 + 20 + col * 38
        mz = 0
        bus_parts.append(line([mx, -25, mz - mod_d / 2 - 0.5],
                              [mx, -25, mz + mod_d / 2 + 0.5], weight=0.05))
    for mi in range(4):
        col = mi
        mx = -tray_w / 2 + 20 + col * 38
        mz = 0
        bus_parts.append(line([mx, 1, mz - mod_d / 2 - 0.5],
                              [mx, 1, mz + mod_d / 2 + 0.5], weight=0.05))
    busbars = HiPart("Busbars", _pick(bus_parts), (0.85, 0.45, 0.15), 1.5,
                     "Copper busbar interconnects with cell-level fusing", (0, 30, 0), 0.5)

    # ── Cooling plate ─────────────────────────────────────────────────
    cool_parts = []
    cool_parts.append(quad([-tray_w / 2, -tray_h / 2, -tray_d / 2],
                           [tray_w / 2, -tray_h / 2, -tray_d / 2],
                           [tray_w / 2, -tray_h / 2, tray_d / 2],
                           [-tray_w / 2, -tray_h / 2, tray_d / 2]))
    for i in range(6):
        zi = -tray_d / 2 + 5 + i * (tray_d - 10) / 5
        cool_parts.append(line([-tray_w / 2, -tray_h / 2, zi],
                               [tray_w / 2, -tray_h / 2, zi], weight=0.04))
    cooling = HiPart("Cooling plate", _pick(cool_parts), (0.5, 0.6, 0.65), 1.5,
                     "Liquid-cooled thermal plate with serpentine channels", (0, -40, 0), 0.4)

    # ── BMS box ───────────────────────────────────────────────────────
    bms_parts = []
    bms_parts.append(box_surface([tray_w / 2 + 15, 5, -30], [20, 8, 25]))
    bms_parts.append(box_edges([tray_w / 2 + 15, 5, -30], [20, 8, 25], weight=0.1))
    for i in range(4):
        bz = -37 + i * 5
        bms_parts.append(line([tray_w / 2 + 25, 5, bz],
                              [tray_w / 2 + 25, 5, bz + 15], weight=0.02))
    bms = HiPart("BMS box", _pick(bms_parts), (0.2, 0.25, 0.3), 0.8,
                 "Battery management system with contactors and fuses", (0, 60, 0), 0.5)

    # ── High-voltage cables ───────────────────────────────────────────
    cable_parts = []
    cable_parts.append(capsule([tray_w / 2 + 15, 10, -5], [tray_w / 2 + 15, -10, -5], 1.5, 0.3))
    cable_parts.append(capsule([tray_w / 2 + 15, 10, 5], [tray_w / 2 + 15, -10, 5], 1.5, 0.3))
    for side in [-1, 1]:
        cz = side * 5
        cable_parts.append(line([tray_w / 2 + 15, -10, cz],
                                [tray_w / 2 + 15, -15, cz], weight=0.06))
    cables = HiPart("HV cables", _pick(cable_parts), (1.0, 0.5, 0.1), 0.6,
                    "Orange high-voltage DC cables with connectors", (0, 40, 0), 0.6)

    # ── Module interconnects ──────────────────────────────────────────
    intconn_parts = []
    for mi in range(7):
        col = mi % 4
        row2 = mi // 4
        mx = -tray_w / 2 + 20 + col * 38
        my = row2 * 26 - 25
        if mi == 3:
            intconn_parts.append(line([mx + mod_w / 2, my, 0],
                                      [mx + mod_w / 2 + 10, my, 0], weight=0.03))
        else:
            next_row = (mi + 1) // 4
            next_col = (mi + 1) % 4
            nx = -tray_w / 2 + 20 + next_col * 38
            ny = next_row * 26 - 25
            intconn_parts.append(line([mx + mod_w / 2, my, 0],
                                      [nx - mod_w / 2, ny, 0], weight=0.03))
    interconnects = HiPart("Module interconnects", _pick(intconn_parts), (0.7, 0.4, 0.1), 0.5,
                           "Series connections between modules", (0, 20, 0), 0.6)

    parts = [tray, modules, busbars, cooling, bms, cables, interconnects]
    return assemble("EV Battery Pack", "machine", parts, n, seed=35, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════════════

VEHICLES = [
    {"name": "Sports Car", "build": sports_car, "color": (1.0, 0.25, 0.25), "beat_bpm": None},
    {"name": "Motorcycle", "build": motorcycle, "color": (1.0, 0.35, 0.25), "beat_bpm": None},
    {"name": "Airliner", "build": airliner, "color": (0.6, 0.8, 1.0), "beat_bpm": None},
    {"name": "Saturn V Rocket", "build": saturn_v, "color": (0.95, 0.95, 1.0), "beat_bpm": None},
]

MACHINES = [
    {"name": "Mechanical Watch", "build": wristwatch, "color": (1.0, 0.8, 0.4), "beat_bpm": None},
    {"name": "EV Battery Pack", "build": battery_pack, "color": (0.4, 0.8, 1.0), "beat_bpm": None},
]