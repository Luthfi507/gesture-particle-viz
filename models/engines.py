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
    segs = n if closed else n - 1
    if closed:
        seg_pts = pts
    else:
        seg_pts = pts
    diffs = np.diff(seg_pts, axis=0)
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


def _sub(a, b):
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def _mul(a, s):
    return np.asarray(a, dtype=np.float64) * s


# ═══════════════════════════════════════════════════════════════════════════
#  1  Inline-4 Engine (DOHC 16V, mm scale)
# ═══════════════════════════════════════════════════════════════════════════

def inline4(n):
    xs = [-136.5, -45.5, 45.5, 136.5]
    bore = 43
    thr = 43
    rod = 145
    ch = 30
    deck = 222
    bx = 205
    bz = 95
    by0 = -40
    headTop = deck + 100
    camY = deck + 128
    cb = 16  # cam base circle radius

    # ── Cylinder block ────────────────────────────────────────────────
    block_parts = []
    block_parts.append(box_surface([0, (by0 + deck) / 2, 0], [bx * 2, deck - by0, bz * 2]))
    block_parts.append(box_edges([0, (by0 + deck) / 2, 0], [bx * 2, deck - by0, bz * 2], weight=0.3))

    def _deck_keep(x, y, z):
        for cx in xs:
            if (x - cx) ** 2 + z ** 2 < bore ** 2:
                return False
        return True

    block_parts.append(quad([bx, deck - 10, bz], [-bx, deck - 10, bz],
                            [-bx, deck - 10, -bz], [bx, deck - 10, -bz], keep=_deck_keep, weight=0.5))

    for cx in xs:
        block_parts.append(revolve([0, deck], [bore, bore], density=0.8,
                                   center=[cx, 0, 0], a0=0, a1=TAU))
    for i in range(5):
        mx = -182 + i * 91
        block_parts.append(box_filled([mx, -16, 0], [32, 28, bz * 1.5], density=0.5, weight=0.15))
        block_parts.append(box_edges([mx, -16, 0], [32, 28, bz * 1.5], weight=0.05))

    block = HiPart("Cylinder block", _pick(block_parts), (0.42, 0.62, 1.0), 6,
                   "Cast iron block with 4 cylinder bores and 5 main bearing caps", (0, 0, 0), 0.1)

    # ── Oil pan ───────────────────────────────────────────────────────
    pan_parts = []
    pan_parts.append(box_surface([0, -70, 0], [bx * 2, 60, bz * 1.8]))
    pan_parts.append(box_edges([0, -70, 0], [bx * 2, 60, bz * 1.8], weight=0.3))
    pan_parts.append(box_filled([bx - 20, -105, 0], [16, 16, 14], density=2, weight=0.1))
    pan_parts.append(ring([bx - 20, -105, 0], 6))
    pan = HiPart("Oil pan", _pick(pan_parts), (0.66, 0.4, 1.0), 2,
                 "Stamped steel oil reservoir with drain plug", (0, -360, 0), 0.5)

    # ── Crankshaft ────────────────────────────────────────────────────
    crank_parts = []
    crank_parts.append(tube_shape([-210, 0, 0], [210, 0, 0], 15, 15, 0.8))
    phis = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for i, cx in enumerate(xs):
        ph = phis[i]
        px = cx
        py = -thr * np.cos(ph)
        pz = thr * np.sin(ph)
        crank_parts.append(tube_shape([cx - 16, py, pz], [cx + 16, py, pz], 14, 14, 0.6))
        for side in [-1, 1]:
            wa = ph + np.pi + side * 1.15
            wb = ph + np.pi - side * 1.15
            crank_parts.append(revolve([py - 10, py + 10], [14, 58], density=0.4,
                                       center=[cx, 0, 0], a0=min(wa, wb), a1=max(wa, wb)))
    crank_parts.append(disc([210, 0, 0], 12, 70, None, 0.5))
    crank_parts.append(ring([210, 0, 0], 70))
    crank = HiPart("Crankshaft", _pick(crank_parts), (0.75, 0.8, 0.95), 4,
                   "Forged steel crankshaft with 4 throws", (0, -175, 0), 0.6)

    # ── Connecting rods (4) ───────────────────────────────────────────
    rod_parts = []
    for i, cx in enumerate(xs):
        big_y = -thr * np.cos(phis[i])
        big_z = thr * np.sin(phis[i])
        pin_y = ch
        rod_parts.append(tube_shape([cx, big_y, big_z], [cx, pin_y, 0], 6, 5, 0.8))
        rod_parts.append(torus_shape([cx, big_y, big_z], 25, 5, None, 0.6))
        rod_parts.append(torus_shape([cx, pin_y, 0], 12, 5, None, 0.6))
        rod_parts.append(line([cx, big_y + 22, big_z], [cx, big_y + 22, big_z + 0.05], jitter=0.5, weight=0.02))
        rod_parts.append(line([cx, pin_y + 10, 0], [cx, pin_y + 10, 0.05], jitter=0.5, weight=0.02))
    rods = HiPart("Connecting rods", _pick(rod_parts), (0.92, 0.95, 1.0), 3,
                  "Four forged steel I-beam connecting rods", (0, 75, 0), 0.7)

    # ── Pistons (4) ───────────────────────────────────────────────────
    piston_parts = []
    for i, cx in enumerate(xs):
        py = ch
        piston_parts.append(cylinder_s([cx, py, 0], bore - 1, 18, axis=1, cap=True))
        for ri in range(3):
            ry = py - 6 + ri * 5
            piston_parts.append(ring([cx, ry, 0], bore + 1))
    pistons = HiPart("Pistons", _pick(piston_parts), (1.0, 0.8, 0.2), 3,
                     "Four aluminum alloy pistons with ring packs", (0, 160, 0), 0.6)

    # ── Cylinder head ─────────────────────────────────────────────────
    head_parts = []
    head_parts.append(box_surface([0, deck + 50, 0], [bx * 2, 100, bz * 2]))
    head_parts.append(box_edges([0, deck + 50, 0], [bx * 2, 100, bz * 2], weight=0.3))
    for cx in xs:
        head_parts.append(ring([cx, deck + 1, 0], bore + 2))
    for cx in xs:
        head_parts.append(capsule([cx, deck + 50, bz + 2], [cx + 50, deck + 70, bz + 50], 8, 0.3))
        head_parts.append(capsule([cx, deck + 50, -bz - 2], [cx + 50, deck + 70, -bz - 50], 8, 0.3))
        head_parts.append(capsule([cx, deck + 50, bz + 2], [cx - 50, deck + 70, bz + 50], 8, 0.3))
        head_parts.append(capsule([cx, deck + 50, -bz - 2], [cx - 50, deck + 70, -bz - 50], 8, 0.3))
    head = HiPart("Cylinder head", _pick(head_parts), (0.2, 1.0, 0.8), 4,
                  "Aluminum DOHC cylinder head with 4 valves per cylinder", (0, 230, 0), 0.5)

    # ── Valves & springs (16) ────────────────────────────────────────
    valve_parts = []
    for ci, cx in enumerate(xs):
        for bank in [-1, 1]:
            for is_intake in [True, False]:
                vx = cx + bank * 18
                vz = 16 if is_intake else -16
                vy = deck + 20
                valve_parts.append(disc([vx, vy, vz], 0, 9 if is_intake else 8, None, 0.5))
                valve_parts.append(line([vx, vy, vz], [vx, camY - 5, vz], jitter=0.2, weight=0.3))
                valve_parts.append(helix_shape([vx, vy + 18, vz], 6, 22, 6, None, 0.6))
                valve_parts.append(ring([vx, vy + 10, vz], 7))
    valves = HiPart("Valves & springs", _pick(valve_parts), (1.0, 0.42, 0.75), 3,
                    "16 stainless steel valves with dual valve springs", (0, 310, 0), 0.7)

    # ── Camshafts (2) ─────────────────────────────────────────────────
    cam_parts = []
    for bank in [-1, 1]:
        cz = bank * 35
        cam_parts.append(tube_shape([-190, camY, cz], [190, camY, cz], cb, cb, 0.8))
        for ci, cx in enumerate(xs):
            lobe_phase = ci * np.pi / 2
            for is_exh in [True, False]:
                lx = cx + (15 if is_exh else -15)
                phase = lobe_phase + (0 if is_exh else np.pi * 1.25)
                cam_parts.append(revolve([camY - 8, camY + 8],
                                         [cb, cb + 7 * max(0, 1)],
                                         density=0.5,
                                         center=[lx, 0, cz],
                                         a0=phase - 0.6, a1=phase + 0.6))
        for i in range(5):
            bx2 = -190 + i * 95
            cam_parts.append(box_edges([bx2, camY, cz], [20, cb * 2.5, cb * 2.5], weight=0.06))
    cams = HiPart("Camshafts", _pick(cam_parts), (0.75, 1.0, 0.3), 3,
                  "Two billet steel camshafts with 8 lobes each", (0, 450, 0), 0.5)

    # ── Valve cover ───────────────────────────────────────────────────
    vc_parts = []
    vc_y = headTop - 10
    vc_parts.append(box_surface([0, vc_y + 18, 0], [bx * 2, 36, bz * 2]))
    vc_parts.append(box_edges([0, vc_y + 18, 0], [bx * 2, 36, bz * 2], weight=0.3))
    vc_parts.append(disc([0, vc_y + 36, 0], 8, 16, None, 0.3))
    vc_parts.append(ring([0, vc_y + 36, 0], 16))
    vc_parts.append(line([0, vc_y + 36, 0], [30, vc_y + 36, 30], jitter=0.3, weight=0.05))
    vc = HiPart("Valve cover", _pick(vc_parts), (1.0, 0.25, 0.22), 2,
                "Stamped aluminum valve cover with oil fill cap", (0, 640, 0), 0.3)

    # ── Coils & spark plugs ───────────────────────────────────────────
    coil_parts = []
    for ci, cx in enumerate(xs):
        coil_parts.append(tube_shape([cx, headTop, 0], [cx, headTop + 35, 0], 10, 10, 0.5))
        coil_parts.append(box_filled([cx, headTop + 40, 0], [18, 16, 18], density=1, weight=0.3))
        coil_parts.append(line([cx, headTop, 5], [cx, headTop - 30, 5], jitter=0.2, weight=0.05))
    coils = HiPart("Coils & spark plugs", _pick(coil_parts), (0.6, 0.85, 1.0), 2,
                   "Four coil-on-plug ignition units", (0, 820, 0), 0.3)

    # ── Intake manifold ───────────────────────────────────────────────
    intake_parts = []
    intake_parts.append(tube_shape([0, deck + 70, bz + 50], [0, deck + 30, bz + 120], 22, 28, 0.6))
    for ci, cx in enumerate(xs):
        intake_parts.append(_plumb([[cx, deck + 50, bz + 5],
                                  [cx, deck + 20, bz + 30],
                                  [cx, deck + 45, bz + 60]], 10, 0.5))
    intake_parts.append(disc([0, deck + 30, bz + 120], 18, 28, None, 0.3))
    intake_parts.append(torus_shape([0, deck + 30, bz + 120], 28, 4, None, 0.3))
    intake = HiPart("Intake manifold", _pick(intake_parts), (0.3, 0.88, 1.0), 2,
                    "Aluminum intake plenum with tuned runners and throttle body", (0, 40, -300), 0.5)

    # ── Exhaust manifold ──────────────────────────────────────────────
    exhaust_parts = []
    for ci, cx in enumerate(xs):
        exhaust_parts.append(_plumb([[cx, deck + 50, -bz - 5],
                                   [cx, deck + 10, -bz - 50],
                                   [20, deck - 20, -bz - 100]], 9, 0.5))
    exhaust_parts.append(tube_shape([-15, deck - 20, -bz - 100],
                                    [20, deck - 90, -bz - 120], 18, 22, 0.5))
    exhaust_parts.append(torus_shape([20, deck - 90, -bz - 120], 22, 3, None, 0.3))
    exhaust = HiPart("Exhaust manifold", _pick(exhaust_parts), (1.0, 0.5, 0.12), 2,
                     "Cast iron 4-into-1 header", (0, 0, 300), 0.5)

    # ── Flywheel ──────────────────────────────────────────────────────
    fly_parts = []
    fly_parts.append(disc([220, 0, 0], 15, 75, None, 0.8))
    fly_parts.append(gear_shape([220, 0, 0], 78, 110, None, 12, 4, None, 0.6))
    for i in range(6):
        a = i * TAU / 6
        wx = 220 + 50 * np.cos(a)
        wz = 50 * np.sin(a)
        fly_parts.append(cylinder_s([wx, 0, wz], 8, 12, axis=0, cap=True))
    flywheel = HiPart("Flywheel", _pick(fly_parts), (1.0, 0.65, 0.25), 2,
                      "Steel flywheel with starter ring gear and counterweights", (260, 0, 0), 0.5)

    # ── Timing belt & gears ───────────────────────────────────────────
    timing_parts = []
    gear_positions = [(-220, 0), (0, camY + 40), (0, camY - 40)]
    for gx, gy in gear_positions:
        timing_parts.append(gear_shape([gx, gy, 0], 38, 22, None, 10, 4, None, 0.4))
    belt_pts = []
    for a in linspace(0, TAU, 72):
        belt_pts.append([-220 + 38 * np.cos(a), 0, 38 * np.sin(a)])
    for a in linspace(0, TAU, 72):
        belt_pts.append([0, camY + 40 + 38 * np.cos(a), 38 * np.sin(a)])
    timing_parts.append(_plines(belt_pts, jitter=0.3, weight=0.3))
    timing_parts.append(disc([-80, 0, 0], 4, 14, None, 0.2))
    timing_parts.append(torus_shape([-80, 0, 0], 14, 3, None, 0.2))
    timing = HiPart("Timing belt & gears", _pick(timing_parts), (0.35, 1.0, 0.4), 2,
                    "Rubber timing belt with cam and crank sprockets, tensioner", (-260, 0, 0), 0.4)

    parts = [block, pan, crank, rods, pistons, head, valves, cams, vc, coils, intake, exhaust, flywheel, timing]
    return assemble("Inline-4 Engine", "engine", parts, n, seed=20, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  2  Supercharged HEMI V8 Engine (90 deg V8 pushrod, mm scale)
# ═══════════════════════════════════════════════════════════════════════════

def hemi_v8(n):
    xs = [-168, -56, 56, 168]
    bore = 50
    thr = 45
    rod = 157
    ch = 32
    deck = 238
    headTop = 330
    beta = 45
    beta_r = deg(beta)
    cb = 18

    # ── Engine block ──────────────────────────────────────────────────
    block_parts = []
    block_parts.append(box_surface([0, deck / 2 - 15, 0], [240, deck + 30, 130]))
    block_parts.append(box_edges([0, deck / 2 - 15, 0], [240, deck + 30, 130], weight=0.3))
    for bank_sign in [-1, 1]:
        bank_cy = deck + 40
        bank_cz = bank_sign * 55
        bank_angle = bank_sign * beta_r
        bank_block = box_surface([0, bank_cy, bank_cz], [240, 80, 80])
        bank_block = orient(bank_block, [0, np.cos(bank_angle), -np.sin(bank_angle * bank_sign)], [0, deck, 0])
        bank_edges = box_edges([0, bank_cy, bank_cz], [240, 80, 80], weight=0.2)
        bank_edges = orient(bank_edges, [0, np.cos(bank_angle), -np.sin(bank_angle * bank_sign)], [0, deck, 0])
        block_parts.extend(bank_block)
        block_parts.extend(bank_edges)
    block = HiPart("Engine block", _pick(block_parts), (1.0, 0.5, 0.12), 5,
                   "Cast iron 90-degree V8 block", (0, 0, 0), 0.1)

    # ── Oil pan ───────────────────────────────────────────────────────
    pan_parts = []
    pan_parts.append(box_surface([0, -90, 0], [260, 80, 140]))
    pan_parts.append(box_edges([0, -90, 0], [260, 80, 140], weight=0.2))
    pan2 = HiPart("Oil pan", _pick(pan_parts), (0.66, 0.4, 1.0), 2,
                  "Deep sump oil pan", (0, -380, 0), 0.5)

    # ── Crankshaft ────────────────────────────────────────────────────
    crank_parts = []
    crank_parts.append(tube_shape([-230, 0, 0], [230, 0, 0], 17, 17, 0.8))
    crank_phas = [0, np.pi / 2, 3 * np.pi / 2, np.pi]
    for i, cx in enumerate(xs):
        ph = crank_phas[i]
        py = -thr * np.cos(ph)
        pz = thr * np.sin(ph)
        crank_parts.append(tube_shape([cx - 18, py, pz], [cx + 18, py, pz], 16, 16, 0.6))
        for side in [-1, 1]:
            wa = ph + np.pi + side * 1.15
            wb = ph + np.pi - side * 1.15
            crank_parts.append(revolve([py - 12, py + 12], [16, 62], density=0.4,
                                       center=[cx, 0, 0], a0=min(wa, wb), a1=max(wa, wb)))
    crank = HiPart("Crankshaft", _pick(crank_parts), (0.75, 0.8, 0.95), 4,
                   "Cross-plane forged crankshaft", (0, -180, 0), 0.6)

    # ── Connecting rods ───────────────────────────────────────────────
    rod_parts = []
    for i, cx in enumerate(xs):
        ph = crank_phas[i]
        big_y = -thr * np.cos(ph)
        big_z = thr * np.sin(ph)
        pin_y = ch
        rod_parts.append(tube_shape([cx, big_y, big_z], [cx, pin_y, 0], 7, 6, 0.7))
        rod_parts.append(torus_shape([cx, big_y, big_z], 28, 6, None, 0.5))
        rod_parts.append(torus_shape([cx, pin_y, 0], 14, 5, None, 0.5))
    rods = HiPart("Connecting rods", _pick(rod_parts), (0.92, 0.95, 1.0), 3,
                  "Forged connecting rods", (0, 80, 0), 0.3)

    # ── Pistons ───────────────────────────────────────────────────────
    piston_parts = []
    for i, cx in enumerate(xs):
        bank_sign = 1 if i < 2 else -1
        bank_z = bank_sign * 55
        py = deck + 20
        p_parts = []
        p_parts.append(cylinder_s([cx, py + ch, bank_z], bore - 1, 18, axis=1, cap=True))
        for ri in range(3):
            ry = py + ch - 6 + ri * 5
            p_parts.append(ring([cx, ry, bank_z], bore + 1))
        p_transformed = orient(p_parts, [0, np.cos(bank_sign * beta_r),
                                         -np.sin(bank_sign * beta_r)], [cx, deck, bank_z])
        piston_parts.extend(p_transformed)
    pistons = HiPart("Pistons", _pick(piston_parts), (1.0, 0.8, 0.2), 3,
                     "Forged aluminum HEMI pistons", (0, 165, 0), 0.25)

    # ── Cylinder heads (2) ────────────────────────────────────────────
    head_parts = []
    for bank_sign in [-1, 1]:
        bank_z = bank_sign * 55
        bank_angle = bank_sign * beta_r
        hh_parts = []
        hh_parts.append(box_surface([0, deck + 50, bank_z], [250, 80, 40]))
        hh_parts.append(box_edges([0, deck + 50, bank_z], [250, 80, 40], weight=0.2))
        for cx in xs:
            hh_parts.append(ring([cx, deck + 1, bank_z], bore + 2))
        dome_fn = lambda u, v: [u * 100, bank_z + bank_sign * np.sqrt(max(0, bore ** 2 - (u * 100) ** 2 - v ** 2)) * 0.3, v]
        hh_parts.append(param_surface(dome_fn, -bore / 100, bore / 100, -bore / 100, bore / 100, density=0.3))
        for i, cx in enumerate(xs):
            for is_exh in [True, False]:
                vx = cx + (12 if is_exh else -12)
                vz = bank_z + bank_sign * (-8 if is_exh else 8)
                hh_parts.append(disc([vx, deck + 25, vz], 0, 8 if is_exh else 10, None, 0.4))
                hh_parts.append(line([vx, deck + 25, vz], [vx, deck + 65, vz], jitter=0.2, weight=0.2))
                hh_parts.append(helix_shape([vx, deck + 35, vz], 7, 24, 6, None, 0.5))
                hh_parts.append(ring([vx, deck + 30, vz], 8))
            hh_parts.append(box_surface([cx, deck + 85, bank_z + bank_sign * 5], [30, 30, 16]))
        hh_transformed = orient(flatten([hh_parts]), [0, np.cos(bank_angle), -np.sin(bank_sign * beta_r)], [0, deck, 0])
        head_parts.extend(hh_transformed)
    heads = HiPart("Cylinder heads", _pick(head_parts), (1.0, 0.25, 0.22), 4,
                   "HEMI aluminum heads with hemispherical combustion chambers and rocker gear", (0, 350, 0), 0.5)

    # ── Valve covers (2) ──────────────────────────────────────────────
    vc_parts = []
    for bank_sign in [-1, 1]:
        bank_z = bank_sign * 55
        bank_angle = bank_sign * beta_r
        vc2_parts = []
        vc2_parts.append(box_surface([0, deck + 110, bank_z], [250, 50, 30]))
        vc2_parts.append(box_edges([0, deck + 110, bank_z], [250, 50, 30], weight=0.2))
        badge_pts = []
        for bi in range(40):
            badge_pts.append([-15 + bi, deck + 135, bank_z])
        vc2_parts.append(_plines(badge_pts, jitter=0.4, weight=0.1))
        vc2_transformed = orient(flatten([vc2_parts]), [0, np.cos(bank_angle), -np.sin(bank_sign * beta_r)], [0, deck, 0])
        vc_parts.extend(vc2_transformed)
    vc2 = HiPart("Valve covers", _pick(vc_parts), (1.0, 0.3, 0.95), 2,
                 "Cast aluminum HEMI valve covers", (0, 440, 0), 0.3)

    # ── Valves & pushrods ─────────────────────────────────────────────
    vp_parts = []
    for bank_sign in [-1, 1]:
        bank_z = bank_sign * 55
        bank_angle = bank_sign * beta_r
        for ci, cx in enumerate(xs):
            for is_exh in [True, False]:
                vx = cx + (12 if is_exh else -12)
                vz = bank_z + bank_sign * (-6 if is_exh else 6)
                vp_parts.append(disc([vx, deck + 20, vz], 0, 9 if is_exh else 11, None, 0.4))
                vp_parts.append(line([vx, deck + 20, vz], [vx, deck + 65, vz], jitter=0.2, weight=0.2))
                vp_parts.append(helix_shape([vx, deck + 35, vz], 7, 22, 5, None, 0.5))
        for ci, cx in enumerate(xs):
            for is_exh in [True, False]:
                px = cx + (12 if is_exh else -12)
                pz = bank_z + bank_sign * (-6 if is_exh else 6)
                vp_parts.append(line([px, deck + 68, pz], [px, 0, pz], jitter=0.3, weight=0.15))
    valves_push = HiPart("Valves & pushrods", _pick(vp_parts), (1.0, 0.42, 0.75), 3,
                         "Stainless valves with pushrod actuation from cam-in-V", (0, 320, 0), 0.5)

    # ── Camshaft (in the V) ───────────────────────────────────────────
    cam_parts = []
    cam_parts.append(tube_shape([-200, deck + 50, 0], [200, deck + 50, 0], cb, cb, 0.8))
    for ci, cx in enumerate(xs):
        lobe_phase = ci * np.pi / 2
        for offs in [-1, 1]:
            lx = cx + offs * 18
            for is_exh in [True, False]:
                phase = lobe_phase + (0 if is_exh else np.pi * 1.25)
                cam_parts.append(revolve([deck + 38, deck + 62],
                                         [cb, cb + 9],
                                         density=0.5,
                                         center=[lx, 0, 0],
                                         a0=phase - 0.6, a1=phase + 0.6))
    cam_parts.append(gear_shape([-200, deck + 50, 0], 32, 20, None, 10, 4, None, 0.3))
    camshaft = HiPart("Camshaft (in the V)", _pick(cam_parts), (0.75, 1.0, 0.3), 2,
                      "Single camshaft nestled in the valley, driving lifters and pushrods", (-520, 0, 0), 0.5)

    # ── Intake manifold ───────────────────────────────────────────────
    int_parts = []
    int_parts.append(box_surface([0, deck + 120, 0], [220, 50, 60]))
    for ci, cx in enumerate(xs):
        bank_sign = 1 if ci < 2 else -1
        bz = bank_sign * 30
        int_parts.append(_plumb([[cx, deck + 95, bz],
                               [cx, deck + 80, bank_sign * 55],
                               [cx, deck + 70, bank_sign * 45]], 8, 0.4))
    intake2 = HiPart("Intake manifold", _pick(int_parts), (0.3, 0.88, 1.0), 2,
                     "High-rise intake with tuned runners", (0, 430, 0), 0.5)

    # ── Supercharger ──────────────────────────────────────────────────
    sc_parts = []
    sc_parts.append(box_surface([0, deck + 190, 0], [180, 60, 70]))
    sc_parts.append(box_edges([0, deck + 190, 0], [180, 60, 70], weight=0.2))
    for rotor_x in [-35, 35]:
        sc_parts.append(tube_shape([rotor_x, deck + 160, 0], [rotor_x, deck + 220, 0], 22, 22, 0.6))
        for i in range(5):
            ry = deck + 160 + i * 15
            sc_parts.append(line([rotor_x - 22, ry, 0], [rotor_x + 22, ry, 0], jitter=0.2, weight=0.03))
    sc_parts.append(gear_shape([-50, deck + 220, 0], 32, 18, None, 10, 4, None, 0.3))
    sc_parts.append(disc([-50, deck + 240, 0], 6, 28, None, 0.3))
    badge2_pts = []
    for bi in range(30):
        badge2_pts.append([-25 + bi * 1.7, deck + 220, 20])
    sc_parts.append(_plines(badge2_pts, jitter=0.3, weight=0.08))
    sc = HiPart("Supercharger", _pick(sc_parts), (0.35, 1.0, 0.4), 3,
                "Roots-type supercharger with twin rotors", (0, 760, 0), 0.5)

    # ── Supercharger belt ─────────────────────────────────────────────
    sb_parts = []
    sb_parts.append(disc([-170, deck + 200, 0], 6, 36, None, 0.5))
    sb_parts.append(torus_shape([-170, deck + 200, 0], 36, 3, None, 0.3))
    belt2_pts = []
    for a in linspace(0, TAU, 60):
        belt2_pts.append([-170 + 36 * np.cos(a), deck + 200, 36 * np.sin(a)])
    for a in linspace(TAU, 0, 60):
        belt2_pts.append([-50 + 32 * np.cos(a), deck + 220, 32 * np.sin(a)])
    sb_parts.append(_plines(belt2_pts, jitter=0.3, weight=0.2))
    sb_parts.append(torus_shape([-110, deck + 180, 0], 14, 3, None, 0.15))
    sbelt = HiPart("Supercharger belt", _pick(sb_parts), (0.92, 0.95, 1.0), 2,
                   "Serpentine belt driving the supercharger", (-300, 0, 0), 0.4)

    # ── Timing cover ──────────────────────────────────────────────────
    tc_parts = []
    tc_parts.append(box_surface([-210, deck / 2, 0], [30, deck + 40, 140]))
    tc = HiPart("Timing cover", _pick(tc_parts), (0.2, 1.0, 0.8), 2,
                "Front timing cover with oil pump", (-200, 0, 0), 0.4)

    # ── Exhaust headers (R) ───────────────────────────────────────────
    exhR_parts = []
    for ci, cx in enumerate(xs):
        if ci < 2:
            exhR_parts.append(_plumb([[cx, deck + 40, 35],
                                    [cx + 20, deck - 10, 90],
                                    [40, deck - 50, 140]], 10, 0.4))
    exhR_parts.append(tube_shape([20, deck - 50, 140], [20, deck - 120, 160], 22, 26, 0.4))
    exhR = HiPart("Exhaust headers", _pick(exhR_parts), (1.0, 0.65, 0.25), 2,
                  "Right bank stainless steel headers and collector", (0, -60, 280), 0.5)

    # ── Exhaust headers (L) ───────────────────────────────────────────
    exhL_parts = []
    for ci, cx in enumerate(xs):
        if ci >= 2:
            exhL_parts.append(_plumb([[cx, deck + 40, -35],
                                    [cx - 20, deck - 10, -90],
                                    [-40, deck - 50, -140]], 10, 0.4))
    exhL_parts.append(tube_shape([-20, deck - 50, -140], [-20, deck - 120, -160], 22, 26, 0.4))
    exhL = HiPart("Exhaust headers (L)", _pick(exhL_parts), (1.0, 0.65, 0.25), 2,
                  "", (0, -60, -280), 0.5)

    # ── Flywheel ──────────────────────────────────────────────────────
    fly2_parts = []
    fly2_parts.append(disc([240, 0, 0], 16, 82, None, 0.8))
    fly2_parts.append(gear_shape([240, 0, 0], 85, 120, None, 14, 5, None, 0.6))
    fly2 = HiPart("Flywheel", _pick(fly2_parts), (0.42, 0.62, 1.0), 2,
                  "Steel flywheel with ring gear", (280, 0, 0), 0.5)

    parts = [block, pan2, crank, rods, pistons, heads, vc2, valves_push, camshaft,
             intake2, sc, sbelt, tc, exhR, exhL, fly2]
    return assemble("Supercharged HEMI V8", "engine", parts, n, seed=21, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  3  Turbofan Jet Engine (cm scale)
# ═══════════════════════════════════════════════════════════════════════════

def turbofan(n):
    UPPER = lambda a: np.pi / 2 <= a <= 3 * np.pi / 2
    LOWER = lambda a: -np.pi / 2 <= a <= np.pi / 2

    def _revX(xs, rs, density=1, a0=0, a1=TAU, keep=None):
        rev = revolve(xs, rs, density=density, a0=a0, a1=a1, keep=keep)
        return orient(rev, [1, 0, 0], [0, 0, 0])[0]

    def _ringX(x, r, density=1, a0=0, a1=TAU):
        rng = ring([0, 0, 0], r, density=density, a0=a0, a1=a1)
        return orient(rng, [1, 0, 0], [x, 0, 0])[0]

    def _bladeRow(x0, rHub, rTip, nBlades, chord, twRoot_deg, twTip_deg, density=1):
        rows = []
        twRoot = deg(twRoot_deg)
        twTip = deg(twTip_deg)
        for bi in range(nBlades):
            a0_c = bi * TAU / nBlades
            blade_pts = []
            for si in range(40):
                cc = np.random.uniform(-0.5, 0.5)
                r = rHub + np.random.uniform(0, 1) * (rTip - rHub)
                tw = twRoot + (twTip - twRoot) * (r - rHub) / (rTip - rHub)
                a = a0_c + TAU * si / 40 + cc * chord * np.sin(tw) / r
                blade_pts.append([x0 + cc * chord * np.cos(tw), r * np.cos(a), r * np.sin(a)])
            rows.append(_plines(blade_pts, jitter=0.15, weight=0.3))
            lead_pts = []
            for si in range(20):
                r = rHub + (rTip - rHub) * si / 20
                a = a0_c
                lead_pts.append([x0, r * np.cos(a), r * np.sin(a)])
            rows.append(_plines(lead_pts, jitter=0.1, weight=0.15))
        rows.append(_ringX(x0, rHub, density * 0.5))
        rows.append(_ringX(x0 + chord, rHub, density * 0.5))
        hub_rev = revolve([x0, x0 + chord], [rHub, rHub], density=density * 0.4)
        rows.extend(orient(hub_rev, [1, 0, 0], [0, 0, 0]))
        w = sum(r.weight for r in rows)
        return Part(w, _pick(rows))

    # ── Spinner ───────────────────────────────────────────────────────
    spin_parts = []
    spin_surf = revolve([-160, -155, -120], [0, 28, 0], density=0.8)
    spin_parts.extend(orient(spin_surf, [1, 0, 0], [0, 0, 0]))
    for i in range(6):
        a0 = i * TAU / 6
        spiral_pts = []
        for t in range(30):
            a = a0 + t * TAU * 0.8 / 30
            x = -160 + t * 45 / 30
            r = 28 * (1 - t / 30)
            spiral_pts.append([x, r * np.cos(a), r * np.sin(a)])
        spin_parts.append(_plines(spiral_pts, jitter=0.1, weight=0.05))
    spinner = HiPart("Spinner", _pick(spin_parts), (0.92, 0.95, 1.0), 1,
                     "Aerodynamic nose cone", (-330, 0, 0), 0.5)

    # ── Fan (22 blades) ───────────────────────────────────────────────
    fan_parts = [_bladeRow(-138, 26, 98, 22, 30, 58, 22, 1.5)]
    fan = HiPart("Fan (22 blades)", _pick(fan_parts), (0.3, 0.88, 1.0), 3,
                 "Wide-chord hollow titanium fan", (-250, 0, 0), 0.5)

    # ── LP compressor ─────────────────────────────────────────────────
    lpc_parts = []
    for stage in range(3):
        xb = -130 + stage * 15
        rh = 38 - stage * 2
        rt = 95 - stage * 4
        lpc_parts.append(_bladeRow(xb, rh, rt, 38 + stage * 4, 12, 50, 28, 0.6))
    lpcomp = HiPart("LP compressor", _pick(lpc_parts), (0.2, 1.0, 0.8), 2,
                    "3-stage low-pressure compressor", (-175, 0, 0), 0.6)

    # ── HP compressor ─────────────────────────────────────────────────
    hpc_parts = []
    for stage in range(9):
        xb = -85 + stage * 8
        rh = 28 - stage
        rt = 80 - stage * 3
        hpc_parts.append(_bladeRow(xb, rh, rt, 44 + stage * 2, 8, 42, 20, 0.5))
    hpc_parts.append(_revX([-85, -13], [80, 68], density=0.4))
    hpcomp = HiPart("HP compressor", _pick(hpc_parts), (0.75, 1.0, 0.3), 3,
                    "9-stage high-pressure compressor with casing", (-80, 0, 0), 0.6)

    # ── Combustion chamber ────────────────────────────────────────────
    comb_parts = []
    comb_parts.append(_revX([-10, 60], [68, 68], density=0.5))
    comb_parts.append(_revX([-10, 60], [48, 48], density=0.3))
    comb_parts.append(_ringX(-8, 68, 0.3))
    comb_parts.append(_ringX(58, 68, 0.3))
    for i in range(20):
        a = i * TAU / 20
        fl_pts = []
        for t in range(15):
            x = -8 + t * 66 / 15
            r = 56
            fl_pts.append([x, r * np.cos(a), r * np.sin(a)])
        comb_parts.append(_plines(fl_pts, jitter=0.2, weight=0.04))
    combust = HiPart("Combustion chamber", _pick(comb_parts), (1.0, 0.5, 0.12), 2,
                     "Annular combustor with flame tubes", (20, 0, 0), 0.7)

    # ── HP turbine ────────────────────────────────────────────────────
    hpt_parts = []
    for stage in range(2):
        xb = 68 + stage * 12
        hpt_parts.append(_bladeRow(xb, 36, 64, 56 + stage * 4, 8, 30, 30, 0.5))
    hpturb = HiPart("HP turbine", _pick(hpt_parts), (1.0, 0.25, 0.22), 2,
                    "2-stage high-pressure turbine with cooling", (110, 0, 0), 0.6)

    # ── LP turbine ────────────────────────────────────────────────────
    lpt_parts = []
    for stage in range(5):
        xb = 100 + stage * 14
        rh = 28 - stage
        rt = 68 - stage * 5
        lpt_parts.append(_bladeRow(xb, rh, rt, 48 + stage * 2, 10, 38, 18, 0.5))
    lpt_parts.append(_revX([100, 170], [68, 80], density=0.4))
    lpturb = HiPart("LP turbine", _pick(lpt_parts), (1.0, 0.3, 0.95), 3,
                    "5-stage low-pressure turbine", (190, 0, 0), 0.6)

    # ── Exhaust nozzle ────────────────────────────────────────────────
    exh_parts = []
    exh_parts.append(_revX([170, 240], [80, 62], density=0.6))
    exh_parts.append(_ringX(240, 62, 0.3))
    exh_parts.append(_revX([200, 270], [30, 55], density=0.3))
    exhaust_jet = HiPart("Exhaust nozzle", _pick(exh_parts), (0.66, 0.4, 1.0), 2,
                         "Convergent exhaust nozzle with centerbody", (290, 0, 0), 0.5)

    # ── Spools ────────────────────────────────────────────────────────
    spool_parts = []
    spool_parts.append(_revX([-140, 20], [18, 14], density=0.3))
    spool_parts.append(_revX([-80, 170], [14, 10], density=0.3))
    spools = HiPart("Spools", _pick(spool_parts), (1.0, 0.8, 0.2), 2,
                    "LP and HP concentric shafts", (0, -150, 0), 0.7)

    # ── Nacelle (upper) ───────────────────────────────────────────────
    nac_up_parts = []
    nac_up_parts.append(_revX([-158, 130], [102, 98], density=0.4, a0=np.pi / 2, a1=3 * np.pi / 2))
    for i in range(6):
        xi = -150 + i * 48
        nac_up_parts.append(_ringX(xi, 100, 0.2, a0=np.pi / 2, a1=3 * np.pi / 2))
    nacelle_u = HiPart("Nacelle (upper)", _pick(nac_up_parts), (0.42, 0.62, 1.0), 2,
                       "Composite fan nacelle, upper half", (0, 190, 0), 0.3)

    # ── Nacelle (lower) ───────────────────────────────────────────────
    nac_lo_parts = []
    nac_lo_parts.append(_revX([-158, 130], [102, 98], density=0.4, a0=-np.pi / 2, a1=np.pi / 2))
    for i in range(6):
        xi = -150 + i * 48
        nac_lo_parts.append(_ringX(xi, 100, 0.2, a0=-np.pi / 2, a1=np.pi / 2))
    nacelle_l = HiPart("Nacelle (lower)", _pick(nac_lo_parts), (0.42, 0.62, 1.0), 2,
                       "", (0, -250, 0), 0.3)

    # ── Core cowl (upper) ─────────────────────────────────────────────
    cc_up_parts = []
    cc_up_parts.append(_revX([-10, 80], [72, 76], density=0.3, a0=np.pi / 2, a1=3 * np.pi / 2))
    cc_u = HiPart("Core cowl (upper)", _pick(cc_up_parts), (0.6, 0.85, 1.0), 1.5,
                  "Core engine cowling, upper", (0, 110, 0), 0.5)

    # ── Core cowl (lower) ─────────────────────────────────────────────
    cc_lo_parts = []
    cc_lo_parts.append(_revX([-10, 80], [72, 76], density=0.3, a0=-np.pi / 2, a1=np.pi / 2))
    cc_l = HiPart("Core cowl (lower)", _pick(cc_lo_parts), (0.6, 0.85, 1.0), 1.5,
                  "", (0, -190, 0), 0.5)

    parts = [spinner, fan, lpcomp, hpcomp, combust, hpturb, lpturb, exhaust_jet,
             spools, nacelle_u, nacelle_l, cc_u, cc_l]
    return assemble("Turbofan Jet Engine", "engine", parts, n, seed=22, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  4  Radial 9-Cylinder Aircraft Engine (mm scale, crank along z)
# ═══════════════════════════════════════════════════════════════════════════

def radial9(n):
    n_cyl = 9
    bore_r = 35
    stroke = 38
    cyl_length = 120
    crank_r = 105

    def _cyl_at(angle, cyl_idx):
        cx = crank_r * np.cos(angle)
        cy = crank_r * np.sin(angle)
        cp = []
        cp.append(cylinder_s([cx, cy, 0], bore_r, cyl_length, axis=2, cap=True))
        for ri in range(18):
            rz = -cyl_length / 2 + ri * cyl_length / 18
            cp.append(ring([cx, cy, rz], bore_r + 4))
        head_rev = revolve([-cyl_length / 2 - 10, -cyl_length / 2], [bore_r, bore_r + 6], density=0.5)
        head_pos = orient(head_rev, [0, 0, 1], [cx, cy, 0])
        cp.append(head_pos)
        cp.append(box_surface([cx + 8, cy + 6, -cyl_length / 2 + 5], [16, 10, 8]))
        cp.append(box_surface([cx - 8, cy + 6, -cyl_length / 2 + 5], [16, 10, 8]))
        for bank in [-1, 1]:
            cp.append(line([cx + bank * 8, cy + 6, -cyl_length / 2 + 5],
                           [cx + bank * 8, cy - 15, 10], jitter=0.3, weight=0.05))
        transformed = orient(flatten(cp), [0, 0, 1], [cx, cy, 0])
        return transformed

    def _piston_at(angle, cyl_idx):
        cx = crank_r * np.cos(angle)
        cy = crank_r * np.sin(angle)
        pz = cyl_length / 2 - stroke / 2
        pp = []
        pp.append(cylinder_s([cx, cy, pz], bore_r - 1, 16, axis=2, cap=True))
        for ri in range(3):
            rz = pz - 5 + ri * 5
            pp.append(ring([cx, cy, rz], bore_r + 1))
        return orient(flatten(pp), [0, 0, 1], [cx, cy, 0])

    def _intake_at(angle, cyl_idx):
        cx = crank_r * np.cos(angle)
        cy = crank_r * np.sin(angle)
        ip_parts = []
        radial_out = 1.3 * crank_r
        ip_parts.append(_plumb([[cx, cy, -cyl_length / 2 + 10],
                              [cx * 0.8, cy * 0.8, -cyl_length / 2 + 5],
                              [cx * 0.4, cy * 0.4, -cyl_length / 2]], 6, 0.3))
        return ip_parts

    def _exhaust_at(angle, cyl_idx):
        cx = crank_r * np.cos(angle)
        cy = crank_r * np.sin(angle)
        ep_parts = []
        ep_parts.append(_plumb([[cx * 1.2, cy * 1.2, cyl_length / 2],
                              [cx * 1.5, cy * 1.5, cyl_length / 2 + 20]], 8, 0.3))
        return ep_parts

    # ── Crankcase ─────────────────────────────────────────────────────
    case_parts = []
    case_rev = revolve([-80, 80], [55, 55], density=0.8)
    case_parts.extend(orient(case_rev, [0, 0, 1], [0, 0, 0]))
    nose_rev = revolve([-100, -80], [55, 38], density=0.5)
    case_parts.extend(orient(nose_rev, [0, 0, 1], [0, 0, 0]))
    case_parts.append(ring([0, 0, -80], 55))
    case_parts.append(ring([0, 0, 80], 55))
    crankcase = HiPart("Crankcase", _pick(case_parts), (0.75, 0.8, 0.95), 3,
                       "Forged aluminum crankcase housing", (0, 0, 0), 0.3)

    # ── Master & link rods ────────────────────────────────────────────
    rod_parts = []
    rod_parts.append(disc([0, 0, 0], 8, 35, None, 0.5))
    rod_parts.append(tube_shape([0, 0, 0], [crank_r * 0.9, 0, 0], 8, 7, 0.4))
    rod_parts.append(torus_shape([crank_r * 0.9, 0, 0], 24, 6, None, 0.3))
    for i in range(1, n_cyl):
        a = i * TAU / n_cyl
        lx = crank_r * 0.9
        ly = 0
        ex = crank_r * np.cos(a)
        ey = crank_r * np.sin(a)
        rod_parts.append(tube_shape([lx, ly, 0], [ex, ey, 0], 6, 5, 0.3))
        rod_parts.append(torus_shape([ex, ey, 0], 24, 5, None, 0.2))
    rods_radial = HiPart("Master & link rods", _pick(rod_parts), (0.92, 0.95, 1.0), 2,
                         "One master rod with 8 articulated link rods", (0, 0, 260), 0.8)

    # ── Crankshaft ────────────────────────────────────────────────────
    crank_parts_r = []
    crank_parts_r.append(tube_shape([0, 0, -70], [0, 0, 70], 14, 14, 0.6))
    cw_rev = revolve([-15, 15], [14, 80], density=0.3, a0=np.pi / 2, a1=3 * np.pi / 2)
    crank_parts_r.extend(orient(cw_rev, [0, 0, 1], [0, 0, 0]))
    crank_radial = HiPart("Crankshaft", _pick(crank_parts_r), (1.0, 0.65, 0.25), 2,
                          "Single-throw forged crankshaft with counterweight", (0, 0, 480), 0.7)

    # ── Propeller hub ─────────────────────────────────────────────────
    hub_parts = []
    hub_rev = revolve([330, 470], [22, 50], density=0.6)
    hub_parts.extend(orient(hub_rev, [0, 0, 1], [0, 0, 0]))
    hub_parts.extend(orient(gear_shape([0, 0, 380], 48, 30, None, 12, 5, None, 0.4), [0, 0, 1], [0, 0, 0]))
    prophub = HiPart("Propeller hub", _pick(hub_parts), (1.0, 0.25, 0.22), 1.5,
                     "Constant-speed propeller hub with governor gear", (0, 0, 760), 0.4)

    # ── Accessory section ─────────────────────────────────────────────
    acc_parts = []
    acc_parts.extend(orient(box_surface([0, 0, -160], [100, 100, 60]), [0, 0, 1], [0, 0, 0]))
    acc_parts.extend(orient(box_surface([0, 40, -180], [40, 30, 30]), [0, 0, 1], [0, 0, 0]))
    acc_parts.extend(orient(box_surface([40, -20, -180], [30, 40, 30]), [0, 0, 1], [0, 0, 0]))
    accessory = HiPart("Accessory section", _pick(acc_parts), (0.2, 1.0, 0.8), 1.5,
                       "Magneto, fuel pump and oil pump housing", (0, 0, -380), 0.5)

    # ── Intake pipes (all 9) ──────────────────────────────────────────
    int_all_parts = []
    for i in range(n_cyl):
        a = i * TAU / n_cyl
        int_all_parts.extend(_intake_at(a, i))
    intake_all = HiPart("Intake pipes", _pick(int_all_parts), (0.3, 0.88, 1.0), 2,
                        "Aluminum intake manifold pipes to each cylinder", (0, 0, -240), 0.6)

    # ── Exhaust stacks (all 9) ────────────────────────────────────────
    exh_all_parts = []
    for i in range(n_cyl):
        a = i * TAU / n_cyl
        exh_all_parts.extend(_exhaust_at(a, i))
    exhaust_all = HiPart("Exhaust stacks", _pick(exh_all_parts), (1.0, 0.5, 0.12), 2,
                         "Short individual exhaust stacks", (0, 0, 330), 0.6)

    # ── Cylinders (9 groups) ──────────────────────────────────────────
    cyl_all_parts = []
    for i in range(n_cyl):
        a = i * TAU / n_cyl
        cyl_all_parts.extend(_cyl_at(a, i))
    cylinders_all = HiPart("Cylinders", _pick(cyl_all_parts), (0.42, 0.62, 1.0), 4,
                           "9 finned steel cylinders with rocker boxes", (0, 0, 0), 0.5)

    # ── Pistons (9) ───────────────────────────────────────────────────
    pis_all_parts = []
    for i in range(n_cyl):
        a = i * TAU / n_cyl
        pis_all_parts.extend(_piston_at(a, i))
    pistons_all = HiPart("Pistons", _pick(pis_all_parts), (1.0, 0.8, 0.2), 2,
                         "9 forged aluminum pistons", (0, 0, 0), 0.7)

    parts = [crankcase, rods_radial, crank_radial, prophub, accessory, intake_all, exhaust_all,
             cylinders_all, pistons_all]
    return assemble("Radial Aircraft Engine", "engine", parts, n, seed=23, beat_bpm=None)


# ═══════════════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════════════

ENGINES = [
    {"name": "Inline-4 Engine", "build": inline4, "color": (0.35, 0.75, 1.0), "beat_bpm": None},
    {"name": "Supercharged HEMI V8", "build": hemi_v8, "color": (1.0, 0.45, 0.15), "beat_bpm": None},
    {"name": "Turbofan Jet Engine", "build": turbofan, "color": (0.3, 0.9, 1.0), "beat_bpm": None},
    {"name": "Radial Aircraft Engine", "build": radial9, "color": (1.0, 0.78, 0.35), "beat_bpm": None},
]