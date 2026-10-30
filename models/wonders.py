import numpy as np
from .base import Part as HiPart, assemble
from .shapes import (Part, line, polyline, tri, quad, box_surface, box_edges, box_filled,
                     revolve, cylinder_s, ring, disc, tube_shape, pipe, param_surface,
                     torus_shape, helix_shape, ellipsoid, capsule, blob, dots,
                     transform, orient, _pick, _pick_with_keep, flatten,
                     rot_x, rot_z, deg, TAU, _spline, linspace, range_n,
                     sphere, vessel, path_tube, bone, gear_shape, badge)


# ── shared helpers ───────────────────────────────────────────────────────

def _archMask(u, v, arches):
    for au, aw, ah in arches:
        if abs(u - au) <= aw / 2 and 0 <= v <= ah:
            if v <= ah - aw / 2:
                return True
            dx = u - au
            dy = v - (ah - aw / 2)
            if dx * dx + dy * dy <= (aw / 2) ** 2:
                return True
    return False


def _wall(along_axis, fixed_val, u0, u1, y0, y1, arches, density=1):
    parts = []
    if arches is None:
        arches = []
    if along_axis == 0:
        keep_fn = (lambda x, y, z: not _archMask(x - u0, y - y0, arches)) if arches else None
        parts.append(quad([u0, y0, fixed_val], [u1, y0, fixed_val],
                          [u1, y1, fixed_val], [u0, y1, fixed_val], density * 0.55, keep=keep_fn if arches else None))
        parts.append(line([u0, y0, fixed_val], [u1, y0, fixed_val], 0, density * 0.15))
        parts.append(line([u0, y1, fixed_val], [u1, y1, fixed_val], 0, density * 0.15))
        parts.append(line([u1, y0, fixed_val], [u1, y1, fixed_val], 0, density * 0.075))
        parts.append(line([u0, y0, fixed_val], [u0, y1, fixed_val], 0, density * 0.075))
    elif along_axis == 2:
        keep_fn = (lambda x, y, z: not _archMask(z - u0, y - y0, arches)) if arches else None
        parts.append(quad([fixed_val, y0, u0], [fixed_val, y0, u1],
                          [fixed_val, y1, u1], [fixed_val, y1, u0], density * 0.55, keep=keep_fn if arches else None))
        parts.append(line([fixed_val, y0, u0], [fixed_val, y0, u1], 0, density * 0.15))
        parts.append(line([fixed_val, y1, u0], [fixed_val, y1, u1], 0, density * 0.15))
        parts.append(line([fixed_val, y0, u1], [fixed_val, y1, u1], 0, density * 0.075))
        parts.append(line([fixed_val, y0, u0], [fixed_val, y1, u0], 0, density * 0.075))
    return parts


def _rectWalls(center, width, height, arches_spec, depth=3.0, density=1):
    cx, cy, cz = center
    hw, hd = width / 2, depth / 2
    y0, y1 = cy - height / 2, cy + height / 2
    front = _wall(0, cz + hd, cx - hw, cx + hw, y0, y1, arches_spec.get('front'), density)
    back = _wall(0, cz - hd, cx - hw, cx + hw, y0, y1, arches_spec.get('back'), density)
    left = _wall(2, cx - hw, cz - hd, cz + hd, y0, y1, arches_spec.get('left'), density)
    right = _wall(2, cx + hw, cz - hd, cz + hd, y0, y1, arches_spec.get('right'), density)
    return flatten([front, back, left, right])


# ═══════════════════════════════════════════════════════════════════════════
#  1  Turtle Tower
# ═══════════════════════════════════════════════════════════════════════════

def turtle_tower(n):
    base_w, base_d = 8.2, 7.0
    hw, hd = base_w / 2, base_d / 2
    slabs = [0.15, 3.65, 6.35]
    storey_h = [(slabs[1] - slabs[0]), (slabs[2] - slabs[1]), (8.9 - slabs[2])]
    tops = [slabs[0] + storey_h[0], slabs[1] + storey_h[1], 8.9]
    arch_door = [(0, 1.6, 2.8)]
    arch_win = [(-1.1, 1.0, 1.5), (1.1, 1.0, 1.5)]

    # Porch & base
    porch_parts = []
    for i, yb in enumerate(slabs):
        sw, sd = base_w * (1 - 0.08 * i), base_d * (1 - 0.08 * i)
        shw, shd = sw / 2, sd / 2
        porch_parts.append(box_surface([0, yb, 0], [sw, 0.08, sd]))
        porch_parts.append(box_edges([0, yb, 0], [sw, 0.08, sd], weight=0.8))
    for i in range(3):
        sw, sd = base_w * (1 - 0.08 * i), base_d * (1 - 0.08 * i)
        shw, shd = sw / 2, sd / 2
        yb, yt = slabs[i], tops[i]
        my = (yb + yt) / 2
        mh = yt - yb
        arch_spec = {'front': arch_door if i == 0 else arch_win,
                     'back': arch_win,
                     'left': arch_win,
                     'right': arch_win}
        porch_parts.extend(_rectWalls([0, my, 0], sw, mh, arch_spec, sd, density=1.2))
        porch_parts.append(polyline([[shw, yb, shd], [-shw, yb, shd], [-shw, yb, -shd], [shw, yb, -shd]], closed=True, weight=0.15))
    porch_sampler = _pick(porch_parts)
    porch = HiPart("Porch & base", porch_sampler, (1.0, 0.72, 0.35), 1.5,
                   "Three storeys of brick with arched openings", (0, 0.3, 0.2), 0.5)

    # Shrine hall
    shrine_parts = []
    for i in range(3):
        sw, sd = base_w * (1 - 0.08 * i) * 0.78, base_d * (1 - 0.08 * i) * 0.78
        shw, shd = sw / 2, sd / 2
        yb, yt = slabs[i] + 0.15, tops[i] - 0.3
        my = (yb + yt) / 2
        mh = yt - yb
        shrine_parts.extend(_rectWalls([0, my, 0], sw, mh, {'front': [(0, 1.8, 4)]}, sd, density=0.8))
    shrine_sampler = _pick(shrine_parts)
    shrine = HiPart("Shrine hall", shrine_sampler, (1.0, 0.65, 0.2), 1.8,
                    "Inner sanctum with devotional space", (0, 0.5, 0), 0.8)

    # Doorway & finial (hip roof)
    roof_parts = []
    rw, rd = base_w * 0.76 / 2, base_d * 0.76 / 2
    apex_y = 8.9 + 2.3
    apex = [0, apex_y, 0]
    roof_corners = [[rw, 8.9, rd], [-rw, 8.9, rd], [-rw, 8.9, -rd], [rw, 8.9, -rd]]
    ridge_front = [0, 9.6, rd]
    ridge_back = [0, 9.6, -rd]
    roof_parts.append(tri(roof_corners[0], apex, roof_corners[1]))
    roof_parts.append(tri(roof_corners[1], apex, roof_corners[2]))
    roof_parts.append(tri(roof_corners[2], apex, roof_corners[3]))
    roof_parts.append(tri(roof_corners[3], apex, roof_corners[0]))
    roof_parts.append(polyline([ridge_front, apex, ridge_back], weight=0.6))
    roof_parts.append(line(ridge_front, ridge_back, weight=0.3))
    roof_parts.append(line(apex, [0, apex_y + 1.5, 0], weight=0.3))
    roof_parts.append(line([apex[0], apex[1] - 0.3, apex[2]], [apex[0], apex[1], apex[2]], weight=0.02))
    for i in range(4):
        roof_parts.append(line(roof_corners[i], roof_corners[(i + 1) % 4], weight=0.15))
    roof_sampler = _pick(roof_parts)
    roof_hp = HiPart("Doorway & finial", roof_sampler, (1.0, 1.0, 0.85), 1.2,
                     "Hip roof with ridge and finial", (0, 1.0, 0), 0.3)

    # Balustrade posts
    post_parts = []
    for i in range(3):
        sw, sd = base_w * (1 - 0.08 * i) * 1.02 / 2, base_d * (1 - 0.08 * i) * 1.02 / 2
        y = slabs[i]
        for sx in [-sw + 0.3, sw - 0.3]:
            for sz in [-sd + 0.3, sd - 0.3]:
                post_parts.append(line([sx, y, sz], [sx, y + 0.5, sz], weight=0.04))
        n_posts = 8 + i * 2
        for pi in range(n_posts):
            ang = TAU * pi / n_posts
            px = sw * 1.01 * np.cos(ang)
            pz = sd * 1.01 * np.sin(ang)
            post_parts.append(line([px, y, pz], [px, y + 0.5, pz], weight=0.035))
    post_sampler = _pick(post_parts)
    post_hp = HiPart("Balustrade posts", post_sampler, (1.0, 0.88, 0.3), 0.6,
                     "Decorative posts ringing each terrace", (0, -0.2, 0), 1.0)

    parts = [porch, shrine, roof_hp, post_hp]
    return assemble("Turtle Tower", "wonder", parts, n, seed=10, beat_bpm=68.0)


# ═══════════════════════════════════════════════════════════════════════════
#  2  Eiffel Tower
# ═══════════════════════════════════════════════════════════════════════════

def _eiffel_w(y):
    return 58.0 * np.exp(-y / 85.0) + 4.5 * (1 - y / 320.0)


def _eiffel_facePt(side, u, y):
    w = _eiffel_w(y)
    d0 = [np.cos(deg(45)), np.cos(deg(135)), np.cos(deg(225)), np.cos(deg(315))]
    d1 = [np.sin(deg(45)), np.sin(deg(135)), np.sin(deg(225)), np.sin(deg(315))]
    x0 = d0[side] * w
    z0 = d1[side] * w
    x1 = d0[(side + 1) % 4] * w
    z1 = d1[(side + 1) % 4] * w
    x = x0 + (x1 - x0) * u
    z = z0 + (z1 - z0) * u
    return x, z


def _eiffel_leg(k, side, rng):
    y = rng.random(k) * 300.0
    pts = np.zeros((k, 3), dtype=np.float64)
    for i in range(k):
        x, z = _eiffel_facePt(side, 0.05, y[i])
        pts[i] = [x, y[i], z]
    return (pts + rng.normal(0, 0.3, (k, 3))).astype(np.float32)


def _eiffel_inner_leg(k, side, rng):
    y = rng.random(k) * 276.0
    pts = np.zeros((k, 3), dtype=np.float64)
    for i in range(k):
        x, z = _eiffel_facePt(side, 0.2, y[i])
        pts[i] = [x, y[i], z]
    return (pts + rng.normal(0, 0.2, (k, 3))).astype(np.float32)


def eiffel_tower(n):
    # Legs
    leg_parts = []
    for s in range(4):
        def _make_leg(side=s):
            return lambda k, rng: _eiffel_leg(k, side, rng)
        def _make_inner(side=s):
            return lambda k, rng: _eiffel_inner_leg(k, side, rng)
        leg_parts.append(Part(1.0, _make_leg()))
        leg_parts.append(Part(0.4, _make_inner()))
    legs_sampler = _pick(leg_parts)
    legs_hp = HiPart("Legs", legs_sampler, (1.0, 0.72, 0.35), 3.0,
                     "Four curved wrought-iron legs", (0, 0.2, 0), 1.2)

    # Lattice
    lattice_parts = []
    for s in range(4):
        y_vals = linspace(0, 115, 24)
        for j in range(len(y_vals) - 1):
            y0, y1 = y_vals[j], y_vals[j + 1]
            for u0_u1 in [(0.05, 0.35, 'x'), (0.05, 0.25, 'diag')]:
                ua, ub, kind = u0_u1
                x0, z0 = _eiffel_facePt(s, ua, y0)
                x1, z1 = _eiffel_facePt(s, ub, y1)
                x2, z2 = _eiffel_facePt(s, ua, y1)
                x3, z3 = _eiffel_facePt(s, ub, y0)
                if kind == 'x':
                    lattice_parts.append(line([x0, y0, z0], [x1, y1, z1], weight=0.03))
                    lattice_parts.append(line([x3, y0, z3], [x2, y1, z2], weight=0.03))
                else:
                    lattice_parts.append(line([x0, y0, z0], [x2, y1, z2], weight=0.02))
        for j in range(6):
            yb = 115 + j * 27
            yt = min(276, yb + 27)
            for ua in np.linspace(0.08, 0.42, 4):
                x0, z0 = _eiffel_facePt(s, ua, yb)
                x1, z1 = _eiffel_facePt(s, ua + 0.1, yt)
                lattice_parts.append(line([x0, yb, z0], [x1, yt, z1], weight=0.025))
    decorative_arch_parts = []
    for s in range(4):
        x0, z0 = _eiffel_facePt(s, 0.0, 0)
        x1, z1 = _eiffel_facePt(s, 0.0, 57)
        arch_pts = _spline([[x0, 0, z0], [(x0 + x1) / 2, 28, (z0 + z1) / 2], [x1, 57, z1]], 12)
        decorative_arch_parts.append(polyline(arch_pts, weight=0.25))
        x0b, z0b = _eiffel_facePt(s, 0.5, 0)
        x1b, z1b = _eiffel_facePt(s, 0.5, 57)
        arch_pts2 = _spline([[x0b, 0, z0b], [(x0b + x1b) / 2, 28, (z0b + z1b) / 2], [x1b, 57, z1b]], 12)
        decorative_arch_parts.append(polyline(arch_pts2, weight=0.15))
    lattice_sampler = _pick(lattice_parts + decorative_arch_parts)
    lattice_hp = HiPart("Lattice", lattice_sampler, (1.0, 0.55, 0.2), 4.0,
                        "Criss-cross bracing and decorative arches", (0, 0.1, 0), 1.5)

    # Platforms
    platform_parts = []
    for py in [57, 115, 276]:
        pw = _eiffel_w(py)
        psize = pw * 1.35
        platform_parts.append(box_edges([0, py, 0], [psize * 2, 0.5, psize * 2], weight=0.8))
        platform_parts.append(quad([-psize, py, -psize], [psize, py, -psize],
                                   [psize, py, psize], [-psize, py, psize], 0.4))
    plat_sampler = _pick(platform_parts)
    plat_hp = HiPart("Platforms", plat_sampler, (1.0, 0.85, 0.5), 1.5,
                     "Three observation platforms", (0, 0.4, 0), 0.8)

    # Spire
    spire_parts = []
    spire_parts.append(tube_shape([0, 276, 0], [0, 300, 0], 3.5, 1.5, 0.5))
    spire_parts.append(line([0, 300, 0], [0, 330, 0], jitter=0.05, weight=0.3))
    spire_sampler = _pick(spire_parts)
    spire_hp = HiPart("Spire", spire_sampler, (1.0, 1.0, 0.9), 0.4,
                      "Broadcast antenna at the summit", (0, 1.0, 0), 0.3)

    parts = [legs_hp, lattice_hp, plat_hp, spire_hp]
    return assemble("Eiffel Tower", "wonder", parts, n, seed=11, beat_bpm=78.0)


# ═══════════════════════════════════════════════════════════════════════════
#  3  Statue of Liberty
# ═══════════════════════════════════════════════════════════════════════════

def statue_of_liberty(n):
    # Pedestal
    pedestal_parts = []
    tiers = [(0, 3.0, 22.0, 22.0), (3.0, 9.0, 20.0, 20.0),
             (9.0, 23.0, 18.0, 18.0), (23.0, 47.0, 16.5, 16.5)]
    for y0, y1, w, d in tiers:
        my = (y0 + y1) / 2
        h = y1 - y0
        pedestal_parts.append(box_surface([0, my, 0], [w, h, d]))
        pedestal_parts.append(box_edges([0, my, 0], [w, h, d], weight=0.6))
        pedestal_parts.append(quad([-w / 2, y0, -d / 2], [w / 2, y0, -d / 2],
                                   [w / 2, y0, d / 2], [-w / 2, y0, d / 2], 0.3))
    ped_sampler = _pick(pedestal_parts)
    ped_hp = HiPart("Pedestal", ped_sampler, (0.95, 0.9, 0.85), 3.0,
                    "Granite pedestal in four tiers", (0, 0.1, 0), 0.5)

    # Robe & body
    robe_parts = []
    robe_ys = [47.0, 47.5, 52.0, 58.0, 62.0, 66.0, 68.0, 70.0]
    robe_rs = [6.8, 7.0, 7.5, 7.8, 7.2, 6.0, 4.0, 2.5]
    robe_parts.append(revolve(robe_ys, robe_rs, density=2.0, sx=1, sz=0.9))
    robe_parts.append(revolve([70.0, 71.0, 72.5, 74.0], [2.5, 2.8, 2.4, 1.8], density=1.0, sx=1, sz=0.95))
    robe_sampler = _pick(robe_parts)
    robe_hp = HiPart("Robe & body", robe_sampler, (0.3, 1.0, 0.45), 4.0,
                     "Draped copper robe and torso", (0, 0.3, 0), 0.8)

    # Crown & right arm
    crown_parts = []
    for i in range(7):
        ang = deg(360 * i / 7 - 90)
        base_x = 1.3 * np.cos(ang)
        base_z = 1.3 * np.sin(ang)
        crown_parts.append(tube_shape([base_x, 74.0, base_z], [2.3 * np.cos(ang), 76.5, 2.3 * np.sin(ang)], 0.2, 0.08, density=0.4))
    right_shoulder = [1.2, 72.0, 1.8]
    right_elbow = [6.5, 76.0, 4.0]
    right_wrist = [10.5, 80.0, 5.5]
    crown_parts.append(tube_shape(right_shoulder, right_elbow, 0.8, 0.6, density=0.8))
    crown_parts.append(tube_shape(right_elbow, right_wrist, 0.6, 0.4, density=0.6))
    torch_ys = [79.5, 81.0, 81.8, 82.5]
    torch_rs = [0.5, 0.9, 1.2, 1.3]
    crown_parts.append(revolve(torch_ys, torch_rs, density=0.5, center=[10.5, 0, 5.5]))
    crown_sampler = _pick(crown_parts)
    crown_hp = HiPart("Crown & right arm", crown_sampler, (1.0, 0.9, 0.5), 1.5,
                      "Seven-ray crown and raised right arm with torch", (0.6, 0.8, 0.5), 0.6)

    # Left arm & tablet
    left_parts = []
    left_shoulder = [-1.2, 72.0, 1.8]
    left_elbow = [-4.0, 66.0, 3.0]
    left_hand = [-5.5, 62.0, 4.5]
    left_parts.append(tube_shape(left_shoulder, left_elbow, 0.8, 0.5, density=0.7))
    left_parts.append(tube_shape(left_elbow, left_hand, 0.5, 0.35, density=0.5))
    tablet_c = [-6.0, 60.5, 5.0]
    tablet_init = [box_surface(tablet_c, [2.0, 3.0, 0.6]), box_edges(tablet_c, [2.0, 3.0, 0.6], weight=0.3)]
    tablet_trans = transform(tablet_init, rot_z(deg(-12)), tablet_c)
    tablet_trans2 = transform(tablet_trans, rot_x(deg(-10)), tablet_c)
    left_parts.extend(tablet_trans2)
    left_sampler = _pick(left_parts)
    left_hp = HiPart("Left arm & tablet", left_sampler, (0.65, 0.8, 1.0), 1.5,
                     "Left arm cradling the tablet of law", (-0.6, -0.4, 0.5), 0.6)

    parts = [ped_hp, robe_hp, crown_hp, left_hp]
    return assemble("Statue of Liberty", "wonder", parts, n, seed=12, beat_bpm=72.0)


# ═══════════════════════════════════════════════════════════════════════════
#  4  Burj Khalifa
# ═══════════════════════════════════════════════════════════════════════════

def burj_khalifa(n):
    W = 19.0
    dirs = np.array([[np.cos(deg(90)), 0, np.sin(deg(90))],
                     [np.cos(deg(210)), 0, np.sin(deg(210))],
                     [np.cos(deg(330)), 0, np.sin(deg(330))]], dtype=np.float64)
    L0 = 58.0
    Lmin = 9.0
    tower_h = 585.0
    setbacks = 27

    def wing_len(y):
        frac = np.clip(y / tower_h, 0, 1)
        step = np.floor(frac * setbacks) / setbacks
        return L0 - (L0 - Lmin) * step

    tower_parts = []
    n_rings = 40
    for ri in range(n_rings):
        y0 = ri * tower_h / n_rings
        y1 = (ri + 1) * tower_h / n_rings
        L = wing_len(y0)
        actual_L0 = L0 * (1 - 0.08 * (y0 / tower_h))
        actual_Lmin = Lmin * (1 - 0.05 * (y0 / tower_h)) if y0 / tower_h > 0.5 else Lmin
        Lnext = wing_len(min(y1, tower_h))
        for wi in range(3):
            d = dirs[wi]
            p0_base = [0, y0, 0]
            p0_tip = d * L
            p0_tip[1] = y0
            p1_tip = d * Lnext
            p1_tip[1] = y1
            p1_base = [0, y1, 0]
            inner_w = 9.0 * (1 - y0 / tower_h)
            p0_inner = d * inner_w
            p0_inner[1] = y0
            p1_inner = d * inner_w * 0.95
            p1_inner[1] = y1
            tower_parts.append(quad(p0_base, p0_tip, p1_tip, p1_base, weight=0.5 / n_rings))
            tower_parts.append(quad(p0_inner, p0_tip, p1_tip, p1_inner, weight=0.3 / n_rings))
            tower_parts.append(line(p0_tip, p1_tip, weight=0.03 / n_rings))
    for wi in range(3):
        d0 = dirs[wi]
        d1 = dirs[(wi + 1) % 3]
        nv = 6
        for vi in range(1, nv):
            u = vi / nv
            for ri in range(n_rings):
                y0 = ri * tower_h / n_rings
                y1 = (ri + 1) * tower_h / n_rings
                L0_cur = wing_len(y0)
                p0_a = d0 * (L0_cur * u) + [0, y0, 0]
                p0_b = d1 * (L0_cur * (1 - u)) + [0, y0, 0]
                p0 = (p0_a + p0_b) / 2
                L1_cur = wing_len(y1)
                p1_a = d0 * (L1_cur * u) + [0, y1, 0]
                p1_b = d1 * (L1_cur * (1 - u)) + [0, y1, 0]
                p1 = (p1_a + p1_b) / 2
                tower_parts.append(line(p0, p1, weight=0.015 / n_rings))
    tower_sampler = _pick(tower_parts)
    tower_hp = HiPart("Tower", tower_sampler, (0.55, 0.8, 1.0), 6.0,
                      "Y-plan tiers stepping back to a needle", (0, 0.1, 0), 0.5)

    # Spire
    spire_parts = []
    spire_y = [585, 640, 700, 760, 828]
    spire_r = [11, 8, 5, 2.5, 0.3]
    spire_parts.append(revolve(spire_y, spire_r, density=0.6))
    spire_parts.append(line([0, 760, 0], [0, 828, 0], jitter=0.02, weight=0.2))
    spire_sampler = _pick(spire_parts)
    spire_hp = HiPart("Spire", spire_sampler, (1.0, 1.0, 1.0), 0.6,
                      "Telescoping steel spire", (0, 1.0, 0), 0.4)

    parts = [tower_hp, spire_hp]
    return assemble("Burj Khalifa", "wonder", parts, n, seed=13, beat_bpm=80.0)


# ═══════════════════════════════════════════════════════════════════════════
#  5  Great Pyramid
# ═══════════════════════════════════════════════════════════════════════════

def great_pyramid(n):
    H = 146.6
    B = 230.3

    # Casing
    casing_parts = []
    apex = [0, H, 0]
    corners = [[B / 2, 0, B / 2], [-B / 2, 0, B / 2],
               [-B / 2, 0, -B / 2], [B / 2, 0, -B / 2]]
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        casing_parts.append(tri(a, apex, b, weight=1.2))
        casing_parts.append(line(a, apex, weight=0.15))
    casing_parts.append(quad(corners[0], corners[1], corners[2], corners[3], weight=0.3))
    for y in np.arange(12, H - 5, 12):
        s = B / 2 * (1 - y / H)
        pts = [[s, y, s], [-s, y, s], [-s, y, -s], [s, y, -s]]
        casing_parts.append(polyline(pts, closed=True, weight=0.18))
    casing_sampler = _pick(casing_parts)
    casing_hp = HiPart("Casing", casing_sampler, (1.0, 0.8, 0.4), 6.0,
                       "Limestone courses and smooth outer faces", (0, 0.1, 0), 0.3)

    # Capstone
    cap_parts = [sphere([0, H, 0], 3.5, density=1.0)]
    cap_sampler = _pick(cap_parts)
    cap_hp = HiPart("Capstone", cap_sampler, (1.0, 0.95, 0.7), 0.3,
                    "Gilded pyramidion", (0, 1.0, 0), 0.2)

    # King's Chamber
    kc_parts = [box_filled([0, 0.8 * H, 0], [8, 4, 5], density=1.0, weight=1.0)]
    kc_sampler = _pick(kc_parts)
    kc_hp = HiPart("King's Chamber", kc_sampler, (1.0, 0.45, 0.25), 0.8,
                   "Granite burial chamber", (0, -0.2, 0.6), 1.0)

    # Grand Gallery
    gg_parts = []
    base_z = -0.6 * B
    n_seg = 20
    for i in range(n_seg):
        t0 = i / n_seg
        t1 = (i + 1) / n_seg
        p0 = [0, 0.3 * H + 0.5 * H * t0, base_z + 0.6 * B * t0]
        p1 = [0, 0.3 * H + 0.5 * H * t1, base_z + 0.6 * B * t1]
        gg_parts.append(tube_shape(p0, p1, 1.2 - 0.3 * t0, 1.2 - 0.3 * t1, density=0.6))
    gg_sampler = _pick(gg_parts)
    gg_hp = HiPart("Grand Gallery", gg_sampler, (0.9, 0.6, 0.3), 0.8,
                   "Sloping corbelled passage", (0, 0.3, -0.6), 1.2)

    parts = [casing_hp, cap_hp, kc_hp, gg_hp]
    return assemble("Great Pyramid", "wonder", parts, n, seed=14, beat_bpm=60.0)


# ═══════════════════════════════════════════════════════════════════════════
#  6  Colosseum
# ═══════════════════════════════════════════════════════════════════════════

def colosseum(n):
    ell_a, ell_b = 94.5, 78.0
    H = 48.5
    sz_scale = ell_b / ell_a

    def _outer_keep(x, y, z):
        el_r = np.sqrt((x / ell_a) ** 2 + (z / ell_b) ** 2)
        if el_r > 1.02:
            return False
        if el_r < 0.92:
            return True
        if z < -30 and y > 30:
            return False
        if y < 28:
            arch_col = int(np.floor((np.arctan2(z, x) + np.pi) / (TAU / 80)))
            if arch_col % 3 != 0 and el_r > 0.96:
                return False
        return True

    # Outer wall
    outer_parts = [revolve([0, H], [ell_a, ell_a], density=1.5, sz=sz_scale, keep=_outer_keep)]
    outer_sampler = _pick(outer_parts)
    outer_hp = HiPart("Outer wall", outer_sampler, (1.0, 0.45, 0.3), 4.0,
                      "Travertine facade with arched arcades", (0, 0.05, 0), 0.3)

    # Inner cavea
    inner_parts = []
    inner_parts.append(revolve([0, 36], [75, 75], density=0.8, sz=sz_scale))
    for s in np.arange(0.5, 40, 3):
        inner_r = 75 - s * 1.1
        inner_parts.append(revolve([s, s + 0.3], [inner_r, inner_r], density=0.15, sz=sz_scale))
    inner_sampler = _pick(inner_parts)
    inner_hp = HiPart("Inner cavea", inner_sampler, (1.0, 0.55, 0.4), 2.5,
                      "Tiered seating for 50000 spectators", (0, 0.2, 0), 0.5)

    # Arena floor
    arena_parts = []
    arena_r = 42.0
    for ri in range(5):
        r = arena_r + ri
        arena_parts.append(ring([0, 0.5, 0], r, density=0.5, normal=None))
    arena_sampler = _pick(arena_parts)
    arena_hp = HiPart("Arena floor", arena_sampler, (0.9, 0.8, 0.7), 1.0,
                      "Elliptical arena floor", (0, -0.8, 0), 0.8)

    # Radial walls
    radial_parts = []
    for ang_deg in range(0, 360, 10):
        ang = deg(ang_deg)
        ox = 42 * np.cos(ang)
        oz = 42 * np.sin(ang) * sz_scale
        ix = 88 * np.cos(ang)
        iz = 88 * np.sin(ang) * sz_scale
        radial_parts.append(line([ox, 0, oz], [ox, 36, oz], weight=0.02))
        if ang_deg % 30 == 0:
            radial_parts.append(line([ox, 0, oz], [ix, 0, iz], weight=0.04))
    radial_sampler = _pick(radial_parts)
    radial_hp = HiPart("Radial walls", radial_sampler, (0.85, 0.65, 0.5), 1.5,
                       "Structural masonry radiating from arena", (0, 0, 0), 1.0)

    parts = [outer_hp, inner_hp, arena_hp, radial_hp]
    return assemble("Colosseum", "wonder", parts, n, seed=15, beat_bpm=70.0)


# ═══════════════════════════════════════════════════════════════════════════
#  7  Leaning Tower of Pisa
# ═══════════════════════════════════════════════════════════════════════════

def leaning_tower_pisa(n):
    R = 7.75
    H = 56.7
    tilt = rot_z(deg(-3.97))

    # Base & arcades
    base_parts = []
    base_parts.append(revolve([0, 5.5], [R + 0.3, R + 0.3], density=1.2))
    base_parts.append(revolve([0, 5.5], [R - 0.4, R - 0.4], density=0.6))
    for i in range(6):
        y0 = 5.5 + i * 7.0
        y1 = y0 + 5.5
        base_parts.append(revolve([y0, y1], [R + 0.15, R + 0.15], density=1.0))
        base_parts.append(revolve([y0, y1], [R - 0.5, R - 0.5], density=0.4))
        for a in range(0, 360, 15):
            ang = deg(a)
            bx = (R - 0.2) * np.cos(ang)
            bz = (R - 0.2) * np.sin(ang)
            tx = (R + 0.05) * np.cos(ang)
            tz = (R + 0.05) * np.sin(ang)
            base_parts.append(line([bx, y0, bz], [tx, y0, tz], weight=0.02))
            base_parts.append(line([bx, y0, bz], [bx, y1, bz], weight=0.015))
        base_parts.append(ring([0, y0, 0], R + 0.1, density=0.15))
        base_parts.append(ring([0, y1, 0], R + 0.1, density=0.15))
    base_trans = transform(base_parts, tilt)
    base_sampler = _pick(list(base_trans) if isinstance(base_trans[0], Part) else flatten(base_trans))
    base_hp = HiPart("Base & arcades", base_sampler, (0.3, 0.95, 1.0), 3.0,
                     "Six levels of colonnaded galleries", (0.5, 0.1, 0), 0.4)

    # Belfry
    belfry_parts = []
    belfry_parts.append(revolve([41.5, 46.0], [R - 0.3, R - 0.3], density=0.8))
    belfry_parts.append(revolve([46.0, 46.8], [R - 0.3, R + 0.1], density=0.3))
    belfry_parts.append(revolve([46.8, 47.2], [R + 0.1, R + 0.1], density=0.2))
    belfry_parts.append(ring([0, 46.8, 0], R + 0.15, density=0.2))
    belfry_parts.append(revolve([47.2, 51.0], [R - 1.5, R - 1.5], density=0.7))
    belfry_parts.append(revolve([51.0, 52.5], [R - 1.5, R - 0.3], density=0.3))
    belfry_parts.append(revolve([52.5, 54.0], [R - 0.3, R - 1.5], density=0.25))
    belfry_parts.append(revolve([54.0, 54.8], [R - 1.2, R - 1.0], density=0.2))
    belfry_parts.append(revolve([54.8, 55.5], [R - 1.0, R - 0.8], density=0.2))
    belfry_parts.append(revolve([55.5, 56.7], [R - 0.8, R], density=0.3))
    belfry_trans = transform(belfry_parts, tilt)
    belfry_sampler = _pick(list(belfry_trans) if isinstance(belfry_trans[0], Part) else flatten(belfry_trans))
    belfry_hp = HiPart("Belfry", belfry_sampler, (1.0, 0.78, 0.6), 2.0,
                       "Smaller-diameter bell chamber at top", (0, 1.0, 0), 0.5)

    # Column galleries
    col_parts = []
    for i in range(6):
        y0 = 5.5 + i * 7.0
        y1 = y0 + 5.5
        col_parts.append(revolve([y0, y1], [R - 1.8, R - 1.8], density=0.5))
        for a in range(0, 360, 15):
            ang = deg(a)
            cx = (R - 1.0) * np.cos(ang)
            cz = (R - 1.0) * np.sin(ang)
            col_parts.append(line([cx, y0, cz], [cx, y1, cz], weight=0.012))
        for a in range(0, 360, 30):
            ang = deg(a)
            ca1 = deg(a - 5)
            ca2 = deg(a + 5)
            arc_r = 1.4
            n_pts = 6
            for j in range(n_pts):
                at0 = ca1 + (ca2 - ca1) * j / n_pts
                at1 = ca1 + (ca2 - ca1) * (j + 1) / n_pts
                p0 = [arc_r * np.cos(at0), y1 - 1.2, arc_r * np.sin(at0)]
                p1 = [arc_r * np.cos(at1), y1 - 1.2, arc_r * np.sin(at1)]
                col_parts.append(line(p0, p1, weight=0.008))
    col_trans = transform(col_parts, tilt)
    col_sampler = _pick(list(col_trans) if isinstance(col_trans[0], Part) else flatten(col_trans))
    col_hp = HiPart("Column galleries", col_sampler, (0.95, 0.9, 0.8), 2.5,
                    "Inner cylinder with colonnade arches", (0, 0.3, 0), 0.6)

    parts = [base_hp, belfry_hp, col_hp]
    return assemble("Leaning Tower of Pisa", "wonder", parts, n, seed=16, beat_bpm=66.0)


# ═══════════════════════════════════════════════════════════════════════════
#  8  Taj Mahal
# ═══════════════════════════════════════════════════════════════════════════

def taj_mahal(n):
    plinth_w, plinth_h = 95.0, 6.0
    hw = plinth_w / 2

    # Plinth & terrace
    plinth_parts = []
    plinth_parts.append(box_surface([0, plinth_h / 2, 0], [plinth_w, plinth_h, plinth_w]))
    plinth_parts.append(box_edges([0, plinth_h / 2, 0], [plinth_w, plinth_h, plinth_w], weight=0.6))
    plinth_sampler = _pick(plinth_parts)
    plinth_hp = HiPart("Plinth & terrace", plinth_sampler, (1.0, 0.55, 0.35), 3.0,
                       "Sandstone plinth with marble terrace", (0, 0.02, 0), 0.3)

    # Main hall
    main_parts = []
    main_w = 57.0
    main_hw = main_w / 2
    main_h = 33.0
    main_y0 = plinth_h
    my = main_y0 + main_h / 2
    main_arches = [(0, 16.0, 22.0)]
    main_parts.extend(_rectWalls([0, my, 0], main_w, main_h,
                                 {'front': main_arches, 'back': main_arches, 'left': [(0, 10, 15)], 'right': [(0, 10, 15)]},
                                 main_w, density=1.2))
    chamfer_w = main_w * 0.25
    for sx in [-main_hw, main_hw]:
        for sz in [-main_hw, main_hw]:
            main_parts.append(quad([sx, main_y0, sz + chamfer_w], [sx + chamfer_w, main_y0, sz],
                                   [sx + chamfer_w, main_y0 + main_h, sz], [sx, main_y0 + main_h, sz + chamfer_w], weight=0.25))
    drum_y = main_y0 + main_h
    drum_h = 3.0
    main_parts.append(revolve([drum_y, drum_y + drum_h], [main_hw - 1, main_hw - 1], density=0.8))
    dome_base_y = drum_y + drum_h
    dome_apex_y = dome_base_y + 40.0
    dome_ys = [dome_base_y, dome_base_y + 5, dome_base_y + 15, dome_base_y + 28, dome_apex_y]
    dome_rs = [main_hw - 1, main_hw - 0.5, main_hw - 3, main_hw - 8, 0.5]
    main_parts.append(revolve(dome_ys, dome_rs, density=2.0, a0=0, a1=TAU))
    main_parts.append(line([0, dome_apex_y, 0], [0, dome_apex_y + 8, 0], weight=0.3))
    main_sampler = _pick(main_parts)
    main_hp = HiPart("Main hall", main_sampler, (0.95, 0.9, 0.85), 3.5,
                     "Marble iwan arches and iconic onion dome", (0, 0.5, 0), 0.4)

    # Chattris & pinnacles
    chattri_parts = []
    chat_r = 6.0
    for sx in [-hw + 12, hw - 12]:
        for sz in [-hw + 12, hw - 12]:
            chattri_parts.append(cylinder_s([sx, plinth_h, sz], chat_r, 10, density=0.4, cap=False))
            chattri_parts.append(revolve([plinth_h + 10, plinth_h + 14, plinth_h + 16],
                                         [chat_r, chat_r - 1, 0.3], density=0.3, center=[sx, 0, sz]))
            chattri_parts.append(line([sx, plinth_h + 10, sz], [sx, plinth_h + 10, sz], weight=0.06))
    for sx in [-hw + 6, hw - 6]:
        for sz in [-hw + 6, hw - 6]:
            chattri_parts.append(line([sx, plinth_h + 36, sz], [sx, plinth_h + 43, sz], weight=0.06))
    chattri_sampler = _pick(chattri_parts)
    chattri_hp = HiPart("Chattris & pinnacles", chattri_sampler, (1.0, 0.8, 0.5), 2.0,
                        "Four corner pavilions with small domes", (0.3, 0.6, 0.3), 0.5)

    # Minarets
    minaret_parts = []
    min_r = 3.5
    min_h = 42.0
    for sx in [-hw + 4, hw - 4]:
        for sz in [-hw + 4, hw - 4]:
            minaret_parts.append(revolve([plinth_h, plinth_h + min_h],
                                         [min_r, min_r], density=0.6, center=[sx, 0, sz]))
            for ry in [plinth_h + 10, plinth_h + 20, plinth_h + 30]:
                minaret_parts.append(ring([sx, ry, sz], min_r + 0.3, density=0.15))
            minaret_parts.append(revolve([plinth_h + min_h, plinth_h + min_h + 5],
                                         [min_r, 0.3], density=0.25, center=[sx, 0, sz]))
            minaret_parts.append(line([sx, plinth_h + min_h + 5, sz], [sx, plinth_h + min_h + 8, sz], weight=0.05))
    minaret_sampler = _pick(minaret_parts)
    minaret_hp = HiPart("Minarets", minaret_sampler, (0.8, 0.85, 0.95), 2.5,
                        "Four slender minarets at plinth corners", (0.5, 0.2, 0.5), 0.6)

    parts = [plinth_hp, main_hp, chattri_hp, minaret_hp]
    return assemble("Taj Mahal", "wonder", parts, n, seed=17, beat_bpm=64.0)


# ═══════════════════════════════════════════════════════════════════════════
#  9  Big Ben
# ═══════════════════════════════════════════════════════════════════════════

def big_ben(n):
    shaft_w, shaft_h = 12.0, 56.0
    hw = shaft_w / 2

    # Shaft
    shaft_parts = []
    shaft_parts.append(box_surface([0, shaft_h / 2, 0], [shaft_w, shaft_h, shaft_w]))
    shaft_parts.append(box_edges([0, shaft_h / 2, 0], [shaft_w, shaft_h, shaft_w], weight=0.5))
    for sx in [-hw + 1.5, hw - 1.5]:
        for sz in [-hw + 1.5, hw - 1.5]:
            shaft_parts.append(line([sx, 0, sz], [sx, shaft_h, sz], weight=0.08))
    for cy in np.arange(8, shaft_h - 5, 10):
        shaft_parts.append(polyline([[hw, cy, -hw], [-hw, cy, -hw], [-hw, cy, hw], [hw, cy, hw]], closed=True, weight=0.1))
    shaft_sampler = _pick(shaft_parts)
    shaft_hp = HiPart("Shaft", shaft_sampler, (0.95, 0.9, 0.7), 3.0,
                      "Limestone shaft with Gothic pilasters", (0, 0.03, 0), 0.3)

    # Clock stage
    clock_parts = []
    clock_y0 = shaft_h
    clock_h = 15.0
    clock_c = [0, clock_y0 + clock_h / 2, 0]
    clock_parts.append(box_surface(clock_c, [shaft_w - 1, clock_h, shaft_w - 1]))
    clock_parts.append(box_edges(clock_c, [shaft_w - 1, clock_h, shaft_w - 1], weight=0.3))
    dial_r = 3.5
    for face_z in [-(hw - 0.5), (hw - 0.5)]:
        dial_c = [0, clock_y0 + clock_h / 2, face_z]
        clock_parts.append(disc(dial_c, 0, dial_r, [0, 0, 1], density=0.5))
        clock_parts.append(ring(dial_c, dial_r, density=0.3))
        clock_parts.append(ring(dial_c, dial_r - 0.5, density=0.15))
        for hi in range(12):
            ang = deg(30 * hi - 90)
            mx = dial_r * 0.85 * np.cos(ang)
            my = clock_y0 + clock_h / 2 + dial_r * 0.85 * np.sin(ang)
            clock_parts.append(line([dial_c[0] + mx, my, dial_c[2]], [dial_c[0] + mx * 1.05, my - dial_r * 0.05, dial_c[2]], weight=0.01))
        m_ang = deg(150)
        h_ang = deg(210)
        clock_parts.append(line(dial_c, [dial_c[0] + dial_r * 0.6 * np.cos(m_ang), clock_y0 + clock_h / 2 + dial_r * 0.6 * np.sin(m_ang), dial_c[2]], weight=0.02))
        clock_parts.append(line(dial_c, [dial_c[0] + dial_r * 0.4 * np.cos(h_ang), clock_y0 + clock_h / 2 + dial_r * 0.4 * np.sin(h_ang), dial_c[2]], weight=0.02))
    for face_x in [-(hw - 0.5), (hw - 0.5)]:
        dial_c = [face_x, clock_y0 + clock_h / 2, 0]
        clock_parts.append(disc(dial_c, 0, dial_r, [0, 0, 1], density=0.5))
        clock_parts.append(ring(dial_c, dial_r, density=0.3))
        clock_parts.append(ring(dial_c, dial_r - 0.5, density=0.15))
        for hi in range(12):
            ang = deg(30 * hi - 90)
            mx = dial_r * 0.85 * np.cos(ang)
            my = clock_y0 + clock_h / 2 + dial_r * 0.85 * np.sin(ang)
            clock_parts.append(line([dial_c[0], my, dial_c[2] + mx], [dial_c[0], my - dial_r * 0.05, dial_c[2] + mx * 1.05], weight=0.01))
        m_ang = deg(150)
        h_ang = deg(210)
        clock_parts.append(line(dial_c, [dial_c[0], clock_y0 + clock_h / 2 + dial_r * 0.6 * np.sin(m_ang), dial_c[2] + dial_r * 0.6 * np.cos(m_ang)], weight=0.02))
        clock_parts.append(line(dial_c, [dial_c[0], clock_y0 + clock_h / 2 + dial_r * 0.4 * np.sin(h_ang), dial_c[2] + dial_r * 0.4 * np.cos(h_ang)], weight=0.02))
    clock_sampler = _pick(clock_parts)
    clock_hp = HiPart("Clock stage", clock_sampler, (1.0, 0.8, 0.45), 2.5,
                      "Four illuminated 7 m dials", (0, 0.6, 0), 0.5)

    # Belfry & spire
    belfry_parts = []
    belfry_y0 = shaft_h + clock_h
    belfry_h = 18.0
    belfry_w = shaft_w - 2
    belfry_c = [0, belfry_y0 + belfry_h / 2, 0]
    belfry_arches = [(0, 5.0, 9.0)]
    arch_spec = {'front': belfry_arches, 'back': belfry_arches, 'left': belfry_arches, 'right': belfry_arches}
    belfry_parts.extend(_rectWalls(belfry_c, belfry_w, belfry_h, arch_spec, belfry_w, density=0.8))
    for sx in [-belfry_w / 2 + 0.5, belfry_w / 2 - 0.5]:
        for sz in [-belfry_w / 2 + 0.5, belfry_w / 2 - 0.5]:
            pin_ys = [belfry_y0, belfry_y0 + 6, belfry_y0 + 8]
            pin_rs = [0.4, 0.4, 0.15]
            belfry_parts.append(revolve(pin_ys, pin_rs, density=0.2, center=[sx, 0, sz]))
    spire_y0 = belfry_y0 + belfry_h
    spire_apex = 96.0
    spire_hw = belfry_w / 2
    spire_corners = [[spire_hw, spire_y0, spire_hw], [-spire_hw, spire_y0, spire_hw],
                     [-spire_hw, spire_y0, -spire_hw], [spire_hw, spire_y0, -spire_hw]]
    spire_apex_pt = [0, spire_apex, 0]
    ridge_n = [0, spire_apex - 4, spire_apex_pt[2]]
    ridge_s = [0, spire_apex - 4, -spire_apex_pt[2]]
    belfry_parts.append(tri(spire_corners[0], spire_apex_pt, spire_corners[1], weight=0.3))
    belfry_parts.append(tri(spire_corners[1], spire_apex_pt, spire_corners[2], weight=0.3))
    belfry_parts.append(tri(spire_corners[2], spire_apex_pt, spire_corners[3], weight=0.3))
    belfry_parts.append(tri(spire_corners[3], spire_apex_pt, spire_corners[0], weight=0.3))
    belfry_parts.append(polyline([spire_corners[0], ridge_n, spire_corners[1]], weight=0.12))
    belfry_parts.append(polyline([spire_corners[2], ridge_s, spire_corners[3]], weight=0.12))
    belfry_parts.append(polyline([ridge_n, spire_apex_pt, ridge_s], weight=0.1))
    for i in range(4):
        belfry_parts.append(line(spire_corners[i], spire_apex_pt, weight=0.06))
    for sx in [-spire_hw + 0.6, spire_hw - 0.6]:
        for sz in [-spire_hw + 0.6, spire_hw - 0.6]:
            belfry_parts.append(line([sx, spire_y0, sz], [sx * 0.3, spire_apex - 3, sz * 0.3], weight=0.03))
    belfry_sampler = _pick(belfry_parts)
    belfry_hp = HiPart("Belfry & spire", belfry_sampler, (0.9, 0.75, 0.4), 2.0,
                       "Gothic belfry and gilded spire", (0, 0.8, 0), 0.7)

    # Flèche & cross
    fleche_parts = []
    fleche_parts.append(line(spire_apex_pt, [0, spire_apex + 4, 0], weight=0.1))
    fleche_parts.append(line([-0.5, spire_apex + 2, 0], [0.5, spire_apex + 2, 0], weight=0.05))
    fleche_sampler = _pick(fleche_parts)
    fleche_hp = HiPart("Flèche & cross", fleche_sampler, (1.0, 0.95, 0.5), 0.5,
                       "Crowning flèche", (0, 1.0, 0), 0.4)

    parts = [shaft_hp, clock_hp, belfry_hp, fleche_hp]
    return assemble("Big Ben", "wonder", parts, n, seed=18, beat_bpm=76.0)


# ═══════════════════════════════════════════════════════════════════════════
# 10  Christ the Redeemer
# ═══════════════════════════════════════════════════════════════════════════

def christ_redeemer(n):
    pedestal_h = 8.0

    # Pedestal & summit
    ped_parts = []
    summit_ys = [0, 8, 14, 20]
    summit_rs = [12, 10, 7, 5]
    ped_parts.append(revolve(summit_ys, summit_rs, density=0.6))
    ped_parts.append(box_surface([0, pedestal_h + 0.5, 0], [10, 1, 10]))
    ped_parts.append(box_edges([0, pedestal_h + 0.5, 0], [10, 1, 10], weight=0.3))
    ped_sampler = _pick(ped_parts)
    ped_hp = HiPart("Pedestal & summit", ped_sampler, (0.8, 0.75, 0.7), 2.0,
                    "Corcovado summit and pedestal base", (0, 0.05, 0), 0.3)

    # Robe
    robe_parts = []
    robe_ys = [pedestal_h, pedestal_h + 1, pedestal_h + 5, pedestal_h + 10,
               pedestal_h + 14, pedestal_h + 17, pedestal_h + 19, pedestal_h + 20]
    robe_rs = [3.5, 4.0, 4.8, 5.0, 4.5, 3.5, 2.5, 1.5]
    robe_parts.append(revolve(robe_ys, robe_rs, density=1.8, sx=1, sz=0.75))
    for i in range(6):
        ang = deg(60 * i - 30)
        rx = 4.0 * np.cos(ang)
        rz = 4.0 * np.sin(ang) * 0.75
        robe_parts.append(line([rx, pedestal_h + 2, rz], [rx * 1.2, pedestal_h + 16, rz * 1.2], weight=0.04))
    robe_sampler = _pick(robe_parts)
    robe_hp = HiPart("Robe", robe_sampler, (0.85, 0.92, 1.0), 3.0,
                     "Soapstone robe with cascading folds", (0, 0.4, 0), 0.5)

    # Head & arms
    head_parts = []
    head_y0 = pedestal_h + 20
    head_parts.append(revolve([head_y0, head_y0 + 1.5, head_y0 + 3.0, head_y0 + 3.8],
                              [1.2, 1.5, 1.4, 0.3], density=0.8, sx=1, sz=0.85))
    right_shoulder = [1.5, head_y0 - 3, 2.0]
    right_elbow = [6.5, head_y0 - 1, 5.0]
    right_hand = [10.5, head_y0 + 2, 7.0]
    head_parts.append(tube_shape(right_shoulder, right_elbow, 0.8, 0.6, density=0.5))
    head_parts.append(tube_shape(right_elbow, right_hand, 0.6, 0.4, density=0.4))
    head_parts.append(ellipsoid(right_hand, [0.8, 0.6, 0.4], density=0.3))
    left_shoulder = [-1.5, head_y0 - 3, 2.0]
    left_elbow = [-4.0, head_y0 - 5, 3.5]
    left_hand = [-5.5, head_y0, 5.0]
    head_parts.append(tube_shape(left_shoulder, left_elbow, 0.8, 0.5, density=0.5))
    head_parts.append(tube_shape(left_elbow, left_hand, 0.5, 0.35, density=0.4))
    head_parts.append(ellipsoid(left_hand, [0.7, 0.5, 0.35], density=0.3))
    for side, shoulder in [(1, right_shoulder), (-1, left_shoulder)]:
        sleeve_w = 2.0
        sleeve_h = 2.5
        head_parts.append(quad(
            [shoulder[0], shoulder[1] - sleeve_h / 2, shoulder[2] - sleeve_w / 2 * side],
            [shoulder[0], shoulder[1] + sleeve_h / 2, shoulder[2] - sleeve_w / 2 * side],
            [shoulder[0], shoulder[1] + sleeve_h / 2, shoulder[2] + sleeve_w / 2 * side],
            [shoulder[0], shoulder[1] - sleeve_h / 2, shoulder[2] + sleeve_w / 2 * side],
            weight=0.08))
    head_sampler = _pick(head_parts)
    head_hp = HiPart("Head & arms", head_sampler, (1.0, 0.85, 0.7), 2.5,
                     "Head, outstretched arms and hands", (0.3, 0.7, 0.3), 0.6)

    # Cross lines
    cross_parts = []
    hand_mid = [(right_hand[0] + left_hand[0]) / 2, (right_hand[1] + left_hand[1]) / 2, (right_hand[2] + left_hand[2]) / 2]
    cross_parts.append(line(right_hand, left_hand, weight=0.2))
    for i in range(12):
        ang = deg(30 * i)
        rx = 3.8 * np.cos(ang)
        rz = 3.8 * np.sin(ang) * 0.75
        cross_parts.append(line([rx, pedestal_h + 8, rz], [rx * 0.9, pedestal_h + 18, rz * 0.9], weight=0.03))
    cross_sampler = _pick(cross_parts)
    cross_hp = HiPart("Cross lines", cross_sampler, (0.5, 0.6, 0.8), 0.5,
                      "Outstretched arms forming a cross", (0.5, 0.2, 0), 0.8)

    parts = [ped_hp, robe_hp, head_hp, cross_hp]
    return assemble("Christ the Redeemer", "wonder", parts, n, seed=19, beat_bpm=62.0)


# ═══════════════════════════════════════════════════════════════════════════
# 11  Sydney Opera House
# ═══════════════════════════════════════════════════════════════════════════

def sydney_opera_house(n):
    podium_w, podium_d, podium_h = 183.0, 120.0, 12.0

    # Podium
    podium_parts = []
    podium_parts.append(box_surface([0, podium_h / 2, 0], [podium_w, podium_h, podium_d]))
    podium_parts.append(box_edges([0, podium_h / 2, 0], [podium_w, podium_h, podium_d], weight=0.4))
    for z_step in np.linspace(-podium_d / 2 + 5, podium_d / 2 - 5, 12):
        podium_parts.append(line([-podium_w / 2, podium_h, z_step], [podium_w / 2, podium_h, z_step], weight=0.02))
    pod_sampler = _pick(podium_parts)
    pod_hp = HiPart("Podium", pod_sampler, (0.5, 0.45, 0.4), 2.5,
                    "Granite podium with monumental steps", (0, 0.02, 0), 0.3)

    # Shared shell helper
    def _make_shell(origin_x, origin_z, span_x, span_z, apex_y, spine_lean_x, n_ribs=6):
        parts = []
        spine_top = [origin_x + spine_lean_x, podium_h + apex_y, origin_z]

        def shell_skin(t, s):
            side_x = origin_x + span_x * t - span_x * 0.5 + spine_lean_x * (1 - s)
            side_z = origin_z + span_z * (s - 0.5)
            ridge_x = origin_x + spine_lean_x * (1 - s)
            ridge_y = podium_h + apex_y * (1 - s)
            bx = side_x + (ridge_x - side_x) * t
            by = ridge_y * t
            bz = side_z + (origin_z - side_z) * t
            return [bx, by, bz]

        parts.append(param_surface(shell_skin, 0.05, 0.95, 0.05, 0.95, density=1.2))
        n_s = 15
        for si in range(n_s + 1):
            s = si / n_s
            ridge_x = origin_x + spine_lean_x * (1 - s)
            ridge_y = podium_h + apex_y * (1 - s)
            edge_pts = []
            for ti in range(n_s + 1):
                t = ti / n_s
                edge_x = origin_x + span_x * t - span_x * 0.5 + spine_lean_x * (1 - s)
                edge_z = origin_z + span_z * (s - 0.5)
                edge_y = ridge_y * t
                edge_pts.append([edge_x, edge_y, edge_z])
            parts.append(polyline(edge_pts, weight=0.015))
        for ri in range(n_ribs + 1):
            t = 0.1 + 0.8 * ri / n_ribs
            rib_pts = []
            for si in range(n_s + 1):
                s = si / n_s
                side_x = origin_x + span_x * t - span_x * 0.5 + spine_lean_x * (1 - s)
                side_z = origin_z + span_z * (s - 0.5)
                ridge_x = origin_x + spine_lean_x * (1 - s)
                ridge_y = podium_h + apex_y * (1 - s)
                bx = side_x + (ridge_x - side_x) * t
                by = ridge_y * t
                bz = side_z + (origin_z - side_z) * t
                rib_pts.append([bx, by, bz])
            parts.append(polyline(rib_pts, weight=0.02))
        ridge_pts = []
        for si in range(n_s + 1):
            s = si / n_s
            ridge_x = origin_x + spine_lean_x * (1 - s)
            ridge_y = podium_h + apex_y * (1 - s)
            ridge_pts.append([ridge_x, ridge_y, origin_z])
        parts.append(polyline(ridge_pts, weight=0.05))
        glass_pts = []
        for ti in range(n_s + 1):
            t = ti / n_s
            glass_pts.append([origin_x + span_x * t - span_x * 0.5, podium_h, origin_z + span_z * (t - 0.5)])
        parts.append(quad(glass_pts[0], glass_pts[-1], ridge_pts[-1], ridge_pts[0], weight=0.3))
        return parts

    # Concert Hall shells (right side, 4 major shells)
    concert_parts = []
    concert_specs = [
        (38, 0, 42, 40, 67, -15),
        (82, 8, 38, 36, 58, -10),
        (118, 14, 32, 30, 46, -6),
        (146, 18, 24, 22, 32, -3),
    ]
    for ox, oz, sx, sz, ay, slx in concert_specs:
        concert_parts.extend(_make_shell(ox, oz, sx, sz, ay, slx))
    concert_sampler = _pick(concert_parts)
    concert_hp = HiPart("Concert Hall shells", concert_sampler, (0.95, 0.95, 0.8), 4.0,
                        "Four soaring shells over the Concert Hall", (0.4, 0.6, 0.2), 0.5)

    # Theatre shells (left side, 4 shells mirrored)
    theatre_parts = []
    theatre_specs = [
        (-38, 0, 42, 40, 67, 15),
        (-82, 8, 38, 36, 58, 10),
        (-118, 14, 32, 30, 46, 6),
        (-146, 18, 24, 22, 32, 3),
    ]
    for ox, oz, sx, sz, ay, slx in theatre_specs:
        theatre_parts.extend(_make_shell(ox, oz, sx, sz, ay, slx))
    theatre_sampler = _pick(theatre_parts)
    theatre_hp = HiPart("Theatre shells", theatre_sampler, (0.9, 0.9, 0.75), 3.5,
                        "Four shells covering the Opera Theatre", (-0.4, 0.6, 0.2), 0.5)

    # Restaurant shells
    resto_parts = []
    resto_specs = [
        (8, 50, 15, 18, 18, -1),
        (-8, 50, 15, 18, 18, 1),
    ]
    for ox, oz, sx, sz, ay, slx in resto_specs:
        resto_parts.extend(_make_shell(ox, oz, sx, sz, ay, slx))
    resto_sampler = _pick(resto_parts)
    resto_hp = HiPart("Restaurant shells", resto_sampler, (0.85, 0.85, 0.7), 2.0,
                      "Two smaller shells for the restaurant", (0, 0.5, 0.8), 0.5)

    parts = [pod_hp, concert_hp, theatre_hp, resto_hp]
    return assemble("Sydney Opera House", "wonder", parts, n, seed=20, beat_bpm=74.0)


# ═══════════════════════════════════════════════════════════════════════════

WONDERS = [
    {"name": "Turtle Tower", "build": turtle_tower, "color": (1.0, 0.84, 0.1)},
    {"name": "Eiffel Tower", "build": eiffel_tower, "color": (1.0, 0.35, 0.95)},
    {"name": "Statue of Liberty", "build": statue_of_liberty, "color": (0.3, 1.0, 0.45)},
    {"name": "Burj Khalifa", "build": burj_khalifa, "color": (0.55, 0.8, 1.0)},
    {"name": "Great Pyramid", "build": great_pyramid, "color": (1.0, 0.8, 0.4)},
    {"name": "Colosseum", "build": colosseum, "color": (1.0, 0.45, 0.3)},
    {"name": "Leaning Tower of Pisa", "build": leaning_tower_pisa, "color": (0.3, 0.95, 1.0)},
    {"name": "Taj Mahal", "build": taj_mahal, "color": (0.95, 0.9, 0.85)},
    {"name": "Big Ben", "build": big_ben, "color": (1.0, 0.8, 0.45)},
    {"name": "Christ the Redeemer", "build": christ_redeemer, "color": (0.85, 0.92, 1.0)},
    {"name": "Sydney Opera House", "build": sydney_opera_house, "color": (0.95, 0.95, 0.8)},
]