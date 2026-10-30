import numpy as np
from .base import Part as HiPart, assemble
from .shapes import (Part, line, polyline, tri, quad, box_surface, box_edges, box_filled,
                     revolve, cylinder_s, ring, disc, tube_shape, param_surface,
                     torus_shape, helix_shape, ellipsoid, capsule, blob, dots,
                     transform, orient, _pick, flatten,
                     rot_x, rot_z, deg, TAU, sphere, bone,
                     linspace, range_n, _spline, gear_shape)


def _vessel(ctrl, r0, r1, density=1, per=8):
    pts = _spline(ctrl, per)
    segs = [tube_shape(pts[i], pts[i + 1],
                       r0 + (r1 - r0) * i / (len(pts) - 1),
                       r0 + (r1 - r0) * (i + 1) / (len(pts) - 1),
                       density) for i in range(len(pts) - 1)]
    return Part(1, sample=_pick(segs))


def _path_tube(pts, r, density=1):
    pts = np.asarray(pts, dtype=np.float64)
    segs = [tube_shape(pts[i], pts[i + 1], r, r, density) for i in range(len(pts) - 1)]
    return Part(1, sample=_pick(segs))


def _polyline(pts, jitter=0, weight=1, closed=False):
    pts = np.asarray(pts, dtype=np.float64)
    segs = len(pts) - 1
    if closed:
        segs = len(pts)
    diff_arr = np.diff(pts, axis=0)
    if closed:
        diff_arr = np.vstack([diff_arr, pts[0] - pts[-1]])
    lengths = np.linalg.norm(diff_arr, axis=1)

    def sample(k, rng):
        seg = rng.choice(segs, size=k, p=lengths / lengths.sum())
        t = rng.random(k)
        a = pts[seg]
        b = pts[(seg + 1) % len(pts)]
        p = a + (b - a) * t[:, None]
        return np.asarray(p + rng.normal(0, max(jitter, 1e-9), size=(k, 3)), dtype=np.float32)
    return Part(weight * lengths.sum(), sample)


def _parts(*args):
    all_items = []
    for a in args:
        if isinstance(a, list):
            all_items.extend(a)
        elif isinstance(a, Part):
            all_items.append(a)
    return all_items


def _tree(out, tips, p, d, L, r, depth, rng, spread=0.5):
    if depth <= 0:
        tips.append(p + d * L)
        return
    pp = p + d * L
    out.append(tube_shape(p, pp, r, r * 0.8))
    tips.append(pp)
    for s in (-1, 1):
        angle = spread * (0.8 + 0.4 * rng.random())
        axis = np.array(d) / np.linalg.norm(d)
        if abs(axis[0]) < 0.9:
            perp = np.cross(axis, [1, 0, 0])
        else:
            perp = np.cross(axis, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        rot_axis = perp if s > 0 else -perp
        c = np.cos(angle)
        s = np.sin(angle)
        k = 1 - c
        ux, uy, uz = rot_axis
        R = np.array([[c + ux*ux*k, ux*uy*k - uz*s, ux*uz*k + uy*s],
                      [uy*ux*k + uz*s, c + uy*uy*k, uy*uz*k - ux*s],
                      [uz*ux*k - uy*s, uz*uy*k + ux*s, c + uz*uz*k]])
        new_d = R @ axis
        _tree(out, tips, pp, new_d, L * 0.74, r * 0.72, depth - 1, rng, spread * 0.85)


def _lung_surface(s):
    def fn(u, v):
        rx = 52 + np.cos(2 * u) * 6
        ry = 65 + np.cos(v * 1.5) * 8
        rz = 48 * (1 + 0.15 * np.sin(u))
        x = s * rx * np.cos(u) + 92 * s
        y_base = ry * np.sin(v)
        y = np.where(y_base < 0, y_base * 0.62, y_base)
        z = rz * np.cos(v) * np.cos(u * 0.7)
        if s < 0:
            if z > 5 - 1.1 * (y_base):
                z *= 0.6
            if abs(x + 92) < 25 and v > 0.15:
                notch = 1.0 - 0.3 * np.exp(-((x + 92) / 25) ** 2) * np.exp(-((v - 0.3) / 0.2) ** 2)
                y *= notch
        return np.asarray([x, y, z])
    return param_surface(fn, 0, TAU, -0.55, PI / 2 - 0.25, density=0.8)


PI = np.pi


def lungs(n):
    rng = np.random.default_rng(42)

    rl_surface = _lung_surface(1)
    ll_surface = _lung_surface(-1)

    trachea_pts = np.array([[0, 72, 0], [0, 55, 0], [0, 38, 1], [0, 25, 3]])
    trachea_pipe = _path_tube(trachea_pts, 6)
    glottis = ellipsoid([0, 78, 0], [8, 6, 8])
    trachea_rings = []
    for yi, (a, b) in enumerate([(70, 72), (62, 64), (54, 56), (46, 48)]):
        for yi_ring in np.linspace(a, b, 4):
            trachea_rings.extend(orient([ring([0, yi_ring, 0], 7, density=0.5)], (1, 0.3, 0),
                                        [0, yi_ring, 0]))

    main_bronchi = []
    main_bronchi.append(tube_shape([0, 38, 0], [15, 30, 5], 5, 3.5))
    main_bronchi.append(tube_shape([0, 38, 0], [-15, 30, -5], 5, 3.5))

    tree_tips = []
    tree_parts = []
    _tree(tree_parts, tree_tips, [15, 30, 5],
          [0.7, -0.6, 0.3] / np.linalg.norm([0.7, -0.6, 0.3]), 12, 3.5, 5, rng)
    _tree(tree_parts, tree_tips, [-15, 30, -5],
          [-0.7, -0.6, -0.3] / np.linalg.norm([-0.7, -0.6, -0.3]), 12, 3.5, 5, rng)

    alveoli_groups = []
    for tip in tree_tips:
        for _ in range(3):
            off = rng.normal(0, 2, 3)
            alveoli_groups.append(tip + off)

    arteries_ctrl_r = [[0, 55, 3], [5, 45, 2], [10, 35, 1]]
    arteries_ctrl_l = [[0, 55, -3], [-5, 45, -2], [-10, 35, -1]]
    artery_rtree_tips = []
    artery_rtree_parts = []
    artery_ltree_tips = []
    artery_ltree_parts = []
    _tree(artery_rtree_parts, artery_rtree_tips, [10, 35, 1],
          [0.5, -0.7, 0.3] / np.linalg.norm([0.5, -0.7, 0.3]), 14, 3, 4, rng, 0.5)
    _tree(artery_ltree_parts, artery_ltree_tips, [-10, 35, -1],
          [-0.5, -0.7, -0.3] / np.linalg.norm([-0.5, -0.7, -0.3]), 14, 3, 4, rng, 0.5)

    veins_ctrl_r = [[0, 55, 6], [8, 42, 5], [14, 30, 4]]
    veins_ctrl_l = [[0, 55, -6], [-8, 42, -5], [-14, 30, -4]]
    vein_rtree_tips = []
    vein_rtree_parts = []
    vein_ltree_tips = []
    vein_ltree_parts = []
    _tree(vein_rtree_parts, vein_rtree_tips, [14, 30, 4],
          [0.5, -0.6, 0.2] / np.linalg.norm([0.5, -0.6, 0.2]), 12, 2.5, 4, rng, 0.5)
    _tree(vein_ltree_parts, vein_ltree_tips, [-14, 30, -4],
          [-0.5, -0.6, -0.2] / np.linalg.norm([-0.5, -0.6, -0.2]), 12, 2.5, 4, rng, 0.5)

    def _diaphragm_fn(u, v):
        r = 100 * (1 + 0.05 * np.sin(3 * u))
        x = r * np.cos(u)
        y = -55 - 15 * np.cos(v * 0.8)
        z = r * np.sin(u) * 0.7
        return np.asarray([x, y, z])

    parts = [
        HiPart("Right upper lobe", _pick([rl_surface]), (0.55, 0.8, 1.0), weight=1.5,
               direction=(-160, 90, 0), spread=0.6),
        HiPart("Right middle lobe", _pick([rl_surface]), (0.7, 0.55, 1.0), weight=1.0,
               direction=(-170, 0, 110), spread=0.7),
        HiPart("Right lower lobe", _pick([rl_surface]), (0.45, 0.65, 1.0), weight=2.0,
               direction=(-160, -90, -60), spread=0.6),
        HiPart("Left upper lobe", _pick([ll_surface]), (0.55, 0.8, 1.0), weight=1.5,
               direction=(160, 90, 0), spread=0.6),
        HiPart("Left lower lobe", _pick([ll_surface]), (0.45, 0.65, 1.0), weight=1.5,
               direction=(160, -90, -60), spread=0.6),
        HiPart("Trachea", _pick([trachea_pipe, glottis] + trachea_rings), (0.95, 0.95, 1.0), weight=0.8,
               direction=(0, 170, 0), spread=0.4),
        HiPart("Bronchial tree", _pick(main_bronchi + tree_parts), (1.0, 0.85, 0.4), weight=2.0,
               direction=(0, 20, 150), spread=0.8),
        HiPart("Alveoli", _pick([dots(alveoli_groups, 3, density=1.5)]), (1.0, 0.55, 0.7), weight=1.5,
               direction=(0, -20, 230), spread=1.0),
        HiPart("Pulmonary arteries",
               _pick(flatten([[_vessel(arteries_ctrl_r, 3, 1.5)],
                              [_vessel(arteries_ctrl_l, 3, 1.5)],
                              artery_rtree_parts,
                              artery_ltree_parts])),
               (0.35, 0.5, 1.0), weight=1.5,
               direction=(0, 40, -150), spread=0.8),
        HiPart("Pulmonary veins",
               _pick(flatten([[_vessel(veins_ctrl_r, 3, 1.5)],
                              [_vessel(veins_ctrl_l, 3, 1.5)],
                              vein_rtree_parts,
                              vein_ltree_parts])),
               (1.0, 0.3, 0.3), weight=1.5,
               direction=(0, -40, -190), spread=0.8),
        HiPart("Diaphragm", _pick([param_surface(_diaphragm_fn, 0, TAU, -0.3, 0.3, density=0.6)]),
               (1.0, 0.5, 0.45), weight=1.0,
               direction=(0, -180, 0), spread=0.3),
    ]
    return assemble("Lungs", "anatomy2", parts, n, seed=100)


def eye(n):
    cornea_arc = revolve([-24, -2, 2, 12], [0, 40, 44, 38], density=0.7,
                         center=[0, 0, 88], a0=-0.8, a1=0.8)
    cornea_ring = orient([ring([0, 0, 95], 42, density=0.5)], (0, 0, 1), [0, 0, 95])

    def _sclera_fn(u, v):
        return [120 * np.cos(v) * np.cos(u), 120 * np.cos(v) * np.sin(u), 120 * np.sin(v)]

    def _sclera_keep(x, y, z):
        if z >= 100:
            return False
        return True

    sclera_surf = param_surface(_sclera_fn, 0, TAU, -PI / 2, PI / 2, density=0.5)

    iris_disc = orient([disc([0, 0, 100], 5, 52, None, density=0.6)], (0, 0, 1), [0, 0, 100])
    iris_ring = orient([ring([0, 0, 100], 48, density=0.4)], (0, 0, 1), [0, 0, 100])
    iris_lines = []
    for i in range(28):
        a = i * 2 * PI / 28
        iris_lines.append(line([5 * np.cos(a), 5 * np.sin(a), 100],
                               [48 * np.cos(a), 48 * np.sin(a), 100], jitter=0.5, weight=0.2))

    ciliary_ring = orient([torus_shape([0, 0, 92], 55, 5, [0, 1, 0], density=0.5)], (0, 0, 1), [0, 0, 92])
    zonule_lines = []
    for i in range(40):
        a = i * 2 * PI / 40
        zonule_lines.append(line([55 * np.cos(a), 55 * np.sin(a), 92],
                                 [46 * np.cos(a), 46 * np.sin(a), 84], jitter=1, weight=0.15))

    lens = ellipsoid([0, 0, 82], [46, 46, 22], density=0.6)
    vitreous = ellipsoid([0, 0, -10], [108, 108, 96], density=0.14)

    def _choroid_fn(u, v):
        return [117 * np.cos(v) * np.cos(u), 117 * np.cos(v) * np.sin(u), 117 * np.sin(v)]

    def _choroid_keep(x, y, z):
        return z <= 70

    choroid = param_surface(_choroid_fn, 0, TAU, -PI / 2, PI / 2, density=0.4, gu=30, gv=30)

    def _retina_fn(u, v):
        return [114 * np.cos(v) * np.cos(u), 114 * np.cos(v) * np.sin(u), 114 * np.sin(v)]

    def _retina_keep(x, y, z):
        return z <= 55

    retina_surf = param_surface(_retina_fn, 0, TAU, -PI / 2, PI / 2, density=0.5)
    retina_vessels = []
    for i in range(7):
        a0 = i * 2 * PI / 7
        pts = []
        for t in np.linspace(0, 1, 8):
            r = t * 100
            a = a0 + 0.3 * np.sin(t * PI)
            pts.append([r * np.cos(a), r * np.sin(a), -20 + t * 50])
        retina_vessels.append(_polyline(pts, jitter=1.5, weight=0.3))
    macula = sphere([0, 0, -40], 4, density=1.0)

    optic_nerve = _vessel([[0, 0, -120], [10, 3, -200], [25, 8, -280], [45, 12, -330]], 3, 2, density=0.6)
    optic_nerve2 = _vessel([[0, 0, -120], [-10, -3, -200], [-25, -8, -280], [-45, -12, -330]], 3, 2, density=0.6)

    muscles = []
    for y_sign in (-1, 1):
        for z_sign in (-1, 1):
            start = [y_sign * 90, z_sign * 90, -80]
            end = [y_sign * 15, z_sign * 15, -320]
            muscles.append(quad(start,
                                [start[0] + z_sign * 20, start[1], start[2] - 10],
                                [end[0] + z_sign * 10, end[1], end[2] - 5],
                                end, weight=0.5))

    parts = [
        HiPart("Cornea", _pick([cornea_arc] + cornea_ring), (0.6, 0.9, 1.0), weight=1.5,
               direction=(0, 0, 190), spread=0.5),
        HiPart("Sclera", _pick([sclera_surf]), (0.95, 0.95, 1.0), weight=2.0,
               direction=(180, 0, -20), spread=0.4),
        HiPart("Iris & pupil", _pick(iris_disc + iris_ring + iris_lines), (0.35, 0.9, 0.55), weight=1.2,
               direction=(0, 0, 135), spread=0.5),
        HiPart("Ciliary body & zonules", _pick(ciliary_ring + zonule_lines), (1.0, 0.6, 0.4), weight=1.0,
               direction=(0, 0, 80), spread=0.6),
        HiPart("Lens", _pick([lens]), (1.0, 0.92, 0.5), weight=1.0,
               direction=(0, 0, 30), spread=0.6),
        HiPart("Vitreous humour", _pick([vitreous]), (0.5, 0.7, 1.0), weight=1.5,
               direction=(0, -170, 0), spread=0.7),
        HiPart("Choroid", _pick([choroid]), (0.9, 0.35, 0.5), weight=1.0,
               direction=(0, 160, -40), spread=0.6),
        HiPart("Retina", _pick([retina_surf] + retina_vessels + [macula]), (1.0, 0.55, 0.3), weight=1.5,
               direction=(0, 0, -90), spread=0.7),
        HiPart("Optic nerve", _pick([optic_nerve, optic_nerve2]), (1.0, 0.9, 0.3), weight=1.0,
               direction=(0, 0, -160), spread=0.5),
        HiPart("Eye muscles", _pick(muscles), (1.0, 0.3, 0.3), weight=1.0,
               direction=(-150, 150, -150), spread=0.5),
    ]
    return assemble("Human Eye", "anatomy2", parts, n, seed=101)


def ear(n):
    def _pinna_fn(u, v):
        r = 70 + 25 * np.sin(u * 2) + 15 * np.cos(u * 3)
        x = r * np.cos(u)
        y = r * np.sin(u) * 0.7 + 20
        z = v * 8 - 4
        return np.asarray([x, y, z])

    pinna = param_surface(_pinna_fn, 0, TAU, 0, 1, density=0.6)
    pinna_outline = _polyline(
        [[70 * np.cos(u), 70 * np.sin(u) * 0.7 + 20, 0] for u in np.linspace(0, TAU, 40)],
        closed=True, weight=0.3)
    lobule = ellipsoid([32, -38, 0], [15, 8, 5], density=0.5)
    concha = ellipsoid([20, 10, 0], [18, 12, 6], density=0.25)

    ear_canal = _vessel([[5, -5, 0], [18, -2, 3], [32, 4, 6]], 5, 4, density=0.6)

    eardrum_revolve = orient([revolve([-1, 0, 1], [38, 36, 0], density=0.6,
                                      center=[0, 0, 0], a0=-0.9, a1=0.9)],
                             (1, 0, 0.3), [38, 8, 7])
    eardrum_ring = orient([ring([38, 8, 7], 36, density=0.5)], (1, 0, 0.3), [38, 8, 7])

    malleus_vessel = _vessel([[42, 14, 8], [48, 20, 10], [54, 22, 10]], 2, 1, density=0.5)
    malleus_head = ellipsoid([48, 20, 10], [5, 6, 4], density=0.5)
    incus_body = ellipsoid([56, 22, 11], [5, 5, 4], density=0.5)
    incus_process = _vessel([[58, 24, 12], [62, 27, 13]], 1.5, 1, density=0.4)
    stapes_arch1 = _polyline([[64, 28, 13], [66, 31, 13], [70, 34, 13]], jitter=0.5, weight=0.3)
    stapes_arch2 = _polyline([[64, 28, 13], [66, 25, 13], [70, 22, 13]], jitter=0.5, weight=0.3)
    stapes_footplate = orient([disc([70, 28, 13], 0, 4, None, density=0.3)],
                              (1, 0, 0), [70, 28, 13])

    middle_ear_space = ellipsoid([32, 18, 0], [40, 44, 28], density=0.14)

    eustachian = _vessel([[30, 15, -5], [25, -15, -10], [15, -40, -15], [0, -70, -18]], 3, 2.5, density=0.6)

    cochlea_pts = []
    for i in range(90):
        t = i / 89 * 2.75 * TAU
        r = 40 - 32 * (i / 89)
        cochlea_pts.append([75 + r * np.cos(t), 25 + r * np.sin(t), 15 + (i / 89) * 5])
    cochlea_spiral = _polyline(cochlea_pts, jitter=1, weight=0.5)
    cochlea_tube = _path_tube(cochlea_pts, 2.5, density=0.5)

    canals = []
    canals.append(torus_shape([75, 35, 18], 18, 3, [0, 0, 1], density=0.3))
    canals.append(torus_shape([75, 28, 22], 14, 3, [1, 0, -0.3], density=0.3))
    canals.append(torus_shape([78, 20, 15], 16, 3, [0, 1, 0], density=0.3))
    canals.append(ellipsoid([75, 28, 18], [6, 6, 6], density=0.3))

    auditory_nerve = _vessel([[70, 25, 10], [90, 20, 5], [110, 10, 0], [140, -5, -5]], 2, 1.5, density=0.5)

    parts = [
        HiPart("Outer ear (pinna)", _pick([pinna, pinna_outline, lobule, concha]), (1.0, 0.72, 0.6),
               weight=1.5, direction=(-150, 0, 0), spread=0.5),
        HiPart("Ear canal", _pick([ear_canal]), (1.0, 0.6, 0.5), weight=1.0,
               direction=(-80, -40, 0), spread=0.5),
        HiPart("Eardrum", _pick(eardrum_revolve + eardrum_ring), (1.0, 0.45, 0.6), weight=1.0,
               direction=(-20, 70, 0), spread=0.5),
        HiPart("Ossicles",
               _pick([malleus_vessel, malleus_head, incus_body, incus_process,
                      stapes_arch1, stapes_arch2] + stapes_footplate),
               (1.0, 0.95, 0.7), weight=1.5, direction=(10, 130, 60), spread=0.7),
        HiPart("Middle ear", _pick([middle_ear_space]), (0.6, 0.7, 1.0), weight=1.0,
               direction=(0, -90, 0), spread=0.5),
        HiPart("Eustachian tube", _pick([eustachian]), (0.8, 0.6, 1.0), weight=1.0,
               direction=(0, -130, 70), spread=0.6),
        HiPart("Cochlea", _pick([cochlea_spiral, cochlea_tube]), (0.6, 1.0, 0.4), weight=1.5,
               direction=(90, -70, 0), spread=0.7),
        HiPart("Semicircular canals", _pick(canals), (0.3, 0.9, 1.0), weight=1.0,
               direction=(90, 120, 0), spread=0.7),
        HiPart("Auditory nerve", _pick([auditory_nerve]), (1.0, 0.85, 0.3), weight=1.0,
               direction=(150, 0, 0), spread=0.5),
    ]
    return assemble("Human Ear", "anatomy2", parts, n, seed=102)


def tooth(n):
    cl = np.cos
    sl = np.sin

    def _enamel_side(u, v):
        r_base = 48 * (1 + 0.06 * np.cos(4 * u))
        r = r_base
        x = r * np.cos(u)
        y = v * 100
        z = r * np.sin(u)
        return np.asarray([x, y, z])

    def _enamel_top(u, v):
        r_base = 48 * (1 + 0.06 * np.cos(4 * u))
        top_height = 0.85
        r = r_base * (1 - v)
        h = 100 + v * 18 * top_height
        x = r * np.cos(u)
        z = r * np.sin(u)
        return np.asarray([x, h, z])

    enamel_side = param_surface(_enamel_side, 0, TAU, 0, 0.7, density=0.6)
    enamel_top = param_surface(_enamel_top, 0, TAU, 0, 0.3, density=0.5)

    def _gum_fn(u, v):
        r = 52 * (1 + 0.03 * sl(2 * u))
        h_val = v * 20 + 52
        x = r * cl(u)
        z = r * sl(u)
        return np.asarray([x, h_val, z])

    gum = param_surface(_gum_fn, 0, TAU, 0, 1, density=0.5)

    def _tooth_keep(x, y, z):
        r_check = np.sqrt(x * x + z * z)
        if r_check < 40 and -80 < y < 60:
            return False
        return True

    jaw = box_surface([0, -92, 0], [155, 166, 115], density=0.4, keep=_tooth_keep)
    jaw_edges = box_edges([0, -92, 0], [155, 166, 115], jitter=1, weight=0.3)
    marrow_pts = [[x, y, z] for x in np.linspace(-60, 60, 6)
                  for y in np.linspace(-170, -20, 5)
                  for z in np.linspace(-45, 45, 5)
                  if np.sqrt(x * x + z * z) > 45 or y < -120]
    marrow = dots(marrow_pts, 4, density=0.4)

    def _dentin_side(u, v):
        r = 38 * (1 + 0.05 * cl(4 * u))
        x = r * cl(u)
        y = v * 85
        z = r * sl(u)
        return np.asarray([x, y, z])

    def _dentin_inner(u, v):
        r = 36 * (1 + 0.04 * cl(4 * u)) * (1 - v * 0.3)
        x = r * cl(u)
        y = v * 75
        z = r * sl(u)
        return np.asarray([x, y, z])

    dentin_side = param_surface(_dentin_side, 0, TAU, 0, 0.65, density=0.5)
    dentin_inner = param_surface(_dentin_inner, 0, TAU, 0, 0.6, density=0.5)
    root_tube1 = _vessel([[10, -15, 8], [12, -50, 10], [14, -75, 12]], 8, 4, density=0.5)
    root_tube2 = _vessel([[-10, -15, -8], [-12, -50, -10], [-14, -75, -12]], 8, 4, density=0.5)

    root_tip1 = ellipsoid([14, -80, 12], [6, 6, 5], density=0.5)
    root_tip2 = ellipsoid([-14, -80, -12], [6, 6, 5], density=0.5)

    pulp_chamber = ellipsoid([0, 20, 0], [14, 16, 12], density=0.5)
    root_canal1 = _vessel([[0, 5, 0], [3, -20, 2], [6, -50, 4], [8, -70, 6]], 4, 2, density=0.4)
    root_canal2 = _vessel([[0, 5, 0], [-3, -20, -2], [-6, -50, -4], [-8, -70, -6]], 4, 2, density=0.4)

    nerve_lines = []
    rng_n = np.random.default_rng(777)
    for _ in range(6):
        pts = [[rng_n.uniform(-5, 5), y, rng_n.uniform(-5, 5)]
               for y in np.linspace(-85, 10, 10)]
        nerve_lines.append(_polyline(pts, jitter=1.5, weight=0.3))

    parts = [
        HiPart("Enamel", _pick([enamel_side, enamel_top]), (0.9, 0.95, 1.0), weight=2.0,
               direction=(0, 150, 0), spread=0.4),
        HiPart("Gum", _pick([gum]), (1.0, 0.5, 0.6), weight=1.5,
               direction=(0, 40, 140), spread=0.6),
        HiPart("Jaw bone", _pick([jaw, jaw_edges, marrow]), (1.0, 0.9, 0.7), weight=2.0,
               direction=(0, -60, -180), spread=0.3),
        HiPart("Dentin", _pick([dentin_side, dentin_inner, root_tube1, root_tube2]),
               (1.0, 0.82, 0.45), weight=2.0, direction=(0, 75, 0), spread=0.6),
        HiPart("Cementum & roots", _pick([root_tube1, root_tube2, root_tip1, root_tip2]),
               (0.85, 0.7, 0.5), weight=1.5, direction=(0, -80, 0), spread=0.6),
        HiPart("Pulp", _pick([pulp_chamber, root_canal1, root_canal2]),
               (1.0, 0.35, 0.4), weight=1.0, direction=(0, 30, 120), spread=0.8),
        HiPart("Nerves & vessels", _pick(nerve_lines), (1.0, 0.9, 0.3), weight=1.0,
               direction=(0, -30, 180), spread=0.9),
    ]
    return assemble("Tooth (Molar)", "anatomy2", parts, n, seed=103)


def skull(n):
    def _vault_fn(u, v):
        f = 1 + 0.02 * np.sin(3 * u)
        x = 96 * f * np.cos(v) * np.cos(u)
        y = 30 + 88 * np.sin(v)
        z = 76 * f * np.cos(v) * np.sin(u)
        return np.asarray([x, y, z])

    vault = param_surface(_vault_fn, 0, TAU, -0.45, PI / 2, density=0.5)

    suture_pts = []
    for t in np.linspace(0, 1, 30):
        u = PI + t * PI + 0.15 * np.sin(t * 6 * PI)
        v = 0.3 + 0.8 * t
        f = 1 + 0.02 * np.sin(3 * u)
        suture_pts.append([96 * f * np.cos(v) * np.cos(u),
                           30 + 88 * np.sin(v),
                           76 * f * np.cos(v) * np.sin(u)])
    suture_line = _polyline(suture_pts, jitter=1.5, weight=0.2)

    def _vault_cap_keep(x, y, z):
        return y >= 72

    def _vault_frontal_keep(x, y, z):
        return y <= 72 and x > 38

    def _vault_occipital_keep(x, y, z):
        return y <= 72 and x < -45

    def _vault_base_keep(x, y, z):
        return y <= 72 and -45 <= x <= 38

    def _orbit_keep(x, y, z):
        x2 = (x - 32) ** 2 + (z - 24) ** 2
        x2b = (x + 32) ** 2 + (z - 24) ** 2
        if y > 70 or y < 8:
            return x2 > 10 and x2b > 10
        return True

    vault_cap_parts = [vault, suture_line]

    vault_frontal = param_surface(_vault_fn, 0, TAU, -0.45, PI / 2, density=0.4)
    vault_occipital = param_surface(_vault_fn, 0, TAU, -0.45, PI / 2, density=0.4)
    vault_base = param_surface(_vault_fn, 0, TAU, -0.45, PI / 2, density=0.4)

    ear_openings = []
    for s in (-1, 1):
        ear_openings.extend(orient([ring([s * 40, 35, -60], 10, density=0.3)],
                                   (1, 0, 0.1), [s * 40, 35, -60]))

    eye_sockets = []
    for s in (-1, 1):
        eye_sockets.append(torus_shape([s * 32, 50, 24], 28, 5, [0, 0, 1], density=0.3))
        eye_sockets.append(torus_shape([s * 32, 50, 24], 28, 5, [0, 1, 0], density=0.3))
        eye_sockets.extend(orient([revolve([-15, 0, 15], [5, 30, 5], density=0.4,
                                           center=[s * 32, 50, 24], a0=-0.6, a1=0.6)],
                                  (0.5 * s, 0.2, 0.6), [s * 32, 50, 24]))

    def _make_maxilla_fn(s):
        def fn(u, v):
            t = u * TAU / 2 + 0.8
            rx = 55 + 10 * np.sin(t)
            x = s * rx * np.cos(t * 0.6) + s * 15
            y = 8 + v * 35
            z = 15 * np.sin(t) * 0.8 + 20
            return np.asarray([x, y, z])
        return fn

    maxilla_rs = []
    for s in (-1, 1):
        maxilla_rs.append(param_surface(_make_maxilla_fn(s), 0, 1.3, 0, 1, density=0.4))

    nose_opening = _polyline([[0, 22, 52], [-12, 0, 48], [12, 0, 48]], closed=True, weight=0.2)
    cheekbone_vessels = []
    for s in (-1, 1):
        cheekbone_vessels.append(_vessel([[s * 45, 30, 8], [s * 55, 18, 15], [s * 60, 8, 12]], 3, 2, density=0.4))

    upper_teeth = []
    for i in range(14):
        t = i / 13 * 1.2 - 0.6
        for s in (-1, 1):
            x = s * (50 + 10 * np.sin(t * PI)) * np.cos(t * 0.8)
            yy = -5 + 12 * np.sin(t * 0.6)
            z = 45 + 8 * np.sin(t * 0.4)
            upper_teeth.append(ellipsoid([x, yy, z], [6, 12, 7], density=0.4))

    lower_teeth = []
    for i in range(14):
        t = i / 13 * 1.2 - 0.6
        for s in (-1, 1):
            x = s * (48 + 10 * np.sin(t * PI)) * np.cos(t * 0.8)
            yy = -18 + 12 * np.sin(t * 0.6)
            z = 43 + 8 * np.sin(t * 0.4)
            lower_teeth.append(ellipsoid([x, yy, z], [6, 12, 7], density=0.4))

    mandible = _vessel([[0, -15, 45], [35, -20, 42], [58, -10, 30], [45, 5, 15]], 5, 3, density=0.5)
    mandible_l = _vessel([[0, -15, 45], [-35, -20, 42], [-58, -10, 30], [-45, 5, 15]], 5, 3, density=0.5)
    ramus_r = _vessel([[45, 5, 15], [40, 38, 8], [30, 50, 0]], 4, 2.5, density=0.4)
    ramus_l = _vessel([[-45, 5, 15], [-40, 38, 8], [-30, 50, 0]], 4, 2.5, density=0.4)

    parts = [
        HiPart("Skull cap", _pick(vault_cap_parts), (0.95, 0.92, 0.82), weight=2.0,
               direction=(0, 150, 0), spread=0.4),
        HiPart("Frontal bone", _pick([vault_frontal]), (1.0, 0.85, 0.6), weight=1.5,
               direction=(120, 60, 0), spread=0.5),
        HiPart("Occipital bone", _pick([vault_occipital]), (0.8, 0.85, 1.0), weight=1.0,
               direction=(-150, 20, 0), spread=0.5),
        HiPart("Cranial base & temporal", _pick([vault_base] + ear_openings), (0.75, 0.8, 0.95),
               weight=1.5, direction=(0, -10, 0), spread=0.4),
        HiPart("Eye sockets", _pick(eye_sockets), (0.5, 0.9, 1.0), weight=1.0,
               direction=(80, 20, 0), spread=0.7),
        HiPart("Maxilla & cheekbones",
               _pick(maxilla_rs + [nose_opening] + cheekbone_vessels),
               (1.0, 0.7, 0.5), weight=1.5, direction=(100, -50, 0), spread=0.6),
        HiPart("Teeth (upper)", _pick(upper_teeth), (1.0, 1.0, 1.0), weight=1.5,
               direction=(130, -80, 0), spread=0.8),
        HiPart("Teeth (lower)", _pick(lower_teeth), (1.0, 1.0, 1.0), weight=1.5,
               direction=(110, -140, 0), spread=0.8),
        HiPart("Mandible", _pick([mandible, mandible_l, ramus_r, ramus_l]), (1.0, 0.8, 0.45),
               weight=1.5, direction=(60, -180, 0), spread=0.6),
    ]
    return assemble("Skull", "anatomy2", parts, n, seed=104)


SPINE = [
    [0, 340, 0],
    [0, 335, 0],
    [0, 318, 0],
    [0, 300, 0],
    [0, 282, 0],
    [0, 264, 0],
    [0, 246, 1],
    [3, 228, 2],
    [5, 210, 3],
    [7, 192, 2],
    [5, 174, 0],
    [2, 160, -2],
    [0, 146, -4],
    [-2, 132, -3],
    [0, 118, -2],
    [0, 104, 0],
    [0, 90, 0],
    [0, 76, 0],
]
_spine_pts = np.asarray(SPINE)


def skeleton(n):

    def _spine_param(index):
        p0 = _spine_pts[index]
        p1 = _spine_pts[min(index + 1, len(_spine_pts) - 1)]
        return p0, p1

    def _make_vertebra(center, prev_center, size=8):
        direction = center - prev_center
        d_norm = np.linalg.norm(direction)
        if d_norm < 0.01:
            direction = np.array([0, 1, 0])
        else:
            direction = direction / d_norm
        r = size
        h = 6

        def _cyl_fn(u, v):
            x = r * np.cos(u)
            y = v * h - h / 2
            z = r * np.sin(u)
            return np.asarray([x, y, z])

        cyl = param_surface(_cyl_fn, 0, TAU, 0, 1, density=0.5)
        sp_line = line([0, h / 2, 0], [0, h / 2 + 10, 2 * size], jitter=0.5, weight=0.3)
        sp_line2 = line([0, -h / 2, 0], [0, -h / 2 - 10, -2 * size], jitter=0.5, weight=0.3)
        oriented = orient([cyl, sp_line, sp_line2], direction, center)
        return oriented

    def _make_rib(vertebra_center, prev_center, side, index):
        direction = vertebra_center - prev_center
        d_norm = np.linalg.norm(direction)
        if d_norm < 0.01:
            direction = np.array([0, 1, 0])
        else:
            direction = direction / d_norm

        r_out = 30 + index * 0.5
        rise = 6 + index * 2
        n_pts = 12
        pts = []
        for t2 in np.linspace(0, 1, n_pts):
            angle = t2 * PI * 0.7
            x = side * r_out * np.sin(angle)
            y_val = vertebra_center[1] + rise * t2 - 5
            z = 4 + r_out * (1 - np.cos(angle)) * 0.5
            pts.append([x, y_val, z])
        return _path_tube(pts, 2, density=0.4)

    def skeletonParts():
        all_parts = []

        cranium = blob([0, 350, 0], [28, 24, 26], density=0.5,
                       bump=lambda u, phi: 1 + 0.03 * np.sin(8 * u))
        face = blob([-5, 335, 18], [18, 20, 16], density=0.4)
        jaw_bone_r = bone([5, 328, 20], [28, 322, 18], 3, density=0.4)
        jaw_bone_l = bone([5, 328, 20], [-18, 322, 18], 3, density=0.4)
        eye_r = orient([ring([25, 340, -5], 7, density=0.3)], (0, 0, 1), [25, 340, -5])
        eye_r2 = orient([ring([25, 340, -5], 7, density=0.3)], (0, 0, 1), [25, 340, -5])
        eye_l = orient([ring([-20, 340, -5], 7, density=0.3)], (0, 0, 1), [-20, 340, -5])
        eye_l2 = orient([ring([-20, 340, -5], 7, density=0.3)], (0, 0, 1), [-20, 340, -5])

        skull_parts = [cranium, face, jaw_bone_r, jaw_bone_l] + eye_r + eye_r2 + eye_l + eye_l2
        all_parts.append(HiPart("Skull", _pick(skull_parts), (0.95, 0.93, 0.85),
                                weight=2.0, direction=(0, 70, 20), spread=0.4))

        rib_parts = []
        for i in range(12):
            for side in (-1, 1):
                d2 = np.asarray([side * 15, -1, 3])
                d2 = d2 / np.linalg.norm(d2)
                rib_parts.append(_make_rib(_spine_pts[max(0, min(len(_spine_pts) - 2, i + 1))],
                                          _spine_pts[max(0, min(len(_spine_pts) - 2, i))],
                                          side, i))
            if i >= 5:
                cx = -5 if i >= 8 else 0
                rib_parts.append(line([-40, _spine_pts[i + 1][1] + 3, 10],
                                      [cx, _spine_pts[i][1] - 2, 15], jitter=1, weight=0.2))
                rib_parts.append(line([40, _spine_pts[i + 1][1] + 3, 10],
                                      [cx, _spine_pts[i][1] - 2, 15], jitter=1, weight=0.2))
        sternum = blob([0, 270, 8], [8, 45, 4], density=0.4)
        rib_parts.append(sternum)
        all_parts.append(HiPart("Rib cage & sternum", _pick(rib_parts), (0.8, 0.9, 1.0),
                                weight=2.5, direction=(0, 15, 70), spread=0.5))

        spine_parts_list = []
        for i in range(min(24, len(_spine_pts) - 1)):
            vert_parts = _make_vertebra(_spine_pts[i], _spine_pts[max(i - 1, 0)])
            spine_parts_list.extend(vert_parts)
        sacrum = blob([0, 80, -5], [16, 25, 12], density=0.4)
        spine_parts_list.append(sacrum)
        all_parts.append(HiPart("Spine", _pick(spine_parts_list), (1.0, 0.8, 0.5),
                                weight=2.0, direction=(0, 0, -70), spread=0.5))

        shoulder_parts = []
        for s in (-1, 1):
            clavicle = _vessel([[0, 310, -5], [s * 25, 315, -10], [s * 55, 308, -12]], 3, 2.5, density=0.5)
            scapula_tri = tri([s * 55, 308, -12], [s * 70, 290, -20], [s * 45, 280, -25], weight=0.5)
            shoulder_parts.append(clavicle)
            shoulder_parts.append(scapula_tri)
        all_parts.append(HiPart("Shoulder girdle", _pick(shoulder_parts), (0.7, 1.0, 0.8),
                                weight=1.5, direction=(0, 35, -45), spread=0.5))

        for side_label, s in [("", 1), (" (L)", -1)]:
            arm_parts = []
            humerus = bone([s * 55, 308, -12], [s * 55, 250, -15], 8, density=0.5)
            radius = bone([s * 55, 250, -15], [s * 58, 210, -10], 6, density=0.5)
            ulna = bone([s * 55, 250, -15], [s * 52, 210, -10], 6, density=0.5)
            arm_parts.append(humerus)
            arm_parts.append(radius)
            arm_parts.append(ulna)
            all_parts.append(HiPart("Arm bones" + side_label, _pick(arm_parts), (0.6, 0.85, 1.0),
                                    weight=1.5, direction=(s * 70, 10, 0), spread=0.6))

            hand_parts = []
            wrist = ellipsoid([s * 55, 205, -10], [8, 6, 8], density=0.5)
            hand_parts.append(wrist)
            for f_idx in range(5):
                x_off = (f_idx - 2) * 6
                f_pts = [[s * 55 + x_off, 205, -8],
                         [s * 58 + x_off, 195, -10],
                         [s * 62 + x_off, 188, -11]]
                hand_parts.append(_polyline(f_pts, jitter=1, weight=0.3))
            all_parts.append(HiPart("Hand bones" + side_label, _pick(hand_parts), (1.0, 0.7, 0.9),
                                    weight=1.0, direction=(s * 110, -20, 30), spread=0.7))

        pelvis_ring = param_surface(lambda u, v: [
            50 * (1 + 0.15 * np.sin(u)) * np.cos(u),
            75 + v * 15,
            -10 + 30 * np.sin(u)], 0, TAU, 0, 1, density=0.5)
        hip_r = orient([ring([28, 82, -5], 16, density=0.4)], (0.5, -0.3, 0.8), [28, 82, -5])
        hip_l = orient([ring([-28, 82, -5], 16, density=0.4)], (-0.5, -0.3, 0.8), [-28, 82, -5])
        all_parts.append(HiPart("Pelvis", _pick([pelvis_ring] + hip_r + hip_l), (1.0, 0.9, 0.6),
                                weight=1.5, direction=(0, -15, 50), spread=0.5))

        for side_label2, s in [("", 1), (" (L)", -1)]:
            leg_parts = []
            femur = bone([s * 28, 82, -5], [s * 30, 20, -8], 10, density=0.5)
            tibia = bone([s * 30, 20, -8], [s * 28, -30, -5], 8, density=0.5)
            fibula = bone([s * 30, 20, -8], [s * 32, -30, -5], 5, density=0.5)
            leg_parts.append(femur)
            leg_parts.append(tibia)
            leg_parts.append(fibula)
            all_parts.append(HiPart("Leg bones" + side_label2, _pick(leg_parts), (0.75, 0.75, 1.0),
                                    weight=2.0, direction=(s * 35, -45, 0), spread=0.5))

            foot_parts = []
            ankle = ellipsoid([s * 30, -35, -5], [7, 6, 7], density=0.5)
            foot_parts.append(ankle)
            for t_idx in range(5):
                x_off2 = (t_idx - 2) * 5
                t_pts = [[s * 30 + x_off2, -35, -3],
                         [s * 32 + x_off2, -48, -4],
                         [s * 35 + x_off2, -55, -5]]
                foot_parts.append(_polyline(t_pts, jitter=1, weight=0.3))
            all_parts.append(HiPart("Foot bones" + side_label2, _pick(foot_parts), (1.0, 0.85, 0.5),
                                    weight=1.0, direction=(s * 45, -80, 40), spread=0.7))

        return all_parts

    parts = skeletonParts()
    return assemble("Skeleton", "anatomy2", parts, n, seed=105)


def human_body(n):
    all_sparts = skeleton(n=0).parts

    def skeletonParts():
        return all_sparts

    head_blob = blob([0, 340, 0], [24, 26, 24], density=0.6)
    neck_cyl = orient([cylinder_s([0, 310, -5], 10, 15, density=0.6, axis=1)], (0, 1, 0), [0, 310, -5])

    def _torso_fn(u, v):
        rx = 30 + 5 * np.cos(2 * u)
        ry = 18 + 4 * np.cos(u)
        x = rx * np.cos(u)
        y_val = 130 + v * 180
        z = ry * np.sin(u) * 0.8
        return np.asarray([x, y_val, z])

    torso = param_surface(_torso_fn, 0, TAU, 0, 1, density=0.4)

    front_parts = [head_blob] + neck_cyl + [torso]
    back_parts = [head_blob] + neck_cyl + [torso]

    for side, s in [("", 1), (" (L)", -1)]:
        arm_tube = _vessel([[s * 45, 295, 0], [s * 50, 255, -5], [s * 52, 215, -8]], 10, 7, density=0.4)
        hand_blob = blob([s * 55, 200, -8], [8, 6, 5], density=0.5)
        front_parts.append(arm_tube)
        front_parts.append(hand_blob)
        back_parts.append(arm_tube)
        back_parts.append(hand_blob)

        leg_tube = _vessel([[s * 25, 90, -5], [s * 27, 45, -5], [s * 28, -10, -2]], 12, 9, density=0.4)
        foot_blob = blob([s * 32, -35, -5], [7, 5, 8], density=0.5)
        front_parts.append(leg_tube)
        front_parts.append(foot_blob)
        back_parts.append(leg_tube)
        back_parts.append(foot_blob)

    brain_blob = blob([0, 340, -2], [18, 16, 18], density=0.6,
                      bump=lambda u, phi: 1 + 0.04 * np.sin(10 * u))

    lung_r = blob([15, 255, 0], [12, 18, 10], density=0.5)
    lung_l = blob([-15, 255, 0], [12, 18, 10], density=0.5)

    heart_blob = blob([4, 246, 7], [8, 10, 7], density=0.5)
    liver_blob = blob([-10, 224, 4], [21, 9, 13], density=0.5)
    stomach_blob = blob([13, 220, 7], [11, 8, 8], density=0.5)

    intestine_pts = [[13, 218, 7], [18, 210, 8], [-15, 205, 7], [-20, 200, 6],
                     [10, 196, 5], [15, 191, 5], [-5, 187, 4]]
    intestines = _path_tube(intestine_pts, 4, density=0.5)

    kidney_r = blob([12, 214, -10], [4, 7, 3], density=0.5)
    kidney_l = blob([-12, 214, -10], [4, 7, 3], density=0.5)
    bladder_blob = blob([0, 186, 8], [6, 5, 5], density=0.5)

    spine_nerve = _path_tube(_spine_pts[:15], 2, density=0.4)
    arm_nerves = []
    for s in (-1, 1):
        arm_nerves.append(_polyline([[0, 300, 0], [s * 20, 295, -3], [s * 45, 260, -8],
                                    [s * 52, 230, -10], [s * 55, 210, -10]], jitter=2, weight=0.3))
    leg_nerves = []
    for s2 in (-1, 1):
        leg_nerves.append(_polyline([[0, 80, -5], [s2 * 15, 60, -5], [s2 * 25, 30, -5],
                                    [s2 * 27, -5, -3]], jitter=2, weight=0.3))
    intercostal_lines = []
    for ii in range(10):
        y_ii = 280 - ii * 12
        intercostal_lines.append(line([-30, y_ii, 5], [30, y_ii, 5], jitter=2, weight=0.15))

    aorta_pts = [[4, 246, 7], [3, 230, 5], [0, 210, 3], [0, 180, 1], [-2, 150, 0]]
    aorta = _vessel(aorta_pts, 3, 2, density=0.5)
    neck_arteries = []
    for s3 in (-1, 1):
        neck_arteries.append(_vessel([[3, 290, 0], [s3 * 8, 310, -2], [s3 * 12, 330, -3]], 2.5, 1.5, density=0.4))
    arm_arteries = []
    for s4 in (-1, 1):
        arm_arteries.append(_vessel([[s4 * 15, 285, -4], [s4 * 40, 250, -8], [s4 * 52, 215, -10]], 2, 1.5, density=0.4))
    leg_arteries = []
    for s5 in (-1, 1):
        leg_arteries.append(_vessel([[s5 * 8, 180, 1], [s5 * 20, 90, -5], [s5 * 26, 0, -2]], 2.5, 2, density=0.4))

    vena_cava = _vessel([[4, 246, 9], [6, 200, 6], [5, 150, 4], [3, 100, 2]], 3.5, 3, density=0.5)
    neck_veins = []
    for s6 in (-1, 1):
        neck_veins.append(_vessel([[3, 290, 2], [s6 * 10, 312, 0], [s6 * 14, 332, -1]], 2.5, 1.5, density=0.4))
    arm_veins = []
    for s7 in (-1, 1):
        arm_veins.append(_vessel([[s7 * 17, 285, -2], [s7 * 42, 250, -6], [s7 * 54, 215, -8]], 2, 1.5, density=0.4))
    leg_veins = []
    for s8 in (-1, 1):
        leg_veins.append(_vessel([[s8 * 7, 180, 3], [s8 * 18, 90, -3], [s8 * 24, 0, 0]], 2.5, 2, density=0.4))

    skel_shapes = [Part(p.weight, p.sampler) for p in all_sparts]
    all_skel_sampler = _pick(skel_shapes)

    parts = [
        HiPart("Skin (front)", _pick(front_parts), (1.0, 0.75, 0.62), weight=3.0,
               direction=(0, 0, 190), spread=0.3),
        HiPart("Skin (back)", _pick(back_parts), (1.0, 0.75, 0.62), weight=3.0,
               direction=(0, 0, -190), spread=0.3),
        HiPart("Brain", _pick([brain_blob]), (1.0, 0.5, 0.7), weight=2.0,
               direction=(0, 55, 100), spread=0.7),
        HiPart("Lungs", _pick([lung_r, lung_l]), (0.55, 0.75, 1.0), weight=2.0,
               direction=(0, 25, 110), spread=0.7),
        HiPart("Heart", _pick([heart_blob]), (1.0, 0.25, 0.3), weight=1.5,
               direction=(0, 5, 135), spread=0.7),
        HiPart("Liver", _pick([liver_blob]), (0.8, 0.4, 0.25), weight=1.5,
               direction=(-35, -5, 115), spread=0.7),
        HiPart("Stomach", _pick([stomach_blob]), (1.0, 0.65, 0.5), weight=1.0,
               direction=(35, -5, 115), spread=0.7),
        HiPart("Intestines", _pick([intestines]), (1.0, 0.8, 0.55), weight=1.5,
               direction=(0, -25, 115), spread=0.8),
        HiPart("Kidneys", _pick([kidney_r, kidney_l]), (1.0, 0.55, 0.45), weight=1.0,
               direction=(0, -10, 85), spread=0.7),
        HiPart("Bladder", _pick([bladder_blob]), (1.0, 0.95, 0.5), weight=0.8,
               direction=(0, -40, 105), spread=0.8),
        HiPart("Nervous system",
               _pick([spine_nerve] + arm_nerves + leg_nerves + intercostal_lines),
               (1.0, 0.9, 0.3), weight=2.0, direction=(-140, 0, 0), spread=0.8),
        HiPart("Arteries",
               _pick([aorta] + neck_arteries + arm_arteries + leg_arteries),
               (1.0, 0.3, 0.25), weight=2.0, direction=(140, 0, 0), spread=0.8),
        HiPart("Veins",
               _pick([vena_cava] + neck_veins + arm_veins + leg_veins),
               (0.35, 0.5, 1.0), weight=2.0, direction=(270, 0, 0), spread=0.8),
        HiPart("Skeleton", all_skel_sampler, (0.95, 0.93, 0.85), weight=2.5,
               direction=(0, 0, -120), spread=0.9),
    ]
    return assemble("Human Body", "anatomy2", parts, n, seed=106)


ANATOMY2 = [
    {"name": "Lungs", "build": lungs, "color": (0.55, 0.75, 1.0), "beat_bpm": None},
    {"name": "Human Eye", "build": eye, "color": (0.55, 0.9, 1.0), "beat_bpm": None},
    {"name": "Human Ear", "build": ear, "color": (1.0, 0.75, 0.55), "beat_bpm": None},
    {"name": "Tooth (Molar)", "build": tooth, "color": (0.92, 0.95, 1.0), "beat_bpm": None},
    {"name": "Skull", "build": skull, "color": (0.95, 0.9, 0.8), "beat_bpm": None},
]
BODY = [
    {"name": "Skeleton", "build": skeleton, "color": (0.95, 0.93, 0.85), "beat_bpm": None},
    {"name": "Human Body", "build": human_body, "color": (1.0, 0.72, 0.6), "beat_bpm": None},
]