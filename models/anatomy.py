import numpy as np
from .base import Part as HiPart, assemble
from .shapes import (Part, line, polyline, tri, quad, box_surface, box_edges,
                     revolve, ring, disc, tube_shape, pipe, param_surface,
                     torus_shape, ellipsoid, capsule, blob, dots,
                     transform, orient, _pick, flatten,
                     rot_z, TAU, sphere, vessel,
                     linspace, range_n, deg, _spline, path_tube)


def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _arr(a):
    return np.asarray(a, dtype=np.float64)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ═══════════════════════════════════════════════════════════════════════════
#  Brain
# ═══════════════════════════════════════════════════════════════════════════

def brain(n):
    def _hemi(s, u, v):
        f = 1 + 0.028 * np.sin(11 * u + 3 * np.sin(5 * v)) * np.cos(9 * v + 2 * np.sin(4 * u))
        x = 165 * np.cos(v) * np.cos(u) * f
        y = 95 * np.sin(v) * f
        z = s * 140 * np.cos(v) * np.sin(u) * f
        return x, y, z

    def _hemi_fn(s):
        return lambda u, v: np.array(_hemi(s, u, v))

    def _region(x, y, z):
        if y < -22:
            return "temporal" if x > -35 else "cerebellar"
        if x > 15:
            return "frontal"
        if x > -60:
            return "parietal"
        return "occipital"

    def _lobe_with_sulci(name, s, u0, u1, v0, v1, color, direction, weight=3):
        surf_fn = _hemi_fn(s)

        def combined_sample(k, rng):
            k_surf = int(k * 0.65)
            k_sulci = k - k_surf
            pts = []

            u_s = rng.uniform(np.float64(u0), np.float64(u1), k_surf)
            v_s = rng.uniform(np.float64(v0), np.float64(v1), k_surf)
            sph = np.array([surf_fn(ui, vi) for ui, vi in zip(u_s, v_s)])
            pts.append(sph)

            n_lines = 12
            n_steps = 14
            du_range = u1 - u0
            dv_range = v1 - v0
            for _ in range(n_lines):
                uu = rng.uniform(u0 + 0.12 * du_range, u1 - 0.12 * du_range)
                vv = rng.uniform(v0 + 0.12 * dv_range, v1 - 0.12 * dv_range)
                line_pts = []
                for __ in range(n_steps + 1):
                    line_pts.append(_hemi(s, uu, vv))
                    uu += rng.normal(0, du_range * 0.04)
                    vv += rng.normal(0, dv_range * 0.04)
                    uu = np.clip(uu, u0 + 0.02, u1 - 0.02)
                    vv = np.clip(vv, v0 + 0.02, v1 - 0.02)
                pts.append(np.array(line_pts))
            all_extra = np.concatenate(pts)
            all_pts = np.concatenate([sph, all_extra])
            idx = rng.choice(len(all_pts), size=k, replace=len(all_pts) < k)
            return _f32(all_pts[idx])

        return HiPart(name=name, sampler=combined_sample, color=color, weight=weight,
                      info=name, direction=direction, spread=1.0)

    PINK = (1.0, 0.45, 0.62)
    BLUE = (0.5, 0.6, 1.0)
    GREEN = (0.4, 1.0, 0.6)
    AMBER = (1.0, 0.75, 0.3)

    parts = []
    pi = np.pi

    # Cortical lobes — right hemisphere
    parts.append(_lobe_with_sulci("Frontal lobe", 1, 0, pi * 0.33, -pi / 10, pi / 2, PINK, (120, 30, 170)))
    parts.append(_lobe_with_sulci("Parietal lobe", 1, pi * 0.33, pi * 0.62, -pi / 14, pi / 2, BLUE, (-10, 125, 170)))
    parts.append(_lobe_with_sulci("Temporal lobe", 1, pi * 0.14, pi * 0.52, -pi / 2, -pi / 10, GREEN, (40, -90, 200)))
    parts.append(_lobe_with_sulci("Occipital lobe", 1, pi * 0.62, pi, -pi / 7, pi / 2, AMBER, (-150, 30, 150)))

    # Cortical lobes — left hemisphere
    parts.append(_lobe_with_sulci("Frontal lobe (L)", -1, 0, pi * 0.33, -pi / 10, pi / 2, PINK, (120, 30, -170)))
    parts.append(_lobe_with_sulci("Parietal lobe (L)", -1, pi * 0.33, pi * 0.62, -pi / 14, pi / 2, BLUE, (-10, 125, -170)))
    parts.append(_lobe_with_sulci("Temporal lobe (L)", -1, pi * 0.14, pi * 0.52, -pi / 2, -pi / 10, GREEN, (40, -90, -200)))
    parts.append(_lobe_with_sulci("Occipital lobe (L)", -1, pi * 0.62, pi, -pi / 7, pi / 2, AMBER, (-150, 30, -150)))

    # Cerebellum
    cb_center = np.array([-105, -110, 0], dtype=np.float64)
    cb_radii = np.array([60, 42, 68], dtype=np.float64)

    def _folia_bump(u, phi):
        return 1 + 0.045 * np.sin(20 * phi) + 0.025 * np.cos(3 * u)

    cb_ellip = ellipsoid(cb_center, cb_radii, density=1.0)
    cb_folia_part = blob(cb_center, cb_radii, bump=_folia_bump, density=0.5)

    cb_ring_parts = []
    for i in range(13):
        phi = (i / 12.0 - 0.5) * pi
        y_at = cb_center[1] + cb_radii[1] * np.sin(phi) * 1.03
        r_at = np.sqrt(cb_radii[0] ** 2 * np.cos(phi) ** 2 + cb_radii[2] ** 2 * np.cos(phi) ** 2)
        r_at *= (0.5 + 0.5 * (1 + 0.04 * np.sin(18 * phi)))
        cb_ring_parts.append(ring(center=(cb_center[0], y_at, cb_center[2]), r=r_at, density=0.15))

    cb_all = [cb_ellip, cb_folia_part] + cb_ring_parts
    cb_sampler = _pick(cb_all)
    part_cb = Part(1, sample=cb_sampler)
    parts.append(HiPart(name="Cerebellum", sampler=part_cb.sample, color=(1.0, 0.55, 0.2),
                        weight=3, info="Cerebellum", direction=(-170, -120, 0), spread=1.0))

    # Brainstem
    bs_pts = [(0, -155, 0), (0, -200, 0), (0, -245, 0)]
    bs = vessel(bs_pts, 14, 18, density=1.0, per=10)
    parts.append(HiPart(name="Brainstem", sampler=bs.sample, color=(1.0, 0.85, 0.35),
                        weight=2, info="Brainstem", direction=(-40, -210, 0), spread=1.0))

    # Corpus callosum
    def _cc_fn(u, v):
        angle = u * pi * 0.55
        x = -15 + 28 * np.sin(angle)
        y = 125 + 12 * (1 - np.cos(angle))
        z = v * 18
        return np.array([x, y, z], dtype=np.float64)

    cc_surf = param_surface(_cc_fn, 0, 1, -1, 1, density=0.8)
    cc_lines = []
    for i in range(7):
        t = i / 6.0
        angle = t * pi * 0.55
        cx = -15 + 28 * np.sin(angle)
        cy = 125 + 12 * (1 - np.cos(angle))
        cc_lines.append(line([cx, cy, -18], [cx, cy, 18], jitter=0.4, weight=0.2))
    cc_sampler = _pick([cc_surf] + cc_lines)
    cc_part = Part(1, sample=cc_sampler)
    parts.append(HiPart(name="Corpus callosum", sampler=cc_part.sample, color=(0.95, 0.95, 1.0),
                        weight=2, info="Corpus callosum", direction=(0, 150, 0), spread=1.0))

    # Thalamus
    thal_r = ellipsoid((-10, 48, 18), (16, 20, 13))
    thal_l = ellipsoid((-10, 48, -18), (16, 20, 13))
    thal_sampler = _pick([thal_r, thal_l])
    thal_part = Part(1, sample=thal_sampler)
    parts.append(HiPart(name="Thalamus", sampler=thal_part.sample, color=(0.3, 0.9, 1.0),
                        weight=2, info="Thalamus", direction=(-20, 55, 0), spread=1.0))

    # Hypothalamus & pituitary
    hypo = ellipsoid((0, -95, 0), (7, 9, 9))
    stalk = vessel([(0, -85, 0), (0, -95, 0), (-8, -115, 0), (-12, -135, 0)], 3, 4, per=6)
    hp_sampler = _pick([hypo, stalk])
    hp_part = Part(1, sample=hp_sampler)
    parts.append(HiPart(name="Hypothalamus & pituitary", sampler=hp_part.sample,
                        color=(0.75, 1.0, 0.3), weight=2, info="Hypothalamus & pituitary",
                        direction=(70, -120, 0), spread=1.0))

    # Hippocampus (right)
    def _hippocampus_parts(pts, sgn):
        hipp = vessel(pts, 6, 8, per=8)
        rings = []
        for i in range(5):
            t = i / 4.0
            p = np.array(pts[0]) + t * (np.array(pts[-1]) - np.array(pts[0]))
            rings.append(ring(center=tuple(p), r=7 + 1.5 * np.sin(t * pi), density=0.25))
        return _pick([hipp] + rings)

    hipp_r_pts = [(-40, -28, 50 * 1), (-48, -50, 60 * 1), (-58, -68, 52 * 1), (-68, -88, 55 * 1)]
    hipp_r_sampler = _hippocampus_parts(hipp_r_pts, 1)
    hipp_r_part = Part(1, sample=hipp_r_sampler)
    parts.append(HiPart(name="Hippocampus", sampler=hipp_r_part.sample, color=(1.0, 0.95, 0.3),
                        weight=2, info="Hippocampus (right)", direction=(-50, -60, 120), spread=1.0))

    hipp_l_pts = [(-40, -28, -50), (-48, -50, -60), (-58, -68, -52), (-68, -88, -55)]
    hipp_l_sampler = _hippocampus_parts(hipp_l_pts, -1)
    hipp_l_part = Part(1, sample=hipp_l_sampler)
    parts.append(HiPart(name="Hippocampus (L)", sampler=hipp_l_part.sample, color=(1.0, 0.95, 0.3),
                        weight=2, info="Hippocampus (left)", direction=(-50, -60, -120), spread=1.0))

    # Amygdala
    amyg_r = ellipsoid((35, -38, 50), (11, 13, 9))
    parts.append(HiPart(name="Amygdala", sampler=amyg_r.sample, color=(1.0, 0.3, 0.3),
                        weight=2, info="Amygdala (right)", direction=(80, -60, 120), spread=1.0))
    amyg_l = ellipsoid((35, -38, -50), (11, 13, 9))
    parts.append(HiPart(name="Amygdala (L)", sampler=amyg_l.sample, color=(1.0, 0.3, 0.3),
                        weight=2, info="Amygdala (left)", direction=(80, -60, -120), spread=1.0))

    # Ventricles
    v_pts = [(0, 55, 0), (18, 35, 28), (8, -5, 32), (0, -30, 28), (-10, -55, 18), (-38, -85, 14)]
    vent_r = vessel(v_pts, 5, 7, per=8)
    v_pts_l = [(0, 55, 0), (18, 35, -28), (8, -5, -32), (0, -30, -28), (-10, -55, -18), (-38, -85, -14)]
    vent_l = vessel(v_pts_l, 5, 7, per=8)
    vent_sampler = _pick([vent_r, vent_l])
    vent_part = Part(1, sample=vent_sampler)
    parts.append(HiPart(name="Ventricles", sampler=vent_part.sample, color=(0.35, 0.5, 1.0),
                        weight=2, info="Ventricles", direction=(30, 90, 0), spread=1.0))

    # Basal ganglia (right)
    bg_r_vessel = vessel([(15, 8, 38), (25, -10, 42), (30, -30, 40)], 7, 9, per=8)
    bg_r_ellip = ellipsoid((25, -5, 42), (14, 17, 11))
    bg_r_sampler = _pick([bg_r_vessel, bg_r_ellip])
    bg_r_part = Part(1, sample=bg_r_sampler)
    parts.append(HiPart(name="Basal ganglia", sampler=bg_r_part.sample, color=(0.72, 0.45, 1.0),
                        weight=2, info="Basal ganglia (right)", direction=(30, -10, 95), spread=1.0))

    # Basal ganglia (left)
    bg_l_vessel = vessel([(15, 8, -38), (25, -10, -42), (30, -30, -40)], 7, 9, per=8)
    bg_l_ellip = ellipsoid((25, -5, -42), (14, 17, 11))
    bg_l_sampler = _pick([bg_l_vessel, bg_l_ellip])
    bg_l_part = Part(1, sample=bg_l_sampler)
    parts.append(HiPart(name="Basal ganglia (L)", sampler=bg_l_part.sample, color=(0.72, 0.45, 1.0),
                        weight=2, info="Basal ganglia (left)", direction=(30, -10, -95), spread=1.0))

    # Optic nerves & chiasm
    optic_r = vessel([(-8, -115, 10), (25, -85, 48), (75, -55, 78)], 3, 5, per=6)
    optic_l = vessel([(-8, -115, -10), (25, -85, -48), (75, -55, -78)], 3, 5, per=6)
    chiasm = ellipsoid((-8, -115, 0), (11, 5, 15))
    optic_sampler = _pick([optic_r, optic_l, chiasm])
    optic_part = Part(1, sample=optic_sampler)
    parts.append(HiPart(name="Optic nerves & chiasm", sampler=optic_part.sample,
                        color=(0.4, 1.0, 0.95), weight=2, info="Optic nerves & chiasm",
                        direction=(160, -60, 0), spread=1.0))

    return assemble("Brain", "biology", parts, n, seed=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Heart
# ═══════════════════════════════════════════════════════════════════════════

def heart(n):
    def _egg(center, abc, rot_mat, taper):
        a, b, c = abc
        cen = _arr(center)
        mat = _arr(rot_mat)

        def fn(u, v):
            frac = (v + np.pi / 2) / np.pi
            s = 1.0
            if taper > 0:
                s = 1 - 0.28 * (1 - frac) ** taper
            x = a * s * np.cos(v) * np.cos(u)
            y = b * s * np.sin(v)
            z = c * s * np.cos(v) * np.sin(u)
            return _f32(mat @ np.array([x, y, z]) + cen)

        return fn

    def _clip_surf(fn, u0, u1, v0, v1, keep_fn, density=1):
        du, dv = u1 - u0, v1 - v0

        def sample(k, rng):
            mul = 5
            u = rng.uniform(np.float64(u0), np.float64(u1), k * mul)
            v = rng.uniform(np.float64(v0), np.float64(v1), k * mul)
            pts = np.array([fn(ui, vi) for ui, vi in zip(u, v)])
            mask = np.array([keep_fn(*p) for p in pts])
            p = pts[mask]
            while len(p) < k:
                u2 = rng.uniform(np.float64(u0), np.float64(u1), k)
                v2 = rng.uniform(np.float64(v0), np.float64(v1), k)
                extra = np.array([fn(ui, vi) for ui, vi in zip(u2, v2)])
                m2 = np.array([keep_fn(*pt) for pt in extra])
                p = np.concatenate([p, extra[m2]])
            return _f32(p[:k])

        return Part(density * du * dv * 0.5, sample)

    parts = []
    pi = np.pi

    # Rotations for heart chambers (tilted heart)
    rot_lv = rot_z(deg(-15))
    rot_rv = rot_z(deg(5))
    rot_la = rot_z(deg(-10))
    rot_ra = rot_z(deg(8))

    # Egg surfaces
    lv_egg = _egg((38, -48, -5), (32, 68, 28), rot_lv, 2.5)
    rv_egg = _egg((-12, -28, 22), (42, 52, 33), rot_rv, 2.0)

    # ── Heart wall (myocardium) ──
    lv_front = _clip_surf(lv_egg, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z > 0, density=1.0)
    rv_front = _clip_surf(rv_egg, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z > 0, density=1.0)
    wall_sampler = _pick([lv_front, rv_front])
    wall_part = Part(1, sample=wall_sampler)
    parts.append(HiPart(name="Heart wall (myocardium)", sampler=wall_part.sample,
                        color=(1.0, 0.4, 0.45), weight=4, info="Heart wall (myocardium)",
                        direction=(0, -10, 240), spread=0.5))

    # ── Coronary arteries ──
    rca_pts = [(-20, 70, 35), (0, 55, 50), (25, 20, 55), (40, -20, 45), (45, -50, 25)]
    rca = vessel(rca_pts, 3.5, 2.5, per=10)
    lad_pts = [(-5, 65, 40), (20, 40, 35), (45, 5, 25), (48, -30, 15), (50, -60, 5)]
    lad = vessel(lad_pts, 3.5, 2.5, per=10)
    circ_pts = [(-5, 65, 40), (-25, 50, 15), (-40, 30, -5), (-35, 5, -15)]
    circ = vessel(circ_pts, 3, 2, per=10)
    coron_sampler = _pick([rca, lad, circ])
    coron_part = Part(1, sample=coron_sampler)
    parts.append(HiPart(name="Coronary arteries", sampler=coron_part.sample,
                        color=(1.0, 0.82, 0.3), weight=2, info="Coronary arteries",
                        direction=(0, 30, 330), spread=0.6))

    # ── Aorta ──
    ao_pts = [(-8, 70, 35), (-5, 95, 38), (0, 130, 30), (-10, 160, 10), (-25, 185, -10)]
    ao = vessel(ao_pts, 8, 10, per=12)
    br1 = vessel([(0, 130, 30), (30, 150, 55), (60, 165, 70)], 4, 3, per=8)
    br2 = vessel([(0, 130, 30), (-10, 155, 65), (-15, 175, 90)], 4, 3, per=8)
    br3 = vessel([(0, 130, 30), (-45, 150, 40), (-70, 160, 50)], 3.5, 2.5, per=8)
    ao_sampler = _pick([ao, br1, br2, br3])
    ao_part = Part(1, sample=ao_sampler)
    parts.append(HiPart(name="Aorta", sampler=ao_part.sample, color=(1.0, 0.35, 0.2),
                        weight=2, info="Aorta", direction=(70, 190, -50), spread=0.5))

    # ── Pulmonary artery ──
    pa_main = vessel([(-25, 70, 40), (-35, 85, 45), (-45, 100, 35)], 7, 6, per=10)
    pa_right = vessel([(-45, 100, 35), (-60, 115, 55), (-75, 130, 65)], 4, 3, per=8)
    pa_left = vessel([(-45, 100, 35), (-55, 120, 15), (-65, 135, -5)], 4, 3, per=8)
    pa_sampler = _pick([pa_main, pa_right, pa_left])
    pa_part = Part(1, sample=pa_sampler)
    parts.append(HiPart(name="Pulmonary artery", sampler=pa_part.sample,
                        color=(0.4, 0.55, 1.0), weight=1.5, info="Pulmonary artery",
                        direction=(-60, 170, 90), spread=0.5))

    # ── Vena cavae ──
    svc = vessel([(-55, 110, 25), (-55, 90, 20), (-45, 70, 18)], 8, 7, per=10)
    ivc = vessel([(10, -115, 10), (-10, -70, 12), (-30, -20, 15), (-42, 30, 17), (-45, 70, 18)], 9, 8, per=12)
    vc_sampler = _pick([svc, ivc])
    vc_part = Part(1, sample=vc_sampler)
    parts.append(HiPart(name="Vena cavae", sampler=vc_part.sample,
                        color=(0.5, 0.42, 1.0), weight=1.5, info="Vena cavae",
                        direction=(-180, 40, 0), spread=0.5))

    # ── Pulmonary veins ──
    pv1 = vessel([(-30, 45, -35), (-50, 35, -50), (-70, 25, -65)], 4, 3, per=8)
    pv2 = vessel([(-25, 50, -30), (-45, 40, -20), (-65, 30, -10)], 4, 3, per=8)
    pv3 = vessel([(-20, 40, -40), (-40, 30, -60), (-60, 20, -75)], 4, 3, per=8)
    pv4 = vessel([(-15, 35, -35), (-35, 25, -25), (-55, 15, -15)], 4, 3, per=8)
    pv_sampler = _pick([pv1, pv2, pv3, pv4])
    pv_part = Part(1, sample=pv_sampler)
    parts.append(HiPart(name="Pulmonary veins", sampler=pv_part.sample,
                        color=(1.0, 0.45, 0.8), weight=1.5, info="Pulmonary veins",
                        direction=(130, 60, -160), spread=0.6))

    # ── Left atrium ──
    la_ellip = ellipsoid((-22, 48, -28), (28, 26, 24))
    parts.append(HiPart(name="Left atrium", sampler=la_ellip.sample,
                        color=(1.0, 0.4, 0.75), weight=1.5, info="Left atrium",
                        direction=(120, 120, -90), spread=0.6))

    # ── Right atrium ──
    ra_ellip = ellipsoid((-52, 42, 22), (30, 28, 26))
    ra_auricle = ellipsoid((-62, 60, 38), (12, 15, 10))
    ra_sampler = _pick([ra_ellip, ra_auricle])
    ra_part = Part(1, sample=ra_sampler)
    parts.append(HiPart(name="Right atrium", sampler=ra_part.sample,
                        color=(0.55, 0.5, 1.0), weight=1.5, info="Right atrium",
                        direction=(-160, 90, 30), spread=0.6))

    # ── Left ventricle ──
    lv_back = _clip_surf(lv_egg, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z <= 0, density=1.0)
    lv_inner = _egg((35, -45, -5), (24, 55, 20), rot_lv, 2.0)
    lv_inner_s = _clip_surf(lv_inner, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z <= 0, density=1.0)
    lv_sampler = _pick([lv_back, lv_inner_s])
    lv_part = Part(1, sample=lv_sampler)
    parts.append(HiPart(name="Left ventricle", sampler=lv_part.sample,
                        color=(1.0, 0.25, 0.3), weight=2, info="Left ventricle",
                        direction=(100, -70, -70), spread=0.7))

    # ── Right ventricle ──
    rv_back = _clip_surf(rv_egg, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z <= 0, density=1.0)
    rv_inner = _egg((-10, -25, 20), (32, 40, 24), rot_rv, 1.8)
    rv_inner_s = _clip_surf(rv_inner, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z <= 0, density=1.0)
    rv_sampler = _pick([rv_back, rv_inner_s])
    rv_part = Part(1, sample=rv_sampler)
    parts.append(HiPart(name="Right ventricle", sampler=rv_part.sample,
                        color=(0.45, 0.6, 1.0), weight=2, info="Right ventricle",
                        direction=(-120, -70, 0), spread=0.7))

    # ── Septum ──
    def _septum_fn(u, v):
        x = -5 + u * 60
        y = -55 + v * 105
        z = 3 * np.sin(u * pi) * np.cos(v * pi * 0.5)
        return np.array([x, y, z], dtype=np.float64)

    septum = param_surface(_septum_fn, 0, 1, 0, 1, density=0.7)
    parts.append(HiPart(name="Septum", sampler=septum.sample,
                        color=(1.0, 0.6, 0.7), weight=1, info="Septum",
                        direction=(0, -45, 20), spread=0.8))

    # ── Heart valves ──
    def _orient_torus(center, major_r, minor_r, normal):
        local = torus_shape((0, 0, 0), major_r, minor_r, (0, 1, 0))
        return orient([local], normal, center=center)[0]

    def _orient_disc(center, inner_r, outer_r, normal):
        local = disc((0, 0, 0), inner_r, outer_r, (0, 1, 0))
        return orient([local], local_normal=None)[0] if False else local

    def _valve_torus_disc(center, major_r, minor_r, normal):
        tor = _orient_torus(center, major_r, minor_r, normal)
        d_center = np.array(center, dtype=np.float64)
        local_disc = disc((0, 0, 0), 0, major_r - minor_r * 0.8, (0, 1, 0))
        dsc = orient([local_disc], normal, center=d_center)[0]
        return [tor, dsc]

    def _cusp_domes(center, major_r, normal, n_cusps, dome_h):
        domes = []
        for i in range(n_cusps):
            ang = i * TAU / n_cusps
            cp = np.array(center) + major_r * np.array([np.cos(ang), 0, np.sin(ang)])
            def dome_fn(u, v):
                x = major_r * 0.25 * np.cos(v) * np.cos(u)
                y = dome_h * np.sin(v)
                z = major_r * 0.25 * np.cos(v) * np.sin(u)
                return np.array([x, y, z], dtype=np.float64)
            dome = param_surface(dome_fn, 0, TAU, 0, pi / 2, density=0.3)
            oriented = orient([dome], normal, center=tuple(cp))[0]
            domes.append(oriented)
        return domes

    def _cusp_boundary_lines(center, major_r, normal, n_cusps):
        lines = []
        for i in range(n_cusps):
            ang = i * TAU / n_cusps
            cp = np.array(center) + major_r * np.array([np.cos(ang), 0, np.sin(ang)])
            peak = np.array(center) + major_r * 0.15 * _arr(normal)
            lines.append(line(tuple(cp), tuple(peak), jitter=0.3, weight=0.15))
        return lines

    valve_parts = []

    # Aortic valve — 3 cusps
    ao_v_center = (-5, 70, 35)
    ao_v_normal = _norm((0, 1, 0.15))
    valve_parts.extend(_valve_torus_disc(ao_v_center, 10, 2.5, ao_v_normal))
    valve_parts.extend(_cusp_domes(ao_v_center, 10, ao_v_normal, 3, 6))
    valve_parts.extend(_cusp_boundary_lines(ao_v_center, 10, ao_v_normal, 3))

    # Pulmonary valve — 3 cusps
    pv_center = (-32, 72, 40)
    pv_normal = _norm((-0.2, 1, 0.2))
    valve_parts.extend(_valve_torus_disc(pv_center, 9, 2.3, pv_normal))
    valve_parts.extend(_cusp_domes(pv_center, 9, pv_normal, 3, 5.5))
    valve_parts.extend(_cusp_boundary_lines(pv_center, 9, pv_normal, 3))

    # Mitral valve — 2 cusps
    mv_center = (2, 15, -20)
    mv_normal = _norm((0, 0.7, -0.7))
    valve_parts.extend(_valve_torus_disc(mv_center, 11, 2.5, mv_normal))
    valve_parts.extend(_cusp_domes(mv_center, 11, mv_normal, 2, 7))
    valve_parts.extend(_cusp_boundary_lines(mv_center, 11, mv_normal, 2))

    # Tricuspid valve — 3 cusps
    tv_center = (-28, 5, 12)
    tv_normal = _norm((0, 0.7, 0.7))
    valve_parts.extend(_valve_torus_disc(tv_center, 12, 2.5, tv_normal))
    valve_parts.extend(_cusp_domes(tv_center, 12, tv_normal, 3, 6.5))
    valve_parts.extend(_cusp_boundary_lines(tv_center, 12, tv_normal, 3))

    valve_sampler = _pick(valve_parts)
    valve_part = Part(1, sample=valve_sampler)
    parts.append(HiPart(name="Heart valves", sampler=valve_part.sample,
                        color=(1.0, 0.95, 0.55), weight=1.5, info="Heart valves",
                        direction=(0, 70, 100), spread=0.7))

    # ── Papillary muscles & chordae ──
    pap_parts = []
    # LV papillary
    lv_pap = vessel([(42, -70, -10), (35, -40, -15), (28, -10, -18)], 3, 4, per=8)
    pap_parts.append(lv_pap)
    for i in range(6):
        t = i / 5.0
        end_pt = (2 + t * 10, 15 + t * 10, -20 - t * 15)
        pap_parts.append(line((28, -10, -18), end_pt, jitter=1.0, weight=0.15))

    # RV papillary
    rv_pap = vessel([(-25, -60, 15), (-30, -30, 18), (-32, -5, 18)], 2.5, 3.5, per=8)
    pap_parts.append(rv_pap)
    for i in range(6):
        t = i / 5.0
        end_pt = (-28 - t * 10, t * 15, 12 + t * 5)
        pap_parts.append(line((-32, -5, 18), end_pt, jitter=1.0, weight=0.15))

    pap_sampler = _pick(pap_parts)
    pap_part = Part(1, sample=pap_sampler)
    parts.append(HiPart(name="Papillary muscles & chordae", sampler=pap_part.sample,
                        color=(0.95, 1.0, 0.85), weight=1.5, info="Papillary muscles & chordae",
                        direction=(100, -40, 40), spread=0.8))

    return assemble("Heart", "biology", parts, n, seed=1, beat_bpm=72.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Kidney
# ═══════════════════════════════════════════════════════════════════════════

def kidney(n):
    def _bean(sc):
        def fn(u, v):
            x = (66 * np.cos(v) * np.cos(u) +
                 40 * np.exp(-(_wrap_pi(u - np.pi)) ** 2 / 0.35) * np.cos(v) ** 4) * sc
            y = 118 * np.sin(v) * sc
            z = 40 * np.cos(v) * np.sin(u) * sc
            return np.array([x, y, z], dtype=np.float64)
        return fn

    def _clip_bean(fn, u0, u1, v0, v1, keep_fn, density=1):
        du, dv = u1 - u0, v1 - v0

        def sample(k, rng):
            mul = 4
            u = rng.uniform(np.float64(u0), np.float64(u1), k * mul)
            v = rng.uniform(np.float64(v0), np.float64(v1), k * mul)
            pts = np.array([fn(ui, vi) for ui, vi in zip(u, v)])
            mask = np.array([keep_fn(*p) for p in pts])
            p = pts[mask]
            while len(p) < k:
                u2 = rng.uniform(np.float64(u0), np.float64(u1), k)
                v2 = rng.uniform(np.float64(v0), np.float64(v1), k)
                extra = np.array([fn(ui, vi) for ui, vi in zip(u2, v2)])
                m2 = np.array([keep_fn(*pt) for pt in extra])
                p = np.concatenate([p, extra[m2]])
            return _f32(p[:k])

        return Part(density * du * dv * 0.5, sample)

    parts = []
    pi = np.pi

    bean1 = _bean(1.0)
    bean09 = _bean(0.9)

    # ── Renal capsule ──
    cap_front = _clip_bean(bean1, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z > 0, density=1.0)
    parts.append(HiPart(name="Renal capsule", sampler=cap_front.sample,
                        color=(1.0, 0.5, 0.45), weight=2, info="Renal capsule",
                        direction=(0, 0, 210), spread=0.6))

    cap_back = _clip_bean(bean1, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z <= 0, density=1.0)
    parts.append(HiPart(name="Renal capsule (back)", sampler=cap_back.sample,
                        color=(1.0, 0.5, 0.45), weight=2, info="Renal capsule",
                        direction=(0, 0, -210), spread=0.6))

    # ── Renal cortex ──
    ctx_front = _clip_bean(bean09, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z > 0, density=1.0)
    parts.append(HiPart(name="Renal cortex", sampler=ctx_front.sample,
                        color=(1.0, 0.7, 0.35), weight=1.5, info="Renal cortex",
                        direction=(0, 0, 125), spread=0.6))

    ctx_back = _clip_bean(bean09, 0, TAU, -pi / 2, pi / 2, lambda x, y, z: z <= 0, density=1.0)
    parts.append(HiPart(name="Renal cortex (back)", sampler=ctx_back.sample,
                        color=(1.0, 0.7, 0.35), weight=1.5, info="Renal cortex",
                        direction=(0, 0, -125), spread=0.6))

    # ── Medulla (pyramids) ──
    S = np.array([-22, 0, 0], dtype=np.float64)
    dirs = [
        _norm([1, 0.95, 0]),
        _norm([1, -0.95, 0]),
        _norm([0.65, 0.5, 0.75]),
        _norm([0.65, 0.5, -0.75]),
        _norm([0.65, -0.5, 0.75]),
        _norm([0.65, -0.5, -0.75]),
        _norm([0.3, 0.2, 0.95]),
        _norm([0.3, 0.2, -0.95]),
    ]

    pyramid_parts = []

    def _build_pyramid(d, idx):
        L = 34 * (1 + 0.8 * abs(d[1]))
        tip = S + d * 24
        cone = revolve(ys=[0, L], rs=[4, 17], a0=0, a1=TAU)
        ori_cone = orient([cone], d, center=tuple(tip))[0]

        ray_lines = []
        for i in range(10):
            frac = 0.1 + i * 0.085
            r_at = 4 + (17 - 4) * frac
            y_at = frac * L
            p = tip + d * y_at
            ray_lines.append(line(tuple(p - _arr(d) * 5), tuple(p + _arr(d) * 5), jitter=0.8, weight=0.08))

        return [ori_cone] + ray_lines

    for idx, d in enumerate(dirs):
        pyr_sampler = _pick(_build_pyramid(d, idx))
        pyr_part = Part(1, sample=pyr_sampler)
        pyramid_parts.append(pyr_part)

    pyr_combined = _pick(pyramid_parts)
    pyr_all = Part(1, sample=pyr_combined)
    parts.append(HiPart(name="Medulla (pyramids)", sampler=pyr_all.sample,
                        color=(0.75, 0.45, 1.0), weight=2, info="Medulla (pyramids)",
                        direction=(45, 0, 0), spread=0.7))

    # ── Nephrons ──
    nephron_parts = []
    for idx, d in enumerate(dirs):
        if idx % 2 != 0:
            continue
        L = 34 * (1 + 0.8 * abs(d[1]))
        tip = S + d * 24
        base = tip + d * L

        for gi in range(3):
            offset = np.cross(d, [0, 1, 0]) if abs(d[1]) < 0.99 else np.cross(d, [1, 0, 0])
            offset *= 0.7 / np.linalg.norm(offset)
            glom_pos = base + offset * (6 + gi * 3)
            glom = ellipsoid(tuple(glom_pos), (3, 3.5, 3))

            henle_pts = [tuple(glom_pos),
                         tuple(tip + d * L * 0.5 + offset * 2),
                         tuple(tip + d * L * 0.2),
                         tuple(tip + d * L * 0.5 - offset * 2),
                         tuple(glom_pos + offset * 4)]
            henle = polyline(henle_pts, jitter=1.0, weight=0.4)
            nephron_parts.extend([glom, henle])

    neph_sampler = _pick(nephron_parts)
    neph_part = Part(1, sample=neph_sampler)
    parts.append(HiPart(name="Nephrons", sampler=neph_part.sample,
                        color=(0.5, 1.0, 0.45), weight=1.5, info="Nephrons",
                        direction=(180, 0, 70), spread=0.9))

    # ── Calyces & pelvis ──
    calyx_parts = []
    for idx, d in enumerate(dirs):
        L = 34 * (1 + 0.8 * abs(d[1]))
        tip = S + d * 24
        cup = revolve(ys=[0, 7, 9], rs=[3, 5.5, 7], a0=0, a1=TAU)
        ori_cup = orient([cup], d, center=tuple(tip))[0]
        calyx_parts.append(ori_cup)

        cup_stalk = vessel([tuple(tip + d * 3), tuple(tip - d * 5), tuple(S)], 2.5, 3.5, per=6)
        calyx_parts.append(cup_stalk)

    pelvis = ellipsoid(S, (14, 12, 14))
    calyx_parts.append(pelvis)

    calyx_sampler = _pick(calyx_parts)
    calyx_part = Part(1, sample=calyx_sampler)
    parts.append(HiPart(name="Calyces & pelvis", sampler=calyx_part.sample,
                        color=(1.0, 0.9, 0.35), weight=1.5, info="Calyces & pelvis",
                        direction=(-70, 0, 0), spread=0.7))

    # ── Ureter ──
    ureter = vessel([tuple(S), tuple(S + np.array([-10, -40, -5])),
                     tuple(S + np.array([-5, -90, -8])),
                     tuple(S + np.array([0, -140, -10]))], 4.5, 3.5, per=10)
    parts.append(HiPart(name="Ureter", sampler=ureter.sample,
                        color=(1.0, 0.95, 0.65), weight=1, info="Ureter",
                        direction=(-70, -70, 0), spread=0.5))

    # ── Renal artery ──
    artery_pts = [tuple(S + np.array([-80, 40, 25])), tuple(S + np.array([-45, 15, 12])), tuple(S)]
    main_artery = vessel(artery_pts, 5, 5.5, per=10)
    artery_parts = [main_artery]
    for idx, d in enumerate(dirs):
        L = 34 * (1 + 0.8 * abs(d[1]))
        tip = S + d * 24
        branch = vessel([tuple(S), tuple(S + d * 12), tuple(tip + d * L * 0.3)], 3, 2, per=6)
        artery_parts.append(branch)

    art_sampler = _pick(artery_parts)
    art_part = Part(1, sample=art_sampler)
    parts.append(HiPart(name="Renal artery", sampler=art_part.sample,
                        color=(1.0, 0.25, 0.25), weight=1.5, info="Renal artery",
                        direction=(-120, 80, 45), spread=0.6))

    # ── Renal vein ──
    vein_pts = [tuple(S + np.array([-80, -40, -25])), tuple(S + np.array([-45, -15, -12])), tuple(S)]
    main_vein = vessel(vein_pts, 6, 6.5, per=10)
    vein_parts = [main_vein]
    for idx, d in enumerate(dirs):
        L = 34 * (1 + 0.8 * abs(d[1]))
        tip = S + d * 24
        branch = vessel([tuple(S + d * 3), tuple(S + d * 15), tuple(tip + d * L * 0.35)], 3.5, 2.5, per=6)
        vein_parts.append(branch)

    vein_sampler = _pick(vein_parts)
    vein_part = Part(1, sample=vein_sampler)
    parts.append(HiPart(name="Renal vein", sampler=vein_part.sample,
                        color=(0.4, 0.5, 1.0), weight=1.5, info="Renal vein",
                        direction=(-120, -80, -45), spread=0.6))

    # ── Adrenal gland ──
    def _adrenal_fn(u, v):
        sc = 0.85 + 0.15 * np.sin(5 * u) * np.cos(3 * v)
        x = 18 * sc * np.cos(v) * np.cos(u)
        y = 22 * sc * np.sin(v) + 5
        z = 12 * sc * np.cos(v) * np.sin(u)
        return np.array([x, y, z], dtype=np.float64)

    top_pos = np.array([-18, 115, 0], dtype=np.float64)
    def _adrenal_translated_fn(u, v):
        return _adrenal_fn(u, v) + top_pos

    adrenal = param_surface(_adrenal_translated_fn, 0, TAU, -pi / 2, pi / 2, density=0.8)
    parts.append(HiPart(name="Adrenal gland", sampler=adrenal.sample,
                        color=(1.0, 0.82, 0.25), weight=1, info="Adrenal gland",
                        direction=(0, 150, 0), spread=0.4))

    return assemble("Kidney", "biology", parts, n, seed=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════

ANATOMY = [
    {"name": "Human Brain", "build": brain, "color": (1.0, 0.5, 0.7), "beat_bpm": None},
    {"name": "Human Heart", "build": heart, "color": (1.0, 0.28, 0.32), "beat_bpm": 72.0},
    {"name": "Kidney", "build": kidney, "color": (1.0, 0.55, 0.45), "beat_bpm": None},
]