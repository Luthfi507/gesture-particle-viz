import numpy as np
from .base import Part as HiPart, assemble
from .shapes import (Part, line, polyline, tri, quad, box_surface, box_edges,
                     sphere, ring, disc, tube_shape, param_surface,
                     ellipsoid, capsule, blob, dots,
                     torus_shape, revolve,
                     transform, orient, _pick, flatten,
                     rot_z, deg, TAU, vessel, path_tube,
                     linspace, range_n, _spline, _norm)
from .shapes import _add, _sub, _mul, _cross


def _normalize_or_zero(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array(v)


def dna_helix(n):
    turns = 2
    H = 480.0
    R = 70.0
    num_pairs = turns * 10
    rng = np.random.default_rng(21)

    def _s1(t):
        a = 2 * np.pi * turns * t
        return np.array([R * np.cos(a), -H / 2 + H * t, R * np.sin(a)])

    def _s2(t):
        a = 2 * np.pi * turns * t + np.pi * 0.8
        return np.array([R * np.cos(a), -H / 2 + H * t, R * np.sin(a)])

    dense = linspace(0, 1, 160)
    backbone1 = path_tube([_s1(t) for t in dense], 6, 1.1)
    backbone2 = path_tube([_s2(t) for t in dense], 6, 1.1)

    PAIR = {"A": "T", "T": "A", "G": "C", "C": "G"}
    bases = {"1A": [], "1T": [], "1G": [], "1C": [],
             "2A": [], "2T": [], "2G": [], "2C": []}
    hbonds = []
    phosphates1 = []
    phosphates2 = []

    for i in range(num_pairs):
        t = (i + 0.5) / num_pairs
        p1 = _s1(t)
        p2 = _s2(t)
        mid = (p1 + p2) * 0.5
        b = "ATGC"[rng.integers(0, 4)]
        phosphates1.append(sphere(p1, 9, 1.2))
        phosphates2.append(sphere(p2, 9, 1.2))

        gap = _normalize_or_zero(p2 - p1) * 4
        bases["1" + b].append(tube_shape(p1, mid - gap, 5, 5, 1.3))
        bases["2" + PAIR[b]].append(tube_shape(p2, mid + gap, 5, 5, 1.3))

        k = 3 if b in "GC" else 2
        side = _normalize_or_zero(np.cross(p2 - p1, np.array([0.0, 1.0, 0.0])))
        for h in range(k):
            offset = side * (h - (k - 1) / 2) * 5
            hbonds.append(line((mid - gap) + offset, (mid + gap) + offset, 0.6, 4))

    COL = {"A": (1.0, 0.35, 0.35), "T": (0.4, 1.0, 0.45), "G": (0.35, 0.6, 1.0), "C": (1.0, 0.85, 0.3)}
    NAME = {"A": "Adenine (A)", "T": "Thymine (T)", "G": "Guanine (G)", "C": "Cytosine (C)"}
    INFO = {"A": "always pairs with thymine (2 bonds)", "T": "always pairs with adenine",
            "G": "pairs with cytosine (3 bonds)", "C": "always pairs with guanine"}

    backbone_parts = [backbone1]
    backbone_parts.extend(phosphates1)
    backbone_parts.extend(phosphates2)

    parts = [
        HiPart("Backbone (sugar-phosphate)", _pick(backbone_parts), (0.8, 0.85, 1.0),
               5.0, "the two rails of the twisted ladder", (-190, 0, 0), 0.5),
        HiPart("Backbone (L)", _pick([backbone2]), (0.8, 0.85, 1.0),
               5.0, "complementary backbone", (190, 0, 0), 0.5),
        HiPart("Hydrogen bonds", _pick(hbonds), (1.0, 1.0, 1.0),
               2.0, "weak links: DNA unzips here to copy itself", (0, 0, 60), 0.6),
    ]

    for b in "ATGC":
        k1 = "1" + b
        k2 = "2" + b
        if bases[k1]:
            parts.append(HiPart(NAME[b], _pick(bases[k1]), COL[b],
                                2.0, INFO[b], (-150, 0, 0), 0.6))
        if bases[k2]:
            show_label = not bool(bases[k1])
            parts.append(HiPart(NAME[b] + (" (L)" if bases[k1] else ""), _pick(bases[k2]),
                                COL[b], 2.0, INFO[b], (150, 0, 0), 0.6))

    return assemble("DNA Double Helix", "biology", parts, n, seed=5)


def animal_cell(n):
    rng = np.random.default_rng(8)

    def _mem(center, radii, front):
        def keep(x, y, z):
            return (z >= 0) if front else (z < 0)
        def bump(u, v):
            return 1 + 0.03 * np.sin(5 * u + 2 * np.sin(3 * v))
        return blob(center, radii, 0.4, keep=keep, bump=bump)

    mem_front = [_mem(np.zeros(3), [200, 170, 185], True)]
    mem_back = [_mem(np.zeros(3), [200, 170, 185], False)]

    nuc = np.array([-20.0, 10.0, 0.0])
    nucleus = [sphere(nuc, 70, 0.7), sphere(nuc, 74, 0.3)]

    pores = []
    for _ in range(40):
        d = _normalize_or_zero(np.array([rng.normal(), rng.normal(), rng.normal()]))
        pores.append(nuc + d * 74)
    nucleus.append(dots(pores, 2.5, 1.4))

    for _ in range(18):
        p = nuc + np.array([rng.normal() * 25, rng.normal() * 25, rng.normal() * 25])
        pts = [p.tolist()]
        for __ in range(12):
            p = p + np.array([rng.normal() * 9, rng.normal() * 9, rng.normal() * 9])
            if np.linalg.norm(p - nuc) < 60:
                pts.append(p.tolist())
        if len(pts) > 1:
            nucleus.append(polyline(pts, 0, 2.5))

    nucleolus = [sphere(nuc + np.array([10, 15, 10]), 22, 1.4)]

    mito = []
    for i in range(7):
        a = (i / 7) * 2 * np.pi + 0.3
        c = np.array([140 * np.cos(a) + 10, 95 * np.sin(a) - 10, (rng.random() - 0.5) * 120])
        d = _normalize_or_zero(np.array([np.cos(a + 1.3), np.sin(a + 1.3), rng.random() - 0.5]))
        p0 = c + d * (-26)
        p1 = c + d * 26
        mito.append(capsule(p0, p1, 15, 0.9))
        side = _normalize_or_zero(np.cross(d, np.array([0.0, 0.0, 1.0])))
        zig = []
        for k in range(11):
            t = k / 10
            zig.append((c + d * (52 * t) + side * (11 if k % 2 else -11)).tolist())
        mito.append(polyline(zig, 0, 4))

    def _er_shell(r, u0, u1, v0, v1):
        def fn(u, v):
            rr = r + 3 * np.sin(12 * u)
            return nuc + np.array([rr * np.cos(v) * np.cos(u), rr * np.sin(v), rr * np.cos(v) * np.sin(u)])
        return param_surface(fn, u0, u1, v0, v1, 0.7, gu=20, gv=8)

    rough = [_er_shell(90, 0.3, 2.2, -0.6, 0.6),
             _er_shell(102, 0.3, 2.2, -0.6, 0.6),
             _er_shell(114, 0.3, 2.2, -0.6, 0.6)]
    ribo = []
    for i in range(220):
        rr = [92, 104, 116][i % 3] + 2
        u = 0.3 + rng.random() * 1.9
        v = -0.6 + rng.random() * 1.2
        ribo.append((nuc + np.array([rr * np.cos(v) * np.cos(u), rr * np.sin(v), rr * np.cos(v) * np.sin(u)])).tolist())
    rough.append(dots(ribo, 1.8, 1.5))

    smooth = []
    for _ in range(6):
        p = nuc + np.array([-95 + rng.normal() * 10, rng.normal() * 40, rng.normal() * 40])
        pts = [p.tolist()]
        for __ in range(6):
            p = p + np.array([-10 + rng.normal() * 10, rng.normal() * 18, rng.normal() * 18])
            pts.append(p.tolist())
        smooth.append(vessel(pts, 4, 4, 1.2))

    golgi_c = np.array([100.0, -70.0, 40.0])
    golgi = []
    for k in range(5):
        def _golgi_fn(rr, kk):
            def fn(r, a):
                return golgi_c + np.array([rr * r * np.cos(a), kk * 8 - 16 + 0.012 * (rr * r) ** 2, rr * 0.55 * r * np.sin(a)])
            return fn
        golgi.append(param_surface(_golgi_fn(40 - k * 3, k), 0.15, 1, 0, 2 * np.pi, 1.1, gu=6, gv=24))
    golgi_vesicles = []
    for t in linspace(0, 1, 14):
        golgi_vesicles.append(golgi_c + np.array([45 * np.cos(t * 2 * np.pi), 30 + 8 * np.sin(t * 9), 25 * np.sin(t * 2 * np.pi)]))
    golgi.append(dots(golgi_vesicles, 3.5, 1.2))

    lyso = []
    for _ in range(6):
        a = rng.random() * 2 * np.pi
        lyso.append(sphere(np.array([120 * np.cos(a), 110 * np.sin(a) * 0.6 + 40, (rng.random() - 0.5) * 140]), 11, 1.3))

    cen = np.array([60, 95, -30])
    centrioles = [
        param_surface(lambda u, h: cen + np.array([8 * np.cos(u), -15 + 30 * h, 8 * np.sin(u)]), 0, 2 * np.pi, 0, 1, 1.5, gu=18, gv=4),
        param_surface(lambda u, h: cen + np.array([5 + 30 * h, 8 * np.cos(u), 8 * np.sin(u)]), 0, 2 * np.pi, 0, 1, 1.5, gu=18, gv=4),
    ]

    skel = []
    for _ in range(34):
        d = _normalize_or_zero(np.array([rng.normal(), rng.normal(), rng.normal()]))
        skel.append(line(cen, np.array([d[0] * 185, d[1] * 155, d[2] * 170]), 0, 0.7))

    free_ribo = []
    for _ in range(180):
        d = _normalize_or_zero(np.array([rng.normal(), rng.normal(), rng.normal()]))
        r = 90 + rng.random() * 90
        free_ribo.append(np.array([d[0] * r, d[1] * r * 0.85, d[2] * r * 0.9]))

    parts = [
        HiPart("Cell membrane", _pick(mem_front), (0.5, 0.85, 1.0),
               4.0, "gatekeeper: controls what gets in and out", (0, 0, 270), 0.4),
        HiPart("Cell membrane (back)", _pick(mem_back), (0.5, 0.85, 1.0),
               4.0, "", (0, 0, -270), 0.4),
        HiPart("Nucleus", _pick(nucleus), (0.75, 0.5, 1.0),
               3.0, "holds the DNA: the control centre", (-90, 60, 0), 0.6),
        HiPart("Nucleolus", _pick(nucleolus), (1.0, 0.4, 0.8),
               1.5, "builds the ribosomes", (-60, 170, 60), 0.7),
        HiPart("Mitochondria", _pick(mito), (1.0, 0.55, 0.25),
               3.0, "power plants: turn food into ATP energy", (160, -20, 70), 0.6),
        HiPart("Rough ER", _pick(rough), (0.4, 0.9, 1.0),
               2.5, "ribosome-studded folds that make proteins", (-40, -150, 0), 0.7),
        HiPart("Smooth ER", _pick(smooth), (0.4, 1.0, 0.7),
               2.0, "makes fats, clears toxins", (-190, -60, 0), 0.7),
        HiPart("Golgi apparatus", _pick(golgi), (1.0, 0.85, 0.35),
               2.5, "packs and ships proteins", (190, -120, 40), 0.6),
        HiPart("Lysosomes", _pick(lyso), (1.0, 0.35, 0.35),
               1.5, "recycling bins full of enzymes", (60, 170, 0), 0.7),
        HiPart("Centrioles", _pick(centrioles), (0.9, 1.0, 0.5),
               1.5, "organise the split when a cell divides", (110, 170, -60), 0.7),
        HiPart("Cytoskeleton", _pick(skel), (0.8, 0.8, 1.0),
               2.0, "scaffolding and transport rails", (0, 0, -80), 0.7),
        HiPart("Free ribosomes", _pick([dots(free_ribo, 2, 1.2)]), (0.95, 0.95, 0.95),
               2.0, "read RNA and build proteins", (0, -40, 140), 0.7),
    ]
    return assemble("Animal Cell", "biology", parts, n, seed=8)


BIOLOGY = [
    {"name": "DNA Double Helix", "build": dna_helix, "color": (0.55, 0.75, 1.0), "beat_bpm": None},
    {"name": "Animal Cell", "build": animal_cell, "color": (0.5, 0.9, 1.0), "beat_bpm": None},
]