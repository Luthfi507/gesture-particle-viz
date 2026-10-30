import numpy as np
from typing import Callable, Optional, Sequence

TAU = 2 * np.pi

Sampler = Callable[[int, np.random.Generator], np.ndarray]


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _arr(a):
    return np.asarray(a, dtype=np.float64)


# ── basic primitives ────────────────────────────────────────────

class Part:
    def __init__(self, weight, sample):
        self.weight = float(weight) if weight is not None else 1.0
        self.sample = sample


def line(a, b, jitter=0, density=None, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    def sample(k, rng):
        t = rng.random((k, 1))
        a2, b2 = _arr(a), _arr(b)
        return _f32(a2 + (b2 - a2) * t + rng.normal(0, max(jitter, 1e-9), size=(k, 3)))
    return Part(w, sample)


def polyline(pts, jitter=0, density=None, weight=None, closed=False):
    w = density if density is not None else (weight if weight is not None else 1.0)
    pts = _arr(pts)
    segs = len(pts) - 1
    if closed:
        segs = len(pts)
    if segs <= 0:
        return Part(0.001, lambda k, rng: _f32(np.zeros((k, 3))))
    if closed:
        diff = pts[(np.arange(segs) + 1) % len(pts)] - pts[np.arange(segs)]
    else:
        diff = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(diff, axis=1)

    def sample(k, rng):
        seg = rng.choice(segs, size=k, p=lengths / lengths.sum())
        t = rng.random(k)
        a_v = pts[seg]
        b_v = pts[(seg + 1) % len(pts)]
        p = a_v + (b_v - a_v) * t[:, None]
        return _f32(p + rng.normal(0, max(jitter, 1e-9), size=(k, 3)))
    return Part(w * lengths.sum(), sample)


def tri(a, b, c, density=None, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    a_v, b_v, c_v = _arr(a), _arr(b), _arr(c)

    def sample(k, rng):
        u = rng.random(k)
        v = rng.random(k)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        return _f32(a_v + (b_v - a_v) * u[:, None] + (c_v - a_v) * v[:, None])
    return Part(w, sample)


def quad(a, b, c, d, density=None, weight=None, keep=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    p = [tri(a, b, c, weight=w / 2), tri(a, c, d, weight=w / 2)]
    return Part(w, sample=_pick(p))


def sphere(center, r, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    def sample(k, rng):
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return _f32(v * r + _arr(center))
    return Part(w * 4 * np.pi * r * r, sample)


def revolve(ys, rs, density=1, center=None, sx=1, sz=1, keep=None, a0=0, a1=TAU, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    ys = _arr(ys)
    rs = _arr(rs)
    cen = _arr(center) if center else np.zeros(3)
    da = a1 - a0
    segments = len(ys) - 1
    heights = np.diff(ys)
    areas = heights * (rs[:-1] + rs[1:]) / 2
    total_area = da * areas.sum()

    def sample(k, rng):
        seg = rng.choice(segments, size=k, p=areas / areas.sum())
        t = rng.random(k)
        y = ys[seg] + t * heights[seg]
        rt = rs[seg] + t * (rs[seg + 1] - rs[seg])
        a = rng.uniform(a0, a1, k)
        pts = np.stack([rt * np.cos(a) * sx, y, rt * np.sin(a) * sz], axis=1)
        if keep:
            mask = np.array([keep(*p) for p in pts])
            pts = pts[mask]
        return _f32(pts + cen)
    return Part(w * total_area, sample)


def ring(center, r, density=1, a0=0, a1=TAU, weight=None, normal=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    da = a1 - a0

    def sample(k, rng):
        a = rng.uniform(a0, a1, k)
        return _f32(np.stack([r * np.cos(a) + center[0],
                              r * np.sin(a) + center[1],
                              np.zeros(k) + center[2]], axis=1))
    return Part(w * r * da, sample)


def disc(center, inner_r, outer_r, normal, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    area = np.pi * (outer_r ** 2 - inner_r ** 2)

    def sample(k, rng):
        a = rng.uniform(0, TAU, k)
        rr = np.sqrt(inner_r ** 2 + rng.random(k) * (outer_r ** 2 - inner_r ** 2))
        p = np.stack([rr * np.cos(a), rr * np.sin(a), np.zeros(k)], axis=1)
        return _f32(p + _arr(center))
    return Part(w * area, sample)


def cylinder_s(center, r, h, density=1, axis=1, cap=True, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    side_area = TAU * r * h
    cap_area = (TAU * r * r) if cap else 0
    total = side_area + cap_area

    def sample(k, rng):
        on_cap = rng.random(k) < cap_area / total
        ang = rng.uniform(0, TAU, k)
        rr = np.where(on_cap, r * np.sqrt(rng.random(k)), r)
        along = np.where(on_cap, np.where(rng.random(k) < 0.5, -h / 2, h / 2), rng.uniform(-h / 2, h / 2, k))
        p = np.stack([rr * np.cos(ang), along, rr * np.sin(ang)], axis=1)
        return _f32(p + _arr(center))
    return Part(w * total, sample)


def box_surface(center, size, density=1, keep=None, top=True, bottom=True, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    cx, cy, cz = center
    hw, hh, hd = size[0] / 2, size[1] / 2, size[2] / 2
    faces = []
    if top:
        faces.append(quad([cx - hw, cy + hh, cz - hd], [cx + hw, cy + hh, cz - hd],
                          [cx + hw, cy + hh, cz + hd], [cx - hw, cy + hh, cz + hd], weight=w * 4 * hw * hd))
    if bottom:
        faces.append(quad([cx - hw, cy - hh, cz - hd], [cx + hw, cy - hh, cz - hd],
                          [cx + hw, cy - hh, cz + hd], [cx - hw, cy - hh, cz + hd], weight=w * 4 * hw * hd))
    for c1, c2, c3, c4 in [
        ([cx + hw, cy - hh, cz - hd], [cx + hw, cy + hh, cz - hd], [cx + hw, cy + hh, cz + hd], [cx + hw, cy - hh, cz + hd]),
        ([cx - hw, cy - hh, cz - hd], [cx - hw, cy + hh, cz - hd], [cx - hw, cy + hh, cz + hd], [cx - hw, cy - hh, cz + hd]),
        ([cx - hw, cy - hh, cz + hd], [cx - hw, cy + hh, cz + hd], [cx + hw, cy + hh, cz + hd], [cx + hw, cy - hh, cz + hd]),
        ([cx - hw, cy - hh, cz - hd], [cx - hw, cy + hh, cz - hd], [cx + hw, cy + hh, cz - hd], [cx + hw, cy - hh, cz - hd]),
    ]:
        side_w = w * 2 * hh * (hd if 'hd' in str(c1) else hw)
        if keep:
            faces.append(quad(c1, c2, c3, c4, weight=side_w, keep=keep))
        else:
            faces.append(quad(c1, c2, c3, c4, weight=side_w))
    return Part(1, sample=_pick(faces))


def box_edges(center, size, jitter=0, density=None, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    cx, cy, cz = center
    hw, hh, hd = size[0] / 2, size[1] / 2, size[2] / 2
    edges = []
    for y in [-hh, hh]:
        for z in [-hd, hd]:
            edges.append(line([cx - hw, cy + y, cz + z], [cx + hw, cy + y, cz + z], jitter, weight=w / 12))
    for x in [-hw, hw]:
        for z in [-hd, hd]:
            edges.append(line([cx + x, cy - hh, cz + z], [cx + x, cy + hh, cz + z], jitter, weight=w / 12))
    for x in [-hw, hw]:
        for y in [-hh, hh]:
            edges.append(line([cx + x, cy + y, cz - hd], [cx + x, cy + y, cz + hd], jitter, weight=w / 12))
    return Part(w, sample=_pick(edges))


def box_filled(center, size, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    cx, cy, cz = center
    hw, hh, hd = size[0] / 2, size[1] / 2, size[2] / 2

    def sample(k, rng):
        return _f32(np.stack([rng.uniform(cx - hw, cx + hw, k),
                              rng.uniform(cy - hh, cy + hh, k),
                              rng.uniform(cz - hd, cz + hd, k)], axis=1))
    return Part(w, sample)


def tube_shape(a, b, r0, r1, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    a_v, b_v = _arr(a), _arr(b)
    axis_v = b_v - a_v
    length = np.linalg.norm(axis_v)
    area = np.pi * (r0 + r1) * np.sqrt(length ** 2 + (r1 - r0) ** 2)

    if length < 1e-12:
        perp1 = np.array([1.0, 0.0, 0.0])
        perp2 = np.array([0.0, 1.0, 0.0])
    else:
        axis_v = axis_v / length
        if abs(axis_v[0]) < 0.9:
            perp1 = np.array([axis_v[1], -axis_v[0], 0.0])
        else:
            perp1 = np.array([0.0, axis_v[2], -axis_v[1]])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis_v, perp1)

    def sample(k, rng):
        t = rng.random(k)
        center = a_v + axis_v * (t[:, None] * length)
        r = r0 + (r1 - r0) * t
        angles = rng.uniform(0, TAU, k)
        pts = center + perp1 * (r[:, None] * np.cos(angles)[:, None]) + perp2 * (r[:, None] * np.sin(angles)[:, None])
        return _f32(pts)
    return Part(w * area, sample)


def pipe(pts, r, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    pts_a = _arr(pts)
    segs = [tube_shape(pts_a[i], pts_a[i + 1], r, r, weight=w / (len(pts) - 1)) for i in range(len(pts) - 1)]
    return Part(w, sample=_pick(segs))


def param_surface(fn, u0, u1, v0, v1, density=1, gu=30, gv=30, keep=None, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    du = u1 - u0
    dv = v1 - v0
    area_approx = du * dv

    def sample(k, rng):
        u = rng.uniform(u0, u1, k)
        v = rng.uniform(v0, v1, k)
        pts = np.array([fn(ui, vi) for ui, vi in zip(u, v)])
        if keep:
            mask = np.array([keep(*p) for p in pts])
            pts = pts[mask]
        return _f32(pts)
    return Part(w * area_approx, sample)


def torus_shape(center, major_r, minor_r, normal=None, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    area = TAU * major_r * TAU * minor_r

    def sample(k, rng):
        th = rng.uniform(0, TAU, k)
        ph = rng.uniform(0, TAU, k)
        cr = major_r + minor_r * np.cos(ph)
        x = cr * np.cos(th)
        y = minor_r * np.sin(ph)
        z = cr * np.sin(th)
        return _f32(np.stack([x, y, z], axis=1) + _arr(center))
    return Part(w * area, sample)


def helix_shape(center, r, length, turns, axis=None, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    total_angle = turns * TAU
    arc_length = np.hypot(total_angle * r, length)

    def sample(k, rng):
        t = rng.random(k)
        angle = t * total_angle
        h = t * length
        x = r * np.cos(angle)
        y = h - length / 2
        z = r * np.sin(angle)
        return _f32(np.stack([x, y, z], axis=1) + _arr(center))
    return Part(w * arc_length, sample)


def gear_shape(center, outer_r, teeth, axis=None, width=10, depth=5, hub=None, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    da = TAU / teeth
    perimeter = teeth * (outer_r * da + 2 * outer_r * np.sin(da / 2))

    def sample(k, rng):
        t = rng.integers(0, teeth, k)
        base_angle = t * da
        a = rng.uniform(-da / 2, da / 2, k)
        angle = base_angle + a
        is_tooth = np.abs(a) < da * 0.35
        r = np.where(is_tooth, outer_r, outer_r - depth)
        h = rng.uniform(-width / 2, width / 2, k)
        pts = np.stack([r * np.cos(angle), h, r * np.sin(angle)], axis=1)
        if hub:
            mask = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2) < hub
            pts[mask] = np.column_stack([rng.uniform(-hub, hub, mask.sum()),
                                         rng.uniform(-width / 2, width / 2, mask.sum()),
                                         rng.uniform(-hub, hub, mask.sum())])
        return _f32(pts + _arr(center))
    return Part(w * perimeter * width, sample)


def ellipsoid(center, radii, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    r = _arr(radii)
    area_approx = 4 * np.pi * np.power(np.prod(r), 2 / 3)

    def sample(k, rng):
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sr = rng.random(k) ** (1 / 3)
        pts = v * r * sr[:, None]
        return _f32(pts + _arr(center))
    return Part(w * area_approx, sample)


def capsule(a, b, r, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    a_v, b_v = _arr(a), _arr(b)
    length = np.linalg.norm(b_v - a_v)
    area = TAU * r * length + 4 * np.pi * r * r

    def sample(k, rng):
        t = rng.random(k)
        center = a_v + (b_v - a_v) * t[:, None]
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return _f32(center + v * r)
    return Part(w * area, sample)


def blob(center, radii, density=1, keep=None, bump=None, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    r = _arr(radii)
    area_approx = 4 * np.pi * np.power(np.prod(r), 2 / 3)

    def sample(k, rng):
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = np.ones(k)
        if bump:
            u = np.arctan2(v[:, 1], v[:, 0])
            phi = np.arccos(v[:, 2])
            f = np.array([bump(ui, phi_i) for ui, phi_i in zip(u, phi)])
        pts = v * r * f[:, None]
        if keep:
            mask = np.array([keep(*p) for p in pts])
            pts = pts[mask]
        return _f32(pts + _arr(center))
    return Part(w * area_approx, sample)


def dots(pts_list, r, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    pts = _arr(pts_list)
    n_pts = len(pts)
    area = n_pts * 4 * np.pi * r * r

    def sample(k, rng):
        idx = rng.integers(0, n_pts, k)
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return _f32(pts[idx] + v * r)
    return Part(w * area, sample)


# ── skeletal helpers ────────────────────────────────────────────

def vessel(ctrl, r0, r1, density=1, per=8, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    pts = _spline(ctrl, per)
    segs = [tube_shape(pts[i], pts[i + 1], r0 + (r1 - r0) * i / (len(pts) - 1),
                       r0 + (r1 - r0) * (i + 1) / (len(pts) - 1), weight=w / (len(pts) - 1))
            for i in range(len(pts) - 1)]
    return Part(w, sample=_pick(segs))


def path_tube(pts, r, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    pts_a = _arr(pts)
    segs = [tube_shape(pts_a[i], pts_a[i + 1], r, r, weight=w / (len(pts) - 1)) for i in range(len(pts_a) - 1)]
    return Part(w, sample=_pick(segs))


def bone(a, b, r0, density=1, weight=None):
    w = density if density is not None else (weight if weight is not None else 1.0)
    a_v, b_v = _arr(a), _arr(b)
    axis_v = b_v - a_v
    length = np.linalg.norm(axis_v)
    if length < 1e-12:
        perp1 = np.array([1.0, 0.0, 0.0])
        perp2 = np.array([0.0, 1.0, 0.0])
    else:
        axis_v = axis_v / length
        if abs(axis_v[0]) < 0.9:
            perp1 = np.array([axis_v[1], -axis_v[0], 0.0])
        else:
            perp1 = np.array([0.0, axis_v[2], -axis_v[1]])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis_v, perp1)

    def sample(k, rng):
        t = rng.random(k)
        center = a_v + axis_v * (t[:, None] * length)
        r = r0 * (1.4 - 0.8 * np.abs(t - 0.5) * 2)
        angles = rng.uniform(0, TAU, k)
        pts = center + perp1 * (r[:, None] * np.cos(angles)[:, None]) + perp2 * (r[:, None] * np.sin(angles)[:, None])
        return _f32(pts)
    return Part(w, sample)


def badge(text, origin, right, up, size, density=5, weight=None):
    w = density if density is not None else (weight if weight is not None else 5)
    GLYPH = {
        "H": [[[0, 0], [0, 1.4]], [[1, 0], [1, 1.4]], [[0, 0.7], [1, 0.7]]],
        "E": [[[1, 0], [0, 0], [0, 1.4], [1, 1.4]], [[0, 0.7], [0.8, 0.7]]],
        "M": [[[0, 0], [0, 1.4], [0.5, 0.7], [1, 1.4], [1, 0]]],
        "I": [[[0.5, 0], [0.5, 1.4]], [[0.2, 0], [0.8, 0]], [[0.2, 1.4], [0.8, 1.4]]],
        "V": [[[0, 1.4], [0.5, 0], [1, 1.4]]],
        "8": [[[0, 0], [1, 0], [1, 1.4], [0, 1.4], [0, 0]], [[0, 0.7], [1, 0.7]]],
    }
    strokes = []
    for i, ch in enumerate(text):
        for stroke in GLYPH.get(ch, []):
            pts = [list(_add(_add(_arr(origin), _mul(_arr(right), (p[0] + i * 1.5) * size)),
                             _mul(_arr(up), p[1] * size))) for p in stroke]
            strokes.append(polyline(pts, 0, weight=w))
    return Part(1, sample=_pick(strokes))


# ── transform helpers ───────────────────────────────────────────

def _transform_part(part, matrix, center=None):
    mat = _arr(matrix)
    if center is not None:
        cen = _arr(center)
        orig_sample = part.sample

        def sample(k, rng):
            pts = orig_sample(k, rng)
            return _f32((pts - cen) @ mat.T + cen)
        return Part(part.weight, sample)
    else:
        orig_sample = part.sample

        def sample(k, rng):
            pts = orig_sample(k, rng)
            return _f32(pts @ mat.T)
        return Part(part.weight, sample)


def transform(parts, matrix, center=None):
    items = parts if isinstance(parts, list) else [parts]
    return [_transform_part(p, matrix, center) for p in items]


def orient(parts, direction, center=None):
    d = _arr(direction)
    d = d / np.linalg.norm(d)
    if abs(d[0]) < 0.999:
        side = np.cross(d, [1, 0, 0])
    else:
        side = np.cross(d, [0, 1, 0])
    side /= np.linalg.norm(side)
    up = np.cross(d, side)
    mat = np.eye(4)
    mat[0, :3] = d
    mat[1, :3] = up
    mat[2, :3] = side
    return transform(parts, mat[:3, :3], center)


# ── rotation matrices ───────────────────────────────────────────

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def deg(r):
    return r * np.pi / 180


# ── vec helpers ─────────────────────────────────────────────────

def _norm(v):
    vv = _arr(v)
    n = np.linalg.norm(vv)
    return vv / n if n > 1e-12 else vv


def _cross(a, b):
    return np.cross(_arr(a), _arr(b))


def _add(a, b):
    return _arr(a) + _arr(b)


def _sub(a, b):
    return _arr(a) - _arr(b)


def _mul(a, s):
    return _arr(a) * s


def linspace(a, b, n):
    return np.linspace(a, b, n)


def range_n(n):
    return list(range(n))


# ── catmull-rom spline ──────────────────────────────────────────

def _spline(pts, per=10):
    pts = _arr(pts)
    out = []
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        for k in range(per):
            t = k / per
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(pt)
    out.append(pts[-1])
    return np.array(out)


# ── sampling combinators ────────────────────────────────────────

def flatten(groups_of_parts):
    result = []
    for g in groups_of_parts:
        if isinstance(g, list):
            result.extend(g)
        elif isinstance(g, Part):
            result.append(g)
        else:
            result.extend(flatten(g))
    return result


def _pick(samplers):
    weights = np.array([s.weight for s in samplers], dtype=np.float64)
    probs = weights / weights.sum()

    def sample(k, rng):
        which = rng.choice(len(samplers), size=k, p=probs)
        out = np.zeros((k, 3), dtype=np.float32)
        counts = np.bincount(which, minlength=len(samplers))
        for i, s in enumerate(samplers):
            if counts[i] > 0:
                pts = s.sample(int(counts[i]), rng)
                idx = np.where(which == i)[0]
                n = min(len(pts), len(idx))
                out[idx[:n]] = pts[:n]
        return out
    return sample


def _pick_with_keep(samplers, keep_fn=None):
    weights = np.array([s.weight for s in samplers], dtype=np.float64)
    probs = weights / weights.sum()

    def sample(k, rng):
        which = rng.choice(len(samplers), size=k, p=probs)
        out = np.zeros((k, 3), dtype=np.float32)
        counts = np.bincount(which, minlength=len(samplers))
        for i, s in enumerate(samplers):
            if counts[i] > 0:
                pts = s.sample(int(counts[i]), rng)
                idx = np.where(which == i)[0]
                n = min(len(pts), len(idx))
                out[idx[:n]] = pts[:n]
        if keep_fn:
            mask = np.array([keep_fn(*p) for p in out])
            out = out[mask]
        return out
    return sample

# Backward compat wrappers for old API: (k, rng) -> positions
def box(k, rng, size, center=(0, 0, 0)):
    return box_filled(center, size).sample(k, rng)

def cylinder(k, rng, radius, height, center=(0, 0, 0), axis=1, caps=True):
    return cylinder_s(center, radius, height, axis=axis, cap=caps).sample(k, rng)

def segment(k, rng, a, b, jitter=0.01):
    a_v, b_v = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    t = rng.random((k, 1))
    return _f32(a_v + (b_v - a_v) * t + rng.normal(0, jitter, size=(k, 3)))

def tube(k, rng, path_fn, radius):
    return _f32(path_fn(rng.random(k)) + rng.normal(0, radius, size=(k, 3)))

old_sphere = sphere

def sphere_compat(k, rng, r=1.0, center=(0, 0, 0)):
    v = rng.normal(size=(k, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return _f32(v * r + np.asarray(center))
