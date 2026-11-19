"""Convert a binary ink image into centre-line pen strokes.

Pipeline: tidy mask -> skeletonize -> build a pixel graph (endpoints / junctions /
paths) -> prune short spurs -> merge paths through degree-2 nodes -> smooth and
resample. Each stroke keeps the local ink radius (from the distance transform) so the
original thickness profile can be reproduced if wanted.

Used both for font glyphs and for glyphs cut out of a user's handwritten page.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes, skeletonize

# 8-neighbourhood in ring order: N, NE, E, SE, S, SW, W, NW  as (d_row, d_col)
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

Pixel = tuple[int, int]


@dataclass
class RawStroke:
    """A stroke in pixel coordinates of the source image."""

    points: np.ndarray  # (N, 2) float, columns = x, y
    radius: np.ndarray  # (N,) local ink half-width in pixels
    closed: bool = False


@dataclass
class _Edge:
    a: int | None  # start node id (None for free loops)
    b: int | None  # end node id
    path: list[Pixel]


def mask_to_strokes(
    mask: np.ndarray,
    *,
    spur_length: float | None = None,
    smooth_window: int = 5,
    spacing: float = 1.0,
) -> list[RawStroke]:
    """Extract centre-line strokes from a boolean ink mask (True = ink)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []

    # Fill pin-holes so they do not create tiny skeleton loops.
    mask = remove_small_holes(mask, max_size=max(3, int(mask.sum() * 0.002)))
    dist = ndi.distance_transform_edt(mask)
    skel = skeletonize(mask)
    if not skel.any():
        return []

    if spur_length is None:
        med = float(np.median(dist[skel]))
        spur_length = 2.0 * med + 2.0

    edges = _trace(skel)
    edges = _prune(edges, spur_length)

    strokes: list[RawStroke] = []
    for e in edges:
        pts = np.array([(c, r) for r, c in e.path], dtype=np.float64)
        rad = np.array([max(dist[r, c], 0.5) for r, c in e.path], dtype=np.float64)
        closed = e.a is None and len(pts) > 3 and np.abs(pts[0] - pts[-1]).max() <= 1.5
        pts, rad = _smooth(pts, rad, smooth_window, closed)
        pts, rad = resample(pts, rad, spacing)
        strokes.append(RawStroke(pts, rad, closed))
    return strokes


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _trace(skel: np.ndarray) -> list[_Edge]:
    pix: set[Pixel] = set(zip(*np.nonzero(skel), strict=True))
    pix = {(int(r), int(c)) for r, c in pix}

    def nbrs(p: Pixel) -> list[Pixel]:
        r, c = p
        return [(r + dr, c + dc) for dr, dc in _RING if (r + dr, c + dc) in pix]

    def crossing_number(p: Pixel) -> int:
        r, c = p
        v = [(r + dr, c + dc) in pix for dr, dc in _RING]
        return sum((not v[i]) and v[(i + 1) % 8] for i in range(8))

    # Classify node pixels: endpoints (cn <= 1) and junctions (cn >= 3).
    cn = {p: crossing_number(p) for p in pix}
    junction_px = {p for p, n in cn.items() if n >= 3}
    end_px = {p for p, n in cn.items() if n <= 1}

    node_of: dict[Pixel, int] = {}
    nodes: list[list[Pixel]] = []
    # Cluster adjacent junction pixels into a single node.
    for p in junction_px:
        if p in node_of:
            continue
        nid = len(nodes)
        stack, members = [p], []
        node_of[p] = nid
        while stack:
            q = stack.pop()
            members.append(q)
            for n in nbrs(q):
                if n in junction_px and n not in node_of:
                    node_of[n] = nid
                    stack.append(n)
        nodes.append(members)
    for p in end_px:
        if p not in node_of:
            node_of[p] = len(nodes)
            nodes.append([p])

    visited: set[Pixel] = set()
    edges: list[_Edge] = []
    direct_pairs: set[frozenset[int]] = set()

    for nid, members in enumerate(nodes):
        if len(members) == 1 and not nbrs(members[0]):
            edges.append(_Edge(nid, nid, [members[0]]))  # isolated dot
            continue
        for p in members:
            for q in nbrs(p):
                if q in node_of:
                    other = node_of[q]
                    if other != nid:
                        key = frozenset((nid, other))
                        if key not in direct_pairs:
                            direct_pairs.add(key)
                            edges.append(_Edge(nid, other, [p, q]))
                    continue
                if q in visited:
                    continue
                path = [p, q]
                visited.add(q)
                prev, cur = p, q
                while True:
                    cand = [r for r in nbrs(cur) if r != prev and r not in visited]
                    ends = [
                        r
                        for r in cand
                        if r in node_of and not (node_of[r] == nid and len(path) < 4)
                    ]
                    if ends:
                        path.append(ends[0])
                        edges.append(_Edge(nid, node_of[ends[0]], path))
                        break
                    free = [r for r in cand if r not in node_of]
                    if not free:
                        edges.append(_Edge(nid, None, path))
                        break
                    if len(free) > 1:
                        far = [r for r in free if max(abs(r[0] - prev[0]), abs(r[1] - prev[1])) > 1]
                        free = far or free
                        four = [r for r in free if abs(r[0] - cur[0]) + abs(r[1] - cur[1]) == 1]
                        free = four or free
                    nxt = free[0]
                    visited.add(nxt)
                    path.append(nxt)
                    prev, cur = cur, nxt

    # Whatever is left forms closed loops without any node (e.g. the letter 'o').
    remaining = pix - visited - set(node_of)
    while remaining:
        start = remaining.pop()
        visited.add(start)
        path = [start]
        prev: Pixel | None = None
        cur = start
        while True:
            cand = [r for r in nbrs(cur) if r not in visited and r not in node_of and r != prev]
            if not cand:
                break
            four = [r for r in cand if abs(r[0] - cur[0]) + abs(r[1] - cur[1]) == 1]
            nxt = (four or cand)[0]
            visited.add(nxt)
            remaining.discard(nxt)
            path.append(nxt)
            prev, cur = cur, nxt
        if len(path) > 2 and max(abs(path[-1][0] - start[0]), abs(path[-1][1] - start[1])) <= 1:
            path.append(start)
        edges.append(_Edge(None, None, path))
    return edges


def _path_len(path: list[Pixel]) -> float:
    if len(path) < 2:
        return 0.0
    a = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


def _degrees(edges: list[_Edge]) -> Counter:
    deg: Counter = Counter()
    for e in edges:
        if e.a is not None:
            deg[e.a] += 1
        if e.b is not None:
            deg[e.b] += 1
    return deg


def _prune(edges: list[_Edge], spur_length: float) -> list[_Edge]:
    for _ in range(4):
        deg = _degrees(edges)
        kept, changed = [], False
        for e in edges:
            if e.a is not None and e.b is not None and e.a != e.b:
                da, db = deg[e.a], deg[e.b]
                spur = (da == 1 and db >= 3) or (db == 1 and da >= 3)
                if spur and _path_len(e.path) < spur_length:
                    changed = True
                    continue
            kept.append(e)
        edges = _merge_degree2(kept)
        if not changed:
            break
    return edges


def _merge_degree2(edges: list[_Edge]) -> list[_Edge]:
    """Join the two edges meeting at any node of degree 2 into one longer edge."""
    while True:
        deg = _degrees(edges)
        target = None
        for node, d in deg.items():
            if d == 2:
                inc = [i for i, e in enumerate(edges) if node in (e.a, e.b)]
                if len(inc) == 2:
                    target = (node, inc)
                    break
                if len(inc) == 1:  # self-loop through this node -> free loop
                    e = edges[inc[0]]
                    edges[inc[0]] = _Edge(None, None, e.path)
                    target = ("loop", None)
                    break
        if target is None:
            return edges
        if target[0] == "loop":
            continue
        node, (i, j) = target
        e1, e2 = edges[i], edges[j]
        p1 = e1.path if e1.b == node else e1.path[::-1]
        o1 = e1.a if e1.b == node else e1.b
        p2 = e2.path if e2.a == node else e2.path[::-1]
        o2 = e2.b if e2.a == node else e2.a
        joined = p1 + (p2[1:] if p2 and p1 and p2[0] == p1[-1] else p2)
        if o1 == o2 and o1 is not None and deg[o1] == 2:
            new = _Edge(None, None, joined)
        else:
            new = _Edge(o1, o2, joined)
        edges = [e for k, e in enumerate(edges) if k not in (i, j)] + [new]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _smooth(
    pts: np.ndarray, rad: np.ndarray, window: int, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pts)
    if n < 4 or window < 2:
        return pts, rad
    w = min(window, n if closed else max(2, n // 2))
    kernel = np.ones(w) / w
    if closed:
        pad = w
        ext = np.concatenate([pts[-pad - 1 : -1], pts, pts[1 : pad + 1]])
        sm = np.stack([np.convolve(ext[:, k], kernel, mode="same") for k in range(2)], 1)
        return sm[pad:-pad], rad
    cols = [np.convolve(np.pad(pts[:, k], w, mode="edge"), kernel, mode="same") for k in range(2)]
    sm = np.stack([c[w:-w] for c in cols], 1)
    sm[0], sm[-1] = pts[0], pts[-1]
    return sm, rad


def resample(
    pts: np.ndarray, values: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline (and per-point values) at roughly uniform arc-length spacing."""
    if len(pts) < 2 or spacing <= 0:
        return pts, values
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        return pts[:1], values[:1]
    n = max(2, int(np.ceil(total / spacing)) + 1)
    t = np.linspace(0.0, total, n)
    out = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)
    return out, np.interp(t, s, values)
