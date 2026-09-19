"""精确平面细分库：连通线段网络的交点、拆分边与面。

仅使用 Python 标准库；所有几何计算基于 fractions.Fraction，
不使用浮点容差，不依赖第三方几何库。

接口
----
build_arrangement(segments) -> Arrangement
    segments: [(seg_id, (x1, y1), (x2, y2)), ...]
    - 1..40 条线段，seg_id 唯一且可排序
    - 坐标为整数，绝对值 <= 10**6
    - 零长线段、ID 重复、数量/坐标越界：抛出 ArrangementError
    - 细分后的无向图不连通：返回 status == "disconnected" 的结果

Arrangement 字段
    status        "ok" 或 "disconnected"
    vertices      排序后的顶点 [(Fraction, Fraction), ...]
    edges         排序后的无向子边 [Edge(a, b, sources)]，a < b，sources 排序
    faces         按规范化边界排序分配 id 的 [Face(id, boundary, area, is_outer)]
    intersections 排序后的精确交点（含端点接触与重叠端点）

约定
    - 面边界为左侧面半边游走；桥在同一游走中出现两次，不删除
    - 有向面积：有界面为正，无界面为负，纯树的唯一面面积为 0
    - 自检：每条半边恰属一个游走、V-E+F==2、每条子边源覆盖、交点均为顶点
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

MAX_COORD = 10**6
MIN_SEGMENTS = 1
MAX_SEGMENTS = 40


class ArrangementError(ValueError):
    """输入非法（零长线段、ID 重复、坐标越界、数量越界等）。"""


@dataclass(frozen=True)
class Edge:
    """无向子边；a < b 为精确坐标，sources 为排序后的源线段 ID 元组。"""

    a: tuple
    b: tuple
    sources: tuple


@dataclass(frozen=True)
class Face:
    """面；boundary 为规范化后的顶点坐标环（桥边顶点会重复出现）。"""

    id: int
    boundary: tuple
    area: Fraction
    is_outer: bool


@dataclass
class Arrangement:
    status: str
    vertices: list
    edges: list
    faces: list
    intersections: list


# ---------------------------------------------------------------- 基础精确几何


def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1]


def _on_segment(q, a, b):
    """q 是否在线段 ab 上（精确）。"""
    if _cross(_sub(q, a), _sub(b, a)) != 0:
        return False
    return (
        min(a[0], b[0]) <= q[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= q[1] <= max(a[1], b[1])
    )


def _param(p, p1, d):
    """点 p 在直线 p1 + t*d 上的参数 t（精确）。"""
    return _dot(_sub(p, p1), d) / _dot(d, d)


def _pair_intersection_points(p1, p2, p3, p4):
    """两线段的精确接触点集：真交叉点、端点接触点、共线重叠的端点。"""
    d1 = _sub(p2, p1)
    d2 = _sub(p4, p3)
    denom = _cross(d1, d2)
    out = set()
    if denom != 0:
        r = _sub(p3, p1)
        t = _cross(r, d2) / denom
        s = _cross(r, d1) / denom
        if 0 <= t <= 1 and 0 <= s <= 1:
            out.add((p1[0] + t * d1[0], p1[1] + t * d1[1]))
    else:
        if _cross(_sub(p3, p1), d1) != 0:
            return out  # 平行不共线
        for q in (p3, p4):
            if _on_segment(q, p1, p2):
                out.add(q)
        for q in (p1, p2):
            if _on_segment(q, p3, p4):
                out.add(q)
    return out


def _dir_cmp(u, v):
    """方向向量逆时针排序比较：先按上下半平面（象限），再按叉积。"""
    hu = 0 if (u[1] > 0 or (u[1] == 0 and u[0] > 0)) else 1
    hv = 0 if (v[1] > 0 or (v[1] == 0 and v[0] > 0)) else 1
    if hu != hv:
        return -1 if hu < hv else 1
    c = _cross(u, v)
    if c != 0:
        return -1 if c > 0 else 1
    return 0


# ---------------------------------------------------------------- 输入规范化


def _normalize_input(segments):
    segs = list(segments)
    if not (MIN_SEGMENTS <= len(segs) <= MAX_SEGMENTS):
        raise ArrangementError(
            f"线段数量须在 {MIN_SEGMENTS}..{MAX_SEGMENTS} 之间，实际 {len(segs)}"
        )
    ids = [s[0] for s in segs]
    if len(set(ids)) != len(ids):
        raise ArrangementError("线段 ID 必须唯一")
    out = []
    for sid, p, q in segs:
        pts = []
        for pt in (p, q):
            x, y = pt
            if not isinstance(x, int) or not isinstance(y, int):
                raise ArrangementError(f"坐标必须为整数: {pt!r}")
            if abs(x) > MAX_COORD or abs(y) > MAX_COORD:
                raise ArrangementError(f"坐标绝对值超过 {MAX_COORD}: {pt!r}")
            pts.append((Fraction(x), Fraction(y)))
        if pts[0] == pts[1]:
            raise ArrangementError(f"零长线段被拒绝: id={sid!r}")
        out.append((sid, pts[0], pts[1]))
    return out


# ---------------------------------------------------------------- 主构建


def build_arrangement(segments):
    """构建精确平面细分；非法输入抛 ArrangementError，断开返回 disconnected。"""
    segs = _normalize_input(segments)

    # 1) 逐对求交，收集每条线段的分裂点（O(n^2)，n <= 40）
    splits = [{p, q} for _, p, q in segs]
    intersections = set()
    for i in range(len(segs)):
        _, p1, p2 = segs[i]
        for j in range(i + 1, len(segs)):
            _, p3, p4 = segs[j]
            pts = _pair_intersection_points(p1, p2, p3, p4)
            if pts:
                splits[i] |= pts
                splits[j] |= pts
                intersections |= pts

    # 2) 沿每条线段按参数排序分裂点，生成原子子边并按几何合并、汇集源 ID
    edge_sources = {}
    for (sid, p1, p2), pts in zip(segs, splits):
        d = _sub(p2, p1)
        ordered = sorted(pts, key=lambda q: _param(q, p1, d))
        for a, b in zip(ordered, ordered[1:]):
            if a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            edge_sources.setdefault(key, set()).add(sid)

    edges = [
        Edge(a, b, tuple(sorted(srcs)))
        for (a, b), srcs in sorted(edge_sources.items())
    ]
    vertices = sorted({p for e in edges for p in (e.a, e.b)})
    vset = set(vertices)

    # 3) 无向图连通性
    adj = {v: [] for v in vertices}
    for e in edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
    seen = {vertices[0]}
    stack = [vertices[0]]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    if seen != vset:
        return Arrangement(
            status="disconnected",
            vertices=vertices,
            edges=edges,
            faces=[],
            intersections=sorted(intersections),
        )

    # 4) 每个顶点的出边按象限/叉积精确逆时针排序
    for v in vertices:
        adj[v].sort(key=cmp_to_key(lambda x, y: _dir_cmp(_sub(x, v), _sub(y, v))))
    index = {v: {w: i for i, w in enumerate(adj[v])} for v in vertices}

    # 5) 半边面遍历：到达 v 后取反向边的顺时针前邻作为 next
    visited = set()
    walks = []
    for u in vertices:
        for v in adj[u]:
            if (u, v) in visited:
                continue
            walk = []
            cur = (u, v)
            while cur not in visited:
                visited.add(cur)
                walk.append(cur[0])
                a, b = cur
                nxt = adj[b][(index[b][a] - 1) % len(adj[b])]
                cur = (b, nxt)
            walks.append(walk)

    # 6) 精确有向面积与规范化边界，按边界排序分配面 ID
    def signed_area(walk):
        s = Fraction(0)
        for i in range(len(walk)):
            x1, y1 = walk[i]
            x2, y2 = walk[(i + 1) % len(walk)]
            s += x1 * y2 - x2 * y1
        return s / 2

    def normalize(walk):
        i = min(range(len(walk)), key=lambda k: walk[k])
        return tuple(walk[k % len(walk)] for k in range(i, i + len(walk)))

    raw = [(normalize(w), signed_area(w)) for w in walks]
    raw.sort(key=lambda t: t[0])
    faces = [
        Face(id=k, boundary=b, area=a, is_outer=(a <= 0))
        for k, (b, a) in enumerate(raw)
    ]

    arr = Arrangement(
        status="ok",
        vertices=vertices,
        edges=edges,
        faces=faces,
        intersections=sorted(intersections),
    )
    _verify(arr, segs)
    return arr


# ---------------------------------------------------------------- 自检


def verify(arrangement, segments):
    """对结果做拓扑与源覆盖自检；失败抛 RuntimeError，通过返回 True。"""
    segs = _normalize_input(segments)
    return _verify(arrangement, segs)


def _verify(arr, segs):
    if arr.status != "ok":
        raise RuntimeError("仅能对连通结果做完整自检")

    # 每条有向半边恰好属于一个游走
    directed = set()
    for f in arr.faces:
        b = f.boundary
        for i in range(len(b)):
            e = (b[i], b[(i + 1) % len(b)])
            if e in directed:
                raise RuntimeError(f"半边 {e} 出现在多个游走中")
            directed.add(e)
    if len(directed) != 2 * len(arr.edges):
        raise RuntimeError("半边总数不等于 2E")

    # 欧拉公式（F 含无界面）
    if len(arr.vertices) - len(arr.edges) + len(arr.faces) != 2:
        raise RuntimeError("V - E + F != 2")

    # 每条源线段的子边覆盖：恰好无缝平铺 [0, 1]
    for sid, p1, p2 in segs:
        d = _sub(p2, p1)
        iv = []
        for e in arr.edges:
            if sid in e.sources:
                ta = _param(e.a, p1, d)
                tb = _param(e.b, p1, d)
                iv.append((min(ta, tb), max(ta, tb)))
        iv.sort()
        if not iv or iv[0][0] != 0 or iv[-1][1] != 1:
            raise RuntimeError(f"源线段 {sid!r} 的子边覆盖不完整")
        for (_, r1), (l2, _) in zip(iv, iv[1:]):
            if r1 != l2:
                raise RuntimeError(f"源线段 {sid!r} 的子边覆盖有缝隙或重叠")

    # 所有精确交点都是细分顶点
    vset = set(arr.vertices)
    for p in arr.intersections:
        if p not in vset:
            raise RuntimeError(f"交点 {p} 不是顶点")
    return True
