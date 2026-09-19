"""精确平面细分库（exact planar subdivision）。

输入 1..40 条带唯一 ID 的线段，坐标为绝对值 <= 10**6 的整数；内部全部使用
:class:`fractions.Fraction` 进行精确运算，不使用浮点容差，也不依赖第三方几何库。

处理端点接触、真交叉与共线部分/完全重叠：在所有必要交点处切分原线段，
几何相同的无向子边合并并保留全部来源 ID。结果与输入顺序无关。

细分后的无向图若不连通，:func:`subdivide` 返回 ``status == "disconnected"`` 的
:class:`DisconnectedNetwork`；连通时返回 :class:`PlanarSubdivision`，其中包含
排序后的顶点、无向子边、半边邻接关系与面（边界游走 + 精确有向面积）。

半边约定：

* 无向子边 ``e`` 生成两条半边 ``2*e``（a -> b）与 ``2*e+1``（b -> a），互为孪生
  （twin 为 ``h ^ 1``）；
* 每个顶点的出边按象限 + 叉积精确地逆时针（CCW）排序；
* 沿半边到达顶点后，取其反向边在 CCW 序列中的顺时针前邻（前一个）作为 next，
  因而每条游走的左侧是同一个面；
* 桥在同一游走中以相反方向出现两次，不会被删除；
* 有界面面积为正（CCW），无界面面积为负（顺时针），纯树时唯一面面积为 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Hashable, Iterable, Sequence

__all__ = [
    "Segment",
    "Vertex",
    "SubEdge",
    "Face",
    "PlanarSubdivision",
    "DisconnectedNetwork",
    "PlanarArrangementError",
    "InvalidInput",
    "ZeroLengthSegment",
    "DuplicateSegmentId",
    "subdivide",
    "check_invariants",
    "frac_str",
    "MAX_SEGMENTS",
    "COORD_LIMIT",
]

MAX_SEGMENTS = 40
COORD_LIMIT = 10**6

STATUS_OK = "ok"
STATUS_DISCONNECTED = "disconnected"

Point = tuple[Fraction, Fraction]
Vec = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------------
# 异常类型
# ---------------------------------------------------------------------------


class PlanarArrangementError(ValueError):
    """本库所有输入类错误的基类。"""


class InvalidInput(PlanarArrangementError):
    """线段数量、坐标或 ID 不合法。"""


class ZeroLengthSegment(InvalidInput):
    """线段两个端点重合。"""


class DuplicateSegmentId(InvalidInput):
    """线段 ID 不唯一。"""


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """一条输入线段：唯一 ID 与两个整数端点。"""

    id: Hashable
    p1: tuple[int, int]
    p2: tuple[int, int]


@dataclass(frozen=True)
class Vertex:
    """细分顶点，坐标为精确 Fraction，按 (x, y) 字典序编号。"""

    id: int
    x: Fraction
    y: Fraction

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class SubEdge:
    """无向子边。

    a、b 为顶点编号（按坐标排序后 a < b）；sources 为覆盖该几何边的全部源
    线段 ID，已排序。半边 ``2*id`` 方向为 a -> b，``2*id+1`` 为 b -> a。
    """

    id: int
    a: int
    b: int
    sources: tuple[Hashable, ...]


@dataclass(frozen=True)
class Face:
    """一个面（半边边界游走）。

    ``halfedges`` 为按 next 顺序排列的半边编号（循环）；``vertices`` 为对应
    起点顶点编号。桥会以两个相反半边各出现一次。``area`` 为精确有向面积：
    有界面为正，无界面为负，纯树时为 0。
    """

    id: int
    halfedges: tuple[int, ...]
    vertices: tuple[int, ...]
    area: Fraction
    unbounded: bool


@dataclass(frozen=True)
class PlanarSubdivision:
    """连通输入的细分结果。"""

    status: str
    vertices: tuple[Vertex, ...]
    edges: tuple[SubEdge, ...]
    faces: tuple[Face, ...]
    # 半边表：长度 2*E
    halfedge_origin: tuple[int, ...]
    halfedge_target: tuple[int, ...]
    halfedge_next: tuple[int, ...]

    @property
    def is_connected(self) -> bool:
        return self.status == STATUS_OK

    def twin(self, halfedge: int) -> int:
        """返回半边的反向半边编号。"""
        return halfedge ^ 1

    def edge_of_halfedge(self, halfedge: int) -> int:
        """半边所属无向子边编号。"""
        return halfedge // 2


@dataclass(frozen=True)
class DisconnectedNetwork:
    """输入线段几何上不连通时的返回值。

    components 为按段 ID 归并的连通分量（段之间端点接触、交叉、重叠均算连通），
    每个分量内 ID 已排序，分量之间也已排序。
    """

    status: str
    components: tuple[tuple[Hashable, ...], ...]


# ---------------------------------------------------------------------------
# 输入校验与规范化
# ---------------------------------------------------------------------------


def _coord(value: object) -> Fraction:
    if isinstance(value, float):
        # 刻意拒绝浮点：全部坐标必须精确，避免无意引入容差。
        raise InvalidInput(f"坐标必须是整数（或 Fraction），收到 float: {value!r}")
    if isinstance(value, int) or isinstance(value, Fraction):
        f = Fraction(value)
        if abs(f) > COORD_LIMIT:
            raise InvalidInput(f"坐标绝对值超出 {COORD_LIMIT}: {value!r}")
        return f
    raise InvalidInput(f"无法识别的坐标类型: {value!r}")


def _point(value: object) -> Point:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise InvalidInput(f"点必须是长度为 2 的序列: {value!r}")
    return (_coord(value[0]), _coord(value[1]))


def _as_segment(raw: object) -> Segment:
    if isinstance(raw, Segment):
        seg = raw
    elif isinstance(raw, tuple) and len(raw) == 3:
        sid, p1, p2 = raw
        seg = Segment(sid, tuple(p1), tuple(p2))  # type: ignore[arg-type]
    else:
        raise InvalidInput(
            "线段必须是 Segment(id, p1, p2) 或 (id, (x1,y1), (x2,y2))"
        )
    return Segment(seg.id, _point(seg.p1), _point(seg.p2))


def _normalize_all(segments: Iterable[object]) -> list[tuple[Hashable, Point, Point]]:
    if not isinstance(segments, (list, tuple, set, frozenset)):
        # 允许任意可迭代对象，但先实体化以便计数与报错。
        segments = list(segments)  # type: ignore[assignment]
    segs_raw = list(segments)  # type: ignore[arg-type]
    count = len(segs_raw)
    if count < 1 or count > MAX_SEGMENTS:
        raise InvalidInput(f"线段数量必须在 1..{MAX_SEGMENTS} 之间，收到 {count}")

    normalized: list[tuple[Hashable, Point, Point]] = []
    seen_ids: set[Hashable] = set()
    for raw in segs_raw:
        seg = _as_segment(raw)
        try:
            hash(seg.id)
        except TypeError as exc:
            raise InvalidInput(f"线段 ID 不可哈希: {seg.id!r}") from exc
        if seg.id in seen_ids:
            raise DuplicateSegmentId(f"线段 ID 重复: {seg.id!r}")
        seen_ids.add(seg.id)
        if seg.p1 == seg.p2:
            raise ZeroLengthSegment(f"线段 {seg.id!r} 长度为 0: {seg.p1}")
        normalized.append((seg.id, seg.p1, seg.p2))

    try:
        sorted(seen_ids)
    except TypeError as exc:
        raise InvalidInput("线段 ID 之间必须可相互排序") from exc
    return normalized


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------


class _DSU:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# 精确求交：把交点参数加入每条线段的切点集合
# ---------------------------------------------------------------------------


def _vec(a: Point, b: Point) -> Vec:
    return (b[0] - a[0], b[1] - a[1])


def _cross(u: Vec, v: Vec) -> Fraction:
    return u[0] * v[1] - u[1] * v[0]


def _dot(u: Vec, v: Vec) -> Fraction:
    return u[0] * v[0] + u[1] * v[1]


def _add_pair_cuts(
    si: tuple[Hashable, Point, Point],
    sj: tuple[Hashable, Point, Point],
    cuts_i: set[Fraction],
    cuts_j: set[Fraction],
) -> bool:
    """精确计算两条线段的相交，并登记切点参数。

    返回两条线段是否有任何公共点（端点接触 / 交叉 / 共线重叠）。
    参数约定：线段 i 上的点 = P + t*r，t ∈ [0,1]；j 同理用 u。
    """
    _id_i, p, q = si
    _id_j, s, t_pt = sj
    r = _vec(p, q)
    d = _vec(s, t_pt)
    sp = _vec(p, s)
    den = _cross(r, d)

    if den != 0:
        # 非平行：叉积求参数，0 分母分支已排除。
        t = _cross(sp, d) / den
        u = _cross(sp, r) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            cuts_i.add(t)
            cuts_j.add(u)
            return True
        return False

    # 平行：不共线则无交点。
    if _cross(sp, r) != 0:
        return False

    # 共线：把 j 的端点投影到 i 的参数轴上。
    r2 = _dot(r, r)
    u0 = _dot(sp, r) / r2
    u1 = _dot(_vec(p, t_pt), r) / r2
    lo = max(Fraction(0), min(u0, u1))
    hi = min(Fraction(1), max(u0, u1))
    if hi < lo:
        return False  # 共线但不搭接

    d2 = _dot(d, d)
    if lo == hi:
        # 端点相接（退化搭接为一点）。
        cuts_i.add(lo)
        touch = (p[0] + lo * r[0], p[1] + lo * r[1])
        cuts_j.add(_dot(_vec(s, touch), d) / d2)
        return True

    cuts_i.add(lo)
    cuts_i.add(hi)
    plo = (p[0] + lo * r[0], p[1] + lo * r[1])
    phi = (p[0] + hi * r[0], p[1] + hi * r[1])
    cuts_j.add(_dot(_vec(s, plo), d) / d2)
    cuts_j.add(_dot(_vec(s, phi), d) / d2)
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _angle_cmp(u: Vec, v: Vec) -> int:
    """向量逆时针（CCW，从 +x 轴起）比较器，象限优先、叉积定序，全程精确。"""
    hu = 0 if (u[1] > 0 or (u[1] == 0 and u[0] > 0)) else 1
    hv = 0 if (v[1] > 0 or (v[1] == 0 and v[0] > 0)) else 1
    if hu != hv:
        return -1 if hu < hv else 1
    c = _cross(u, v)
    if c > 0:
        return -1  # u 在 v 的顺时针侧，CCW 序列中 u 在前
    if c < 0:
        return 1
    return 0


def _build_connected(
    geo_sources: dict[tuple[Point, Point], set[Hashable]],
) -> PlanarSubdivision:
    # 1) 顶点按精确坐标排序编号（无向子边已在 subdivide 中完成几何合并）。
    all_points = sorted({pt for key in geo_sources for pt in key})
    vid = {pt: i for i, pt in enumerate(all_points)}

    # 2) 无向子边按端点坐标排序编号，来源 ID 排序。
    edge_keys = sorted(geo_sources, key=lambda k: (vid[k[0]], vid[k[1]]))
    edges: tuple[SubEdge, ...] = tuple(
        SubEdge(
            i,
            vid[a],
            vid[b],
            tuple(sorted(geo_sources[(a, b)])),
        )
        for i, (a, b) in enumerate(edge_keys)
    )

    e_count = len(edges)
    h_count = 2 * e_count
    origin = [0] * h_count
    target = [0] * h_count
    outgoing: list[list[int]] = [[] for _ in all_points]
    for e in edges:
        h_ab, h_ba = 2 * e.id, 2 * e.id + 1
        origin[h_ab], target[h_ab] = e.a, e.b
        origin[h_ba], target[h_ba] = e.b, e.a
        outgoing[e.a].append(h_ab)
        outgoing[e.b].append(h_ba)

    # 4) 各顶点出边 CCW 精确排序。必须先完成全部顶点的排序，再计算 next，
    #    否则会读到尚未排序的邻接表。
    def angle_sort_key(v: int):
        pv = all_points[v]

        def cmp(h1: int, h2: int) -> int:
            w1 = _vec(pv, all_points[target[h1]])
            w2 = _vec(pv, all_points[target[h2]])
            c = _angle_cmp(w1, w2)
            if c != 0:
                return c
            # 同射线（正常切分后不应出现）时以半边编号保证全序确定性。
            return -1 if h1 < h2 else (1 if h1 > h2 else 0)

        return cmp_to_key(cmp)

    for v, hs in enumerate(outgoing):
        hs.sort(key=angle_sort_key(v))

    # next(h) = 到达顶点后反向边在 CCW 序列中的顺时针前邻（前一个）。
    nxt = [0] * h_count
    for hs in outgoing:
        for h in hs:
            end = target[h]
            twin = h ^ 1
            seq = outgoing[end]
            j = seq_index(seq, twin)
            nxt[h] = seq[(j - 1) % len(seq)]

    # 5) next 是一个置换，其循环即面游走；桥在同一循环中正反各出现一次。
    visited = [False] * h_count
    raw_faces: list[tuple[tuple[int, ...], Fraction]] = []
    for start in range(h_count):
        if visited[start]:
            continue
        walk: list[int] = []
        h = start
        while not visited[h]:
            visited[h] = True
            walk.append(h)
            h = nxt[h]
        if h != start:
            # 理论上不可能：next 为每个顶点的出边定义了一一对应，必为置换。
            raise RuntimeError("内部错误：半边 next 不是单一循环置换")
        area = _signed_area(walk, origin, all_points)
        raw_faces.append((tuple(walk), area))

    if not all(visited):
        raise RuntimeError("内部错误：存在未被任何游走覆盖的半边")

    # 6) 面按规范化边界（最小循环旋转）排序后分配 ID。
    def canonical(walk: tuple[int, ...]) -> tuple[int, ...]:
        best = walk
        for k in range(1, len(walk)):
            rot = walk[k:] + walk[:k]
            if rot < best:
                best = rot
        return best

    unbounded_seen = 0
    face_objs: list[Face] = []
    enriched = [(canonical(w), w, area) for w, area in raw_faces]
    enriched.sort(key=lambda item: item[0])
    for fid, (_canon, walk, area) in enumerate(enriched):
        is_unbounded = area <= 0
        if is_unbounded:
            unbounded_seen += 1
        face_objs.append(
            Face(
                id=fid,
                halfedges=walk,
                vertices=tuple(origin[h] for h in walk),
                area=area,
                unbounded=is_unbounded,
            )
        )
    if unbounded_seen != 1:
        raise RuntimeError(f"内部错误：无界面数量应为 1，实际为 {unbounded_seen}")

    vertices = tuple(Vertex(i, pt[0], pt[1]) for i, pt in enumerate(all_points))
    return PlanarSubdivision(
        status=STATUS_OK,
        vertices=vertices,
        edges=edges,
        faces=tuple(face_objs),
        halfedge_origin=tuple(origin),
        halfedge_target=tuple(target),
        halfedge_next=tuple(nxt),
    )


def seq_index(seq: list[int], value: int) -> int:
    # 顶点度数很小（<= 2n），直接线性查找；单列函数以保持上方流程清晰。
    for i, v in enumerate(seq):
        if v == value:
            return i
    raise RuntimeError("内部错误：在顶点出边表中找不到反向半边")


def _signed_area(
    walk: Sequence[int],
    origin: Sequence[int],
    points: Sequence[Point],
) -> Fraction:
    total = Fraction(0)
    for h in walk:
        p = points[origin[h]]
        q = points[origin[h ^ 1]]
        total += p[0] * q[1] - p[1] * q[0]
    return total / 2


def subdivide(
    segments: Iterable[object],
) -> PlanarSubdivision | DisconnectedNetwork:
    """对输入线段网络做精确平面细分。

    参数可为 :class:`Segment` 或三元组 ``(id, (x1, y1), (x2, y2))`` 的可迭代对象。

    成功返回 :class:`PlanarSubdivision`；几何上不连通时返回
    :class:`DisconnectedNetwork`（``status == "disconnected"``）；零长、重复 ID、
    数量越界等非法输入抛出 :class:`InvalidInput` 子类。
    """
    segs = _normalize_all(segments)
    n = len(segs)
    cuts: list[set[Fraction]] = [{Fraction(0), Fraction(1)} for _ in range(n)]
    dsu = _DSU(n)

    for i in range(n):
        for j in range(i + 1, n):
            if _add_pair_cuts(segs[i], segs[j], cuts[i], cuts[j]):
                dsu.union(i, j)

    # 连通性：段自身连通，两段共点（接触/交叉/重叠）即在同一分量；
    # 这与细分后的无向图连通性等价。分量以段 ID 形式上报。
    roots = {dsu.find(i) for i in range(n)}
    if len(roots) > 1:
        groups: dict[int, list[Hashable]] = {}
        for i, (sid, _p, _q) in enumerate(segs):
            groups.setdefault(dsu.find(i), []).append(sid)
        components = tuple(
            tuple(sorted(g))
            for g in sorted(groups.values(), key=lambda g: sorted(g)[0])
        )
        return DisconnectedNetwork(status=STATUS_DISCONNECTED, components=components)

    # 几何切分：在全部切点处切开，几何相同的无向子边合并来源。
    geo_sources: dict[tuple[Point, Point], set[Hashable]] = {}
    for (sid, p, q), ts in zip(segs, cuts):
        r = _vec(p, q)
        pts = [(p[0] + t * r[0], p[1] + t * r[1]) for t in sorted(ts)]
        for a, b in zip(pts, pts[1:]):
            key = (a, b) if a < b else (b, a)
            geo_sources.setdefault(key, set()).add(sid)

    return _build_connected(geo_sources)


# ---------------------------------------------------------------------------
# 结果自检
# ---------------------------------------------------------------------------


def _on_segment(p: Point, seg: tuple[Hashable, Point, Point]) -> bool:
    _sid, a, b = seg
    r = _vec(a, b)
    v = _vec(a, p)
    if _cross(v, r) != 0:
        return False
    t = _dot(v, r) / _dot(r, r)
    return 0 <= t <= 1


def _seg_parameter(p: Point, seg: tuple[Hashable, Point, Point]) -> Fraction:
    _sid, a, b = seg
    r = _vec(a, b)
    return _dot(_vec(a, p), r) / _dot(r, r)


def _pair_intersection_points(
    s1: tuple[Hashable, Point, Point],
    s2: tuple[Hashable, Point, Point],
) -> set[Point]:
    """独立重算两条线段的全部公共点（共线时给重叠区间端点）。"""
    _i, p, q = s1
    _j, s, t_pt = s2
    r, d = _vec(p, q), _vec(s, t_pt)
    sp = _vec(p, s)
    den = _cross(r, d)
    if den != 0:
        t = _cross(sp, d) / den
        u = _cross(sp, r) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return {(p[0] + t * r[0], p[1] + t * r[1])}
        return set()
    if _cross(sp, r) != 0:
        return set()
    r2 = _dot(r, r)
    u0 = _dot(sp, r) / r2
    u1 = _dot(_vec(p, t_pt), r) / r2
    lo = max(Fraction(0), min(u0, u1))
    hi = min(Fraction(1), max(u0, u1))
    if hi < lo:
        return set()
    return {
        (p[0] + lo * r[0], p[1] + lo * r[1]),
        (p[0] + hi * r[0], p[1] + hi * r[1]),
    }


def check_invariants(
    result: PlanarSubdivision,
    segments: Iterable[object],
) -> list[str]:
    """对细分结果做一组独立一致性检查，返回问题描述列表（空列表表示全部通过）。

    检查内容：

    1. 半边表结构：twin、next 合法且构成置换，每条半边恰属于一个面游走；
    2. 欧拉公式 V - E + F = 2（F 含唯一无界面）；
    3. 面积符号与重算值：有界面为正、无界面为负（纯树为 0），各面面积之和为 0；
    4. 子边来源覆盖：每条子边中点恰好落在其 sources 的线段上，且每条源线段
       被所属子边按参数区间无缝铺成 [0, 1]；
    5. 精确交点：任两输入线段的全部公共点（含共线重叠端点）都是细分顶点。
    """
    problems: list[str] = []
    if not isinstance(result, PlanarSubdivision):
        problems.append("结果不是连通的 PlanarSubdivision")
        return problems

    segs = _normalize_all(segments)
    V, E, Fh = len(result.vertices), len(result.edges), len(result.faces)
    H = 2 * E

    # --- 1. 半边结构 ---
    if len(result.halfedge_next) != H:
        problems.append(f"半边表长度 {len(result.halfedge_next)} != 2E({H})")
    used_as_next = [0] * H
    face_cover = [0] * H
    for face in result.faces:
        for h in face.halfedges:
            if not 0 <= h < H:
                problems.append(f"面 {face.id} 含非法半边 {h}")
                continue
            face_cover[h] += 1
    for h in range(H):
        if face_cover[h] != 1:
            problems.append(f"半边 {h} 属于 {face_cover[h]} 个游走（应为 1）")
        nh = result.halfedge_next[h] if h < len(result.halfedge_next) else -1
        if not 0 <= nh < H:
            problems.append(f"半边 {h} 的 next 越界: {nh}")
        else:
            used_as_next[nh] += 1
            if result.halfedge_origin[nh] != result.halfedge_target[h]:
                problems.append(f"半边 {h} 的 next 起点与该边终点不一致")
    for h, c in enumerate(used_as_next):
        if c != 1:
            problems.append(f"半边 {h} 被 next 引用 {c} 次（应为 1）")
    for e in result.edges:
        h0, h1 = 2 * e.id, 2 * e.id + 1
        if (result.halfedge_origin[h0], result.halfedge_target[h0]) != (e.a, e.b):
            problems.append(f"边 {e.id} 的正向半边端点不匹配")
        if (result.halfedge_origin[h1], result.halfedge_target[h1]) != (e.b, e.a):
            problems.append(f"边 {e.id} 的反向半边端点不匹配")

    # --- 2. 欧拉公式 ---
    if V - E + Fh != 2:
        problems.append(f"欧拉公式不成立: V-E+F = {V}-{E}+{Fh} = {V - E + Fh} != 2")

    # 图连通性复查
    seen = {0} if V else set()
    adj: dict[int, set[int]] = {v: set() for v in range(V)}
    for e in result.edges:
        adj[e.a].add(e.b)
        adj[e.b].add(e.a)
    stack = [0] if V else []
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    if len(seen) != V:
        problems.append("细分图不连通")

    # --- 3. 面积 ---
    unbounded = [f for f in result.faces if f.unbounded]
    if len(unbounded) != 1:
        problems.append(f"无界面数量 {len(unbounded)} != 1")
    for face in result.faces:
        recomputed = Fraction(0)
        for h in face.halfedges:
            p = result.vertices[result.halfedge_origin[h]].point
            q = result.vertices[result.halfedge_target[h]].point
            recomputed += p[0] * q[1] - p[1] * q[0]
        recomputed /= 2
        if recomputed != face.area:
            problems.append(f"面 {face.id} 面积 {face.area} 与重算值 {recomputed} 不符")
        if not face.unbounded and face.area <= 0:
            problems.append(f"有界面 {face.id} 面积非正: {face.area}")
    if unbounded:
        u = unbounded[0]
        is_tree = E == V - 1
        if is_tree:
            if u.area != 0:
                problems.append(f"纯树无界面面积应为 0，实际 {u.area}")
        elif u.area >= 0:
            problems.append(f"无界面面积应为负，实际 {u.area}")
    if sum((f.area for f in result.faces), Fraction(0)) != 0:
        problems.append("所有面面积之和不为 0")

    # --- 4. 来源覆盖 ---
    seg_by_id = {sid: (sid, p, q) for sid, p, q in segs}
    point_set = {v.point for v in result.vertices}
    for e in result.edges:
        a = result.vertices[e.a].point
        b = result.vertices[e.b].point
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        covering = tuple(
            sorted(sid for sid, p, q in segs if _on_segment(mid, (sid, p, q)))
        )
        if tuple(e.sources) != covering:
            problems.append(
                f"子边 {e.id} 来源 {e.sources} 与中点实际覆盖 {covering} 不一致"
            )

    intervals: dict[object, list[tuple[Fraction, Fraction]]] = {sid: [] for sid, _, _ in segs}
    for e in result.edges:
        a = result.vertices[e.a].point
        b = result.vertices[e.b].point
        for sid in e.sources:
            seg = seg_by_id[sid]
            intervals[sid].append(
                tuple(sorted((_seg_parameter(a, seg), _seg_parameter(b, seg))))
            )
    for sid, ivs in intervals.items():
        ivs.sort()
        merged: list[list[Fraction]] = []
        for lo, hi in ivs:
            if merged and lo <= merged[-1][1]:
                if hi > merged[-1][1]:
                    merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        expected = [[Fraction(0), Fraction(1)]]
        if merged != expected:
            problems.append(f"源线段 {sid!r} 未被子边无缝铺满: {merged}")

    # --- 5. 精确交点全部为顶点 ---
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            for pt in _pair_intersection_points(segs[i], segs[j]):
                if pt not in point_set:
                    problems.append(
                        f"线段 {segs[i][0]!r} 与 {segs[j][0]!r} 的交点 {pt} 不是细分顶点"
                    )
    for _sid, p, q in segs:
        if p not in point_set or q not in point_set:
            problems.append(f"线段 {_sid!r} 端点不是细分顶点")

    return problems


# ---------------------------------------------------------------------------
# 展示辅助
# ---------------------------------------------------------------------------


def frac_str(value: Fraction) -> str:
    """把 Fraction 格式化为紧凑字符串，如 3、-1/2、13/2。"""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def point_str(p: Point | tuple[object, object]) -> str:
    return f"({frac_str(p[0])}, {frac_str(p[1])})"  # type: ignore[arg-type]
