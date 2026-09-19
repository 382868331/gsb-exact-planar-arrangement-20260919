"""精确平面细分库的单元测试。

覆盖 TASK.md 点名的全部情形：X 交叉、T 接触、共线重叠、重复几何不同 ID、
桥与悬挂边、纯树唯一无界面、断开输入拒绝，以及分数交点、面积符号、
顺序无关、输入校验和固定种子的随机小样本验证。

运行：python -m unittest discover -s tests -v
"""

import io
import random
import sys
import unittest
from fractions import Fraction

# 允许从仓库根目录直接运行本测试文件。
sys.path.insert(0, ".")

from planar_arrangement import (  # noqa: E402
    COORD_LIMIT,
    DisconnectedNetwork,
    DuplicateSegmentId,
    InvalidInput,
    PlanarSubdivision,
    Segment,
    ZeroLengthSegment,
    _DSU,
    _add_pair_cuts,
    check_invariants,
    frac_str,
    subdivide,
)


def _square(prefix, x0, y0, s=1):
    """以 (x0,y0) 为左下角的方框四条边。"""
    return [
        (f"{prefix}b", (x0, y0), (x0 + s, y0)),
        (f"{prefix}r", (x0 + s, y0), (x0 + s, y0 + s)),
        (f"{prefix}t", (x0 + s, y0 + s), (x0, y0 + s)),
        (f"{prefix}l", (x0, y0 + s), (x0, y0)),
    ]


class TestBasicPolygons(unittest.TestCase):
    def test_single_segment_is_pure_tree(self):
        r = subdivide([("a", (0, 0), (1, 0))])
        self.assertIsInstance(r, PlanarSubdivision)
        self.assertEqual(len(r.vertices), 2)
        self.assertEqual(len(r.edges), 1)
        self.assertEqual(len(r.faces), 1)
        face = r.faces[0]
        self.assertTrue(face.unbounded)
        self.assertEqual(face.area, 0)
        self.assertEqual(len(face.halfedges), 2)  # 唯一边正反各走一次
        self.assertEqual(face.halfedges[0] ^ 1, face.halfedges[1])
        self.assertEqual(2 - 1 + 1, 2)
        self.assertEqual(check_invariants(r, [("a", (0, 0), (1, 0))]), [])

    def test_triangle_area_and_orientation(self):
        segs = [
            ("a", (0, 0), (2, 0)),
            ("b", (2, 0), (0, 1)),
            ("c", (0, 1), (0, 0)),
        ]
        r = subdivide(segs)
        self.assertEqual(len(r.vertices), 3)
        self.assertEqual(len(r.edges), 3)
        self.assertEqual(len(r.faces), 2)
        bounded = [f for f in r.faces if not f.unbounded]
        unbounded = [f for f in r.faces if f.unbounded]
        self.assertEqual(len(bounded), 1)
        self.assertEqual(len(unbounded), 1)
        self.assertEqual(bounded[0].area, Fraction(1))  # 底2 高1 => 面积 1
        self.assertEqual(unbounded[0].area, Fraction(-1))
        self.assertEqual(check_invariants(r, segs), [])

    def test_unit_square_area_one(self):
        segs = _square("s", 0, 0)
        r = subdivide(segs)
        areas = sorted(f.area for f in r.faces)
        self.assertEqual(areas, [Fraction(-1), Fraction(1)])
        self.assertEqual(check_invariants(r, segs), [])

    def test_vertices_sorted_by_exact_coordinates(self):
        segs = _square("s", 0, 0)
        r = subdivide(segs)
        coords = [(v.x, v.y) for v in r.vertices]
        self.assertEqual(coords, sorted(coords))
        self.assertEqual(
            coords,
            [(Fraction(0), Fraction(0)), (Fraction(0), Fraction(1)),
             (Fraction(1), Fraction(0)), (Fraction(1), Fraction(1))],
        )

    def test_edges_sorted_and_oriented(self):
        segs = _square("s", 0, 0)
        r = subdivide(segs)
        # 无向子边端点按坐标排序 a < b，边按 (a,b) 排序。
        for e in r.edges:
            self.assertLess(e.a, e.b)
        pairs = [(e.a, e.b) for e in r.edges]
        self.assertEqual(pairs, sorted(pairs))
        self.assertEqual([e.id for e in r.edges], list(range(len(r.edges))))


class TestCrossAndTouches(unittest.TestCase):
    def test_x_crossing_fractional_intersection(self):
        segs = [("a", (0, 0), (1, 1)), ("b", (0, 1), (1, 0))]
        r = subdivide(segs)
        # 交点 (1/2, 1/2) 必须是精确 Fraction 顶点
        coords = [(v.x, v.y) for v in r.vertices]
        self.assertIn((Fraction(1, 2), Fraction(1, 2)), coords)
        self.assertEqual(len(r.vertices), 5)  # 4 端点 + 1 交点
        self.assertEqual(len(r.edges), 4)
        # 两条直线交叉成树：F = E - V + 2 = 1，唯一无界面面积 0
        self.assertEqual(len(r.faces), 1)
        self.assertEqual(r.faces[0].area, 0)
        self.assertEqual(check_invariants(r, segs), [])

    def test_x_crossing_inside_grid_creates_four_faces(self):
        # 方框内加两条对角线：4 个有界面
        segs = _square("s", 0, 0, s=2) + [
            ("d1", (0, 0), (2, 2)),
            ("d2", (0, 2), (2, 0)),
        ]
        r = subdivide(segs)
        # V = 4 角 + 1 中心 = 5；E = 4 边 + 4 个对角半截 = 8；F = 5
        self.assertEqual((len(r.vertices), len(r.edges), len(r.faces)), (5, 8, 5))
        bounded = [f for f in r.faces if not f.unbounded]
        self.assertEqual(len(bounded), 4)
        for f in bounded:
            self.assertEqual(f.area, Fraction(1))  # 每个直角三角形面积 1
        self.assertEqual(check_invariants(r, segs), [])

    def test_t_contact(self):
        segs = [("a", (0, 0), (2, 0)), ("b", (1, 0), (1, 2))]
        r = subdivide(segs)
        coords = {(v.x, v.y) for v in r.vertices}
        self.assertEqual(
            coords,
            {(0, 0), (Fraction(1), 0), (2, 0), (Fraction(1), 2)},
        )
        self.assertEqual(len(r.edges), 3)
        self.assertEqual(len(r.faces), 1)  # T 形是树
        self.assertEqual(r.faces[0].area, 0)
        self.assertEqual(check_invariants(r, segs), [])

    def test_endpoint_to_endpoint_contact_is_connected(self):
        segs = [("a", (0, 0), (1, 0)), ("b", (1, 0), (1, 1))]
        r = subdivide(segs)
        self.assertEqual(r.status, "ok")
        self.assertEqual(len(r.vertices), 3)
        self.assertEqual(check_invariants(r, segs), [])

    def test_two_squares_sharing_one_vertex(self):
        # 8 字形：两方框只共享一个顶点，图连通；两个有界面
        left = _square("l", 0, 0)
        right = _square("r", 1, 0)
        r = subdivide(left + right)
        self.assertEqual(r.status, "ok")
        bounded = [f for f in r.faces if not f.unbounded]
        self.assertEqual(len(bounded), 2)
        self.assertEqual(sum(f.area for f in bounded), Fraction(2))
        self.assertEqual(check_invariants(r, left + right), [])

    def test_fractional_intersection_coordinates(self):
        # 交点 (3/2, 1)：2x - 2 = ... 用 (0,0)-(3,2) 与 (0,2)-(3,0)
        segs = [("a", (0, 0), (3, 2)), ("b", (0, 2), (3, 0))]
        r = subdivide(segs)
        coords = {(v.x, v.y) for v in r.vertices}
        self.assertIn((Fraction(3, 2), Fraction(1)), coords)
        self.assertEqual(check_invariants(r, segs), [])


class TestCollinear(unittest.TestCase):
    def test_partial_overlap_sources_merged(self):
        segs = [("a", (0, 0), (4, 0)), ("b", (1, 0), (3, 0))]
        r = subdivide(segs)
        # 切点 0,1,3,4 => 3 条无向子边
        by_endpoints = {}
        for e in r.edges:
            pa_ = r.vertices[e.a].point
            pb_ = r.vertices[e.b].point
            by_endpoints[(pa_, pb_)] = tuple(e.sources)
        self.assertEqual(by_endpoints[((0, 0), (1, 0))], ("a",))
        self.assertEqual(by_endpoints[((1, 0), (3, 0))], ("a", "b"))
        self.assertEqual(by_endpoints[((3, 0), (4, 0))], ("a",))
        self.assertEqual(len(r.faces), 1)  # 共线重叠不产生面
        self.assertEqual(check_invariants(r, segs), [])

    def test_full_overlap_duplicate_geometry_distinct_ids(self):
        segs = [("a", (0, 0), (1, 1)), ("b", (0, 0), (1, 1))]
        r = subdivide(segs)
        self.assertEqual(len(r.edges), 1)
        self.assertEqual(r.edges[0].sources, ("a", "b"))
        self.assertEqual(check_invariants(r, segs), [])

    def test_three_way_overlap(self):
        segs = [("a", (0, 0), (6, 0)), ("b", (1, 0), (5, 0)), ("c", (2, 0), (4, 0))]
        r = subdivide(segs)
        # 按几何坐标（x 递增）排列来源集合
        ordered = [
            tuple(e.sources)
            for e in sorted(r.edges, key=lambda e: r.vertices[e.a].x)
        ]
        self.assertEqual(
            ordered,
            [("a",), ("a", "b"), ("a", "b", "c"), ("a", "b"), ("a",)],
        )
        self.assertEqual(check_invariants(r, segs), [])

    def test_collinear_endpoint_touch(self):
        segs = [("a", (0, 0), (1, 0)), ("b", (1, 0), (2, 0))]
        r = subdivide(segs)
        self.assertEqual(len(r.vertices), 3)
        self.assertEqual(len(r.edges), 2)
        self.assertEqual(check_invariants(r, segs), [])

    def test_overlap_plus_cross_through(self):
        segs = [("a", (0, 0), (4, 0)), ("b", (1, 0), (3, 0)),
                ("c", (2, -2), (2, 2))]
        r = subdivide(segs)
        # 水平线切点 0,1,2,3,4 => 4 段；竖直线切点 -2,0,2 => 2 段，共 6
        self.assertEqual(len(r.edges), 6)
        center = next(
            e for e in r.edges
            if (r.vertices[e.a].point == (1, 0) or r.vertices[e.b].point == (1, 0))
            and (r.vertices[e.a].point == (2, 0) or r.vertices[e.b].point == (2, 0))
        )
        self.assertEqual(center.sources, ("a", "b"))
        self.assertEqual(check_invariants(r, segs), [])


class TestBridgesAndTrees(unittest.TestCase):
    def test_bridge_appears_twice_in_same_walk(self):
        segs = _square("s", 0, 0) + [("tail", (1, 1), (2, 2))]
        r = subdivide(segs)
        unbounded = next(f for f in r.faces if f.unbounded)
        bridge = next(e for e in r.edges if set(e.sources) == {"tail"})
        # 桥的两条半边在无界面游走中各出现一次
        self.assertEqual(unbounded.halfedges.count(2 * bridge.id), 1)
        self.assertEqual(unbounded.halfedges.count(2 * bridge.id + 1), 1)
        # 方框边只出现一次
        for e in r.edges:
            if e.id != bridge.id:
                self.assertEqual(
                    unbounded.halfedges.count(2 * e.id)
                    + unbounded.halfedges.count(2 * e.id + 1),
                    1,
                )
        self.assertEqual(check_invariants(r, segs), [])

    def test_dangling_edge_on_square(self):
        segs = _square("s", 0, 0) + [("d", (0, 0), (0, -2))]
        r = subdivide(segs)
        # V=5 E=5 F=2，欧拉成立；无界面面积为 -1
        self.assertEqual(
            (len(r.vertices), len(r.edges), len(r.faces)), (5, 5, 2)
        )
        unbounded = next(f for f in r.faces if f.unbounded)
        self.assertEqual(unbounded.area, -1)
        self.assertEqual(check_invariants(r, segs), [])

    def test_pure_tree_unique_unbounded_face_zero_area(self):
        segs = [
            ("a", (0, 0), (1, 0)),
            ("b", (1, 0), (2, 1)),
            ("c", (1, 0), (1, -1)),
            ("d", (2, 1), (3, 1)),
        ]
        r = subdivide(segs)
        self.assertEqual(len(r.faces), 1)
        face = r.faces[0]
        self.assertTrue(face.unbounded)
        self.assertEqual(face.area, 0)
        # 树有 V-1 条边，每条都是桥：每条半边都在唯一游走里
        self.assertEqual(len(face.halfedges), 2 * len(r.edges))
        self.assertEqual(len(r.vertices) - len(r.edges), 1)
        self.assertEqual(check_invariants(r, segs), [])

    def test_every_halfedge_belongs_to_exactly_one_walk(self):
        segs = _square("s", 0, 0) + _square("r", 2, 0) + [
            ("link", (1, 0), (2, 0))
        ]
        r = subdivide(segs)
        counts = [0] * (2 * len(r.edges))
        for f in r.faces:
            for h in f.halfedges:
                counts[h] += 1
        self.assertEqual(counts, [1] * (2 * len(r.edges)))
        # next 必须是半边集上的置换
        self.assertEqual(sorted(r.halfedge_next), list(range(2 * len(r.edges))))
        self.assertEqual(check_invariants(r, segs), [])


class TestTwoBoxesDemoScene(unittest.TestCase):
    def two_box_segments(self):
        return (
            _square("L", 0, 0, s=2)
            + [("bridge", (2, 1), (4, 1))]
            + _square("R", 4, 0, s=2)
        )

    def test_two_boxes_connected_by_edge(self):
        segs = self.two_box_segments()
        r = subdivide(segs)
        self.assertEqual(r.status, "ok")
        # 两方框各 4 顶点，连接点 (2,1) 与 (4,1) 在方框边中点：
        # 左框 4 角 + (2,1)，右框 4 角 + (4,1) = 10 顶点
        self.assertEqual(len(r.vertices), 10)
        # 每框被连接点切成 5 条边 + 桥 = 11
        self.assertEqual(len(r.edges), 11)
        # V - E + F = 2 => F = 3
        self.assertEqual(len(r.faces), 3)
        bounded = [f for f in r.faces if not f.unbounded]
        self.assertEqual(len(bounded), 2)
        self.assertEqual({f.area for f in bounded}, {Fraction(4)})
        unbounded = next(f for f in r.faces if f.unbounded)
        self.assertEqual(unbounded.area, Fraction(-8))
        self.assertEqual(check_invariants(r, segs), [])

    def test_demo_scene_with_crossing_and_overlap(self):
        # 在双框基础上加：一条与桥共线重叠的边、一条真正穿过左框的交叉线
        segs = self.two_box_segments() + [
            ("bridge2", (3, 1), (4, 1)),       # 与桥部分共线重叠
            ("slash", (-1, 0), (1, 2)),        # 穿左框：交左边于(0,1)、顶边于(1,2)
        ]
        r = subdivide(segs)
        self.assertEqual(r.status, "ok")
        self.assertEqual(check_invariants(r, segs), [])
        coords = {(v.x, v.y) for v in r.vertices}
        # 两个真交叉点都是精确顶点
        self.assertIn((Fraction(0), Fraction(1)), coords)
        self.assertIn((Fraction(1), Fraction(2)), coords)
        # 桥几何中至少有一段同时来自 bridge 与 bridge2
        merged = [
            e for e in r.edges
            if "bridge" in e.sources and "bridge2" in e.sources
        ]
        self.assertTrue(merged)
        # 面积守恒：所有面有向面积之和为 0
        self.assertEqual(sum((f.area for f in r.faces), Fraction(0)), 0)


class TestDisconnected(unittest.TestCase):
    def test_two_separate_segments_rejected(self):
        segs = [("a", (0, 0), (1, 0)), ("b", (5, 5), (6, 6))]
        r = subdivide(segs)
        self.assertIsInstance(r, DisconnectedNetwork)
        self.assertEqual(r.status, "disconnected")
        self.assertEqual(r.components, (("a",), ("b",)))

    def test_nested_components_rejected(self):
        outer = _square("o", 0, 0, s=4)
        inner = _square("i", 1, 1, s=2)
        r = subdivide(outer + inner)
        self.assertEqual(r.status, "disconnected")
        self.assertEqual(
            r.components,
            (tuple(sorted(s[0] for s in inner)), tuple(sorted(s[0] for s in outer))),
        )

    def test_components_recomputed_independently(self):
        # 三个分量，其中一个内部交叉相连；白盒复查分量归并
        segs = [
            ("a", (0, 0), (2, 2)), ("a2", (0, 2), (2, 0)),  # 分量1（交叉）
            ("b", (10, 0), (11, 0)),                          # 分量2
            ("c", (-5, -5), (-4, -4)), ("c2", (-4, -4), (-3, -5)),  # 分量3（接触）
        ]
        r = subdivide(segs)
        self.assertEqual(r.status, "disconnected")
        # 独立用库的精确求交原语 + DSU 重算
        norm = [
            ("a", (Fraction(0), Fraction(0)), (Fraction(2), Fraction(2))),
            ("a2", (Fraction(0), Fraction(2)), (Fraction(2), Fraction(0))),
            ("b", (Fraction(10), Fraction(0)), (Fraction(11), Fraction(0))),
            ("c", (Fraction(-5), Fraction(-5)), (Fraction(-4), Fraction(-4))),
            ("c2", (Fraction(-4), Fraction(-4)), (Fraction(-3), Fraction(-5))),
        ]
        dsu = _DSU(len(norm))
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                ci, cj = {Fraction(0), Fraction(1)}, {Fraction(0), Fraction(1)}
                if _add_pair_cuts(norm[i], norm[j], ci, cj):
                    dsu.union(i, j)
        groups = {}
        for i, (sid, *_rest) in enumerate(norm):
            groups.setdefault(dsu.find(i), []).append(sid)
        expected = tuple(sorted(tuple(sorted(g)) for g in groups.values()))
        expected = tuple(sorted(expected))
        self.assertEqual(r.components, expected)
        self.assertEqual(len(r.components), 3)


class TestInputValidation(unittest.TestCase):
    def test_zero_length_rejected(self):
        with self.assertRaises(ZeroLengthSegment):
            subdivide([("z", (3, 3), (3, 3))])

    def test_duplicate_id_rejected(self):
        with self.assertRaises(DuplicateSegmentId):
            subdivide([("a", (0, 0), (1, 0)), ("a", (2, 0), (3, 0))])

    def test_empty_and_too_many(self):
        with self.assertRaises(InvalidInput):
            subdivide([])
        with self.assertRaises(InvalidInput):
            subdivide([(i, (0, 0), (1, 0)) for i in range(41)])

    def test_coordinate_limits(self):
        r = subdivide([("a", (-COORD_LIMIT, -COORD_LIMIT), (COORD_LIMIT, COORD_LIMIT))])
        self.assertEqual(r.status, "ok")
        with self.assertRaises(InvalidInput):
            subdivide([("a", (0, 0), (COORD_LIMIT + 1, 0))])

    def test_float_coordinates_rejected(self):
        # 精确库不接受 float，杜绝隐式容差
        with self.assertRaises(InvalidInput):
            subdivide([("a", (0.0, 0), (1, 0))])

    def test_fraction_coordinates_accepted(self):
        segs = [
            ("a", (Fraction(1, 3), Fraction(0)), (Fraction(1, 3), Fraction(1))),
            ("b", (Fraction(0), Fraction(1, 2)), (Fraction(1), Fraction(1, 2))),
        ]
        r = subdivide(segs)
        self.assertEqual(r.status, "ok")
        coords = {(v.x, v.y) for v in r.vertices}
        self.assertIn((Fraction(1, 3), Fraction(1, 2)), coords)
        self.assertEqual(check_invariants(r, segs), [])

    def test_segment_dataclass_input(self):
        r = subdivide([Segment("s", (0, 0), (1, 0))])
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.edges[0].sources, ("s",))


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        coords = [
            (0, 0, 4, 0), (4, 0, 4, 4), (4, 4, 0, 4), (0, 4, 0, 0),
            (2, 0, 2, 4), (1, 2, 3, 2),
        ]
        self.segs = [(i, (a, b), (c, d)) for i, (a, b, c, d) in enumerate(coords)]

    def _canonical(self, r):
        return (
            tuple((str(v.x), str(v.y)) for v in r.vertices),
            tuple((e.a, e.b, tuple(map(str, e.sources))) for e in r.edges),
            tuple((tuple(f.halfedges), str(f.area)) for f in r.faces),
        )

    def test_input_order_does_not_affect_result(self):
        import itertools
        results = {self._canonical(subdivide(p)) for p in itertools.permutations(self.segs)}
        self.assertEqual(len(results), 1)

    def test_face_ids_assigned_by_canonical_boundary(self):
        r1 = subdivide(self.segs)
        r2 = subdivide(list(reversed(self.segs)))
        self.assertEqual(
            [tuple(f.halfedges) for f in r1.faces],
            [tuple(f.halfedges) for f in r2.faces],
        )
        self.assertEqual(
            [f.area for f in r1.faces], [f.area for f in r2.faces]
        )


class TestHalfedgeTopology(unittest.TestCase):
    def test_walks_close_and_next_is_locally_correct(self):
        segs = _square("s", 0, 0)
        r = subdivide(segs)
        for face in r.faces:
            hs = face.halfedges
            for k, h in enumerate(hs):
                self.assertEqual(r.halfedge_next[h], hs[(k + 1) % len(hs)])
                self.assertEqual(
                    r.halfedge_target[h],
                    r.halfedge_origin[hs[(k + 1) % len(hs)]],
                )

    def test_twin_and_edge_mapping(self):
        segs = _square("s", 0, 0)
        r = subdivide(segs)
        for e in r.edges:
            self.assertEqual(r.twin(2 * e.id), 2 * e.id + 1)
            self.assertEqual(r.twin(2 * e.id + 1), 2 * e.id)
            self.assertEqual(r.edge_of_halfedge(2 * e.id), e.id)

    def test_euler_for_dense_grid(self):
        segs = [("h", (0, 1), (3, 1))]  # 占位，下面整体替换
        segs = []
        for y in (1, 2):
            segs.append((f"h{y}", (0, y), (3, y)))
        for x in (1, 2):
            segs.append((f"v{x}", (x, 0), (x, 3)))
        r = subdivide(segs)
        # 4 条线井字相交：交点 4 个 + 端点 8 个 = 12 顶点，12 边，F = 2
        self.assertEqual(len(r.vertices), 12)
        self.assertEqual(len(r.edges), 12)
        self.assertEqual(len(r.faces), 2)
        bounded = [f for f in r.faces if not f.unbounded]
        self.assertEqual(bounded[0].area, Fraction(1))
        self.assertEqual(check_invariants(r, segs), [])


class TestRandomSmallNetworks(unittest.TestCase):
    """固定种子的小样本随机验证（不做长时间压力测试）。"""

    def test_random_networks(self):
        rng = random.Random(20260920)
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (2, 1),
        ]
        connected_checked = 0
        rejected = 0
        for _trial in range(120):
            n = rng.randint(1, 6)
            # 从一个公共锚点生长：每条新段的起点从已有端点中抽取，
            # 保证网络连通；少量试验额外放一条远处孤立段制造断开。
            endpoints = [(0, 0)]
            segs = []
            for i in range(n):
                p = endpoints[rng.randrange(len(endpoints))]
                dx, dy = rng.choice(directions)
                q = (p[0] + dx, p[1] + dy)
                if p == q:
                    q = (p[0] + 1, p[1])
                segs.append((i, p, q))
                endpoints.append(q)
            if rng.random() < 0.25:
                segs.append((99, (40, 40), (41, 41)))
            r = subdivide(segs)
            if r.status == "disconnected":
                rejected += 1
                continue
            problems = check_invariants(r, segs)
            self.assertEqual(problems, [])
            # 面积守恒
            self.assertEqual(sum((f.area for f in r.faces), Fraction(0)), 0)
            connected_checked += 1
        self.assertGreaterEqual(connected_checked, 60)
        self.assertGreaterEqual(rejected, 1)  # 小样本中应确实出现断开网络

    def test_random_fractional_crossings(self):
        """随机斜线相交：交点必须精确为顶点，不重不漏。"""
        rng = random.Random(4242)
        for _trial in range(40):
            segs = []
            base_x, base_y = rng.randint(-3, 0), rng.randint(-3, 0)
            for i in range(3):
                x1 = base_x + rng.randint(0, 3)
                y1 = base_y + rng.randint(0, 3)
                x2 = x1 + rng.randint(1, 3)
                y2 = y1 + rng.choice([-2, -1, 1, 2])
                segs.append((i, (x1, y1), (x2, y2)))
            r = subdivide(segs)
            if r.status == "ok":
                self.assertEqual(check_invariants(r, segs), [])


class TestFracHelper(unittest.TestCase):
    def test_frac_str(self):
        self.assertEqual(frac_str(Fraction(3)), "3")
        self.assertEqual(frac_str(Fraction(-1, 2)), "-1/2")
        self.assertEqual(frac_str(Fraction(13, 2)), "13/2")


if __name__ == "__main__":
    # Windows 控制台友好输出
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    unittest.main(verbosity=2)
