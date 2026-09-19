"""planar_arrangement 的单元测试。

覆盖：X 交叉、T 接触、共线重叠、重复几何不同 ID、桥与悬挂边、
纯树唯一无界面、断开输入拒绝、分数交点、输入顺序不变性、随机自检。
"""

import random
import unittest
from fractions import Fraction

from planar_arrangement import (
    ArrangementError,
    build_arrangement,
    verify,
)

F = Fraction


def square(x, y, s, ids):
    """左下角 (x,y)、边长 s 的方框四条边。"""
    a, b, c, d = ids
    return [
        (a, (x, y), (x + s, y)),
        (b, (x + s, y), (x + s, y + s)),
        (c, (x + s, y + s), (x, y + s)),
        (d, (x, y + s), (x, y)),
    ]


class TestCrossings(unittest.TestCase):
    def test_x_crossing(self):
        arr = build_arrangement([
            (1, (0, 0), (2, 2)),
            (2, (0, 2), (2, 0)),
        ])
        self.assertEqual(arr.status, "ok")
        self.assertEqual(len(arr.vertices), 5)
        self.assertIn((F(1), F(1)), arr.vertices)
        self.assertEqual(arr.intersections, [(F(1), F(1))])
        self.assertEqual(len(arr.edges), 4)
        # 十字是树：唯一面，面积 0，为无界面
        self.assertEqual(len(arr.faces), 1)
        self.assertEqual(arr.faces[0].area, 0)
        self.assertTrue(arr.faces[0].is_outer)
        self.assertTrue(verify(arr, [
            (1, (0, 0), (2, 2)),
            (2, (0, 2), (2, 0)),
        ]))

    def test_fractional_intersection(self):
        segs = [
            (1, (0, 0), (3, 0)),
            (2, (1, -1), (2, 2)),
        ]
        arr = build_arrangement(segs)
        # y=0 处 x = 1 + 1/3 = 4/3
        self.assertIn((F(4, 3), F(0)), arr.vertices)
        self.assertEqual(arr.intersections, [(F(4, 3), F(0))])
        self.assertEqual(len(arr.edges), 4)
        self.assertTrue(verify(arr, segs))

    def test_t_touch(self):
        segs = [
            (1, (0, 0), (2, 0)),
            (2, (1, 0), (1, 1)),
        ]
        arr = build_arrangement(segs)
        self.assertEqual(arr.status, "ok")
        self.assertEqual(len(arr.vertices), 4)
        self.assertEqual(len(arr.edges), 3)
        self.assertEqual(arr.intersections, [(F(1), F(0))])
        self.assertEqual(len(arr.faces), 1)
        self.assertEqual(arr.faces[0].area, 0)
        self.assertTrue(verify(arr, segs))


class TestOverlapAndDuplicate(unittest.TestCase):
    def test_collinear_partial_overlap(self):
        segs = [
            ("A", (0, 0), (4, 0)),
            ("B", (2, 0), (6, 0)),
        ]
        arr = build_arrangement(segs)
        self.assertEqual(len(arr.edges), 3)
        got = {(e.a, e.b): e.sources for e in arr.edges}
        self.assertEqual(
            got,
            {
                ((F(0), F(0)), (F(2), F(0))): ("A",),
                ((F(2), F(0)), (F(4), F(0))): ("A", "B"),
                ((F(4), F(0)), (F(6), F(0))): ("B",),
            },
        )
        self.assertTrue(verify(arr, segs))

    def test_collinear_full_containment(self):
        segs = [
            (1, (0, 0), (10, 0)),
            (2, (2, 0), (8, 0)),
            (3, (4, 0), (6, 0)),
        ]
        arr = build_arrangement(segs)
        got = [e.sources for e in arr.edges]
        self.assertEqual(
            got, [(1,), (1, 2), (1, 2, 3), (1, 2), (1,)]
        )
        self.assertTrue(verify(arr, segs))

    def test_duplicate_geometry_different_ids(self):
        segs = [
            (7, (0, 0), (3, 0)),
            (9, (0, 0), (3, 0)),
        ]
        arr = build_arrangement(segs)
        self.assertEqual(len(arr.vertices), 2)
        self.assertEqual(len(arr.edges), 1)
        self.assertEqual(arr.edges[0].sources, (7, 9))
        self.assertEqual(len(arr.faces), 1)
        self.assertTrue(verify(arr, segs))


class TestFaces(unittest.TestCase):
    def test_single_square(self):
        segs = square(0, 0, 2, (1, 2, 3, 4))
        arr = build_arrangement(segs)
        self.assertEqual(len(arr.faces), 2)
        areas = sorted(f.area for f in arr.faces)
        self.assertEqual(areas, [F(-4), F(4)])
        outer = [f for f in arr.faces if f.is_outer]
        self.assertEqual(len(outer), 1)
        self.assertEqual(outer[0].area, F(-4))
        self.assertTrue(verify(arr, segs))

    def test_two_boxes_with_bridge(self):
        segs = square(0, 0, 2, (1, 2, 3, 4)) + square(4, 0, 2, (5, 6, 7, 8))
        segs.append((9, (2, 1), (4, 1)))  # 桥
        arr = build_arrangement(segs)
        self.assertEqual(arr.status, "ok")
        self.assertEqual(len(arr.faces), 3)
        areas = sorted(f.area for f in arr.faces)
        self.assertEqual(areas, [F(-8), F(4), F(4)])
        # 桥的两个方向都出现在无界面同一游走中
        outer = next(f for f in arr.faces if f.is_outer)
        b = outer.boundary
        directed = [(b[i], b[(i + 1) % len(b)]) for i in range(len(b))]
        p, q = (F(2), F(1)), (F(4), F(1))
        self.assertIn((p, q), directed)
        self.assertIn((q, p), directed)
        self.assertTrue(verify(arr, segs))

    def test_dangling_edge(self):
        segs = square(0, 0, 2, (1, 2, 3, 4))
        segs.append((5, (2, 1), (4, 1)))  # 悬挂边
        arr = build_arrangement(segs)
        self.assertEqual(arr.status, "ok")
        self.assertEqual(len(arr.faces), 2)  # 悬挂边不产生新面
        outer = next(f for f in arr.faces if f.is_outer)
        b = outer.boundary
        directed = [(b[i], b[(i + 1) % len(b)]) for i in range(len(b))]
        p, q = (F(2), F(1)), (F(4), F(1))
        self.assertIn((p, q), directed)
        self.assertIn((q, p), directed)
        self.assertTrue(verify(arr, segs))

    def test_pure_tree_single_outer_face(self):
        segs = [
            (1, (0, 0), (1, 0)),
            (2, (1, 0), (2, 0)),
            (3, (2, 0), (2, 1)),
        ]
        arr = build_arrangement(segs)
        self.assertEqual(len(arr.faces), 1)
        face = arr.faces[0]
        self.assertTrue(face.is_outer)
        self.assertEqual(face.area, 0)
        # 唯一游走经过每条半边：长度 == 2E
        self.assertEqual(len(face.boundary), 2 * len(arr.edges))
        self.assertTrue(verify(arr, segs))

    def test_face_ids_sorted_by_boundary(self):
        segs = square(0, 0, 2, (1, 2, 3, 4))
        arr = build_arrangement(segs)
        bounds = [f.boundary for f in arr.faces]
        self.assertEqual(bounds, sorted(bounds))
        self.assertEqual([f.id for f in arr.faces], [0, 1])


class TestRejection(unittest.TestCase):
    def test_disconnected(self):
        arr = build_arrangement([
            (1, (0, 0), (1, 0)),
            (2, (5, 5), (6, 6)),
        ])
        self.assertEqual(arr.status, "disconnected")
        self.assertEqual(arr.faces, [])

    def test_zero_length_rejected(self):
        with self.assertRaises(ArrangementError):
            build_arrangement([(1, (1, 1), (1, 1))])

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ArrangementError):
            build_arrangement([
                (1, (0, 0), (1, 0)),
                (1, (0, 0), (0, 1)),
            ])

    def test_count_and_coord_bounds(self):
        with self.assertRaises(ArrangementError):
            build_arrangement([])
        with self.assertRaises(ArrangementError):
            build_arrangement([(i, (i, 0), (i, 1)) for i in range(41)])
        with self.assertRaises(ArrangementError):
            build_arrangement([(1, (0, 0), (10**6 + 1, 0))])


class TestDeterminism(unittest.TestCase):
    CASES = [
        square(0, 0, 2, (1, 2, 3, 4)) + square(4, 0, 2, (5, 6, 7, 8))
        + [(9, (2, 1), (4, 1))],
        [
            (1, (0, 0), (4, 4)),
            (2, (0, 4), (4, 0)),
            (3, (0, 0), (4, 0)),
            (4, (2, 0), (6, 0)),
            (5, (1, -1), (2, 2)),
        ],
    ]

    def signature(self, arr):
        return (
            arr.status,
            arr.vertices,
            [(e.a, e.b, e.sources) for e in arr.edges],
            [(f.boundary, f.area) for f in arr.faces],
            arr.intersections,
        )

    def test_input_order_irrelevant(self):
        rng = random.Random(20260920)
        for segs in self.CASES:
            base = self.signature(build_arrangement(segs))
            for _ in range(5):
                shuffled = segs[:]
                rng.shuffle(shuffled)
                self.assertEqual(self.signature(build_arrangement(shuffled)), base)

    def test_random_small_samples_selfcheck(self):
        rng = random.Random(12345)
        ok = 0
        for _ in range(60):
            n = rng.randint(1, 8)
            segs = []
            for i in range(n):
                x1, y1 = rng.randint(-4, 4), rng.randint(-4, 4)
                while True:
                    x2, y2 = rng.randint(-4, 4), rng.randint(-4, 4)
                    if (x1, y1) != (x2, y2):
                        break
                segs.append((i, (x1, y1), (x2, y2)))
            arr = build_arrangement(segs)
            if arr.status != "ok":
                continue
            ok += 1
            self.assertTrue(verify(arr, segs))
            self.assertEqual(
                len(arr.vertices) - len(arr.edges) + len(arr.faces), 2
            )
            total_half_edges = sum(len(f.boundary) for f in arr.faces)
            self.assertEqual(total_half_edges, 2 * len(arr.edges))
            outers = [f for f in arr.faces if f.is_outer]
            self.assertEqual(len(outers), 1)
        self.assertGreater(ok, 0)


if __name__ == "__main__":
    unittest.main()
