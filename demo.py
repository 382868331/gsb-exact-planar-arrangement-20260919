"""演示：精确平面细分库的正常结果与实际触发的失败。

运行：python demo.py（标准库，离线，约 8 秒内完成）
"""

import sys
import time

from planar_arrangement import ArrangementError, build_arrangement, verify

# 重定向到管道时 Windows 默认区域编码可能无法显示中文，统一用 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fmt_pt(p):
    return f"({p[0]},{p[1]})"


def show_arrangement(arr, segs):
    print(f"  状态: {arr.status}")
    print(f"  顶点 V={len(arr.vertices)}: " + " ".join(fmt_pt(v) for v in arr.vertices))
    print(f"  子边 E={len(arr.edges)}:")
    for e in arr.edges:
        print(f"    {fmt_pt(e.a)} -- {fmt_pt(e.b)}  源ID={list(e.sources)}")
    print(f"  交点: " + (" ".join(fmt_pt(p) for p in arr.intersections) or "无"))
    print(f"  面 F={len(arr.faces)}:")
    for f in arr.faces:
        kind = "无界面" if f.is_outer else "有界面"
        ring = " -> ".join(fmt_pt(v) for v in f.boundary + (f.boundary[0],))
        print(f"    面{f.id} [{kind}] 有向面积={f.area}")
        print(f"      边界: {ring}")
    verify(arr, segs)
    print("  自检通过: 每条半边恰属一个游走、V-E+F=2、源覆盖完整、交点均为顶点")


def main():
    t0 = time.perf_counter()

    print("=" * 72)
    print("演示 1：两个由一条边连接的方框（桥在无界面游走中出现两次）")
    print("=" * 72)
    segs1 = [
        (1, (0, 0), (2, 0)), (2, (2, 0), (2, 2)),
        (3, (2, 2), (0, 2)), (4, (0, 2), (0, 0)),
        (5, (4, 0), (6, 0)), (6, (6, 0), (6, 2)),
        (7, (6, 2), (4, 2)), (8, (4, 2), (4, 0)),
        (9, (2, 1), (4, 1)),  # 连接两个方框的桥边
    ]
    arr1 = build_arrangement(segs1)
    show_arrangement(arr1, segs1)

    print()
    print("=" * 72)
    print("演示 2：真交叉（分数交点）与共线重叠（源 ID 合并）")
    print("=" * 72)
    segs2 = [
        ("diag", (0, 0), (3, 3)),
        ("anti", (0, 3), (3, 0)),
        ("base1", (0, 0), (4, 0)),
        ("base2", (2, 0), (6, 0)),   # 与 base1 在 [2,4] 共线重叠
        ("slant", (1, -1), (2, 2)),  # 与 base 族产生分数交点 (4/3, 0)
    ]
    arr2 = build_arrangement(segs2)
    show_arrangement(arr2, segs2)

    print()
    print("=" * 72)
    print("演示 3：实际触发的失败 —— 断开的输入返回 disconnected")
    print("=" * 72)
    segs3 = [
        (1, (0, 0), (1, 0)),
        (2, (10, 10), (12, 10)),  # 与上面线段不连通
    ]
    arr3 = build_arrangement(segs3)
    print(f"  状态: {arr3.status}（细分后的无向图不连通，按约定不处理断开分量）")
    assert arr3.status == "disconnected"

    print()
    print("=" * 72)
    print("演示 4：实际触发的失败 —— 零长线段被拒绝")
    print("=" * 72)
    try:
        build_arrangement([(1, (5, 5), (5, 5))])
    except ArrangementError as exc:
        print(f"  抛出 ArrangementError: {exc}")

    dt = time.perf_counter() - t0
    print()
    print(f"全部演示完成，耗时 {dt:.3f} 秒（预算约 8 秒）")


if __name__ == "__main__":
    main()
