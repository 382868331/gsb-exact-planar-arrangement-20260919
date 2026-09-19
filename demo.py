"""精确平面细分库演示。

运行：python demo.py

内容：

1. 正常结果：两个方框由一条桥连接，另加共线重叠边与真正穿框的交叉边。
   展示精确交点（含分数坐标）、子边来源合并、半边面游走（桥在同一游走中
   正反出现两次）、精确有向面积与独立一致性自检。
2. 实际触发的失败：一组断开（嵌套但不接触）的线段网络被库拒绝，
   返回 status == "disconnected" 并给出分量；另演示零长线段抛异常。

所有数字均由 planar_arrangement 现场计算，无硬编码结果。
"""

from __future__ import annotations

import io
import sys
import time
from fractions import Fraction

# Windows 控制台默认 GBK，统一切到 UTF-8，保证分数与方框字符正常显示。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from planar_arrangement import (  # noqa: E402
    ZeroLengthSegment,
    check_invariants,
    frac_str,
    point_str,
    subdivide,
)


def _box(prefix: str, x0: int, y0: int, s: int) -> list:
    return [
        (f"{prefix}下边", (x0, y0), (x0 + s, y0)),
        (f"{prefix}右边", (x0 + s, y0), (x0 + s, y0 + s)),
        (f"{prefix}上边", (x0 + s, y0 + s), (x0, y0 + s)),
        (f"{prefix}左边", (x0, y0 + s), (x0, y0)),
    ]


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_success() -> float:
    _print_header("【正常结果】两个由桥连接的方框 + 交叉边 + 共线重叠边")

    segments = (
        _box("左框", 0, 0, s=2)
        + [("桥", (2, 1), (4, 1))]
        + _box("右框", 4, 0, s=2)
        + [
            ("重叠边", (3, 1), (4, 1)),   # 与桥的右半段共线重叠
            ("穿框斜线", (-1, 0), (1, 2)),  # 交左框左边于(0,1)、顶边于(1,2)
        ]
    )

    print("输入线段（共 %d 条，ID 唯一）：" % len(segments))
    for sid, p, q in segments:
        print(f"  {sid:<8} {point_str(p)} -> {point_str(q)}")

    t0 = time.perf_counter()
    result = subdivide(segments)
    elapsed = time.perf_counter() - t0

    assert result.status == "ok", result
    print()
    print(f"细分状态：{result.status}    耗时：{elapsed*1000:.1f} ms")
    print(f"顶点 V = {len(result.vertices)}，"
          f"无向子边 E = {len(result.edges)}，面 F = {len(result.faces)}")
    print(f"欧拉公式检查：V - E + F = "
          f"{len(result.vertices)} - {len(result.edges)} + {len(result.faces)} "
          f"= {len(result.vertices) - len(result.edges) + len(result.faces)}（应为 2）")

    print()
    print("顶点（按精确坐标排序，分数保持精确形式）：")
    line = "  "
    for v in result.vertices:
        token = f"v{v.id}={point_str(v.point)}"
        if len(line) + len(token) + 2 > 74:
            print(line)
            line = "  "
        line += token + "  "
    print(line)

    print()
    print("无向子边（按端点坐标排序；来源为覆盖该几何边的全部源 ID）：")
    for e in result.edges:
        pa_, pb_ = result.vertices[e.a], result.vertices[e.b]
        src = ", ".join(map(str, e.sources))
        print(f"  e{e.id:<2} {point_str(pa_.point)} -- {point_str(pb_.point)}"
              f"    来源[{src}]")

    print()
    print("面（半边边界游走，面在行进方向左侧）：")
    for face in result.faces:
        kind = "无界面" if face.unbounded else "有界面"
        chain = " -> ".join(f"v{vid}" for vid in face.vertices)
        print(f"  面 {face.id} [{kind}] 精确有向面积 = {frac_str(face.area)}")
        print(f"      顶点环：{chain} -> v{face.vertices[0]}")
        print(f"      半边序列：{face.halfedges}（共 {len(face.halfedges)} 条半边）")

    # 桥在无界面游走中应正反各出现一次（桥在 (3,1) 被重叠边切成两段子边，
    # 取仅由“桥”覆盖的左半段 (2,1)--(3,1) 演示）。
    bridge_edge = next(
        e for e in result.edges
        if {result.vertices[e.a].point, result.vertices[e.b].point}
        == {(Fraction(2), Fraction(1)), (Fraction(3), Fraction(1))}
        and tuple(e.sources) == ("桥",)
    )
    unbounded = next(f for f in result.faces if f.unbounded)
    h_fwd, h_back = 2 * bridge_edge.id, 2 * bridge_edge.id + 1
    print()
    print("桥的处理（不删除，同一游走中走两次）：")
    print(f"  桥子边 e{bridge_edge.id} (2,1)--(3,1) 的半边 {h_fwd} 在无界面游走中出现 "
          f"{unbounded.halfedges.count(h_fwd)} 次，"
          f"反向半边 {h_back} 出现 {unbounded.halfedges.count(h_back)} 次")

    print()
    print("独立一致性自检 check_invariants()：")
    t0 = time.perf_counter()
    problems = check_invariants(result, segments)
    check_elapsed = time.perf_counter() - t0
    if problems:
        for p in problems:
            print("  [问题] " + p)
    else:
        print("  全部通过：每条半边恰属一个游走；V-E+F=2；面积符号与面积和为 0；")
        print("  每条子边来源覆盖正确；所有精确交点（含共线重叠端点）均为顶点。")
    print(f"  自检耗时：{check_elapsed*1000:.1f} ms")
    return elapsed + check_elapsed


def demo_failure() -> float:
    _print_header("【实际触发的失败】断开的线段网络被拒绝")

    outer = _box("外框", 0, 0, s=4)
    inner = _box("内框", 1, 1, s=2)
    segments = outer + inner
    print("输入线段：外框与内框几何上嵌套但互不接触（共 %d 条）" % len(segments))
    for sid, p, q in segments:
        print(f"  {sid:<6} {point_str(p)} -> {point_str(q)}")

    t0 = time.perf_counter()
    result = subdivide(segments)
    elapsed = time.perf_counter() - t0

    print()
    print(f"细分状态：{result.status}")
    assert result.status == "disconnected"
    print("库拒绝继续处理（不处理断开分量嵌套），按段 ID 归并出的连通分量：")
    for i, comp in enumerate(result.components):
        print(f"  分量 {i + 1}：{', '.join(comp)}")

    print()
    print("另一类失败——零长线段直接抛 ZeroLengthSegment：")
    try:
        subdivide([("坏段", (1, 1), (1, 1))])
    except ZeroLengthSegment as exc:
        print(f"  捕获异常：{type(exc).__name__}: {exc}")
    return elapsed


def main() -> int:
    print("精确平面细分库演示（Python 标准库 + fractions.Fraction，无浮点容差）")
    t_start = time.perf_counter()
    t_ok = demo_success()
    t_fail = demo_failure()
    total = time.perf_counter() - t_start
    _print_header("演示结束")
    print(f"正常用例计算+自检耗时 {t_ok*1000:.1f} ms；失败用例 {t_fail*1000:.1f} ms；"
          f"总计 {total:.2f} s（要求约 8 秒内）")
    print("运行测试：python -m unittest discover -s tests -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
