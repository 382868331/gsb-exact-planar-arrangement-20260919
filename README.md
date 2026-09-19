# 精确平面细分库（exact planar subdivision）

把一个**连通**的线段网络（1..40 条带唯一 ID 的线段，整数坐标绝对值 ≤ 10⁶）
在所有交点处切分，得到排序后的顶点、无向子边（保留全部来源 ID），并基于双向
半边遍历每个面的边界游走与精确有向面积。

全部计算使用标准库 [`fractions.Fraction`](https://docs.python.org/3/library/fractions.html)，
**不使用浮点容差，也不依赖任何第三方几何库**；仅需 Python 3.14 标准库，
可在 Windows 原生环境离线运行。

## 环境与命令

- Windows 原生 Python 3.14.7（3.10+ 亦可运行），无第三方依赖、无账号/密钥/外部服务。
- 演示（约 0.1 秒，远小于 8 秒预算；展示一个正常结果和一个实际触发的失败）：

  ```
  python demo.py
  ```

- 测试：

  ```
  python -m unittest discover -s tests -v
  ```

## 快速上手

```python
from planar_arrangement import subdivide, check_invariants

# 线段：Segment(id, p1, p2)，或直接写三元组 (id, (x1,y1), (x2,y2))
segments = [
    ("a", (0, 0), (2, 0)),
    ("b", (2, 0), (2, 2)),
    ("c", (2, 2), (0, 2)),
    ("d", (0, 2), (0, 0)),
    ("x", (0, 0), (2, 2)),       # 对角线：分数交点 (1,1)
]

result = subdivide(segments)
assert result.status == "ok"      # 连通
print(len(result.vertices))       # V
print(len(result.edges))          # E（无向子边）
print(len(result.faces))          # F（含唯一无界面）
for face in result.faces:
    print(face.id, face.area, face.unbounded, face.halfedges, face.vertices)

# 独立一致性自检：半边覆盖、V-E+F=2、面积符号、来源覆盖、精确交点
print(check_invariants(result, segments))   # [] 表示全部通过
```

断开的输入（含“外框套内框但互不接触”这类嵌套）不做嵌套处理，直接返回失败：

```python
result = subdivide([("a", (0, 0), (1, 0)), ("b", (9, 9), (10, 10))])
result.status           # "disconnected"
result.components       # (("a",), ("b",)) —— 按源 ID 归并的连通分量
```

非法输入抛出异常（均为 `InvalidInput` 的子类）：`ZeroLengthSegment`、
`DuplicateSegmentId`；数量越界、坐标超界、坐标为 `float`、ID 之间不可排序等
抛出 `InvalidInput`。

## 接口

模块：`planar_arrangement.py`

| 名称 | 说明 |
| --- | --- |
| `subdivide(segments)` | 主入口。连通返回 `PlanarSubdivision`；不连通返回 `DisconnectedNetwork`；非法输入抛 `InvalidInput` 子类。 |
| `Segment(id, p1, p2)` | 输入线段；`id` 可哈希且 ID 之间可互相排序，端点为两个整数（或 `Fraction`）坐标。 |
| `PlanarSubdivision.status` | `"ok"`；另有 `.vertices` / `.edges` / `.faces` / `.halfedge_origin` / `.halfedge_target` / `.halfedge_next`。 |
| `Vertex(id, x, y)` | 顶点，坐标为精确 `Fraction`，按 `(x, y)` 字典序编号。 |
| `SubEdge(id, a, b, sources)` | 无向子边：顶点编号 `a < b`（按坐标），`sources` 为覆盖该几何边的全部源 ID（已排序）。半边 `2*id` 为 `a→b`，`2*id+1` 为其孪生 `b→a`。 |
| `Face(id, halfedges, vertices, area, unbounded)` | 一个面。`halfedges` 为按 `next` 排列的半边循环；`area` 为精确有向面积。 |
| `DisconnectedNetwork(status, components)` | 不连通返回值：`status == "disconnected"`，`components` 为源 ID 分量（分量与内部 ID 均已排序）。 |
| `check_invariants(result, segments)` | 独立重算的一致性自检，返回问题描述列表（空列表 = 全部通过）。 |
| `frac_str(x)` / `point_str(p)` | Fraction / 点的紧凑展示，如 `1/2`、`(3, -1/2)`。 |
| 常量 | `MAX_SEGMENTS = 40`、`COORD_LIMIT = 10**6`。 |

## 约定与语义

- **精确求交（O(n²)）**：非平行线段用叉积解参数 `t, u = 叉积/叉积`，判断
  `0 ≤ t, u ≤ 1`；共线线段把端点投影到参数轴取搭接区间，处理部分/完全重叠与
  端点相接。所有切点（含端点 `0、1`）都是精确 `Fraction`。
- **切分与合并**：每条输入线段在其切点处切开；几何相同的无向子边合并为一条，
  `sources` 取全部源 ID 并集并排序。顶点按精确坐标排序、子边按端点排序，
  **结果与输入顺序无关**（6 段网络的全部 720 种排列结果逐字节相同，有测试覆盖）。
- **半边遍历**：无向边生成一对孪生半边；每个顶点的出边按“象限 + 叉积”精确
  逆时针（CCW）排序；沿半边到达顶点后，取其反向边在 CCW 序列中的**顺时针前邻**
  （前一个）作为 `next`，故每个游走的左侧是同一个面。
- **桥**在同一个面游走中以两个相反方向各出现一次，不删除；纯树网络唯一的面就是
  无界面，其游走覆盖全部半边两次，面积为 0。
- **面积符号**：有界面为正（逆时针），无界面为负（顺时针），所有面有向面积
  之和恒为 0；面 ID 按边界半边循环的最小旋转规范化后排序分配。
- 不做点定位，也不把孔洞单独分类（断开的嵌套分量直接拒绝）。

## 测试覆盖

`tests/test_planar_arrangement.py`（40 个用例）覆盖：

- X 交叉（含分数交点）、T 接触、端点接触、共点方框；
- 共线部分/完全/三重重叠、反向重叠、重叠区再被垂直线穿过；
- 重复几何不同 ID 的来源合并；桥在同一游走出现两次、悬挂边；
- 纯树唯一无界面面积 0；双框由桥连接（含交叉/重叠的演示场景）；
- 断开输入拒绝（两段分离、嵌套双框、三分量独立重算核对）；
- 零长/重复 ID/数量越界/坐标越界/`float` 坐标拒绝，`Fraction` 坐标接受；
- 输入顺序无关、面 ID 规范化、欧拉公式、半边置换与面积守恒；
- 固定种子的小样本随机验证（连通网络生长生成 + 少量断开样本）。

## 文件结构

```
planar_arrangement.py   库实现（精确求交、切分合并、半边、面、自检）
demo.py                 演示：双框+桥+交叉+重叠；断开网络与零长段失败
tests/                  unittest 测试套件
TASK.md                 任务规格
requirements.txt        仅注明标准库，无第三方依赖
```
