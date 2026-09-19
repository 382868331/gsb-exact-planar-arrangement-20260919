# 连通线段网络的精确平面细分库

CAD导入组件要把一个连通的线段网络切分成可追溯的平面边和面边界。

完整规格见 [TASK.md](TASK.md)。实现为 `planar_arrangement.py`，仅标准库，
内部全部用 `fractions.Fraction` 精确计算，无浮点容差、无第三方几何库。

## 环境与命令

Windows 原生 Python 3.14.7，仅标准库，无第三方依赖安装步骤，无账号、密钥、外部服务或 Docker。

演示（约 8 秒内，含正常结果与实际触发的失败）：

```
python demo.py
```

测试：

```
python -m unittest discover -s tests -v
```

## 接口

```python
from planar_arrangement import build_arrangement, verify, ArrangementError

arr = build_arrangement([
    (seg_id, (x1, y1), (x2, y2)),   # 1..40 条，ID 唯一，整数坐标 |c| <= 10**6
    ...
])
```

- 非法输入（零长线段、ID 重复、数量/坐标越界）抛出 `ArrangementError`。
- 细分后的无向图不连通时返回 `arr.status == "disconnected"`（不处理断开分量嵌套）。

`Arrangement` 字段（全部按精确坐标排序，输入顺序不影响结果）：

| 字段 | 含义 |
| --- | --- |
| `status` | `"ok"` 或 `"disconnected"` |
| `vertices` | 顶点 `[(Fraction, Fraction), ...]`，按坐标排序 |
| `edges` | 无向子边 `[Edge(a, b, sources)]`，`a < b`，`sources` 为排序后的源线段 ID |
| `faces` | 面 `[Face(id, boundary, area, is_outer)]`，按规范化边界排序分配 ID |
| `intersections` | 精确交点（真交叉、端点接触、共线重叠端点），排序去重 |

拓扑约定：

- 处理端点接触、真交叉与共线部分/完全重叠；在必要交点处分割，
  相同几何子边合并并保留所有源 ID。
- 双向半边在每个顶点按象限/叉积精确逆时针排序；沿有向边到达顶点后取
  反向边的顺时针前邻作为 next，遍历左侧面边界。
- 桥在同一游走中出现两次，不删除；有界面有向面积为正，无界面为负，
  纯树的唯一面面积为 0。不做点定位或独立孔洞分类。
- 自检（`build_arrangement` 内部执行，也可显式调用 `verify(arr, segments)`）：
  每条半边恰属一个游走、`V - E + F == 2`（F 含无界面）、每条源线段的
  子边无缝覆盖 `[0,1]`、所有交点均为细分顶点。

## 文件

- `planar_arrangement.py` — 库本体
- `demo.py` — 演示：两个由边连接的方框、交叉/重叠边、断开与零长失败
- `tests/test_planar_arrangement.py` — X 交叉、T 接触、共线重叠、重复几何
  不同 ID、桥与悬挂边、纯树唯一无界面、断开拒绝、分数交点、输入顺序不变性、
  固定种子随机小样本自检
