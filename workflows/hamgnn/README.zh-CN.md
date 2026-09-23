# 从 HamGNN 预测结果生成 NAC

[English](README.md) | [简体中文](README.zh-CN.md)

本路线从已有 HamGNN Hamiltonian 预测和匹配的图重叠矩阵生成波函数，
使用 [ABACUS 重叠步骤](../abacus/README.zh-CN.md#生成相邻帧-ao-重叠)生成的
已校准 AO 重叠，最后调用[共用 NAC 投影](../README.zh-CN.md#共用-nac-投影)。

| 脚本 | 作用 |
| --- | --- |
| `wfc_hamgnn.py` | 每帧图数据和预测 Hamiltonian → `eigen.npy`、`wfc.npy` |
| `hamgnn_wfc.py` | Gamma 点矩阵组装和 CPU 本征求解模块 |
| `abacus_basis.py` | HamGNN 图数据采用的 ABACUS 轨道掩码 |

## 依赖与参数设置

使用 Python 3.9+、NumPy、SciPy，以及 HamGNN 的 PyTorch、PyTorch Geometric、
pymatgen 环境。图文件含有 pickle 对象，应使用自己可信流程生成的数据。
模型训练、图数据生成和 Hamiltonian 预测仍在 HamGNN 中完成。

按实际计算修改以下示例路径和帧范围：

```bash
WORKFLOWS=/path/to/CA-NAC/workflows
WFC_PYTHON=/path/to/hamgnn-environment/bin/python
GRAPH_ROOT=/path/to/hamgnn-graphs
PREDICTION_ROOT=/path/to/hamgnn-predictions
MODEL_SOURCE=/path/to/checkpoint-used-for-these-predictions
HAMGNN_ROUTE=/path/to/hamgnn-route
OVERLAP_ROOT=/path/to/direct-overlap
START=1
END=100
NPROC=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## 生成波函数

每个图文件必须恰好包含一帧图。预测数组按与图一致的顺序保存
onsite 块和随后所有 offsite 块。Hamiltonian 的单位为 Hartree，重叠矩阵无量纲。
支持的填充轨道掩码尺寸为 `13`、`15`、`27` 和 `40`；
必须与图生成器的设置及实际轨道文件一致，其他掩码会被拒绝。

```bash
NAO_MAX=13
"$WFC_PYTHON" "$WORKFLOWS/hamgnn/wfc_hamgnn.py" \
  --graph-data-path "$GRAPH_ROOT/{index}/graph_data.npz" \
  --hamiltonian-path "$PREDICTION_ROOT/{index}/prediction_hamiltonian.npy" \
  --model-source "$MODEL_SOURCE" --nao-max "$NAO_MAX" --dtype float32 \
  --save-pattern "$HAMGNN_ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC" \
  --summary "$HAMGNN_ROUTE/wfc-summary.json"
```

默认输入为完整预测 Hamiltonian。仅当上游模型明确预测需要加上真实
`Hon0/Hoff0` 的残差时，才使用 `--fix-edge-index`。
不要给完整 Hamiltonian 重复添加 H0。
可用 `--expected-orbitals` 指定已知的完整 AO 维数。

每个 worker 一次只读取一帧，并定期回收进程。
原子发布和已完成结果的续算检查见[数据约定](../README.zh-CN.md#支持范围与数据约定)。

## 使用 AO 重叠并运行 NAC

`OVERLAP_ROOT` 必须指向声明的 SCF 轨迹经
[`abacus/direct_overlap.py`](../abacus/direct_overlap.py)生成的结果。
若尚未生成，先执行 [ABACUS 重叠步骤](../abacus/README.zh-CN.md#生成相邻帧-ao-重叠)。
结构、AO 基组和排序须与 HamGNN 图数据一致；AO 重叠由这些输入决定。

设置 `ROUTE="$HAMGNN_ROUTE"` 和 `SOURCE=hamgnn`，然后按
[共用 NAC 投影](../README.zh-CN.md#共用-nac-投影)链接重叠文件，
指定物理能带范围和有效帧间隔。
