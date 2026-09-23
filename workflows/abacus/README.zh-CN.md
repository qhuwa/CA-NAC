# 从 ABACUS SCF 输出生成 NAC

[English](README.md) | [简体中文](README.zh-CN.md)

本路线从已完成的 ABACUS SCF 矩阵生成波函数，并从相应 SCF 结构和轨道文件
生成相邻帧 AO 重叠，最后调用[共用 NAC 投影](../README.zh-CN.md#共用-nac-投影)。

| 脚本 | 作用 |
| --- | --- |
| `wfc_dft.py` | SCF `data-HR` / `data-SR` → `eigen.npy`、`wfc.npy` |
| `direct_overlap.py` | SCF 结构与轨道文件 → 已校准的 `tdoverlap.npy` 和来源说明 |
| `abacus_csr.py` | 供 `wfc_dft.py` 使用的矩阵读取模块 |

## 依赖与参数设置

波函数生成和 CA-NAC 需要 Python 3.9+、NumPy 和 SciPy。
直接 AO 重叠还需要包含 `ModuleBase` 和 `ModuleNAO` 的 `pyabacus`。
它应与 SCF 使用的 ABACUS 源码版本、编译器和浮点选项匹配，并保留构建记录。
首次正式运行会使用普通同帧 `data-SR` 校准；仅能成功导入不代表数值实现兼容。

按实际计算修改以下示例路径和帧范围。波函数和 pyabacus 可以使用不同的 Python 环境。

```bash
WORKFLOWS=/path/to/CA-NAC/workflows
WFC_PYTHON=/path/to/numpy-scipy-environment/bin/python
OVERLAP_PYTHON=/path/to/pyabacus-environment/bin/python
SCF_ROOT=/path/to/scf
ORBITAL_DIR=/path/to/orbitals
OVERLAP_ROOT=/path/to/direct-overlap
ABACUS_ROUTE=/path/to/abacus-route
START=1
END=100
NPROC=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

续算行为和支持范围见[数据约定](../README.zh-CN.md#支持范围与数据约定)。

## 生成波函数

```bash
"$WFC_PYTHON" "$WORKFLOWS/abacus/wfc_dft.py" \
  --folder-pattern "$SCF_ROOT/{index}/OUT.ABACUS" \
  --save-pattern "$ABACUS_ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC"
```

每帧输入文件名为 `data-HR-sparse_SPIN0.csr` 和
`data-SR-sparse_SPIN0.csr`。沿用现有 float32 Gamma 点矩阵求和、
Rydberg 到 Hartree 的换算和广义本征求解，输出本征能量的单位为 eV。
`data-H0R`、`data-S0R` 不能替代这些 SCF 矩阵。

## 生成相邻帧 AO 重叠

每个 SCF 帧必须保留自己的 `STRU` 和
`OUT.ABACUS/{INPUT,running_scf.log}`，并有正常完成 SCF 的证据。
用于校准的帧还需要同一次计算生成的普通 `data-SR-sparse_SPIN0.csr`。
不得用轨迹导出文件或其他计算的结构替代缺失的 SCF 结构。

```bash
"$OVERLAP_PYTHON" "$WORKFLOWS/abacus/direct_overlap.py" \
  --scf-root "$SCF_ROOT" --orbital-dir "$ORBITAL_DIR" \
  --output-root "$OVERLAP_ROOT" \
  --start "$START" --end "$END" --index-width 4 --nproc "$NPROC"
```

如果物种标签与轨道文件的 `Element` 不同，须显式传入映射，
例如 `--species-element-map H1=H H2=H`。脚本自动推导积分网格、完成校准，
并写入 `tdoverlap-contract.json`。每个相邻帧对的结果按左帧编号保存在
`<left-frame>/SPIN0/tdoverlap.npy`，来源说明为 `tdoverlap.npy.meta.json`。

续算由生成器内置的输入和来源检查控制；基组、结构或实现改变时使用新输出目录。
不需要另行运行校准脚本或独立数值验证脚本。HamGNN 路线也使用本步骤生成的 AO 重叠。

## 运行 NAC

设置 `ROUTE="$ABACUS_ROUTE"` 和 `SOURCE=abacus`，然后按
[共用 NAC 投影](../README.zh-CN.md#共用-nac-投影)链接重叠文件，
指定物理能带范围和有效帧间隔。
