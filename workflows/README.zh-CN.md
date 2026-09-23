# NAC 生成流程

[English](README.md) | [简体中文](README.zh-CN.md)

根据 Hamiltonian 的来源选择路线。这里从已完成的 ABACUS SCF 输出或已有的
HamGNN 预测结果开始，生成波函数、相邻帧 AO 重叠，再调用 CA-NAC。
模型训练、图数据生成和 Hamiltonian 预测仍由 HamGNN 完成。

| 路线 | 准备阶段 | 最后一步 |
| --- | --- | --- |
| [ABACUS](abacus/README.zh-CN.md) | 从 SCF H/S 求波函数；从 SCF 结构和轨道文件生成跨帧 AO 重叠 | [共用 NAC 投影](#共用-nac-投影) |
| [HamGNN](hamgnn/README.zh-CN.md) | 从预测 H 和图中的 S 求波函数；使用 ABACUS 路线生成的 AO 重叠 | [共用 NAC 投影](#共用-nac-投影) |

```text
workflows/
├── run_canac_route.py       # 两条路线共用的 NAC 投影入口
├── abacus/
│   ├── wfc_dft.py           # ABACUS SCF H/S → eigen.npy、wfc.npy
│   ├── direct_overlap.py    # SCF STRU 与轨道文件 → tdoverlap.npy
│   └── abacus_csr.py        # ABACUS 矩阵读取模块
└── hamgnn/
    ├── wfc_hamgnn.py        # HamGNN 预测结果 → eigen.npy、wfc.npy
    ├── hamgnn_wfc.py        # Hamiltonian 组装与 CPU 求解模块
    └── abacus_basis.py      # HamGNN 图数据使用的 ABACUS 轨道掩码
```

`direct_overlap.py` 的输入来自 ABACUS SCF，因此放在 `abacus/`。
当两条路线的结构、AO 基组和排序一致时，可以共用它生成的跨帧重叠。
`hamgnn/abacus_basis.py` 属于 HamGNN 波函数转换的内部依赖，其文件名表示
模型采用的 ABACUS 轨道约定。

## 支持范围与数据约定

当前入口支持固定晶胞、Gamma 点、实数标量轨道、物理 `SPIN0` 通道和完整秩波函数。
不适用于 SOC、其他 k 点、降秩处理或其他物理自旋通道。
CA-NAC 中的 `HAMNET` 只是已准备 NumPy 文件的读取器名称；
实际 Hamiltonian 来源通过 `--source abacus` 或 `--source hamgnn` 记录。

帧范围包含首尾：`START..END` 对应 `END-START+1` 个波函数结果和
`END-START` 个相邻帧 NAC。默认保留四位编号，例如 `0001`。
帧范围、物理能带、轨道掩码和有效保存帧间隔均应按自己的计算设置。

两个波函数入口都先完成单帧输出，再通过目录重命名发布，并写入
`wfc-source.json`。续算只跳过来源路径、文件大小、修改时间、参数一致，
且数组形状、类型和有限值检查通过的结果。输入文件在运行期间须保持不变。
已有目录若无法验证，会停止而不会覆盖；输入或精度改变时使用新输出目录。
应先生成波函数，再链接重叠文件。脚本不进行本征模删除或重叠矩阵条件化。

矩阵读取与 CPU 本征求解代码提取自现有 ABACUS/HamGNN 工具；
HamGNN 入口采用逐帧读取实现。轨道掩码和单位常数保留了原有来源说明，
见 `hamgnn/abacus_basis.py`。

脚本不内置机器路径或集群环境。长轨迹应在调度系统分配的资源中运行，
并明确设置 BLAS 线程数，避免多个 worker 各自启动大量线程：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## 共用 NAC 投影

根目录的 `run_canac_route.py` 接收任一路线准备好的 NumPy 数据并调用 CA-NAC。
此步骤需要 Python 3.9+、NumPy 和 SciPy。

沿用所选路线中设置的 `WORKFLOWS`、`WFC_PYTHON`、`OVERLAP_ROOT`、
`START`、`END` 和 `NPROC`。每帧须有 `eigen.npy` 和 `wfc.npy`；
除最后一帧外，每个左帧还须有它与下一帧之间的 `tdoverlap.npy` 及
`tdoverlap.npy.meta.json`。结构、AO 基组和排序必须与重叠来源一致。

下面示例选择 ABACUS 路线。使用 HamGNN 时，将前两行分别改为
`ROUTE="$HAMGNN_ROUTE"` 和 `SOURCE=hamgnn`。
能带编号从 1 开始，对应完整保存谱中的物理能带；`POTIM_FS` 是有效保存帧间隔，
单位为 fs。示例数值需要按实际任务修改。

```bash
ROUTE="$ABACUS_ROUTE"
SOURCE=abacus
BAND_MIN=1
BAND_MAX=2
POTIM_FS=1.0

for ((frame=START; frame<END; frame++)); do
  printf -v label '%04d' "$frame"
  for name in tdoverlap.npy tdoverlap.npy.meta.json; do
    ln -s "$OVERLAP_ROOT/$label/SPIN0/$name" "$ROUTE/$label/SPIN0/$name"
  done
done

"$WFC_PYTHON" "$WORKFLOWS/run_canac_route.py" \
  --source "$SOURCE" --run-dir-pattern "$ROUTE/{index}/SPIN0" \
  --start "$START" --end "$END" --index-width 4 \
  --band-min "$BAND_MIN" --band-max "$BAND_MAX" \
  --potim "$POTIM_FS" --nproc "$NPROC" --summary "$ROUTE/nac-summary.json"
```

链接请使用绝对路径；已有且正确的链接可以继续使用，无需再次执行链接循环。
脚本默认寻找当前仓库中的 `CAnac.py`，也可通过 `--ca-nac-root` 或
`CA_NAC_ROOT` 指定另一份仓库。它会检查输入数组、相邻帧重叠的来源说明和可用的
波函数来源信息，重新计算能带投影，并在所有要求的 NAC 输出完成后记录成功状态。

### 状态跟踪与单位

默认输出 `nac_ps.npy`。添加 `--state-tracking` 后输出 `nac_psrd.npy`，
请使用不同的 summary 文件名。若两种结果都需要，应分别运行两次。
状态跟踪从所选首帧开始，必须按轨迹顺序连续运行。再次调用会重新生成所选模式的 NAC。

`nac_ps.npy` 和 `nac_psrd.npy` 都是**无量纲的反对称重叠分子**，
其中 `ps` 表示 pseudopotential（赝势），不是皮秒。
`--potim` 记录帧间隔，不会对这两个文件进行时间缩放。
若需要导数耦合，按以下约定另存新文件，并保留原始数组：

- `D = nac / (2 * POTIM_FS)`，单位为 `fs^-1`。
- `D = 1000 * nac / (2 * POTIM_FS)`，单位为 `ps^-1`。
