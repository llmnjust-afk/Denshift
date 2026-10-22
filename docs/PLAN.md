# Denshift — 文献基础、Baseline 与实验计划

> 状态：研究代码，未投稿。合成基准可复现；真实数据通路已实现但尚未用真实权重验证。
> 仓库：https://github.com/llmnjust-afk/Denshift
> 本文档回答四个问题：**创新点从哪些文献来**、**跟谁比**、**复现谁的代码**、**用哪些数据**。

---

## 0. 一句话定位

现有 3D 测试时适应（TTA）把分布漂移当作一个整体，用熵最小化或伪标签去适应；
我们把**采样密度**单独拎出来，证明它是点云上一个**可从输入直接观测**的漂移轴，
并用成对视图在**纯 logit 空间**闭式地估计并扣除它对决策边界的影响，
最后给出这个估计**在何种条件下可辨识**的边界。

---

## 1. 文献基础

### 1.1 四条脉络

| 脉络 | 代表工作 | 我们从中取什么 |
|---|---|---|
| 2D TTA 的范式 | Tent (arXiv 2006.10726)、CoTTA (2203.13591)、T3A (2201.05718) | 熵最小化 / 权重平均 / 原型调整 三种主流思路，以及它们**都要求访问模型内部**这一共同前提 |
| 3D TTA | MATE (ICCV'23, 2211.11432)、BFTT3D (CVPR'24, 2403.18442)、CloudFixer (ECCV'24, 2407.16193)、PCoTTA (NeurIPS'24, 2411.00632)、Purge-Gate (ICCV'25, 2509.09785)、Graph-Spectral Point Shift (ICCV'25, 2507.18225) | 3D 上已有的 TTA 谱系；**BFTT3D 是无反传路线的直接对手** |
| 采样变化与 3D TTA | **3DD-TTA (WACV'25, 2411.01116)** "Leveraging Sampling Variation with Weight Averaging" | **最接近本文的工作**，必须在论文中正面区分（见 §1.3） |
| 鲁棒性基准与 shortcut learning | ModelNet40-C (2201.12296)、ModelNet-C (2202.03377)、Robo3D (2303.17597)、LiDAR 分割鲁棒性 (2301.00970)；Shortcut to Nowhere (2606.01723)、Clever Hans 综述 (Frontiers AI 2026) | 评测协议与 mCE 归一化约定；"模型依赖伪相关"这一理论框架 |

### 1.2 时间梯度上的最新工作（必须引用，不能只引老三篇）

- **2026**：MAMVI (2606.12939)、SATOR (2610.21367)、Adapting in the Dark (2604.15609，2D 黑盒 TTA)、Shortcut to Nowhere (2606.01723)
- **2025**：Purge-Gate、Graph-Spectral Point Shift、SMART-PC (2505.19546)、3DD-TTA (WACV)
- **2024**：BFTT3D、PCoTTA、CloudFixer
- **2023**：MATE、Point-TTA (2308.16481，注意是**配准**不是分类)

> 写作时按"新→旧"组织：先讲 2025–2026 的黑盒/无反传路线走到了哪，再回溯到 2023 的 MATE，
> 最后说明密度这一轴在此谱系中始终没有被单独处理。

### 1.3 与最接近工作（3DD-TTA）的区别 —— 论文必须写清楚

| | 3DD-TTA (WACV'25) | Denshift（本文） |
|---|---|---|
| 对采样变化的用法 | 把不同采样当作**增强**，用于**权重平均** | 把密度当作**可测量的 nuisance**，估计其**对 logit 的线性响应** |
| 作用位置 | 参数空间（需要权重、需要训练） | **输出空间**（只碰 logits） |
| 是否需要源数据/训练 | 需要再训练/微调 | 不需要，纯推理期闭式解 |
| 是否给出可辨识性分析 | 无 | **有**（设计矩阵秩、条件数、截断曲线） |
| 失效边界是否显式声明 | 无 | 有（全局边际漂移下失效） |

一句话：**3DD-TTA 用采样变化去更新模型，我们用采样变化去测量偏差**。这是本文唯一
必须防守的对比点，审稿人一定会问。

### 1.4 每个创新点对应的文献支撑

| 本文主张 | 文献依据 |
|---|---|
| 模型会把数据集偏差学成捷径 | Shortcut to Nowhere (2026)；Clever Hans 综述 (2026)；Less Learn Shortcut (IJCAI'23) |
| 熵最小化在系统性 logit 偏移下会走错方向 | Tent 的熵目标本身不含偏差项；Ranked Entropy Minimization (2025) 亦指出 EM 的失效模式 |
| 3D TTA 普遍需要模型内部访问 | BFTT3D 明确以"免反传"为卖点，但仍需权重；黑盒 TTA 在 2D 上才刚起步（Adapting in the Dark, 2026） |
| 点云密度是可从输入观测的独立轴 | 采样变化已被用作增强（3DD-TTA），但未被当作可估计的 nuisance |
| mCE 必须按各自基准归一化 | ModelNet-C 与 Robo3D 的 clean error 基准不同，不可混表 |

---

## 2. Baseline 对比方案

### 2.1 我们自己实现、可直接跑的

| 名称 | 对应文献 | 实现说明 |
|---|---|---|
| `source_only` | — | 冻结模型，不改任何东西 |
| `entropy_bias` | Tent (2020) 的黑盒类比 | Tent 改 BN 参数；黑盒拿不到 BN，只能在输出上加偏置下降同一熵目标。**必须在论文中说明这是"受约束的类比"，不是 Tent 复现** |
| `global_bias` | 消融 | 只用一个标量尺度项，不条件于密度 |
| `dsc` | 本文 | 完整闭式估计器 |
| `dsc_top2` | 本文 | 截断到前两个可辨识方向（推荐配置） |

### 2.2 引用论文数字、不重跑的

| 方法 | 出处 | 为什么不自跑 |
|---|---|---|
| MATE | ICCV'23 | 需要 MAE 预训练权重与特定训练流程 |
| CloudFixer | ECCV'24 | 依赖 diffusion 模型，成本远超剩余时间 |
| PCoTTA | NeurIPS'24 | 面向多任务持续适应，设定不同 |
| Purge-Gate / Graph-Spectral | ICCV'25 | 需要 token 级访问，与黑盒设定不同 |

引用时**必须注明**：这些数字来自各自论文，测试协议与 checkpoint 可能不同，不构成严格同条件对比。

### 2.3 可以尝试重跑的（同 checkpoint，条件可比）

| 方法 | 代码 | 可行性 |
|---|---|---|
| **BFTT3D** | `abie-e/BFTT3D`（Point-NN + MATE） | 与我们同为 DGCNN/PointNet 系，最有价值的同条件对比；但环境较乱，预留 2 天 |
| 3DD-TTA | 论文 + 作者代码 | 需要再训练，若时间不足则只引用数字并在 limitation 中说明 |

**判据**：若 BFTT3D 能在同一 DGCNN checkpoint 上跑通，主表就用"我们 vs BFTT3D vs source_only"三行；
若跑不通，主表退化为"我们 vs source_only"，并在文中明确说明未能复现的原因。

---

## 3. 数据集

| 数据集 | 任务 | 用途 | 获取 | 指标 |
|---|---|---|---|---|
| **ModelNet40-C** | 40 类分类 | **主实验** | `ldkong1205/PointCloud-C` 合并发布 | OA、mCE |
| **ModelNet-C** | 40 类分类（7 种 corruption） | 与 ModelNet40-C 二选一，**不要混用** | 同上 | OA、mCE |
| ShapeNet-C | 部件分割 | 可选：证明方法不限于分类 | 同上 | mIoU |
| SemanticKITTI-C (Robo3D) | 19 类 LiDAR 分割 | 可选：真实传感器场景 | `worldbench/Robo3D` | mIoU、mCE |
| KITTI-C / nuScenes-C | 3D 检测 | **不做**（时间不够） | Robo3D | — |

### 关键约定

1. **mCE 归一化基准不同**：ModelNet-C 用 DGCNN clean OA=0.926 作基准（mCE=1.000），
   Robo3D 用 MinkUNet-18 clean mIoU=62.76。**两张表永远不能放在一起**。
2. Robo3D 的官方基线协议**禁止 TTA/ensemble**，我们要用它的数据就必须重新跑基线并在文中声明协议改动。
3. ModelNet-C（Ren et al., 7 种）与 ModelNet40-C（Sun et al., 15 种）是**两个不同基准**，
   合并仓库里都有，选一个并全文统一。

---

## 4. 代码基线来源

| 用途 | 仓库 | 我们用它的什么 |
|---|---|---|
| 数据 + DGCNN 权重 + 评测协议 | `ldkong1205/PointCloud-C` | `dgcnn_cls` 架构、预训练 `.t7`、corruption 生成脚本 |
| 3D TTA 对手 | `abie-e/BFTT3D` | 同条件对比（若跑通） |
| 2D TTA 参照 | `DequanWang/tent`、`qinenergy/cotta`、`mr-eggplant/SAR` | 只用于确认熵最小化的标准实现细节 |
| 真实 LiDAR 鲁棒性 | `worldbench/Robo3D` | 可选扩展 |

**本仓库的 `denshift/backends/dgcnn.py` 是按 `dgcnn_cls` 的参数名逐个对齐写的**，
因此官方 checkpoint 可以 `load_checkpoint` 直接加载，无需转换。

---

## 5. 已完成的证据（合成基准）

`python scripts/smoke_test.py --n-samples 150 --seed 0`

| adapter | none | decoupled | decoupled_hard | global |
|---|---|---|---|---|
| source_only | 1.000 | 0.933 | 0.853 | 0.753 |
| entropy_bias | 1.000 | 0.933 | 0.853 | 0.753 |
| global_bias | 1.000 | 0.927 | 0.853 | 0.753 |
| dsc | 0.933 | 0.920 | 0.887 | 0.613 |
| **dsc_top2** | **0.987** | **0.933** | **0.873** | **0.720** |

可辨识性诊断（`scripts/diagnose_identifiability.py`）：

| 指标 | decoupled | decoupled_hard |
|---|---|---|
| 设计矩阵秩 | 19 / 21 | 19 / 21 |
| 有效条件数 | 2.3e2 | 3.3e2 |
| 成对残差 | 11.0% | 12.0% |
| 绝对空间误差 ‖ΦB−f‖/‖f‖ | 71.8% | 66.1% |
| 逐样本余弦 | 0.48 | 0.55 |

两个已锁进测试的工程修正：分类器截距不能丢（丢了语义通路从 0.99 塌到 0.08）；
偏置不能按样本中心化（会引入按类别不同的偏移）。

---

## 6. 实验计划

### P0 — 真实基线对齐（1–2 天）

用 PointCloud-C 官方 DGCNN 权重跑 clean split。

- **判据**：clean OA 落在 **0.915–0.935**。偏离则先排查点数（1024）、是否 `--normalize`、
  输入是否转置，**不要往下走**。
- 产出：`results/clean.json`

### P1 — 真实数据上密度效应是否存在（2–3 天）

按 corruption 分组跑 `source_only`，同时记录每个 split 的描述子第一维
（log 中位 kNN 距离），检验"密度变化 ↔ 准确率下降"是否单调相关。

- **判据**：至少一组 corruption 显示单调关系。若无 → 走 §8 的备选方案。
- 注意：`jitter`/`dropout` 改变的是噪声与点数，`scale`/`add_global` 更接近纯密度轴，
  预计后者更干净。
- 产出：`results/corruption_vs_density.json` + 散点图

### P2 — 真实数据 TTA 对比（3–4 天）

同一 checkpoint、同一数据顺序，对比 5 个适配器；若 BFTT3D 跑通则加一行。

- **判据**：`dsc_top2` 在 ≥3/5 severity 上不低于 `source_only`，且 mCE 更低；
  `entropy_bias` 基本无效（这是核心对比论点）。
- 产出：主表 + `results/modelnet_c_*.json`

### P3 — 消融（2 天）

视图比例 0.25/0.5/0.75；截断维数 k=1..6；有/无 change detector；
去掉描述子第 1 维（证明是密度而非尺度在起作用）。

### P4 — 写作与图（剩余时间）

- 图 1：密度分位数 vs 熵 vs 置信度（三者解耦）
- 图 2：截断曲线（可辨识性可视化）
- 图 3：mCE 条形图，含 BFTT3D

---

## 7. 如何运行

### 7.1 环境

```bash
git clone https://github.com/llmnjust-afk/Denshift.git
cd Denshift
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch
python -m pytest -q               # 预期 35 passed
```

### 7.2 合成实验（无需数据，30–60 秒）

```bash
python scripts/smoke_test.py --n-samples 150 --seed 0

for s in 0 1 2 3 4; do
  python scripts/smoke_test.py --n-samples 200 --seed $s --json results/synth_seed$s.json
done

python scripts/diagnose_identifiability.py --n-samples 200 --seed 1 --regime decoupled
```

### 7.3 真实数据

```bash
# 1) 数据 data/ModelNet40-C/，权重 weights/dgcnn_modelnet40.t7

# 2) P0 基线
python scripts/eval_modelnet_c.py \
    --root data/ModelNet40-C \
    --checkpoint weights/dgcnn_modelnet40.t7 \
    --corruptions clean --n-samples 200 --json results/clean.json

# 3) P2 全量
for a in source_only entropy_bias global_bias dsc dsc_top2; do
  python scripts/eval_modelnet_c.py \
      --root data/ModelNet40-C \
      --checkpoint weights/dgcnn_modelnet40.t7 \
      --corruptions add_global jitter scale rotation \
      --severity 1 2 3 4 5 \
      --adapter $a --n-samples 200 --json results/mnc_${a}.json
done
```

CPU 约 1 秒/朵，200×4×5×5 约 11 小时；有 GPU 用 `--device cuda`。

### 7.4 接到自己的模型

```python
import numpy as np
from denshift import (BlackBoxModel, DensityBiasModel,
                      density_descriptor, stratified_subsample)

box = BlackBoxModel(module=net, device="cuda", n_classes=40,
                    preprocess=lambda p: torch.from_numpy(p.T[None]).float())
cal = DensityBiasModel(n_classes=40, warmup=32)
rng = np.random.default_rng(0)

for cloud in stream:
    sparse = stratified_subsample(cloud, 0.5, 4, rng)
    z, zs = box(cloud), box(sparse)
    cal.observe_pair(density_descriptor(cloud),
                     density_descriptor(sparse), z, zs)
    pred = np.argmax(cal.correct(z, density_descriptor(cloud)))
```

---

## 8. 风险与决策点

| 风险 | 触发条件 | 应对 |
|---|---|---|
| 真实数据上密度效应弱 | P1 判据不达标 | 单独构造密度流（对同一测试集做确定性下采样），把"效应存在性"和"corruption 鲁棒性"解耦 |
| 增益不显著 | P2 差 < 1 点 | 扩样本量、换 3 个 checkpoint、报置信区间；仍不显著则按负面结果写法 |
| BFTT3D 跑不通 | 环境 2 天内未解决 | 主表退化为 vs source_only，limitation 中说明 |
| 3DD-TTA 审稿质疑 | 必然出现 | 提前在 related work 用 §1.3 表格正面区分 |
| 时间不足 | 剩余 < 10 天仍未跑通 P2 | 砍 Robo3D，只保留 ModelNet40-C，收窄为"可辨识性分析 + 轻量校正" |

**核心写法**：不要写成"我们提升了 X 点"。现有证据支持的是更窄但更站的论点——
*点云 TTA 中密度偏差只有部分可辨识，而可辨识的那部分足以解释并移除主要损害*。
有理论、有负面结果、有可复现证据，比硬凑 SOTA 更抗审稿。

---

## 9. 参考文献（按本文档首次出现顺序）

1. Wang, Shelhamer, Liu, Olshausen, Darrell. *Tent: Fully Test-time Adaptation by Entropy Minimization.* arXiv:2006.10726, 2020.
2. Wang, Fink, Van Gool, Dai. *Continual Test-Time Domain Adaptation.* CVPR 2022. arXiv:2203.13591.
3. Boudiaf, Mueller, Ben Ayed, Bertinetto. *Parameter-free Online Test-time Adaptation.* CVPR 2022. arXiv:2201.05718.
4. Mirza et al. *MATE: Masked Autoencoders are Online 3D Test-Time Learners.* ICCV 2023. arXiv:2211.11432.
5. Wang et al. *Backpropagation-free Network for 3D Test-time Adaptation.* CVPR 2024. arXiv:2403.18442.
6. Shim, Kim, Yang. *CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation.* ECCV 2024. arXiv:2407.16193.
7. Jiang et al. *PCoTTA: Continual Test-Time Adaptation for Multi-Task Point Cloud Understanding.* NeurIPS 2024. arXiv:2411.00632.
8. Bahri et al. *Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging.* WACV 2025. arXiv:2411.01116.
9. Yazdanpanah et al. *Purge-Gate: Backpropagation-Free Test-Time Adaptation for Point Clouds Classification via Token Purging.* ICCV 2025. arXiv:2509.09785.
10. Wei et al. *3D Test-Time Adaptation via Graph Spectral Driven Point Shift.* ICCV 2025. arXiv:2507.18225.
11. Bahri et al. *SMART-PC: Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds.* arXiv:2505.19546, 2025.
12. Kong, Jung, Jung. *MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds.* arXiv:2606.12939, 2026.
13. Kong, Jung, Jung. *SATOR: Backpropagation-Free 3D Test-Time Adaptation via Sample-Adaptive Token Routing.* arXiv:2610.21367, 2026.
14. Zhang, Niu, Cai, Liu, Hamm. *Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models.* arXiv:2604.15609, 2026.
15. Xu, Li, Wang, Yang. *Shortcut to Nowhere: Demystifying Deep Spurious Regression.* arXiv:2606.01723, 2026.
16. Pathak, Gupta, Jain. *Unmasking the Clever Hans effect in AI models.* Frontiers in Artificial Intelligence, 2026.
17. Du et al. *Less Learn Shortcut: Analyzing and Mitigating Learning of Spurious Feature-Label Correlation.* IJCAI 2023.
18. Sun et al. *Benchmarking Robustness of 3D Point Cloud Recognition Against Common Corruptions.* arXiv:2201.12296, 2022.
19. Ren, Pan, Liu. *Benchmarking and Analyzing Point Cloud Classification under Corruptions.* ICML 2022. arXiv:2202.03377.
20. Kong et al. *Robo3D: Towards Robust and Reliable 3D Perception against Corruptions.* ICCV 2023. arXiv:2303.17597.
21. Yan et al. *Benchmarking the Robustness of LiDAR Semantic Segmentation Models.* arXiv:2301.00970, 2023.
22. Hatem, Qian, Wang. *Point-TTA: Test-Time Adaptation for Point Cloud Registration Using Multitask Meta-Auxiliary Learning.* ICCV 2023. arXiv:2308.16481.
23. Liang, He, Tan. *A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts.* IJCV 2024. arXiv:2303.15361.
24. Han, Na, Hwang. *Ranked Entropy Minimization for Continual Test-Time Adaptation.* arXiv:2505.16441, 2025.

> 以上条目均已核对 arXiv 编号或 DOI；引用进 `refs.bib` 时仍需跑 `ref_verify` 做字段级校验。
