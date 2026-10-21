# 基于周期感知动态图优化与扩散–固有信号解耦的交通流量预测方法

基于 `innovation-1-2-3`。这是待实验验证的实现方案，不代表已经证明创新性或精度提升。新增开关默认关闭，关闭时使用原分支逻辑。

## 1. 周期一致性引导的动态图质量优化机制

原方法对边两端的节点周期相似度取平均。本方案比较当前节点表征之间的关系矩阵与周期表征之间的关系矩阵，并结合两个节点相对周期信号的偏离差异评估边的可靠程度。

设归一化当前/周期表征为 U、V，当前信号和历史周期信号为 x、c：

```
relation_ij = 1 - |(U U^T)_ij - (V V^T)_ij| / 2
deviation_i = x_i - c_i
scale_i = mean(|x_i|) + mean(|c_i|) + 1
agreement_ij = exp(-mean(|deviation_i - deviation_j|) / ((scale_i + scale_j) / 2))
quality_ij = relation_ij * agreement_ij
```

将 quality 替换原节点平均可靠性，沿用有效性掩码、质量残差修正和行归一化。无效周期上下文不改变原质量分数。该实现仍为稠密图，计算/存储复杂度仍为 O(N²)。

## 2. 图质量感知的扩散–固有信号解耦

先根据当前输入和隐状态构建动态图，再计算周期一致性修正后的边质量 Q。将入/出两个支持图的非自环边质量按原始图权重汇总，得到节点可靠度 r；不存在非自环边时取 0。这里的 Q 是学习到的可靠性分数，并非经过标定的概率。

```
r_i = sum_s sum_{j != i}(A_s,ij * Q_s,ij) / sum_s sum_{j != i}(A_s,ij)
g_quality = sigmoid(signal_gate(features)) * r
x_diff = g_quality * x
x_inh = x - x_diff
```

可靠度越高，相同输入下允许分配给空间扩散分支的信号越多；可靠度低时，更多信号保留给节点级固有 GRU。两个分量始终相加等于原信号。图质量分数同时影响图优化和信号分配，预测损失可沿两条路径反向更新质量模块。

该实现使用 `use_graph_quality_decouple`，要求同时开启 `use_signal_decouple` 和 `use_dgq`。此前实现的 `use_periodic_residual_decouple` 保留作可选实验，本文主题配置关闭该开关，以直接研究图质量对信号分配的作用。

## 运行与消融

同时开启两项改动，以 PEMSD4 为例：

```bash
python run.py --dataset PEMSD4 --mode train \
  --use_dgq True --dgq_eval_ensemble False \
  --use_periodic_context True --use_context_graph_refine True \
  --use_signal_decouple True \
  --use_periodic_edge_quality True \
  --use_graph_quality_decouple True
```

保持其他参数、数据划分和随机种子相同，比较`use_periodic_edge_quality` 和 `use_graph_quality_decouple` 的四种组合：False/False、True/False、False/True、True/True。运行目录自动记录新增开关名称。建议报告多个随机种子的 MAE、RMSE、MAPE 与参数量、耗时。

上述命令关闭了现有教师集成，也未开启周期后处理，用于评估模型本身。该配置在编码器使用周期上下文；原有解码器周期上下文需要同时开启 `use_decoder_periodic_context` 和 `use_periodic_consistency`，此处不改变其行为。

## 已验证与待验证

已在 CPU 小规模单层单输入通道单元上验证：信号守恒、周期缺失回退、上下文敏感性、图行归一化、前向输出及反向梯度有限，以及门控/图投影梯度非零。图质量解耦另验证可靠度对扩散分量的单调控制、孤立节点回退及端到端梯度。

尚未进行完整数据训练、超参数搜索或相关工作检索。论文贡献表述应结合相关工作与消融结果确定；新结构应重新训练后评估。

## 论文贡献表述

1. 提出一种周期一致性引导的动态图质量优化机制，利用历史周期规律评估和调整动态关联关系。
2. 提出一种图质量感知的扩散–固有信号解耦机制，根据动态图关联可靠性自适应建模空间传播和节点自身动态。
