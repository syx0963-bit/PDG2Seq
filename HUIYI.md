# 基于周期感知动态图优化与扩散–固有信号解耦的交通流量预测方法

## 2026-10-22 性能排查与重训

上一轮 seed=10 的结果为 RMSE 30.4214、MAE 18.1716、MAPE 12.0734%。旧目标为 28.1697、17.0095、11.4857%；旧运行配置包含教师集成、周期一致性和在线校正，当前纯模型配置不同，需区分结构效果和评估流程效果。

已修复 run.py 全局初始化覆盖信号门和融合门偏置的问题。增加 `--graph_quality_strength`，质量控制变为 `gate * (1 - strength * (1 - quality))`，默认 1 保持原公式；0 表示不按质量衰减，0.1 表示最多衰减 10%。此变更仍保证信号守恒和质量单调控制。使用旧检查点时不重新初始化其偏置。

新的保守实验沿用上文两个创新点命令，追加 `--graph_quality_strength 0.1 --batch_size 32 --seed 10 --save_every 5`，日志为 `logs/huiyi_quality_strength01_seed10.out`。已通过偏置恢复、信号守恒、质量单调性、有限梯度检查。该实验尚未完成，不能宣称已达到旧目标。最终以验证集选择的 best_model.pth 为准，不使用 best_test_model.pth 进行选型。

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

## 可选增强评估（2026-10-22）

用户已允许教师集成、周期一致性和在线校正。`tools/eval_after_training.py --checkpoint <best_model.pth> --quality-strength 0.1` 通过 `run.py --mode enhanced_eval` 执行增强评估；可用 `--pid <训练PID>` 等待训练结束。不会重启训练，解码器周期上下文保持关闭，周期参考仅用于后处理。

融合权重在前半段验证集拟合，间隔一个预测长度后在后半段验证集选择是否采用学生、教师、集成、周期融合，以及在线偏置更新速率和截断范围。选择分数为 RMSE + MAE + 1.2 × MAPE百分数。测试集只用于最终报告，结果保存为对应评估目录的 `enhanced_results.json`，记录原模型和最终增强指标及参数。

在线校正要求窗口按时间排序且步长为1；只使用当前输入窗口最后一个已经观测到的流量，对到期的各预测步误差更新偏置。不会读取前一窗口尚未发生的未来标签。旧分支的 `adapt_batch(pred, true)` 在每个滑动样本后直接更新完整预测区间，因此其在线结果与此严格因果协议不能直接视为同一评估条件。

当前训练的自动后续评估日志：`logs/huiyi_quality_strength01_seed10_enhanced.out`。上一轮验证集最优检查点的增强评估日志：`logs/huiyi_previous_enhanced.out`。两者均不使用 best_test_model 选型。

## 第二轮优化：自回归微调与节点残差校正

第一轮增强测试结果 29.8904 / 17.8351 / 11.9375%。增加每节点、每预测步的岭回归残差校正，特征为两模型预测、周期偏差、输入最近值、最近3步与12步均值；沿用验证集前半拟合、后半选择，测试结果为 29.8065 / 17.8209 / 11.9924%，仍未达到目标。

已启动从 20261022101735/best_model.pth 微调的实验，保留两项结构创新。参数：sampling_ratio=0（关闭训练时真实值喂入）、loss_func=mae_rmse、rmse_loss_weight=0.3、lr_init=0.0003、epochs=60、lr_decay_step1=20,40、batch_size=32、seed=10、early_stop_patience=15、grad_norm=True。优化器重新初始化，不宣称恢复原训练状态。混合损失用于验证集选检查点，最终仍以三个真实尺度指标评估。

日志 `logs/huiyi_autoregressive_finetune.out`，后续自动增强评估日志 `logs/huiyi_autoregressive_finetune_enhanced.out`。已验证 sampling_ratio=0 时训练前向不受目标流量变化影响，以及混合损失和梯度有限；节点校正在独立合成样本上恢复已知系数并通过非负输出检查。

## 目标结果溯源及共享节点校准

目标数值出自 `logs/newinnovation-1-2-3_DGQ-DGQEnsemble-PeriodicContext-ContextGraphRefine-MetaReliableGraph-PeriodicConsistency-OnlineGraphAdapt_20260925194837.log`，2026-09-25 19:50。其检查点为 `newinnovation-1-2_DGQ-DGQEnsemble-PeriodicContext-ContextGraphRefine-MetaReliableGraph-PeriodicConsistency_20260813215539/best_test_model.pth`。该旧检查点目前不在当前工作区的 PEMSD4 实验目录中。

当时配置为 DGQ、周期上下文、MetaReliableGraph，未启用信号解耦；融合教师 `pre-trained/PEMSD4.pth`，逐预测步学生权重 0.43～0.77，周期权重 0.01～0.06。采用节点岭回归（日志选择 ridge=1e-5），特征为 ensemble/student/teacher/last/periodic/bias。在线参数含 bias_lr=0.08、scale_lr=0.16、global_lr=0.02、scale_clip=0.4、validation warmup 和 post_bias_shrink=-0.5。日志展示验证校准指标 29.6156 / 17.5650 / 11.8743%，最终在线测试指标才为 28.1697 / 17.0095 / 11.4857%。因此不能将其描述为当前两个创新点单独训练的结果。旧测试选模和完整未来区间在线更新的问题见上文。

新增 `lib/forecast_calibration.py`，各节点跨预测步共享残差系数，使用已知输入和两模型/周期预测。验证集前半拟合，间隔12步后后半选择 ridge、稳健迭代次数和融合强度；选择完毕才在完整验证集重拟合。包括未校准模型的候选比较，不强制启用无效校准；测试标签不参与参数拟合。最终参数保存在评估目录的 shared_node_calibration.npz，元信息和指标在 enhanced_results.json。运行日志 `logs/huiyi_shared_calibration.out`。

进一步查看 20260813215539 训练日志：该模型从 20260804220312/best_test_model.pth 微调，seed=10、batch_size=32、lr=0.0003、epochs=80、early_stop_patience=20、loss_func=mae_rmse、rmse_loss_weight=0.35、select_metric=hybrid。因此目标数值包含旧模型多阶段微调和后续评估增强，不是当前结构从零训练的直接结果。

共享节点校准的实测结果：29.7641 / 17.7741 / 11.8931%。另增加 `run.py --mode calibrated_eval --checkpoint <模型> --calibration_source <共享节点评估目录>`，在共享节点校准之上验证可选 XGBoost 非线性残差修正（使用环境已有 xgboost）。树深度和融合强度在验证集后半选择，再以完整验证集重新拟合；所有输入特征不含未来目标流量。学生、教师、历史周期及输入趋势之外增加预测步和节点编号。源目录的检查点必须与指定模型一致。

非线性评估日志 `logs/huiyi_nonlinear_calibration.out`，结果和预测保存在新评估目录的 calibrated_results.json、test_predictions.npz；校准器分别保存为 shared_node_calibration.npz 和 nonlinear_calibration.ubj（仅选中非线性修正时）。完整复用还需源目录中记录的模型参数、教师路径、周期/解耦开关及 graph_quality_strength=0.1。

最终实测节点+非线性校准：RMSE 29.6824、MAE 17.7447、MAPE 11.9232%。校准前模型为 30.6438 / 18.2036 / 12.0827%。相较上一轮增强评估 29.8012 / 17.8252 / 11.9909%，三项均改善，但未达到目标。相较仅共享节点校准，非线性修正改善 RMSE、MAE，MAPE 略回升。最终目录后缀 20261022141949，已保存参数和测试预测。`lib.boosted_calibration.apply_saved(values, directory)` 可直接应用冻结的校准参数，不重新拟合；输入沿用 collect 的布局，未来目标流量字段不被读取。

## RMSE 28.x 专项优化（不改变两个创新点）

原创新模型检查点冻结为 20261022132015/best_model.pth，SHA256 为 `3c84b60223793670502036cbd152b06311c6ae5a013767341ebd49165fad847a`。以下均为辅助预测/残差补偿，不修改周期边质量计算和质量解耦模块；不改变指标公式，不用测试标签拟合参数。

现有预测的第1步/第12步 RMSE 约26.88/31.43，误差随预测距离增加。新增原始历史特征包含更长流量滞后、训练集相关图邻居历史、时段/星期、已观测的另外两个原始通道。主模型仍为原单通道模型，但增强配置使用多通道历史，必须在论文/对比表中单独注明。

已完成的补偿试验（RMSE / MAE / MAPE百分数）：

| 方法 | 测试结果 | 记录 |
|---|---|---|
| 历史特征树残差 | 29.5410 / 17.6899 / 11.9485 | experiments/PEMSD4/history_calibration_rmse/results.json |
| 冻结主模型的神经残差头 | 29.8722 / 17.8378 / 11.8788 | experiments/PEMSD4/residual_head_rmse/results.json |
| 逐节点空间滞后岭回归 | 29.5766 / 17.7053 / 11.9988 | experiments/PEMSD4/spatial_ridge_rmse/results.json |
| 全路网历史残差岭回归 | 29.6181 / 17.6938 / 11.9172 | experiments/PEMSD4/network_residual_rmse/results.json |

所有方法的参数/强度按验证集选定；以上表格是各自运行的结果记录，不以测试集反选超参数或混合权重。保存了校准参数及完整预测。历史特征树将验证段分为前半拟合、间隔12步、后半选择，再在完整验证集重拟合。各节点/预测步的补偿参数也仅按验证集 RMSE 选择。

进一步新增独立的时空注意力教师 `model/AuxiliaryTeacher.py`：12步三通道已知历史，3层时间注意力和3层空间注意力，直接输出未来12步，采用MSE训练并按验证集RMSE选择检查点。该教师不接收未来流量作为输入。训练脚本 `tools/train_auxiliary_teacher.py`，输出目录 experiments/PEMSD4/auxiliary_teacher_rmse，日志 logs/huiyi_auxiliary_teacher.out。最多100轮、早停15轮、batch_size32、seed10、AdamW lr0.001、lr在40/60/80轮乘0.3。

训练结束自动执行三模型融合（原创新模型、新教师、原预训练教师）和共享节点校准。每个预测步的原创新模型权重至少0.25；融合在验证前半选择，校准超参数在验证后半选择，再全验证集拟合校准，最后才评估测试集。原创新检查点通过哈希确认未修改。结果输出 results.json、test_predictions.npz、fusion_calibration.npz、best_teacher.pth、config.json；未完成前不宣称达到28.x。

验证：`python -m unittest discover -s tests -p test_residual_calibration.py`。覆盖历史特征对未来流量不变性、多步/节点排列、残差头零初始化保持原预测、直接教师形状及有限梯度。

教师训练后还已排队执行 `tools/refine_auxiliary_results.py`：将三模型融合等价转换为历史校准器输入，在固定教师预测上再次验证历史残差补偿。融合等价性已加入测试；主模型不会被替换，原创新模型在初始融合中至少占0.25。最终方案通过两阶段的验证集RMSE比较决定，写入 auxiliary_teacher_rmse/recommended_result.json，不按测试分数反选方案。自动后续日志 logs/huiyi_auxiliary_teacher_refinement.out。

首轮实测教师每轮约43秒，100轮约72分钟；提前停止时更短，校准另需几分钟。进程均使用独立会话运行，SSH断开不影响。最终结果生成前，28.x仅为优化目标。
