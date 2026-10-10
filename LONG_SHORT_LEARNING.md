# 长短期协同的多尺度时空学习

实现位于 `newininvation-4`，基于 `innovation-1-2-3`。原分支的未提交工作保存在 Git stash 中，没有提交，也没有混入本次实现。

## 模型与训练

原 PDG2Seq 的 DGQ、周期上下文、上下文图细化和扩散/内生信号解耦编码器继续使用。通过 `--use_long_short_learning true` 启用新路径；不开启时仍使用原预测路径。

1. **长历史掩码预训练**：一周历史（2016 个五分钟时间步），12 步一个 patch。掩码覆盖节点、连续时间段及每天重复的周期片段。物理道路图和可学习图进行空间交互，Transformer 学习时间表示，重建头只在被遮挡的有效位置计算损失。
2. **长期记忆**：提取最近片段、最近一天均值、整周均值和日内波动表示。`--phase_memory true` 另外保留过去七天同一时段的两个 patch，供预测时检索。历史不足的位置有明确可用性标记。
3. **当前多尺度分解**：最近 12 步分解为短期变化、中间频带和趋势，三个分量相加可恢复原序列。近期时间注意力、节点空间注意力与原图编码器表示共同构成当前状态。
4. **长短期协同及 horizon 融合**：当前状态和预测步长作为 query，检索预训练长期记忆；协同门控制长期信息强度。每个节点、样本、预测步长分别学习 short / trend / periodic / long 四个尺度的融合权重。输出头一次生成未来 12 步，不递归反馈上一步预测。

正式预测阶段冻结原编码器及预训练长期编码器，训练新预测头。损失包含 MAE、MSE、非零流量位置的 MAPE 和既有教师预测蒸馏。原自回归解码器参数保留；新路径用已观测的前一天、前一周参考传递解码端周期信息，用直接预测头输出多个步长。既有周期一致性配置保留，但本次新头训练没有额外执行原 Trainer 的周期一致性损失。

## 数据与评测口径

PeMS04 按时间划分；归一化只使用训练区间。预测训练/验证/测试分别为 10171 / 3383 / 3393 个窗口，划分边界剔除了目标区间重叠窗口。预训练验证也在预测训练区间内，不使用预测验证或测试区间。模型输入包括预测起点前的交通数据和已知时间日历，目标流量输入始终为零。

9 月 25 日平均指标：RMSE **28.1697**、MAE **17.0095**、MAPE **11.4857%**。增益为 `(基线误差 - 本次误差) / 基线误差 × 100%`；三项同时达到 1% 时，对应阈值为 27.888003、16.839405、11.370843%。

用户指定的历史对比采用 **未来标签辅助口径**：输出一个窗口后，立即用该窗口完整未来标签更新后续预测的 bias/scale。更新参数固定为 9 月 25 日设置，不直接复制真值。该口径不是可部署的因果预测性能。报告同时保存新模型直接预测结果、无标签更新的融合结果，以及延迟 12 步更新的因果对照结果。

可选的既有教师融合在验证前 2/3 拟合，在验证后 1/3 选择，随后在完整验证集重拟合一次。系数在测试前固定，测试不用于本轮权重和融合系数选择。前轮测试结果已经查看，后续在同一测试划分报告的结果不是全新的盲测。验证集指纹用于确认评测复用的是已固定的融合参数。教师融合属于整个评测流程，不能把所有流程增益都归因于新模块；单独证明新创新点的贡献还需要同口径消融实验。

## 复现本次训练

使用安装了 CUDA 版 PyTorch 的环境；本服务器为 `/root/miniconda3/envs/PDG2SEQ_CU128/bin/python`。原编码器初始化检查点见 `tools/long_short_common.py` 的 `SIGNAL_CHECKPOINT`，所有原分支模块的参数必须匹配，否则中止。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/root/miniconda3/envs/PDG2SEQ_CU128/bin/python

# 自动完成 6 轮掩码预训练，再训练基础直接预测头 40 轮。
$PYTHON -u run.py --use_long_short_learning true \
  --output_dir experiments/PEMSD4/newininvation4_long_short \
  --attention_layers 0 --epochs 40 --no_test

# 添加近期时间/空间注意力，复用同一份冻结特征，继续训练 50 轮。
$PYTHON -u run.py --use_long_short_learning true \
  --output_dir experiments/PEMSD4/newininvation4_long_short_attention \
  --pretrained_prior experiments/PEMSD4/newininvation4_long_short/pretraining/long_prior.pth \
  --context_cache experiments/PEMSD4/newininvation4_long_short/frozen_context.pt \
  --warmstart_forecaster experiments/PEMSD4/newininvation4_long_short/best_model.pth \
  --attention_layers 2 --epochs 50 --lr 0.0005 --no_test

# 添加每天同一时段的长期记忆，继续训练最多 40 轮。
$PYTHON tools/start_long_short_background.py \
  --logfile experiments/PEMSD4/newininvation4_long_short_phase/train.log -- \
  --output_dir experiments/PEMSD4/newininvation4_long_short_phase \
  --pretrained_prior experiments/PEMSD4/newininvation4_long_short/pretraining/long_prior.pth \
  --context_cache experiments/PEMSD4/newininvation4_long_short/frozen_context.pt \
  --warmstart_forecaster experiments/PEMSD4/newininvation4_long_short_attention/best_model.pth \
  --attention_layers 2 --phase_memory true --epochs 40 --lr 0.0005 --no_test
```

新训练目录不得包含已有 `best_model.pth`，避免覆盖实验。中断后用原参数加 `--resume` 恢复最近完成的一轮，恢复预测头、优化器、学习率调度和随机数状态。本次转后台时从已完成的第 2 轮恢复；早期快照尚未包含 dropout 随机数状态，这次恢复重新初始化了该随机流。后续快照包含 CPU/CUDA 随机数状态。

```bash
tail -f experiments/PEMSD4/newininvation4_long_short_phase/train.log

# 仅验证集选择和保存融合系数。
$PYTHON tools/select_long_short_fusion.py --model_dir <训练目录>

# 加载验证集选出的权重，进行完整测试和生成报告。
$PYTHON run.py --use_long_short_learning true --mode test --output_dir <训练目录>

# 验证数据边界、掩码防泄漏、直接多步路径、长期记忆和历史评测复现。
$PYTHON -m unittest discover -s tests -p 'test_*.py'
```

每个实验目录保存配置、预训练权重（基础目录）、最佳模型、断点状态、逐轮验证指标、每个 horizon 的四尺度平均权重、验证融合选择记录及测试 `report.json`。只有测试报告的 `target_met: true` 才表示三项平均指标均达到至少 1% 增益；训练完成不等于目标达成。

## 40 轮后的延长训练

周期记忆阶段完成 40 轮后，最佳验证权重为第 33 轮。阶段快照保存在该实验目录的 `completed_40_epochs/`。继续训练从第 40 轮的最新状态恢复，将该阶段总上限改为 200 轮，早停耐心值为 20；保留 AdamW 动量，学习率从 0.0002 开始，在剩余 160 轮内按余弦衰减到 0.000006。新增参数 `--restart_lr_schedule` 只在这次延长启动时使用，后续普通断点恢复不要重复添加，避免再次重置学习率。

延长训练日志：`experiments/PEMSD4/newininvation4_long_short_phase/train_to_200.log`。启动参数与周期记忆阶段相同，替换为 `--epochs 200 --patience 20 --lr 0.0002 --resume --restart_lr_schedule`。这表示该阶段从 41 轮继续到最多 200 轮，并非额外再训练 200 轮；验证无改善也可能提前停止。仍使用 `--no_test`，结束后单独处理最终评测。

## 根据前 40 轮改进训练策略

用户要求停止延长训练后，保留原目录所有完成轮次的断点，另建 `experiments/PEMSD4/newininvation4_long_short_balanced/`。以 `completed_40_epochs/best_model.pth`（旧第 33 轮）初始化，不改变网络结构，优化器重新初始化。训练变化：MSE 权重 0.35→1.0、MAPE 权重 0.05→0.02、蒸馏权重 0.03→0.01；前端 horizon 权重由 1.5 逐渐降至末端 1.0，再归一化为均值 1；dropout 0.1→0.15、weight decay 0.001→0.003；新预测头的 EMA decay 为 0.995。

EMA 在每个训练 batch 后更新。验证和最佳模型使用 EMA 权重，断点同时保存实际训练权重、AdamW 状态、EMA 权重和随机数状态；两套权重不会混用。模型选择使用相对旧第 33 轮同口径验证指标的最差比值，加上小幅均值比值作为同分判据，防止一项改善掩盖其他指标退步。旧权重本身作为第 0 轮候选保留，避免新一轮全部退步时覆盖旧结果。

本轮上限 200 轮，早停耐心值 20，初始学习率 0.0002。冻结上下文和周期记忆复用旧缓存，掩码预训练无需重做。日志为新目录的 `train.log`，完整参数可从同目录 `train.log.pid.json` 和 `configuration.json` 查看，改进依据见 `optimization_plan.json`。这是新的训练策略试验，是否达到 1% 增益仍需验证及最终测试。

## 原编码器与新头联合微调

仅调整预测头的策略在新第 20 轮早停，没有超过初始化权重。下一轮保持所有创新模块和直接预测结构不变，使用 `--finetune_core true` 解冻原短期图编码器、节点嵌入和时间日历嵌入，与新多尺度预测头一起训练。长期掩码编码器继续冻结，未使用的旧自回归解码器不参与梯度更新。

本轮完整加载旧第 33 轮模型。原始短期编码器初始来源仍是 8 月 4 日；并非改用 9 月 25 日模型。短期表示在每次前向中重新计算，旧缓存中的 `core` 不再用于预测；冻结长期先验和既有教师仍可复用缓存。EMA 对整个模型更新，验证时短期编码器、新头及用于历史在线评测的节点图都来自同一份 EMA 权重。联合训练断点保存完整模型、EMA、优化器、调度和 RNG，避免恢复时退回初始编码器。

目录：`experiments/PEMSD4/newininvation4_long_short_joint/`。上限 200 轮，早停耐心 20，初始 head LR 0.0003、core LR 0.00003；损失 MSE 权重 0.6、MAPE 权重 0.05、蒸馏权重 0.02，各 horizon 等权，EMA decay 0.995。仍以旧第 33 轮同口径验证结果为参照进行综合选模，保留第 0 轮旧模型候选。日志为该目录 `train.log`，完整启动参数保存在 `train.log.pid.json`。联合微调更费时间，每轮耗时以日志为准。训练期间不评分测试集，完成后再进行固定口径评测。

## FP32 与近期训练样本采样

联合微调在第 47 轮早停，最佳为第 27 轮。其测试报告已经保存，仍未达到目标。后续继续保持网络和创新模块不变，先对第 27 轮权重作 FP32 验证探测（没有再次评分测试），同口径验证 RMSE / MAE / MAPE 为 29.260078 / 17.888248 / 10.777213%。

新目录 `experiments/PEMSD4/newininvation4_long_short_recent_fp32/` 完整加载第 27 轮模型，参数及激活使用 FP32（仍开启 TF32 矩阵加速）。原训练窗口按时间分为前 67% 和后 33%，后段采样权重为前段两倍；每轮有放回采样相同数量的窗口，全部索引限制在原训练划分。验证/测试划分、训练归一化和冻结长期预训练保持不变，没有把测试样本加入训练或验证。该前后段划分用于采样，不是独立盲测：初始化权重已经见过整个训练集。

训练目标为 MAE + 0.35 MSE + 0.25 RMSE + 0.05 MAPE + 0.05 教师蒸馏，RMSE 按加权 MSE 加 1e-8 后开方得到，防止零残差梯度非有限。初始 head LR 0.0002、core LR 0.00003，batch 32，EMA 0.995；上限 200 轮，早停耐心 20。模型选择参照初始化权重在相同 FP32 口径下的验证指标，旧权重仍作为第 0 轮候选保留。本轮运行 `--no_test`，日志为新目录 `train.log`，完整配置/来源/采样方式分别保存于 `configuration.json`、`optimization_plan.json`、`selection_reference.json`。21 项数据边界、梯度、损失、EMA 和采样测试通过。实际效果需训练结束后再评估。
