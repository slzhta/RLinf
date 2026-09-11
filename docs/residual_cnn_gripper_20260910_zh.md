# 六维 residual + 独立 CNN 夹爪：BC → 联合 PPO

## 设计

最终动作前六维为 `clip(OpenPI[:6] + scale[:6] * clip(residual[:6], -1, 1), -1, 1)`；
第七维为夹爪 CNN 的独立绝对命令，`-1=合，+1=开`。OpenPI 仍保留原 checkpoint 的七维输出接口，
但第七维不参与执行，也在 residual/value 网络的 base plan 条件中置零。

夹爪使用独立的小 CNN（共享相机间的卷积编码器、无 dropout/BatchNorm），输入主相机、腕部相机和
14 维本体状态，输出一个 Bernoulli logit。它不接收 OpenPI 的动作，不与 residual 共用可训练编码器。
BC 使用原始 BCE logits，无类别重加权；PPO 时训练采样 Bernoulli，评估使用 `logit >= 0` 的确定性命令。
六维 Gaussian 的 log-prob 与一维 Bernoulli 的 log-prob 求和，使用现有 joint/chunk PPO ratio 与优势。
两路参数均包含在现有 actor AdamW 中，BC 后的 CNN 不冻结。联合 checkpoint/权重同步包含 `gripper.*`。
本实现的第二阶段是 PPO 微调，没有附加在线 BC/replay loss。

## 阶段一：夹爪 BC

现已支持直接使用当前 OpenPI SFT 的混合数据集
`pnp_lerobot_sim500_real250_success_20260905_v1`（500 仿真 + 250 真机成功轨迹，
合计 750 条、74,556 帧，双相机 224×224，meta fps=10）。
只读取原始 `actions[:,6]`，不套用 OpenPI norm_stats，也不做 DeltaActions 或 action chunk 时移。
仍使用双相机与 14 维原始状态；OpenPI SFT 的 nostate 设置不要求独立 BC CNN 也丢弃状态。

在 a4 仓库根目录使用现有 `.venv`。该入口不启动 Ray、仿真器或 OpenPI，默认 CPU；
只有明确选择空闲设备时才传 `--device cuda:N`，避免干扰现有 GPU 实验。

```bash
cd /home/shiliangzhi/work-space/wangyinghan/RLinf
.venv/bin/python -m toolkits.residual.train_gripper_bc \
  --data-root /data/shiliangzhi/work-space/co-training/postprocess/pnp_lerobot_sim500_real250_success_20260905_v1 \
  --data-format lerobot \
  --output-dir outputs/gripper_bc_sft_mixed_v1 \
  --image-keys image wrist_image \
  --state-key state --state-dim 14 \
  --action-key actions --action-encoding signed \
  --epochs 30 --batch-size 128 --device cpu
```

输出目录必须不存在，防止覆盖旧训练。数据按整条 episode 固定随机划分 90% 训练、10% 验证，
保存完整 manifest。输出 `best.pt`（按验证 balanced accuracy 选取）、`last.pt`、`metrics.jsonl`。
记录 BCE、总准确率、开/合召回准确率、切换点前后两帧准确率与样本数。
BC 标签必须来自同一时刻的专家命令 `action[t, 6]`，不能用下一帧夹爪位置代替。

输入为 `episode_*.npz`，默认字段为已检查的 `base_images/wrist_images: [T,H,W,3] uint8`、
`states: [T,14]`、`actions: [T,7]`。可指定字段名；signed 只接受 -1/+1，zero_one 只接受 0/1，
两种编码都规定较大的值表示开。标签不符合约定立即报错。

以上 NPZ 是兼容的旧入口；默认 `--data-format auto` 根据 `meta/info.json` 判断 LeRobot/NPZ。
LeRobot 支持当前使用的 v2.x 每轨迹一个 Parquet 文件、图像内嵌 bytes 格式，默认字段为
`image`、`wrist_image`、`state`、`actions`。不支持 v3 多轨迹文件或外部视频；会明确拒绝。
训练遍历原始帧，不自动将真机轨迹重复扩增或改为 1:1 域采样。

这批旧数据的 metadata 标记 control_freq=5，而当前 PPO 配置是 10；相机图像为 128×128。
代码不假装这种数据分布差异已经解决。正式训练建议优先使用与当前仿真相机裁剪、控制频率、状态定义一致的
成功示范，并在 BC 后运行 `OpenPI 六维 + BC 夹爪` 的确定性基线验证动作时序。
两阶段 CNN 都按相同逻辑缩放 RGB 至 96×96；这不能自动修复视野/裁剪差异。

小规模检查可额外传 `--max-episodes 4 --epochs 1 --max-train-batches 2`，使用独立输出目录；
这种 checkpoint 仅用于接口测试，不能当作训练完成的夹爪。

## 阶段二：联合 PPO

BC 使用两张卡时，在上述 BC 命令上添加 `--device cuda:0 --data-parallel`，并用
`CUDA_VISIBLE_DEVICES=0,1` 限定设备。采用单进程 DataParallel，训练同一个夹爪模型；
`--batch-size 128` 是两卡合计的 global batch，每卡通常 64。保存的 BC checkpoint 不带
`module.` 前缀，与后续 residual PPO 加载兼容。每 50 次更新记录一次进度，可用 `--log-every` 调整。
100 epoch 是初始训练预算；最终使用验证集最优 `best.pt`，不假定最后一步最好。

在已经正确选择 Ray/设备环境的 shell 中使用现有启动脚本；不要为此停止当前正在运行的实验。

```bash
bash examples/embodiment/run_maniskill_residual.sh \
  maniskill_pick_and_place_ppo_residual_cnn_gripper_a4 \
  actor.model.gripper_checkpoint=/absolute/path/to/gripper_bc_sft_mixed_v1/best.pt
```

新配置使用原实验的六维 residual scale=0.4、actor lr=2e-5 与稀疏成功奖励；夹爪 residual scale=0，
第七维由环境显式替换。训练视频每 20 轮记录一次并关闭文字叠加。
`gripper_checkpoint` 是必填项；BC 输入配置与联合策略不匹配会在加载时拒绝。
不支持从旧七维 Gaussian 模型直接 resume。新联合模型可按原有 `runner.resume_dir` 机制恢复；
如保留 `gripper_checkpoint` 字段，该初始化文件仍需可访问，之后完整 checkpoint 会覆盖其权重。

先做 BC 基线（仅评估，前六维 residual 置零，夹爪仍为 BC CNN）：

```bash
bash examples/embodiment/run_maniskill_residual.sh \
  maniskill_pick_and_place_ppo_residual_cnn_gripper_a4 \
  actor.model.gripper_checkpoint=/absolute/path/to/gripper_bc_sft_mixed_v1/best.pt \
  runner.only_eval=true actor.model.enabled=false
```

夹爪 PPO 探索是 Bernoulli，不再是高斯噪声，但仍可能出现采样错误/抖动。
本版未添加滞回或连续多帧确认，避免实际执行动作与 PPO 保存的采样动作不一致。
BC 精度和确定性成功率通过后，再比较联合 PPO 的 train/eval success、drop、KL 和 clip fraction。

## 验证

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 .venv/bin/python -m pytest -q \
  tests/unit_tests/test_split_gripper_policy.py \
  tests/unit_tests/test_residual_policy.py \
  tests/unit_tests/test_residual_gripper_additive.py
```

覆盖 BC 权重和完整联合权重加载、原始第七维隔离、绝对夹爪执行、rollout/update 概率一致性、
真实 PPO loss 对 residual/std/夹爪编码器和头的梯度与参数更新、数据标签与配置校验。

## 2026-09-10 implementation verification

CPU verification on a4: all 6 new tests passed; the combined suite had 43 passes and one pre-existing failure. The legacy test expects gripper initial_logstd=-1.5, while the existing experiment uses -1.3; the same failure was reproduced in the unmodified repository.

A real-data smoke test used 4 NPZ episodes, 1 epoch and 2 training batches on CPU. BC export -> model factory -> joint PPO log-probability replay -> actor optimizer membership all passed. The gripper has 173,313 parameters. The smoke checkpoint is not a trained policy. No full BC run, GPU simulation evaluation or multi-worker PPO job was launched.

## Mixed SFT dataset adapter verification

All 750 episodes / 74,556 frames passed raw binary gripper label validation. Embedded RGB images and 14-D states were decoded from the actual mixed LeRobot dataset. Eight related tests passed, including NPZ/Parquet frame and label equivalence, rejection of normalized labels, and joint PPO regressions. A CPU smoke run used 6 episodes and 2 BC batches; no full BC training was started.

## Two-GPU verification

GPU 0/1 tests passed: global-batch gradient equivalence (TF32 disabled for the strict reference test), portable BC checkpoint export, and NPZ/Parquet compatibility. A real-data two-GPU smoke run completed two optimizer updates and held-out evaluation.

## CUDA rollout device handling

The parent CNN policy stores replay observations on CPU. The split gripper rollout
moves a separate observation dictionary to the gripper device and returns commands
and log-probabilities on their parent tensors' devices. Replay images stay on CPU.
A CUDA regression covers real rollout, replay log-probabilities, and gradients
through both the gripper encoder and arm standard deviation.

## YAML-controlled launch environment

The split-gripper YAML now includes a launch block with GPU 0/1, Ray address
101.6.96.181:6411, node rank, communication interface, thread limits, and graphics
environment cleanup. run_maniskill_residual.sh reads this block before importing
CUDA/Ray and selects the repository .venv automatically. The Ray address refers
to the existing dedicated cluster; the script does not start or stop clusters.

From the repository root, use:

    bash examples/embodiment/run_maniskill_residual.sh maniskill_pick_and_place_ppo_residual_cnn_gripper_a4

No separate activation or export commands are required. The YAML contains the
selected epoch-9 best.pt path and CNN gripper settings. Hydra command-line
overrides for launch.env_vars are supported. Configurations without a launch
block preserve the caller's environment. Shell syntax, settings extraction,
override handling, and compatibility with the old config were checked.

## Gripper output aligned with the seven-axis CNN

The previous pick-and-place CNN configurations use binary_action_indices=[6].
Their gripper is Bernoulli(logits / binary_action_temperature), sampled in train
mode and thresholded at raw logit >= 0 in eval mode, then mapped to -1/+1.
The independent BC gripper now honors the same actor.model.binary_action_temperature
in both rollout and PPO replay, including entropy. The YAML explicitly sets 1.0.
At 1.0 this preserves the previous split-gripper behavior and all BC checkpoints.
Only six arm residuals use Gaussian standard deviations; the gripper replaces the
OpenPI seventh dimension. The CNN outputs a Bernoulli logit, not an opening width.
Lower temperatures sharpen the distribution; they do not add hysteresis.
CPU tests compare replay probabilities and entropy with the legacy seven-axis
CNN at temperatures 0.5, 1, and 2, with eval and gradient checks.
