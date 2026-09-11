# SLZ residual integration

Stable Git baseline: `ae0f31e514eb4cc53e031a3a3320c1a09bad1b4a`.

This sibling repository reuses `../RLinf/.venv`; it does not modify that environment
or the original repository. The original tracked working-tree patch and overlapping
untracked files are archived in `.git/slz-migration`.

PPO actor, losses, advantages, FSDP, buffering, checkpoint handling, environment
controllers and CNN implementation are the stable baseline. Only model/environment
registration, residual observations, rollout dispatch and OpenPI PnP dual-camera
adapters are added. No KL early-stop or alternate epoch shuffle is retained.

The environment emits padded 224x224 images and binary gripper state. The native
CNN resizes its own input to 128x128; OpenPI receives 224x224. Camera constants and
backgrounds match the local co-training assets. Relative safety limits are injected
through `env.sim_real_alignment`, using the stable merge order. Success-only rewards
use the stable task's dense callback with all intermediate reward coefficients zero.

The main configuration preserves the colleague's latest 32-env, four-update-epoch,
actor-lr=2e-5, logstd=-1 setup. It places actors on physical GPU 2 and sampling on
GPUs 2-3. Check availability before running. The original 0-1 job is independent.

```bash
cd /home/shiliangzhi/work-space/wangyinghan/RLinf-slz
source pre_start_ray.sh
# Attach to an independently started Ray cluster for this repository.
export RAY_ADDRESS=101.6.96.181:6397
bash examples/embodiment/run_maniskill_residual.sh maniskill_pick_and_place_ppo_residual_slz_smoke
```

The smoke configuration runs two complete rollout/update rounds, saving a checkpoint
and training video each round. Passing it establishes pipeline compatibility, not
learning performance. Old continuous-gripper-state/128-environment-image residual
checkpoints have different input semantics; do not treat them as aligned checkpoints.

Logstd handling follows the stable CNN/FSDP implementation (forward clamp only),
not the colleague's optimizer projection extension. Keep that deliberate baseline
difference separate from later numerical experiments.

## 迁移与检查记录（2026-09-09）

- 分支 `slz-residual` 从本地 Git bundle 的 `ae0f31e` 建立，不是逐文件选择性覆盖旧版本。
- 新增文件使用 `git add -N` 纳入 `git diff` 展示；没有 commit 或 push。
- `rlinf/workers/actor`、`hybrid_engines`、`algorithms`、`workers/env`、
  `runners`、`scheduler`、原生 CNN 和 digital twin 任务代码均与该 Git 基线一致。
- 只保留 residual 模型/环境、base action cache、数据字段和模型注册，以及 OpenPI PnP 适配。
- 没有保留旧版 KL early-stop、独立 epoch shuffle、单卡 FSDP 绕过或 optimizer logstd 投影。
- 环境采用本地最新相机、224 图像、二值夹爪 state、alignment 合并顺序和相对安全范围。
  CNN 的 128 resize 是基线已有功能，没有重新实现一套。
- `assets/` 先随原目录复制，再用本地最新 `co-training-assets/digital_twin` 更新。
  25 个本地资源逐一核对 SHA256 全部一致，额外旧物体资源保留。
  当前背景为 `thirdview_background.png`，旧背景保留为
  `thirdview_background_pre_20260904_calibration.png`。
- 没有复制旧 `.venv`、日志或 checkpoint；没有安装或改动共享 Python 环境。
  OpenPI checkpoint 和 ResNet 预训练权重继续读取 YAML 中的已有外部绝对路径。

当前独立 Ray 地址为 `101.6.96.181:6397`。需要重新建立时，在确认端口及 2、3 号卡空闲后使用：

```bash
cd /home/shiliangzhi/work-space/wangyinghan/RLinf-slz
source pre_start_ray.sh
ray start --head --node-ip-address=101.6.96.181 --port=6397 \
  --num-gpus=4 --num-cpus=8 --object-store-memory=4294967296 \
  --temp-dir=/tmp/rlinf-slz-residual-20260909 \
  --min-worker-port=24000 --max-worker-port=24999 \
  --dashboard-agent-listen-port=52386 --include-dashboard=false --disable-usage-stats
export RAY_ADDRESS=101.6.96.181:6397
bash examples/embodiment/run_maniskill_residual.sh maniskill_pick_and_place_ppo_residual_slz_smoke
```

不要在有其他实验的 a4 上使用全局 `ray stop`。该命令会影响别人的 Ray 集群。
正式配置为 `maniskill_pick_and_place_ppo_residual_a4`，这里只验证冒烟配置，未启动长训练。

### 已完成的两轮训练

日志：`logs/20260909-144731-maniskill_pick_and_place_ppo_residual_slz_smoke/`。

| 指标 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| train success rate | 0.28125 | 0.25 |
| proximal ratio | 1.001366 | 1.001045 |
| proximal approx KL | 0.001482 | 0.000517 |
| explained variance | 0.251587 | 0.534844 |
| entropy | 2.512824 | 2.511752 |
| 累计 optimizer steps | 60 | 120 |

两份 full_weights.pt 的 111 个 tensor 均有限；两个 worker 每轮的视频均已写入且可解码，
共 4 个 MP4。3 个 CPU 单元测试通过。成功率只用于验证能运行，不能据此判断学习趋势。

这次两轮训练结束后，shell 因执行期间更新了启动脚本的默认配置名而收尾报 EOF。
最终脚本已经通过 `bash -n`；另补跑一轮验证完整命令的退出状态，结果在下方记录。

补跑日志：`logs/20260909-145338-maniskill_pick_and_place_ppo_residual_slz_smoke/`。
使用相同冒烟配置，仅覆盖 `runner.max_epochs=1`。完整命令正常退出（exit code 0），
完成 60 次更新，success rate 0.28125，ratio 约 0.997，KL 约 0.0049，
explained variance 约 0.292。checkpoint 和两个 worker 的视频再次正常写出。
两次测试结束后 GPU worker 均退出，原始目录上的训练未停止。独立 Ray 6397 保留供后续使用。

## 100 轮验证实验

2026-09-09 15:09（a4 本地时间）在 tmux `residual-slz-s04-std0` 启动。
配置：`maniskill_pick_and_place_ppo_residual_slz_s04_std0_n32_ue4.yaml`。
日志：`logs/20260909-150906-maniskill_pick_and_place_ppo_residual_slz_s04_std0_n32_ue4/`。

- 仅使用物理 GPU 2、3，actor 在 2；0、1 上原实验不动。
- 新初始化 residual（不续跑冒烟权重），OpenPI checkpoint 保持不变。
- residual scale 为 `[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0]`，夹爪仍由 OpenPI 控制。
- 初始 logstd 0，范围 `[-3, 0]`；32 环境，每轮 120 步；update epoch 4。
- actor lr `2e-5`，global batch 256，micro batch 128，与原 0、1 实验的已保存配置一致。
- checkpoint 每 50 轮保存，train 视频每 10 轮保存；eval 每 50 轮进行。
- 总计 100 轮，按冒烟速度估计约 2～3 小时，实际时间以训练日志为准。

```bash
tmux attach -t residual-slz-s04-std0
```

首次 15:04 启动使用继承的 eval=20，因 runner 要求 save_interval 必须整除
val_check_interval 而在首轮后退出。改为 eval=50 后重新从头启动，不接续失败启动的权重。
最终配置已对 step 1、50、100 的 `check_progress` 调用做过检查。

## 关闭 entropy bonus 的新实验

按用户要求停止上述 std0 实验，保留日志和第 50 轮 checkpoint，不续跑它的权重。
新配置：`maniskill_pick_and_place_ppo_residual_slz_s04_std_m05_ent0_n32_ue4.yaml`。
tmux：`residual-slz-m05-ent0`。

仅在新 YAML 中覆盖已有的 `algorithm.entropy_bonus: 0.0`，以及
`initial_logstd: -0.5`、`logstd_range: [-4.0, 0.0]`。
基础配置仍为 `entropy_bonus: 0.001`，没有修改公共 PPO/CNN 代码或既有 CNN 实验配置。
其余设置继承上一版：GPU 2、3，100 轮，32 env，update epoch 4，batch 256/128，
actor lr 2e-5，scale 六维 0.4、夹爪 0，视频每 10 轮，checkpoint/eval 每 50 轮。

配置组合和第 1/50/100 轮调度检查通过；调用真实 CNN `_action_std` 的 CPU 梯度检查
确认初始 std 约 0.60653，原始 logstd 可以收到梯度。没有加入 optimizer 参数投影，
因此这只避免从边界开始训练，不保证未来不会再次越界。

现有 actor 在 entropy_bonus=0 时不计算 entropy，日志里的 `actor/entropy_loss=0`
是占位值，不代表真实策略熵。需要从 checkpoint 的 logstd 或单独诊断观察噪声变化。
