# 双视角 no-state SFT：5k / 10k 真机评估

仅准备 checkpoint 与启动入口；本次未启动 Ray 或机械臂。

## 模型来源

a4 实验：
`/data/wangyinghan/openpi_checkpoints/pi05_franka_pnp_mixed_nostate/pnp_sim500_real250_success_20260905_v1_nostate_20k_v1`

取其中 `5000`、`10000`，使用 a4 原有 `examples/convert_jax_model_to_pytorch.py`，
配置 `pi05_franka_pnp_mixed_nostate` 转为 BF16 PyTorch。
norm stats 直接来自各自 checkpoint 的 assets，未借用其他实验。

数据集：`pnp_lerobot_sim500_real250_success_20260905_v1`。
模型：双视角、`discrete_state_input=false`、10 步动作块。
checkpoint 根目录的 `conversion_manifest.json` 记录来源、step 和文件 SHA-256；
4090 的 `transfer_verified.json` 记录逐文件校验结果。

4090 权重目录：

- `pytorch_checkpoints/pi05_pnp_sim500_real250_success_20260905_nostate_5k`
- `pytorch_checkpoints/pi05_pnp_sim500_real250_success_20260905_nostate_10k`

以上路径均相对于 `/home/ubuntu/wangyinghan/wangyitao`。

## 启动

NUC 使用已迁移的 `/home/abc/wangyinghan/RLinf`，先在准备 Ray 的 shell 中执行：

```bash
cd /home/abc/wangyinghan/RLinf
source pre_start_ray.sh
```

沿用之前双机 Ray 拓扑：NUC rank=0，4090 rank=1。
脚本不自动启动、停止或重组 Ray；已经启动的旧 worker 需要另外重启以加载新代码。

两机 Ray 就绪后，在 4090 分别运行，每次只评估一个 checkpoint：

```bash
cd /home/ubuntu/wangyinghan/wangyitao
source openpi-pnp-venv/bin/activate
bash examples/embodiment/run_pi05_realworld_eval.sh realworld_pnp_nostate_5k_eval_50
```

```bash
cd /home/ubuntu/wangyinghan/wangyitao
source openpi-pnp-venv/bin/activate
bash examples/embodiment/run_pi05_realworld_eval.sh realworld_pnp_nostate_10k_eval_50
```

直接使用已有 `run_pi05_realworld_eval.sh` + YAML，不为每个 checkpoint 新建 shell 脚本。
启动前手动激活 `openpi-pnp-venv`，并在准备 Ray 时沿用 4090 rank=1 的环境设置。
共用入口调用 `eval_embodied_agent.py`，不走训练入口，不加载训练 actor。
人工反馈开关由 YAML 决定；本地共用入口已与 4090 对齐，移除强制设为 null 的覆盖。

短试运行可在命令后追加 `algorithm.eval_rollout_epoch=1`。
其他 Hydra 参数同样可追加，但比较两个 checkpoint 时应保持协议一致。

## 参数与人工反馈

配置从 4090 当前已对齐的 state 10k 评估 YAML 复制，仅改变模型路径、实验名及
`discrete_state_input=false`。原 state 配置未修改。

- 第三视角：233622071355，实验中的 wrist_1/main_image_key。
- 腕部视角：141722075170。
- 继续继承 10 Hz、use_relative_frame=false 和环境原有随机 reset。
- 环境仍提供 state 字段以满足原有 PnP 输入变换接口；no-state 由模型的
  discrete_state_input=false 控制，不应随意删除环境 state 字段。
- 保留 S/F/X/R 人工反馈包装器。R 应在 reset-ready 提示出现之后按下。

S 表示成功，F 表示失败，X 表示中止，R 表示下一轮准备完成。
沿用原评估指标输出；文件名中的 50 是 `eval_rollout_epoch=50`，
在 `auto_reset=true` 下不等于严格 50 个有效人工试验。
若需要严格统计 S/(S+F)，应记录 S/F 次数并单列 X；本次没有修改计数实现，
不能将 rollout 进度条直接作为成功率分母。

每次日志在 4090 的 `/tmp/<时间>-realworld_pnp_nostate_<5k或10k>_eval_50/`，
两份配置有独立 experiment_name，避免混淆结果。

## 验证边界

传输完成后逐文件核对 SHA-256，并检查 Hydra 合成参数、原有输入变换的 no-state 行为、
模型文件结构和脚本语法。上述检查不替代完整推理或真机试验。
