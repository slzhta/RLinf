# 七维连续 residual：独立夹爪初始噪声

目标：a4 `/home/shiliangzhi/work-space/wangyinghan/RLinf`；不修改 `RLinf-slz`。
配置名：`maniskill_pick_and_place_ppo_residual_gripper_a4`。

```yaml
residual_action_indices: [0, 1, 2, 3, 4, 5, 6]
residual_action_scale: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 2.0]
initial_logstd: [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1.5]
logstd_range: [-4.0, 0.0]
```

全部七维沿用 Gaussian residual，执行
`clip(base + scale * clip(raw_residual, -1, 1), -1, 1)`。
PPO 保存原始 Gaussian 动作及其概率，第七维参与联合 logprob 和更新。
前六维初始 std=0.60653；第七维初始 std=0.22313。
只改变初始化，不冻结标准差，也没有给夹爪额外设置上限；训练后概率会变化。
配置校验支持标量旧配置和长度恰为 action_dim 的逐维列表，逐项检查有限性与边界。
原生 CNN 和 residual 模型已有逐维初始化支持，不改变其采样或更新实现。

## 初始概率的含义

假设 OpenPI 命令为 -1、夹爪关闭，均值为零，误开要求 raw residual>=0.75。
初始单步误开概率约 0.0387929%；连续 120 步独立采样、base 始终要求保持关闭时，
至少一次误开命令概率约 4.54932%。打开时误闭对称，不把两侧概率相加。
这是命令概率，不是掉落或任务失败概率，也不是整个实际 rollout 的概率保证。

若当前打开、base=-1 过早要求闭合，raw residual>0.25 可以阻止当步闭合，
初始概率约 13.1%。保持区间不能打开一个已经关闭的夹爪。
base 相加前不裁剪，因此 base 超过 ±1 时应按实际值重新计算阈值。

## 运行与验证

其他参数保持不变：actor、env、rollout 均放置到 GPU 0、1；32 train env、
base horizon=10、逐步 residual、update_epoch=4、actor lr=2e-5、batch=256/128、
entropy_bonus=0、100 轮、每 50 轮保存/评估、每轮保存训练视频。
实验日志名称更新为 `residual_s04_gripper_s20_gstd_m15_ent0`。
从头初始化 residual，不续接六维输出头/optimizer checkpoint；OpenPI checkpoint 不变。

在 a4 的 RLinf 根目录，确认 GPU 和 Ray 集群加载的是该目录代码之后：

```bash
source pre_start_ray.sh
bash examples/embodiment/run_maniskill_residual.sh maniskill_pick_and_place_ppo_residual_gripper_a4
```

不要沿用加载 RLinf-slz 代码的 Ray worker。此修改不启动或停止训练/Ray。

CPU 测试：

```bash
PYTHONPATH=. USE_TF=0 TRANSFORMERS_NO_TF=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/unit_tests/test_residual_gripper_additive.py
```

覆盖配置展开、标量兼容、列表长度/数值校验、真实模型逐维 std、rollout/update
概率一致性、实际 PPO 第七维梯度、真实控制器阈值方法及解析概率。
TODO(agent): 仍需仿真训练验证成功率与闭合时机，CPU 测试不证明训练收益。

实验 YAML 已展开为完整配置，包含 base_model、cluster、runner、algorithm、env、rollout、actor、reward 和 critic；仅保留与原实验相同的标准 env/model/backend defaults。
本次展开沿用用户已修改的 `actor,env,rollout: 0,1`；唯一额外参数变化为训练
`record_rollout_interval: 10 -> 1`，其余 Hydra 完整解析结果一致。
视频原来在第 10、20……次 rollout 才开始录制，所以前几轮没有 MP4。
视频在 rollout 结束后异步写入 `runner.logger.log_path/video/train/seed_*/`。
通过启动脚本运行时 log_path 为本仓库 `logs/<时间>-<配置名>`。
已运行任务持有启动时配置：编辑 YAML 不会热更新它的 GPU 放置或录制频率。
此修改不打断正在训练的进程，新设置在下一次启动生效。
