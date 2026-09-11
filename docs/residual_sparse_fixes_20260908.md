# a4 residual PPO：成功奖励、bootstrap 和 logstd 修复

本次修改仅针对 a4 仓库，保留现有 10 Hz 控制频率、32 个训练环境、
global batch 256 和 residual scale `[0.5, 0.5, 0.5, 0.1, 0.1, 0.5, 0]`。

## 纯 sparse reward

Residual PnP 的 `init_params.reward_mode` 使用 `sparse`，走独立的
`compute_sparse_reward`，最终成功奖励为 1，其余情况为 0。
抓取、抬升、放置、掉落仍可作为指标记录，但不参与奖励。
环境包装层使用 `reward_mode: raw` 和 `use_rel_reward: false`。

启动校验和 residual 环境构造会拒绝 dense/normalized_dense、阶段奖惩、
相对奖励以及忽略终止等配置；residual 启动校验也禁止外部奖励模型。
共享 PnP 环境的旧 dense 方法保留供其他实验使用，residual 不允许进入该路径。
成功导致 episode 终止，自动重置后新 episode 重新计奖。

## Bootstrap

- `none`：不往终止/超时的环境奖励中添加 `gamma * V(final_obs)`。
- `standard`：只对超时且没有真正终止的样本补值。
- `always`：保留旧实验显式要求的语义，对所有 done 样本补值。
- 其他值报错，不再隐式当成 `always`。

`none` 不会禁用 GAE，也不会把一个仍在继续的 episode 的 rollout 切片末尾
当作终止。未结束的轨迹仍使用正常的 next value；真正终止/超时处由 done 截断。
价值估计和 entropy 正则不是环境 dense reward。

## Logstd

CNN 的 independent logstd 参数在每次 FSDP 优化器更新后执行原地投影，
保持在 `logstd_range` 内。前向仍保留 clamp，优化算法和高斯概率定义不变。
加载旧权重或恢复 FSDP checkpoint 时同样投影，防止历史越界值持续零梯度。
该方法不重置 Adam 状态，也不改变高斯概率定义。
按用户要求 initial_logstd 保持 0，初始标准差保持 1；
entropy_bonus 保持 0.001。扩大 alpha 带来的更强初始探索仍然保留。

## 与近期 CNN 配置的比较

参考 `/data/shiliangzhi/work-space/co-training/RLinf/results/` 中
`pnp_clean_bc_n32_rl500_ue4_lr1e5_20260906` 与
`pnp_first_update_n32_ue2_20260907_capture` 的启动配置：

| 参数 | CNN 参考实验 | 本次 residual |
|---|---|---|
| train envs | 32 | 32 |
| sim/control frequency | 500 / 10 Hz | 500 / 10 Hz |
| bootstrap | none | none（修复语义） |
| gamma / gae_lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| initial_logstd | -1 | 0（按用户要求保留） |
| global / micro batch | 240 / 120 | 256 / 128 |
| actor / value LR | 1e-5 / 1e-4 | 1e-4 / 2e-4 |
| update_epoch | 4 / 2 | 4 |
| entropy_bonus | 0.002 | 0.001 |
| critic_warmup_steps | 5 | 0 |
| use_state | true | false |

每轮 32×120=3840 条 transition，3840/256=15 个 minibatch，
每个 minibatch 两个 microbatch，四次完整更新共 60 次 optimizer step。
不因 CNN 的 batch=240 就修改合法的 residual batch=256。
参考 CNN 已有 BC 初始化；residual 的动作均值头从零开始，故本次不照搬
其更小学习率及 warmup。是否调低学习率/增加 warmup 仍应由新奖励下的 KL、
critic 和成功率实验判断。

state 输入、夹爪控制权和 OpenPI 编译/轻量推理不属于本次代码语义修复：
state 开关会改变网络输入/权重兼容性，夹爪保持用户指定的 alpha=0，
OpenPI 优化需要独立性能及动作一致性测试。本次没有擅自开启这些功能。
视频仍为每 20 轮评估时保存；训练视频关闭。

## 验证与生效

回归测试覆盖三个 bootstrap 模式、真正终止与超时重合、none 的 GAE 边界、
纯成功奖励、dense/阶段奖励配置拒绝，以及实际 optimizer step 后 logstd
上下界投影和向区间内部恢复梯度、旧权重加载修复。

运行中的 Ray worker 已加载旧 Python 代码和配置，修改文件不会热更新它们。
需要重新启动训练进程及其 worker 才能使用新逻辑。阶段奖励变成纯成功奖励后，
旧 critic 的价值尺度不再相同，不能把旧训练曲线直接接成相同实验。
