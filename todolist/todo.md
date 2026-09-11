# PnP 真机 PPO 与 Sim–Real Co-Training TODO

## 目标

在现有 PnP 纯仿真 CNN PPO 已成功训练的基础上，按以下顺序完成：

1. 建立与仿真任务对齐的真机 PnP 环境。
2. 使用人工终点奖励完成真机 CNN PPO 训练，并观察到稳定成功率提升。
3. 参考已验证的 Push Button `paired` Async PPO 框架，实现 PnP Sim–Real Co-Training。

## 已确认的原则

- [x] 机械臂控制、相机、工作空间、compliance、Ray 拓扑等设置沿用已经验证的 Push Button 实现。
- [x] 不重新设计机械臂运动相关逻辑；PnP 只补充夹爪、方块、盘子、人工奖励和人工复位。
- [x] 不使用自动视觉奖励、RGB-D 奖励模型或自动物体状态判断。
- [x] 真机训练奖励由人工明确给出。
- [x] PPO 训练时关闭 SpaceMouse 动作干预，保证 on-policy；人工干预只能用于 BC 数据采集。
- [x] 不修改 OpenPI 相关代码。
- [x] 当前仿真 PPO `global_step_600` 与 Push Button 一样是单视角 CNN，可作为真机和 co-training 的候选初始化权重。

## 固定的策略接口

- [x] 环境固定采集两路 `128×128 RGB`：一个第三视角相机和一个腕部相机。
- [x] CNN policy 与 Push Button 保持一致，只使用单路主视角，设置 `image_num: 1`。
- [x] 与 Push Button 使用相同的物理主相机，并固定映射为 `main_images`。
- [x] 另一相机固定映射为 `extra_view_images[:, 0]`，用于录像、监控和数据留存，不进入 CNN policy。
- [x] 仿真和真机使用相同的主视角，禁止根据相机字典排序隐式决定主相机。
- [ ] 两路图像分别保存原始帧、策略输入帧和时间戳。
- [x] 沿用 Push Button 已验证的双相机采集和单视角策略设置，并显式记录第三视角/腕部相机的 serial、物理视角和主/辅映射。
- [x] 状态维度固定为 14：

  ```text
  [joint_qpos(7), tcp_xyz(3), tcp_euler_xyz(3), gripper_open_state(1)]
  ```

- [x] 动作维度固定为 7：

  ```text
  [dx, dy, dz, droll, dpitch, dyaw, gripper]
  ```

- [x] 夹爪语义与仿真一致：`-1=close`，`+1=open`。
- [x] 保持 `binary_action_indices: [6]`。
- [x] 保持当前仿真的 state normalization、action std scale 和 Bernoulli gripper 定义。
- [x] 真机状态使用显式字段顺序，禁止依赖字典键排序。

## 1. 新增真机 PnP 环境

目标文件：

```text
rlinf/envs/realworld/franka/tasks/pick_and_place_env.py
```

- [x] 新增 `FrankaPickAndPlaceConfig`。
- [x] 新增 `FrankaPickAndPlaceEnv`。
- [x] 继承 `FrankaCoTrainingBaseEnv`，不继承 `FrankaPushButtonEnv`。
- [ ] 复用 Push Button 已验证的以下最终运行参数：
  - `target_ee_pose`
  - target controller
  - 位置和旋转 action scale
  - compliance 参数
  - EE 工作空间限制
  - joint reset qpos
  - 第三视角相机与腕部相机的序列、裁剪、padding 和 128×128 输出
  - 控制频率和执行节奏
  - 安全退回和 rest pose 逻辑
- [x] 所有关键运动参数显式写入配置，不依赖默认值。
- [x] 初始夹爪状态为打开。
- [x] 输出固定顺序的 14 维策略状态。
- [x] 输出 7 维 action space。
- [x] 输出双相机 observation：Push Button 的主视角为 `main_images`，另一视角为 `extra_view_images`。
- [x] 确认 CNN 在 `image_num: 1` 下只编码 `main_images`，不误用 `extra_view_images`。
- [ ] 两路相机帧时间差必须被记录并限制在可接受范围内。
- [x] 设置 PnP task description。
- [ ] 增加相机失效、控制器异常和安全停止的处理。
- [x] 在 `rlinf/envs/realworld/franka/tasks/__init__.py` 中注册环境。
- [x] 增加对应 Gym 环境 ID，例如 `FrankaPickAndPlaceEnv-v1`。

### 环境继承关系

```text
仿真：
DigitalTwinBaseEnv
  └─ PushButtonEnv
      └─ PickAndPlaceDigitalTwinEnv

真机：
FrankaEnv
  └─ FrankaCoTrainingBaseEnv
      └─ FrankaPickAndPlaceEnv
```

## 2. 人工奖励状态机

建议新增独立 wrapper，避免改变现有 Push Button 和其他真机任务：

```text
HumanPnPRewardDoneWrapper
```

- [x] 借鉴现有 `reward_done_wrapper.py` 的键盘监听结构，但不原样复用现有语义。
- [x] 每个普通 step 默认 `reward=0`、`terminated=False`。
- [x] 支持以下四类人工事件：

  | 事件 | 建议按键/按钮 | 奖励 | 结束类型 | 是否训练 |
  |---|---|---:|---|---|
  | success | `S` / 绿色 | `+1` | terminated | 是 |
  | unrecoverable failure | `F` / 红色 | `0` | terminated | 是 |
  | safety/admin abort | `X` / 黄色 | `0` | truncated | 否 |
  | reset ready | `R` / 蓝色 | 无 | 开始下一局 | 否 |

- [ ] success 标准固定为：方块进入白色目标盘、夹爪释放、方块稳定。
- [ ] failure 只用于不可恢复状态，不能因为暂时动作不理想而提前判失败。
- [x] timeout 为零奖励 truncation。
- [x] abort trajectory 必须从 PPO loss 中排除，不能作为负样本。
- [x] 保留底层硬件 termination/truncation，人工反馈不能覆盖安全终止。
- [x] 每个 episode 的 success/failure 只能触发一次。
- [x] 对按键进行 debounce，防止重复反馈。
- [x] 提供声音或界面确认，确保人工知道反馈已被接收。
- [ ] 记录以下反馈元数据：
  - episode ID
  - step ID
  - feedback 类型
  - reward
  - terminated/truncated
  - 是否进入 PPO
  - 输入时间戳
  - 反馈延迟
  - 视频路径

### 状态机

```text
RUNNING
  ├─ success → TERMINATED
  ├─ failure → TERMINATED
  ├─ abort   → ABORTED
  └─ timeout → TRUNCATED

TERMINATED / ABORTED / TRUNCATED
  └─ robot retreat → WAITING_FOR_RESET

WAITING_FOR_RESET
  └─ ready → RUNNING
```

## 3. 人工物体复位

- [x] episode 结束后立即停止策略动作。
- [x] 打开夹爪。
- [x] 机械臂先竖直抬升，再使用 Push Button 已验证的安全路径返回 rest pose。
- [x] 环境进入 `WAITING_FOR_RESET`，禁止继续执行 policy action。
- [x] 人工将方块放回贴黄色胶带的源盘。
- [x] 人工按 `ready` 后才允许下一个 reset 完成。
- [x] reset 后重新采集相机和 proprioception，再开始 rollout。
- [x] reset 等待期间不得产生训练 transition。
- [ ] 单独记录 reset 耗时和 reset 失败次数。

### Reset case 设计

- [ ] 将源盘划分为中心、四角、四边和若干内部位置。
- [ ] 为方块定义多个 yaw。
- [ ] 生成训练 reset manifest，每轮界面显示待摆放的 case。
- [ ] 每个 episode 记录 `reset_case_id`。
- [ ] 真机和仿真使用统计一致的初始位置/yaw 分布。
- [ ] 固定评估 case 与训练 case 分离。

## 4. Sim–Real 奖励对齐

真机采用人工 terminal sparse reward，因此 co-training 的仿真分支也改为 terminal-only：

```yaml
sparse_grasp_reward: 0.0
sparse_lift_reward: 0.0
sparse_place_reward: 0.0
sparse_success_reward: 1.0
sparse_drop_penalty: 0.0
```

- [ ] 真机 success：人工 `+1` 并 terminate。
- [ ] 仿真 success：稳定放置并释放后 `+1` 并 terminate。
- [ ] 其他普通 step 均为 `0`。
- [ ] terminal-only 奖励切换本身不要求重做模型；PnP 继续采用与 Push Button 一致的 `image_num: 1`，因此当前 step600 架构保持兼容。
- [ ] 暂不增加人工 grasp/lift/place milestone，避免人工负担和反馈时延。

## 5. 新增配置

### 真机环境配置

目标文件：

```text
examples/embodiment/config/env/realworld_pick_and_place.yaml
```

- [x] 使用 `env_type: realworld`。
- [x] 使用 `FrankaPickAndPlaceEnv-v1`。
- [x] `total_num_envs: 1`。
- [x] `include_states_in_obs: true`。
- [x] 显式配置第三视角和腕部相机的语义名称、serial 和主/辅映射。
- [x] `main_image_key` 与 Push Button 使用相同的物理主相机。
- [x] 另一相机作为唯一的 extra view，仅用于录像、监控和数据留存。
- [x] `auto_reset: true`，但 `reset()` 内部等待人工 ready。
- [x] `ignore_terminations: false`。
- [x] `use_spacemouse: false` 用于 PPO。
- [x] 开启训练视频和评估视频。
- [x] 显式配置 Push Button 已验证的机械臂运动参数。

### 真机 PPO 配置

目标文件：

```text
examples/embodiment/config/realworld_pick_and_place_async_ppo_cnn_state.yaml
```

- [x] 参考 `realworld_push_button_async_ppo_cnn_state.yaml`。
- [x] 使用 `train_async.py` 和 `decoupled_actor_critic`。
- [x] 环境采集两路图像，CNN 使用单路 `main_images`，并使用 14 维状态和 7 维动作。
- [x] 设置 `image_num: 1`，与 Push Button 和当前 step600 保持一致。
- [x] 使用二值夹爪分布。
- [x] 支持加载当前 step600 或后续单视角 BC/PPO 权重。
- [x] 开启 checkpoint 保存。
- [ ] 初始建议：

  ```yaml
  algorithm:
    rollout_epoch: 4
    update_epoch: 2
    gamma: 0.99
    gae_lambda: 0.95
    entropy_bonus: 5.0e-4
    target_kl: 0.01

  actor:
    micro_batch_size: 120
    global_batch_size: 480
    optim:
      lr: 1.0e-5
      backbone_lr: 1.0e-6
      value_lr: 1.0e-4
  ```

- [x] 每 5～10 个 PPO update 保存一次 checkpoint。
- [ ] 根据真实吞吐和 rollout 方差再调整 batch size，不预先堆叠复杂超参数。

### Sim–Real Co-Training 配置

目标文件：

```text
examples/embodiment/config/co_rl_pick_and_place_async_ppo_cnn_state_paired.yaml
```

- [ ] 参考 `co_rl_push_button_async_ppo_cnn_state_paired.yaml`。
- [ ] 使用 `sim_real_rl_co_training: true`。
- [ ] 使用 `co_training_rollout_routing_mode: paired`。
- [ ] 一个 real env worker。
- [ ] 一个 sim env worker，初始使用 16 或 32 个并行仿真环境。
- [ ] 两个 paired rollout worker。
- [ ] 一个 actor worker。
- [ ] 节点组名称保持 `real` 和 `sim`。
- [ ] real/sim observation、action、horizon 和 reward 语义完全一致。
- [ ] 初始 domain buffer 建议：

  ```yaml
  co_training_domain_buffer:
    channel_maxsize: 1
    sample_rollout_length: 16
    real_ratio_min: 0.0625
    real_ratio_target: 0.125
    real_ratio_max: 0.25
  ```

- [ ] 初始目标约为每个采样块 2 条真实轨迹和 14 条仿真轨迹。
- [ ] 稳定后再将真机目标比例提升至 25%。

### 独立真机评估配置

- [ ] 新增真机 PnP eval 配置和运行脚本。
- [ ] 使用确定性动作。
- [ ] 使用固定 `reset_case_id` manifest。
- [ ] 保存标准 MP4、逐 episode 结果和汇总成功率。
- [ ] 不依赖 Async PPO runner 内部 validation。

## 6. 数据采集

### Step600 真机基线评估

- [ ] 先以 shadow mode 运行，只推理不发送动作。
- [ ] 检查图像、14 维状态和动作分布。
- [ ] 使用安全限幅进行小规模动作测试。
- [ ] 完整 action scale 下进行固定 case 评估。
- [ ] 记录 grasp、lift、place、release、success 和 drop 的人工统计。
- [ ] 同时保存主视角和辅助视角视频，但只将主视角送入 CNN。

决策标准：

- 真机成功率 `≥40%～50%`：进入真机 PPO。
- 真机成功率 `<40%`：先采集真机专家数据做 BC。
- 真机成功率接近 `0`：优先检查图像、状态顺序、夹爪语义和物体布局对齐。

### 真机专家数据

- [ ] 先采集约 50 条成功轨迹进行第一轮 BC。
- [ ] 如有必要，逐步扩展至 150～300 条。
- [ ] 覆盖源盘不同位置和方块 yaw。
- [ ] 避免重复相同 reset case。
- [ ] 保存实际执行的 7 维专家动作。
- [ ] 保存完整 14 维状态、第三视角 RGB、腕部 RGB、双视角原始视频和人工结果。
- [ ] 按 trajectory 划分 train/validation/test，禁止按帧随机拆分。
- [ ] BC 数据采集时允许 `use_spacemouse: true`。
- [ ] PPO 训练时必须切换为 `use_spacemouse: false`。

### 人工奖励流程验证数据

- [ ] 使用人工或脚本动作运行 50～100 个 episode。
- [ ] 验证 success 只发放一次。
- [ ] 验证 failure 正确结束。
- [ ] 验证 abort trajectory 不进入 PPO。
- [ ] 验证 timeout 是零奖励 truncation。
- [ ] 验证 ready 正确启动下一局。
- [ ] 检查反馈事件与视频时间戳一致。

### 固定真机评估集

- [ ] 建立 60～100 个固定 reset case。
- [ ] 覆盖中心、边缘、角落和多个 yaw。
- [ ] 固定评估 case 不进入 BC 数据。
- [ ] 所有 checkpoint 使用相同人工成功判定标准。

## 7. 执行阶段与验收门槛

### 阶段 A：真机环境 Dry Run

- [ ] 零动作环境运行。
- [ ] 单轴小动作。
- [ ] 夹爪开闭。
- [ ] success/failure/abort/ready 全流程。
- [ ] 人工复位。
- [ ] 视频与 trajectory 保存。
- [ ] 连续 reset 无意外动作和碰撞。

通过后才能加载策略。

### 阶段 B：Step600 真机评估

- [ ] step600 shadow mode。
- [ ] 安全限幅 rollout。
- [ ] 确认双相机采集正常、主视角与 Push Button 一致，CNN 只使用 `main_images`。
- [ ] 完整动作 rollout。
- [ ] 固定测试集成功率报告。
- [ ] 决定直接真机 PPO 或先做真实 BC。

### 阶段 C：真机 BC（按需）

- [ ] 从 step600 或其 policy 权重初始化。
- [ ] 使用真实成功轨迹小学习率微调。
- [ ] 必要时混入唯一的仿真专家轨迹，避免灾难性遗忘。
- [ ] 不通过复制轨迹提高真机样本比例。
- [ ] 使用 domain-balanced sampling。

建议进入 PPO 的门槛：

- 真机固定测试集成功率约 `60%`。
- 仿真成功率保持在 `80%` 以上。
- 夹爪切换稳定，drop rate 可接受。

### 阶段 D：真机 PPO

- [ ] 从最佳 step600/BC 权重启动新训练。
- [ ] 使用新 optimizer，不继承不匹配的旧训练状态。
- [ ] 人工 terminal sparse reward。
- [ ] SpaceMouse 关闭。
- [ ] 每 5～10 个 update 保存 checkpoint。
- [ ] 定期暂停训练并进行独立固定集评估。
- [ ] 保存所有训练 rollout、人工反馈和视频。

完成标准：

- 独立固定测试集成功率达到 `≥80%`。
- 连续三个评估批次没有明显回落。
- 无人工奖励重复、错位或漏记录。
- 安全中止率和 drop rate 可接受。

### 阶段 E：Sim–Real Co-Training

- [ ] 从最佳真机/混合 checkpoint 初始化。
- [ ] 使用 Push Button 已验证的 `paired` Async PPO 拓扑。
- [ ] real/sim 都使用 terminal-only success reward。
- [ ] 初始真机目标采样比例为 12.5%。
- [ ] 确认 batch size 与 `sample_rollout_length × horizon` 可整除。
- [ ] 分别监控 real 和 sim 成功率。
- [ ] 稳定后逐步提高真机采样比例。

重点指标：

- `real/success_rate`
- `sim/success_rate`
- `buffer/train_real_ratio`
- real/sim interaction steps
- PPO KL
- clip fraction
- entropy
- policy version staleness
- abort 数量
- reset 失败数量
- 人工反馈延迟

## 8. 测试清单

- [ ] sim/real observation shape 和 dtype 一致。
- [ ] `main_images` 始终对应 Push Button 已验证的物理主视角，`extra_view_images[:, 0]` 始终对应另一物理视角。
- [ ] 两路相机分辨率、裁剪、padding、RGB/BGR 和时间戳测试。
- [ ] `image_num: 1` 时 CNN 不读取辅助视角的测试。
- [x] 14 维状态顺序测试。
- [x] 7 维动作和夹爪正负语义测试。
- [ ] RGB/BGR 通道测试。
- [x] success/failure/abort/timeout reward 测试。
- [x] 底层安全 termination 保留测试。
- [x] abort trajectory mask 测试。
- [x] 人工按键 debounce 测试。
- [x] reset-ready 状态机测试。
- [x] 真机 dummy environment smoke test。
- [ ] paired co-training domain ratio 测试。
- [ ] batch 可整除测试。
- [ ] step600/BC/PPO 单视角 checkpoint 兼容性测试。

## 9. 非目标

- 不实现自动视觉奖励。
- 不训练奖励模型。
- 不使用 RGB-D 判断成功。
- 不修改 OpenPI。
- 不重写 Push Button 已验证的机械臂控制链路。
- 不把人工动作干预混入 on-policy PPO。
- 不在真机环境闭环通过前启动 co-training。
