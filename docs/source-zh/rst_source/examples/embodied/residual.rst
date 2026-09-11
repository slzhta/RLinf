冻结 OpenPI 的 Residual PPO
==========================

本示例先实现单节点 PnP 仿真 residual RL，使用现有的
``train_embodied_agent.py``、``EmbodiedRunner``、PPO actor、HF rollout
和当前策略采集的 rollout batch。当前仅支持一个 pipeline stage，尚未接入真机和 co-training。

代码组织
--------

* ``rlinf/models/embodiment/residual_policy/``：继承原有 CNNPolicy 的 residual 策略及配置校验。
* ``rlinf/envs/residual.py``：base 动作缓存和动作合成。
* ``rlinf/envs/maniskill/residual_maniskill_env.py``：加载冻结 OpenPI、合成动作、提供 base 条件。
* ``examples/embodiment/config/model/residual_policy.yaml``：继承原有 ``model/cnn_policy.yaml``，补充 residual 条件与探索参数。
* ``examples/embodiment/config/model/pi0_5.yaml``：直接复用现有 OpenPI 默认配置，PnP 专用参数在实验配置的 ``base_model`` 中覆盖。
* ``examples/embodiment/config/maniskill_pick_and_place_ppo_residual.yaml``：仿真实验及 PPO 配置。
* ``examples/embodiment/run_maniskill_residual.sh``：调用现有训练入口并接收 Hydra 覆盖参数。
* ``tests/unit_tests/test_residual_policy.py``：策略、缓存、PPO 概率重算和配置测试。

现有模型注册、环境选择、rollout 模式和观测传输仅增加按新配置启用的分支。
原来的 SFT YAML、OpenPI 模型实现及 checkpoint 没有改动。

执行链路
--------

冻结的 OpenPI 放在 GPU env worker 中，同进程的训练和评估环境共享权重，分别维护缓存。
Actor 和 rollout 中仅有小型 residual 网络，权重同步和 residual
checkpoint 不包含 OpenPI。复现实验需要同时保留完整实验配置及原始 base checkpoint。

OpenPI 仍使用原来的全分辨率图像、状态、输入变换和动作反归一化。
每个环境独立缓存 10 步 base 动作；residual 每个控制步读取当前图像、剩余动作块和掩码，输出修正。
只有缓存用尽或对应环境 reset 时才重新调用 OpenPI。

.. code-block:: text

   executed = clip(base_action + residual_scale * clip(residual_action, -1, 1), -1, 1)

合成发生在 OpenPI 动作反归一化之后、现有归一化增量控制器之前。
默认修正前六个机械臂维度，比例为 0.1；夹爪修正比例为零，完全跟随 base。
这里 0.1 是控制器输入范围的比例，不是米或弧度。当前继承的仿真控制频率为 5 Hz，
尚未验证与真机的时间尺度一致。

状态输入开关
------------

ResidualPolicy 继承原有 CNNPolicy，复用图像归一化、ResNet 编码器、状态投影、
Gaussian 采样、解析熵和 value head。图像沿用 PnP 环境的 128×128 NHWC 输出。
默认沿用 CNN 的冻结 ResNet backbone 设置，训练 pooling/projection、actor 和 value head；
如需同时训练 backbone，设置 ``+actor.model.encoder_config.freeze_backbone=false``。

原有 CNN 的数值 ``states`` 通道用于存放展平的剩余 base 动作和 mask；默认为 80 维。
``actor.model.use_state=false`` 表示不追加机械臂原始状态，并非完全没有数值条件。
OpenPI 仍读取原来训练时的状态，base 动作本身可能携带状态信息。
设置 ``actor.model.use_state=true`` 后再追加机械臂状态；``state_dim``、``state_mean``、
``state_std`` 仅描述这一部分。配置不在运行中被改写，内部 CNN 使用独立的配置副本。
切换开关会改变网络形状，需要对应结构的 residual checkpoint。

Rollout 沿用 CNN 的存储字段：图像、打包后的 ``states``、原始 Gaussian ``action``、旧 logprob 和旧 value。
训练直接复用 CNN 的概率和解析熵计算，以及现有 GAE 和 ``decoupled_actor_critic`` PPO loss。
不修正的动作通道，其 logprob 和 entropy 贡献为零。仅在环境执行时裁剪和缩放 residual，
PPO 保存的采样动作不裁剪。不新增 runner、loss 或 replay buffer，更新时不重新调用 OpenPI。
时间截断通过包含 base 条件的 final observation 估计 bootstrap value，真正终止不 bootstrap。
额外的 value 估计不会消耗 base 动作缓存；只有环境执行一步才推进缓存。


依赖和运行
----------

沿用现有 Linux RLinf + OpenPI + ManiSkill 环境及 PnP 资源，不新增依赖或 Docker target。
复用原有 CNN 示例的 ``resnet10_pretrained.pt``，通过 ``actor.model.model_path`` 指定所在目录，
rollout 自动继承该路径。这与 OpenPI 的 ``base_model.model_path``、训练后 residual 的
``runner.ckpt_path`` 分别对应不同权重。当前尚未确认这份 ResNet 权重在服务器上的目录。
默认 base 路径指向已确认的 4090 SFT checkpoint；其他机器通过
``base_model.model_path`` 覆盖，并保留对应 normalization assets 和 ``openpi_data.repo_id``。

先在合适的仿真 Ray 会话中运行短测试，不复用正在控制机械臂的 Ray 会话：

.. code-block:: bash

   bash examples/embodiment/run_maniskill_residual.sh \
     maniskill_pick_and_place_ppo_residual \
     actor.model.model_path=/path/to/resnet_weights \
     runner.max_epochs=2 algorithm.update_epoch=1 runner.save_interval=1

先在实验配置中填好 CNN 权重目录，再默认训练或启用 residual 状态输入：

.. code-block:: bash

   bash examples/embodiment/run_maniskill_residual.sh
   bash examples/embodiment/run_maniskill_residual.sh \
     maniskill_pick_and_place_ppo_residual actor.model.use_state=true

通过新仿真适配层评估 base-only：

.. code-block:: bash

   bash examples/embodiment/run_maniskill_residual.sh \
     maniskill_pick_and_place_ppo_residual \
     runner.only_eval=true actor.model.enabled=false

评估训练后的 residual 时保留 ``enabled=true``，把 ``runner.ckpt_path`` 指向
residual 的 ``full_weights.pt``，不要填 OpenPI SFT 权重。继续训练使用原有
``runner.resume_dir`` 参数。

验证
----

.. code-block:: bash

   python -m pytest tests/unit_tests/test_residual_policy.py -q

测试覆盖状态开关、初始零修正均值、夹爪屏蔽、部分 reset、chunk 边界、观测传输、
rollout 条件、与原生 CNN 的概率/value 对照、未裁剪动作的概率重算和现有 PPO loss 更新。仍需 GPU 短训练验证 Ray/FSDP、仿真资源和真实 checkpoint
的完整组合。短测试通过不代表成功率提升；训练后需要固定评估种子，对比 base-only 与 residual。

.. note::

   TODO(agent)：初版在本地编写，尚未验证 GPU/Ray 训练以及加载实际 checkpoint 的仿真运行。
   本地 pytest 在导入 PyTorch 时因 Windows ``c10.dll`` 初始化失败而中断，尚无运行测试通过结果。
