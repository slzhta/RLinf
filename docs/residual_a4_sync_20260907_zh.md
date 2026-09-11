# Residual 同步到 a4：结果与使用说明

日期：2026-09-07。此次将本地 residual 实现同步到 a4，同时给 4090 补入同一套 residual 接口。没有启动训练、Ray 集群或机器人服务，没有复制模型权重。

## 同步位置与版本含义

| 机器 | 工作目录 | 同步前 Git HEAD |
| --- | --- | --- |
| 4090 | `/home/ubuntu/wangyinghan/wangyitao` | `75080ece32c426337e0affd0ce627fc6cc3870ab` |
| a4 | `/home/shiliangzhi/work-space/wangyinghan/RLinf`，实际解析为 `/data/shiliangzhi/work-space/wangyinghan/RLinf` | `e89ae7ab7d30c4ce44bf605fa1af435477e4e325` |

两端仍在原来的 `rl-co-online` 分支；没有创建提交、切换分支或改写 Git 历史。这里的“同步”指工作目录中的源码更新，不代表 a4 的 Git HEAD 已前进到 4090 的提交，也不是两个目录的完整镜像。

4090 原有 CNN、PPO、数据结构与本地 residual 所需版本一致，核心同步新增/修改 18 个 residual 文件。a4 核心同步涉及 88 个文件：以 4090 实际工作树中的 `rlinf` 源码、模型配置和 embodied 入口为更新来源，加上本地 residual 模型、缓存、观测传输接口、示例、测试和文档。另补齐文档导航；4090 补齐 residual 单测的 Git 忽略规则例外。

没有删除 a4 独有文件，也没有同步 `.venv`、数字孪生 assets、数据集或其他实验权重。服务器的既有未提交修改在比较和合并后保留。此次未更新本地 `RLinf-a4` 镜像目录；本地审阅副本和逐文件清单位于 RLinf-4090 的 `_remote_staging/residual-sync-20260907`。

## 保留的 a4 差异

- 原 `maniskill_pick_and_place_co_rl.yaml` 保留，包括 a4 自己的托盘位置、随机 reset 和位姿范围设置。
- a4 的 PnP trace 调试代码、相机字段兼容逻辑和 OpenPI 观测 dump 保留。
- a4 原有具名 OpenPI 实验配置及路径保留，同时加入 4090 新增的配置。
- a4 的 `save_full_model_weights` checkpoint 选项保留，并与较新的 FSDP 策略接口合并。
- `use_config_control_mode` 使用 4090 较新的实现，继续支持 a4 的开关，并检查显式控制模式是否有效。

为避免改变 a4 的旧实验，单独新增：

```text
examples/embodiment/config/env/maniskill_pick_and_place_residual.yaml
examples/embodiment/config/maniskill_pick_and_place_ppo_residual_a4.yaml
```

前者复制 4090 的 residual 参考场景参数；后者使用该环境和 a4 的权重路径变量。a4 上运行 residual 请选带 `_a4` 后缀的配置，普通配置保留本地/4090 版本。

## 已完成验证

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| a4 residual 单测 | **11 passed，14.08 秒** | CNN 条件、状态开关、Gaussian 概率重算、缓存与部分 reset、PPO 更新及配置等 |
| a4 专用 Hydra 配置 | 通过 | 实际 compose、OmegaConf.resolve 和 residual validator；检查时未加载模型权重 |
| ManiSkill residual 环境、OpenPI 模块导入 | 通过 | 使用 a4 当前仓库及其 `.venv` |
| a4 Python 源码 AST | 475 个文件通过 | `rlinf` 目录的语法检查 |
| 启动脚本 `bash -n` | 通过 | shell 语法 |
| 同步文件哈希 | 已逐文件校验 | 校验依据为同步清单中的 SHA-256 |
| 既有 Git 暂存内容 | 保留 | a4 原 staged diff 与同步前备份比对一致；4090 核心同步未改变 index |

专用配置初次使用 Hydra 继承覆盖时未通过，已改为独立完整配置并复核通过。同步工具的一次 index 字节检查因 Git 刷新文件状态信息报错；之后比较 staged patch，确认暂存内容没有改变，且专用配置文件已正确写入。

原 residual 说明中“Windows PyTorch DLL 导入失败、单测未通过”的描述是先前本地记录；上表补充本次在 a4 上完成的验证。**单测与导入通过不等于实际 checkpoint 的仿真训练、终止观测/bootstrap 全链路或训练收益已验证。**

## 模型路径与后续启动

a4 已检查到对应 base checkpoint：

```text
/data/wangyinghan/pytorch_checkpoints/pi05_pnp_sim500_real250_success_20260905_state_10k
```

该目录存在 `model.safetensors`（7,233,650,408 字节）、`config.json`，以及 `pnp_lerobot_sim500_real250_success_20260905_v1/norm_stats.json`。此次只检查文件存在，没有加载权重验证推理，也没有对两台服务器的大权重进行完整哈希比较。

在 `/data/wangyinghan`、`/home/shiliangzhi/work-space` 的限定深度搜索中，尚未找到 `resnet10_pretrained.pt`。需要提供真实 CNN encoder 权重所在目录；不要将单测生成的随机 encoder 权重用于实验。

准备好该目录后，在 a4 上使用：

```bash
cd /home/shiliangzhi/work-space/wangyinghan/RLinf
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RESIDUAL_BASE_MODEL_PATH=/data/wangyinghan/pytorch_checkpoints/pi05_pnp_sim500_real250_success_20260905_state_10k
export RESIDUAL_ENCODER_PATH=/实际目录/包含resnet10_pretrained.pt

bash examples/embodiment/run_maniskill_residual.sh \
  maniskill_pick_and_place_ppo_residual_a4
```

这是后续手动启动说明，本轮没有执行该训练命令。现有 `pre_start_ray.sh` 引用另一处 `/home/shiliangzhi/work-space/RLinf`，本次保留了它；上面的命令显式选择当前仓库和 `.venv`，避免依赖该旧脚本的代码路径。

## 备份与审阅

核心同步前的完整待修改文件、Git index、staged/unstaged patch、HEAD 和同步清单在：

```text
a4:
/data/shiliangzhi/work-space/wangyinghan/.codex-residual-backups/20260907-134836

a4 专用配置修正前:
/data/shiliangzhi/work-space/wangyinghan/.codex-residual-backups/20260907-135136

4090:
/home/ubuntu/wangyinghan/.codex-residual-backups/20260907-135338
```

每次补充同步也在同一父目录中生成独立时间戳备份。`before.zip` 保存原文件，`manifest.json` 保存新增/修改路径与前后哈希，`incoming.zip` 保存本次写入内容。若以后要恢复，应先核对后续用户修改，再按清单逐文件恢复；新增文件在原备份中没有对应内容，不能只解压 `before.zip` 就视为完整回滚。

本地逐文件审阅清单：`_remote_staging/residual-sync-20260907/initial-plan.json`；配置复核日志：`_remote_staging/residual-sync-20260907/a4-validate_config_a4.log`。数据链路说明见同目录文档 `residual_dataflow_scale_zh.md`，本报告补充同步范围及实际验证结果。
