# HighSpeedUUV Kinematic

## GPU 声呐/碰撞加速版

这是一个独立实验版本：运动学积分、Dijkstra 和 PPO 逻辑保持不变，只把
三路几何声呐和运动线段碰撞检测迁移到 PyTorch/CUDA。

训练入口新增：

```text
--sensor-backend cpu|gpu
--sensor-device cuda|cuda:0
```

先比较相同 100–150 障碍场景下 CPU/GPU 的吞吐：

```bash
./scripts/benchmark_sensors.sh \
  --min-obstacles 100 --max-obstacles 150 \
  --steps 300 --sensor-device cuda
```

使用 GPU 传感器训练：

```bash
./scripts/train.sh \
  --sensor-backend gpu \
  --sensor-device cuda \
  --device cuda \
  --num-envs 8 \
  --torch-threads 1 \
  --min-obstacles 100 \
  --max-obstacles 150 \
  --output output_model/kinematic_ppo_gpu_sensor \
  --overwrite
```

每个并行环境会创建独立的 CUDA 上下文。若显存不足，优先减少
`--num-envs` 或 `--bc-workers`。在该版本中只有声呐和碰撞计算在 GPU；
运动学积分、Dijkstra 和 Python 控制循环仍由 CPU 执行。

受速度、加速度、转弯半径和俯仰约束的三维运动学 UUV 规划与跟踪工作空间。

本项目的论文定位是：

> 研究复杂静态障碍环境中的高速三维路径规划与跟踪，不讨论推进器、空化、
> 螺旋桨效率和高速水动力参数标定。

训练和主评估均使用运动学模型。HoloOcean 仅用于独立场景回放、碰撞几何检查和
可视化，不作为高速动力学验证平台。

## 运行环境

训练和策略评估使用 CUDA GPU。当前锁定环境为 Python 3.10、PyTorch 2.5.1
CUDA 12.4、Stable-Baselines3 2.7.0。首次部署时运行：

```bash
cd /path/to/HighSpeedUUV_Kinematic
./scripts/setup_env.sh
```

脚本会在项目目录创建 `.venv`，安装依赖，并验证 CUDA 与 GPU。若系统缺少
`python3-venv`，请先安装 `python3.10-venv` 或提供可用的 `virtualenv`。

## 运动学模型

状态：

```text
(x, y, z, yaw, pitch, speed, yaw_rate, pitch_rate)
```

策略动作：

```text
[speed_command, yaw_rate_command, pitch_rate_command]
```

约束：

- 速度范围：默认 5–50 节
- 最大加速度和减速度
- 最大偏航角速度
- 最大横向加速度 `|u r|`
- 最小转弯半径
- 最大俯仰角和俯仰角速度
- 偏航/俯仰角加速度限制

当前默认场景为静态海底山地高度场（heightfield terrain），不再生成悬浮球形
障碍物，也不再让障碍物运动。地形由 18–30 个不同尺度的山体叠加低频起伏生成，
高度范围约为 0–62 m。默认巡航高度为 30 m，起点和目标点保持同一高度。
UUV 需要从山峰之间穿过，或在必要时进行短时高度调整后回到巡航深度。

起终点三维距离默认随机采样为 650–900 m，规划空间为
X ∈ [-100, 1100] m、Y ∈ [-750, 750] m、Z ∈ [0, 100] m。
地形使用 `z <= h(x,y)` 表示固体区域，声呐、碰撞、净距和 Dijkstra 都直接读取
海床高度场。

这样高速时自动限制转弯角速度，避免“高速大角度急转弯”。

## 观测

```text
sonar  : (6, 12, 48) 三路几何声呐
target : 目标俯仰、偏航、速度误差、归一化距离
motion : 当前速度、巡航速度、速度上下限、偏航率、俯仰、俯仰率、深度误差
```

当前学生策略观测不包含 Dijkstra 路径。Dijkstra 只作为教师内部信息，用于静态
BC/DAgger 数据生成；PPO 完全根据声呐历史、目标方向/距离和自身运动状态决策。

## 训练

默认流程：

1. 默认目标速度固定为 50 节，不再随机训练 10–40 节；
2. BC、DAgger 和 PPO 都使用同一类静态海底地形；
3. 教师使用 Dijkstra 路径生成专家动作，但学生观测中不包含路径；
4. BC 和 DAgger 完成后，在静态地形上执行成功率门槛判断；
5. PPO 不再构建或奖励 Dijkstra 代价场，但仍使用静态海底地形；
6. PPO 使用低权重欧氏距离势函数，并保留速度、安全与终端奖励；
7. 默认只在 50 节上评估，并保留最佳 checkpoint。

```bash
cd /path/to/HighSpeedUUV_Kinematic
./scripts/train.sh \
  --output output_model/seabed_terrain_ppo \
  --overwrite
```

当前默认配置为 30 万步 PPO、8 路并行采样、每路 512 步 rollout、1024 的
PPO batch、18–30 个海床山体、50 节目标速度，以及 LSTM + 残差 PPO。
`n_steps` 和 `batch_size` 会根据并行环境数自动选择；使用默认 8 路环境时分别
解析为 512 和 1024。

训练默认使用 `cuda`、8 路并行采样和 A100 的 TF32 加速。可通过
`--device cuda:0`、`--num-envs 8` 和 `--torch-threads 1` 调整。模型、行为克隆、
策略评估和 PPO 更新均在 GPU 上执行；运动学仿真环境仍由 CPU 并行运行。

BC 数据量可通过 `--bc-steps` 和 `--bc-val-steps` 调整。成功率门槛可通过
`--bc-min-success-rate` 修改；仅调试流程时可使用 `--allow-low-bc` 跳过门槛。

DAgger 默认执行 8 轮，每轮采集 30,000 步 BC 实际访问状态，并用教师动作
进行纠正。可通过以下参数调整：

```text
--dagger-rounds 8
--dagger-steps 30000
--dagger-workers 4
--dagger-epochs 5
--dagger-learning-rate 1e-4
--dagger-deterministic
```

设置 `--dagger-rounds 0` 可以关闭 DAgger。

## 从现有模型继续海底地形 PPO

使用 `--resume-model` 可以加载同版无路径观测的 checkpoint，自动跳过 BC 和
DAgger 数据采集。旧版包含 `path` 观测的模型不能直接加载：

```bash
./scripts/train.sh \
  --resume-model output_model/seabed_terrain_ppo/bc_model.zip \
  --timesteps 300000 \
  --num-envs 8 \
  --sensor-backend gpu \
  --sensor-device cuda \
  --device cuda \
  --learning-rate 3e-5 \
  --bc-finetune-learning-rate 3e-5 \
  --resume-log-std -2.0 \
  --target-kl 0.02 \
  --output output_model/seabed_terrain_resume \
  --overwrite
```

`--resume-model` 会自动把 `bc-steps` 和 `dagger-rounds` 置为 0。

PPO 不再使用 Dijkstra 奖励。欧氏距离采用标准势函数：

```text
Phi_d(s) = -distance_to_goal(s) / distance_potential_scale
r_distance = gamma * Phi_d(s_next) - Phi_d(s)
```

默认 `distance_potential_scale=250`。这种写法比直接奖励
`old_distance - new_distance` 更稳定，避免策略通过重复绕行累积距离奖励。
最终 PPO 奖励主要由低权重距离势函数、速度误差、安全净距和到达/碰撞终端项组成。
定深奖励会惩罚相对参考巡航高度的偏差：

```text
r_depth = -depth_hold_weight
          * min((|z - z_reference| / depth_hold_scale)^2, 1)
```

默认 `depth_hold_weight=0.2`、`depth_hold_scale=10 m`。学生观测中的 `motion`
新增归一化深度误差，使策略能够主动回到巡航深度。

Dijkstra 只保留在静态教师采集中。由于 PPO 不再访问代价场，
`--dijkstra-replan-steps` 不会影响 PPO 阶段的速度或奖励。

## LSTM 与残差 PPO

声呐默认保留 8 个采样帧，每 4 个控制步取一帧，形成约 2.8 秒的历史窗口：

```text
--sonar-history-length 8
--sonar-history-stride 4
```

这比连续 3 帧覆盖的范围更长，同时在 BC 数据中以 `float16` 保存
`sonar_history`，避免历史窗口增大后观测数据集占用内存翻倍。

残差 PPO 默认开启：

```text
--residual-policy
--residual-scale 0.30
--critic-warmup-steps 30000
```

BC 训练主策略和特征提取器；进入 PPO 后冻结 BC 主策略，只训练小幅残差动作头。
Critic warmup 期间残差动作头关闭，只训练价值网络，结束后开放残差动作和探索方差。

BC 成功率门槛使用静态场景，只决定是否进入 PPO。进入 PPO 前还会在动态开发种子
上重新评估 BC，作为 PPO checkpoint 的公平初始基线；后续最佳模型按动态成功率、
最终距离和碰撞率选择，不会再拿静态 BC 成绩和动态 PPO 成绩直接比较。

注意：无路径观测、LSTM 和残差动作头都会改变模型结构，旧版包含 Dijkstra 路径
观测的 checkpoint 不能直接用于该配置，需要重新运行 BC。

训练输出：

- `ppo_kinematic.zip`：周期评估选出的最佳模型
- `ppo_kinematic_final.zip`：PPO 最终模型
- `bc_model.zip`：BC 初始化模型
- `bc_initial_model.zip`：DAgger 前的初始 BC 模型
- `dagger_round_*_student.zip`：每轮 DAgger 采样时的学生模型
- `dagger_round_*_model.zip`：每轮 DAgger 再训练后的模型
- `best_eval.json`：最佳模型评估
- `run.json`：完整运动学和训练配置
- `csv/progress.csv`：训练曲线
- `csv/monitors/`：各并行环境的回合记录

## 评估

默认在 50 节运行 20 回合：

```bash
./scripts/evaluate.sh output_model/kinematic_ppo/ppo_kinematic.zip \
  --episodes-per-speed 20 --seed 0 --speeds-knots 50 \
  --output output_model/kinematic_eval
```

策略评估默认使用 `--device cuda`；如需显式指定 GPU，可传入
`--device cuda:0`。

当前场景为静态海底山地，不存在移动障碍物。`--moving-obstacles` 和障碍速度参数
仅保留给旧版球形障碍 checkpoint 的兼容评估。

输出：

- `csv/evaluation_metrics.csv`：逐回合结果
- `csv/evaluation_summary.csv`：按速度分组汇总
- `evaluation_summary.json`
- `evaluation.png`
- `trajectories/*.npz`：可用于 HoloOcean 回放的轨迹

地形轨迹会额外保存 `terrain_height`、`terrain_x` 和 `terrain_y`，用于轨迹诊断、
实时播放和场景复现。

评估同时记录：

- 成功率、碰撞率、越界率、超时率
- 完成时间、路径长度
- 速度跟踪
- 最大横向加速度
- 最大偏航角速度
- 最小障碍物距离
- 最大俯仰角

默认任务成功球半径为 5 m。Dijkstra 规划路径始终以目标球心为终点，
不会再把整个成功球内部都当成零代价区域。

## 轨迹可视化验证

策略评估会生成 `trajectories/*.npz`。可视化程序会为每条轨迹生成一张
诊断图，包含三维轨迹、海床地形、目标球、速度跟踪、俯仰/偏航、目标距离和
海床净距：

```bash
./scripts/evaluate.sh output_model/kinematic_ppo_v2/ppo_kinematic.zip \
  --episodes-per-speed 5 --seed 1000 \
  --output output_model/kinematic_viz_eval

./scripts/visualize.sh output_model/kinematic_viz_eval \
  --output output_model/kinematic_visualization \
  --speeds-knots 10 20 30 40 50 \
  --seeds 0 \
  --max-trajectories 10
```

`--seeds` 可以选择一个或多个评估种子，例如 `--seeds 5` 或
`--seeds 0 5 10`；需要同时播放指定轨迹时，使用对应的 `.npz` 文件即可。

也可以只可视化一个轨迹：

```bash
./scripts/visualize.sh \
  output_model/kinematic_viz_eval/trajectories/speed20_seed1000.npz
```

输出包括 `*_diagnostics.png` 和 `visualization_summary.json`。后者记录成功率、
最终目标距离、最小障碍物净距、最大俯仰角、最大偏航角速度和最大横向加速度。

## 场景分布可视化

批量生成默认海床地形并检查山体分布、直接路径阻挡和 Dijkstra 可达性：

```bash
./scripts/visualize_scenes.sh \
  --output output_model/terrain_preview \
  --num-scenes 6 \
  --seed-start 7000
```

每个场景输出 3D 地形、俯视等高线、起终点直线剖面，并额外生成
`overview_terrain.png` 总览图和 `terrain_summary.json`。

## 实时播放单条轨迹

有桌面的服务器上，可以直接实时播放 AUV 位置、航向箭头、尾迹、速度和姿态：

```bash
./scripts/play.sh \
  output_model/kinematic_viz_eval/trajectories/speed20_seed1000.npz \
  --backend TkAgg --speed 1.0
```

无桌面或远程服务器可导出 MP4/GIF：

```bash
./scripts/play.sh \
  output_model/kinematic_viz_eval/trajectories/speed20_seed1000.npz \
  --save output_model/kinematic_viz_eval/speed20_playback.mp4
```

`--speed 0.5` 为半速播放，`--speed 2` 为二倍速；`--trail-seconds 8` 控制
尾迹时长，`--stride 2` 表示每两帧播放一帧。交互窗口会播放一次，可用
`--repeat` 循环。

## HoloOcean 几何验证

选择一条导出的轨迹：

```bash
./scripts/replay_holoocean.sh \
  output_model/kinematic_eval/trajectories/speed50_seed0.npz
```

回放过程：

- 按相同 NED 场景生成金色障碍物和白色目标球；
- 将运动学轨迹转换为 HoloOcean NWU 坐标；
- 通过智能体位姿接口逐帧回放；
- 检查几何碰撞、最小净距和可视化一致性。

这不验证高速 Torpedo 动力学，只验证场景几何、坐标转换和轨迹结果。

## Sim-to-sim 验证建议

1. 训练环境：本项目的运动学 Gym 环境；
2. 独立验证：不同时间步长、控制延迟和约束扰动的第二套运动学实现；
3. HoloOcean：相同场景的轨迹回放和几何检查；
4. 若需要低速高保真补充，可在 10 节以内使用 Torpedo Fossen 模型对照。

论文中应明确：

> 高速推力、功率、空化和水动力可行性不在本文研究范围内。

## 运动学约束基准

运行速度、加速、转弯和制动基准：

```bash
./scripts/benchmark.sh
```

默认参数下（最大横向加速度 6 m/s²、最小转弯半径 20 m）的结果：

| 目标速度 | 0–90% 加速时间 | 满舵转弯半径 | 制动时间 | 制动距离 |
| -------: | --------------: | -----------: | -------: | -------: |
|    10 节 |          1.50 s |       20.3 m |    1.4 s |    4.1 m |
|    20 节 |          3.40 s |       20.9 m |    3.1 s |   16.9 m |
|    30 节 |          5.20 s |       40.9 m |    4.8 s |   37.9 m |
|    40 节 |          7.10 s |       72.0 m |    6.4 s |   67.2 m |
|    50 节 |          8.90 s |      111.9 m |    8.1 s |  104.9 m |

这些指标直接验证运动学约束，不涉及 RPM、推力、空化或轴功率。

## 测试

```bash
./scripts/test.sh
```
