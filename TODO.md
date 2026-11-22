# HighSpeedUUV 项目交接与 TODO

> 更新时间：2026-11-22  
> 项目目录：`/home/action-mc/code_lyx/HighSpeedUUV_Kinematic_GPU`  
> Python：`/home/action-mc/code_lyx/HighSpeedUUV_Kinematic/.venv/bin/python`  
> 分支：`main`，当前工作区有未提交修改，不要重置或覆盖用户已有改动。

## 一、先看这里

当前研究路线已经固定为：

```text
静态障碍教师
  -> 静态 BC
  -> 静态 DAgger
  -> 静态 BC 成功率门槛
  -> 动态障碍 PPO
  -> 动态多指标评估
```

关键原则：

- Dijkstra 只用于教师生成动作标签；
- 学生的观测中不得出现 Dijkstra 路径、全局地图、障碍物真值、教师动作或未来轨迹；
- BC/DAgger 只在静态障碍中训练；
- PPO 在动态障碍中训练；
- PPO 不构建、不重规划、不使用 Dijkstra 势场奖励；
- 学生观测只有 `sonar / sonar_history / target / motion`。

### 当前运行任务快照

> 截至 2026-11-22 21:52

- `ppo_v13` 正在运行，当前处于第 3 轮 DAgger 附近；
- 输出目录：`output_model/ppo_v13`
- 日志：`nohup_out/ppo_v13.log`
- 继续之前先执行 `ps -eo pid,etime,args | grep training.train`，确认是否仍在运行；
- 不要在未确认的情况下重复启动相同训练。

## 二、当前模型结构

### 观测

```text
sonar_history：8 个采样帧，每 4 步一帧，覆盖约 2.8 秒
target：目标方向、速度误差、归一化距离
motion：自身速度、角速度、姿态
```

学生观测中没有 `path`。BC、DAgger、PPO 训练和评估使用同一套学生观测。

### PPO

- BC 的特征提取器和基础动作网络冻结；
- PPO 只学习残差动作头；
- `residual_scale = 0.30`；
- Critic warmup 为 30,000 步；
- warmup 后残差动作和探索方差一起学习；
- PPO 奖励为低权重欧氏距离势函数，加少量速度、安全净距和终端奖励。

## 三、最新 v12 结果

| 模型 | 场景 | 成功率 | 碰撞率 |
|---|---|---:|---:|
| BC | 20 个动态开发种子 | 65% | 25% |
| PPO 最佳，50k 步 | 20 个动态开发种子 | 70% | 15% |
| PPO 最终，300k 步 | 20 个动态开发种子 | 35% | 40% |
| BC | 50 个独立动态种子 | 64% | 30% |
| PPO 最佳 | 同一 50 个独立种子 | 66% | 32% |
| PPO 最终 | 同一 50 个独立种子 | 46% | 32% |

结论：

- PPO 选模修复已经生效，v12 最终选中的是 50k 附近的 PPO 最佳模型，不再错误选回 BC；
- PPO 确实改变了策略，但独立测试只从 64% 提高到 66%；
- 50 个场景中只有 5 个由失败变成功、4 个由成功变失败，净提升只有 1 个场景；
- 训练到 300k 后严重退化，最佳 checkpoint 出现在约 50k；
- 当前主要问题不是训练轮数，而是残差策略变化幅度小、探索不足和动态域偏移大。

## 四、v12 为什么改善有限

1. **残差动作变化很小。** 最佳 PPO 相对 BC 的平均动作变化约 1.35%。
2. **探索几乎没变。** `log_std` 从约 `-2.5000` 变为 `-2.50018`，基本没有增加探索。
3. **残差学习率偏低。** 当前 Actor/ Critic 共用 `1e-5`，残差头从零开始，学习速度慢。
4. **BC 在动态场景中已经较强。** 静态 BC 为 70%，动态开发集仍有 65%，PPO 可提升空间较小。
5. **长期记忆不足。** 固定 2.8 秒声呐窗口不能记住完整路线、死路或很早看到的通道。
6. **静态到完整动态跨度大。** PPO 直接面对 300–450 个障碍和 1–10 节运动障碍。
7. **训练后期退化。** 动态开发成功率在 50k 达 70%，之后波动并下降，300k 只有 35%。

## 五、已完成事项

- [x] 移除学生观测中的 Dijkstra 全局路径；
- [x] 确认学生观测只包含有限声呐、目标与自身状态；
- [x] BC/DAgger 改为静态障碍；
- [x] PPO 改为动态障碍；
- [x] PPO 移除 Dijkstra 奖励、重规划与路径观测；
- [x] 使用欧氏距离势函数；
- [x] 修复静态 BC 与动态 PPO 混用选模基线的问题；
- [x] 使用 20 个动态开发种子选择 checkpoint；
- [x] 残差范围从 0.15 提高到 0.30；
- [x] Critic warmup 从 10k 提高到 30k；
- [x] warmup 后允许探索方差学习；
- [x] 声呐历史改为 8 帧 × stride 4；
- [x] 声呐历史使用 float16 存储；
- [x] 修复 DAgger 数据集复制导致的内存 OOM；
- [x] 当前完整测试 23 项通过，1 项因普通沙箱无 CUDA 跳过。

## 六、下一阶段 TODO

### P0：先做 PPO 学习能力对照

- [ ] 为残差 Actor 和 Critic 配置独立学习率；
  - 建议残差 Actor：`1e-4`
  - 建议 Critic：`3e-4`
- [ ] 将 `ent_coef` 从 `0` 提高到约 `0.001`；
- [ ] 将 `log_std_init` 从 `-2.5` 提高到约 `-2.0`；
- [ ] Critic warmup 可先保持 20k–30k；
- [ ] 先训练 100k PPO，并加入 early stopping，不要直接跑 300k；
- [ ] 使用 20 个动态开发种子选模，独立 50 个种子只做最终测试。

### P1：动态障碍课程

- [ ] 阶段 1：1–3 节动态障碍，约 50k；
- [ ] 阶段 2：3–6 节动态障碍，约 50k；
- [ ] 阶段 3：6–10 节动态障碍，约 100k；
- [ ] 课程切换时保持环境观测和奖励结构不变，只改变障碍速度。

### P2：长期局部记忆

- [ ] 比较真正的跨时间步 recurrent hidden state；
- [ ] 或加入由有限声呐历史构建的 egocentric local occupancy map；
- [ ] 记忆必须来自学生传感器，不得引入全局地图或障碍真值。

### P3：动作网络容量

- [ ] 如果提高残差学习率和熵后动作变化仍很小，允许解冻 `action_net`；
- [ ] 特征提取器仍可保持冻结，优先只微调最后一层动作网络；
- [ ] 每次只改变一个容量因素，保持可对照。

### P4：碰撞与安全

- [ ] 扩大局部风险检测范围；
- [ ] 对预测碰撞距离增加连续惩罚；
- [ ] 分析中段碰撞与左右绕障决策，避免只依赖碰撞后的终端惩罚。

### P5：速度目标

- [ ] 碰撞率降到 20% 以下后再强化速度；
- [ ] 增加“高速成功”指标，例如成功到达且平均速度超过 40 节；
- [ ] 当前成功回合平均速度仍明显低于 50 节。

## 七、关键代码入口

- 环境、观测和奖励：`kinematic_uuv/env.py`
- 教师动作：`kinematic_uuv/teacher.py`
- CNN + LSTM 和残差策略：`kinematic_uuv/policy.py`
- 静态 BC/动态 PPO 流程：`training/train.py`
- BC 静态数据采集：`training/teacher_data.py`
- 静态 DAgger：`training/dagger.py`
- 独立评估：`training/evaluate.py`
- 技术报告：`技术报告_当前技术路线.md`

## 八、关键结果目录

- v12 最佳模型：`output_model/ppo_v12/best_model.zip`
- v12 最终模型：`output_model/ppo_v12/ppo_kinematic_final.zip`
- v12 选中的模型：`output_model/ppo_v12/ppo_kinematic.zip`
- v12 BC：`output_model/ppo_v12/bc_model.zip`
- v12 训练日志：`nohup_out/ppo_v12.log`
- v12 50 种子评估日志：`nohup_out/ppo_v12_eva.log`
- PPO 最佳模型 50 种子评估：`output_model/ppo_v12/evaluation/`
- PPO 最终模型 50 种子评估：`output_model/ppo_v12_final/evaluation/`
- BC 50 种子评估：`output_model/ppo_v12/evaluation_bc50/`

## 九、容易重复踩的坑

- 不要把静态 BC 成功率直接当成动态 PPO 基线；
- 不要把 `ppo_kinematic_final.zip` 和 `best_model.zip` 混为一谈；
- 不要重新把 Dijkstra 路径加入学生观测；
- 不要只增加训练步数；v12 最佳结果出现在约 50k；
- 不要同时修改网络结构、奖励和课程，否则无法判断收益来源；
- 训练脚本默认使用项目本地 `.venv`，GPU 项目需要显式设置 `PYTHON_BIN`；
- 强制 `kill -9` 时出现 CUDA IPC 清理警告通常是次生现象，先检查是否 OOM 或被信号终止。

## 十、推荐启动方式

仅在确认当前 `ppo_v13` 已结束或明确需要停止后使用：

```bash
cd ~/code_lyx/HighSpeedUUV_Kinematic_GPU

PYTHON_BIN=/home/action-mc/code_lyx/HighSpeedUUV_Kinematic/.venv/bin/python \
nohup ./scripts/train.sh \
  --output output_model/ppo_v13 \
  --overwrite \
  > nohup_out/ppo_v13.log 2>&1 &
```
