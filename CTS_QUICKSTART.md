# CTS 快速入门指南

## 简介

CTS (Concurrent Teacher-Student) 是一种新型的强化学习训练范式，用于足式机器人运动控制。与传统的两阶段师生方法不同，CTS 同时训练教师和学生策略。

## 核心概念

```
教师（Teacher）: 使用特权信息（地形、接触力等）
学生（Student）: 仅使用本体感觉（IMU、关节编码器）
共享策略网络: 两者使用相同的动作网络
不同编码器: 教师编码特权信息，学生编码观测历史
```

## 安装

CTS 已集成到 Isaac Lab 中，无需额外安装。

## 5 分钟快速开始

### 步骤 1: 检查实现文件

```bash
cd /home/tino66/IsaacLab

# 查看核心文件
ls -l source/isaaclab_rl/isaaclab_rl/rsl_rl/cts*.py
ls -l source/isaaclab_rl/isaaclab_rl/rsl_rl/encoder_model.py
```

### 步骤 2: 准备环境配置

你的环境需要提供两组观测：

**1. Policy 观测（学生可见）:**
- 基座角速度（IMU）
- 投影重力（IMU）
- 速度命令
- 关节位置
- 关节速度
- 前一时刻动作

**2. Privileged 观测（教师可见）:**
- 所有 policy 观测
- 基座线速度（地面真值）
- 地形高度
- 接触力
- 其他特权信息

### 步骤 3: 配置现有环境

```python
# 示例：为 Anymal-C 配置 CTS
from isaaclab_rl.rsl_rl import configure_cts_observations
from isaaclab_tasks.manager_based.locomotion.velocity import velocity_env_cfg

# 加载基础配置
cfg = velocity_env_cfg.AnymalCRoughEnvCfg()

# 添加 CTS 观测结构
cfg = configure_cts_observations(cfg)
```

### 步骤 4: 开始训练

```bash
# 基础训练（8192 个环境，教师:学生 = 3:1）
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 8192 \
    --headless

# 自定义训练
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 8192 \
    --teacher_student_ratio 3.0 \
    --max_iterations 5000 \
    --seed 42 \
    --headless
```

### 步骤 5: 部署学生策略

训练完成后，只需部署学生策略：

```python
from isaaclab_rl.rsl_rl import CTSRunner
import torch

# 加载训练好的模型
runner = CTSRunner.load("logs/cts/model_5000.pt")

# 获取学生策略组件
student_encoder, student_policy = runner.alg.get_student_policy()

# 在机器人上使用
def robot_control_loop():
    obs_buffer = []  # 维护 5 步历史
    
    while True:
        # 获取当前观测
        current_obs = robot.get_proprioceptive_observations()
        
        # 更新历史
        obs_buffer.append(current_obs)
        if len(obs_buffer) > 5:
            obs_buffer.pop(0)
        
        # 编码观测历史
        obs_history = torch.cat(obs_buffer, dim=-1)
        latent = student_encoder(obs_history)
        
        # 生成动作
        policy_input = torch.cat([current_obs, latent], dim=-1)
        action = student_policy(policy_input)
        
        # 执行动作
        robot.execute_action(action)
```

## 配置参数说明

### 关键参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `num_envs` | 8192 | 总环境数 |
| `teacher_student_ratio` | 3.0 | 教师:学生比例（3:1） |
| `latent_dim` | 24 | 潜在向量维度 |
| `history_length` | 5 | 观测历史长度 |
| `max_iterations` | 5000 | 训练迭代次数 |
| `reconstruction_loss_coef` | 1.0 | 重建损失权重 |

### 网络结构

```python
# 编码器
Privileged Encoder: [512, 256] → 24
Proprioceptive Encoder: [512, 256] → 24

# 策略网络（共享）
Actor: [512, 256, 128] → num_actions
Critic: [512, 256, 128] → 1
```

## 训练监控

训练过程中会输出以下指标：

```
[CTS] Iteration 100/5000
  Time: 2.35s
  teacher/policy_loss: 0.1234
  teacher/value_loss: 0.5678
  teacher/entropy: 1.2345
  teacher/kl: 0.0089
  student/reconstruction_loss: 0.0456
```

**关键指标：**
- `teacher/policy_loss`: 教师策略损失（PPO）
- `teacher/value_loss`: 价值函数损失
- `student/reconstruction_loss`: 学生重建损失（越低越好）
- `teacher/kl`: KL 散度（用于自适应学习率）

## 常见问题

### Q1: 重建损失不下降？
**A:** 检查观测历史是否正确维护，确保特权观测包含有用信息。

### Q2: 学生性能差？
**A:** 增加教师环境比例，检查观测归一化，验证历史缓冲。

### Q3: 训练不稳定？
**A:** 降低学习率，增加环境数量，检查奖励函数。

### Q4: 如何调整教师/学生比例？
**A:** 使用 `--teacher_student_ratio` 参数。论文推荐 3:1。

### Q5: 训练需要多长时间？
**A:** 论文报告 RTX 4090 上 3000 次迭代约 105 分钟。

## 性能对比

根据论文，CTS 相比两阶段师生方法：
- 斜坡: 速度跟踪误差降低 **17.85%**
- 粗糙斜坡: 降低 **19.12%**
- 楼梯: 降低 **7.9%**
- 离散障碍: 降低 **21.85%**

## 高级用法

### 自定义编码器

```python
encoder_cfg = RslRlEncoderCfg(
    latent_dim=32,              # 更大的潜在空间
    hidden_dims=[1024, 512, 256],  # 更深的网络
    activation="relu",           # 不同的激活函数
    history_length=10,           # 更长的历史
)
```

### 添加视觉输入

```python
# 为特权编码器添加视觉特征
privileged_encoder_cfg = RslRlEncoderCfg(
    latent_dim=48,  # 增大以容纳视觉信息
    hidden_dims=[1024, 512, 256],
)
```

### 调整训练超参数

```python
algorithm_cfg = RslRlCtsAlgorithmCfg(
    num_learning_epochs=10,      # 更多 epoch
    learning_rate=5e-4,          # 调整学习率
    reconstruction_loss_coef=2.0, # 增大重建损失权重
)
```

## 下一步

1. **阅读完整文档**: `docs/CTS_README.md`
2. **查看实现细节**: `CTS_IMPLEMENTATION_SUMMARY.md`
3. **运行测试**: `python scripts/reinforcement_learning/rsl_rl/test_cts.py`
4. **开始训练**: 使用上述命令开始你的第一次训练
5. **调优性能**: 根据你的机器人调整参数

## 获取帮助

- 查看论文: https://arxiv.org/abs/2405.10830
- 项目主页: https://clearlab-sustech.github.io/concurrentTS/
- Isaac Lab 文档: https://isaac-sim.github.io/IsaacLab/

## 引用

如果使用此实现，请引用：

```bibtex
@article{wang2024cts,
  title={CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion},
  author={Wang, Hongxi and Luo, Haoxiang and Zhang, Wei and Chen, Hua},
  journal={IEEE Robotics and Automation Letters},
  year={2024}
}
```

---

**祝训练顺利！** 🚀🤖
