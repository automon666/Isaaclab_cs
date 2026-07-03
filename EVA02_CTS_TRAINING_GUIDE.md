# EVA02 CTS (Concurrent Teacher-Student) 训练指南

## ✅ 配置完成

EVA02 机器人已成功配置为使用你搭建的 CTS 网络进行训练！

## CTS 训练原理

CTS (Concurrent Teacher-Student) 是一种并发训练方法：
- **Teacher**: 使用特权信息（privileged info）训练，性能更好
- **Student**: 只使用本体感受信息（proprioceptive）+ 历史，需要学习重建 Teacher 的潜在表示
- **目标**: Student 在部署时无需特权信息，但能达到接近 Teacher 的性能

## 快速开始

### 1. 基础 CTS 训练（推荐）

```bash
conda activate Isaaclab_csl

# 使用 1536 个环境，Teacher:Student = 2:1
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Flat-EVA02-v0 \
    --num_envs 1536 \
    --teacher_student_ratio 2.0 \
    --max_iterations 2000 \
    --headless
```

### 2. 小规模测试（GPU 内存有限）

```bash
# 使用 512 个环境测试
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Flat-EVA02-v0 \
    --num_envs 512 \
    --teacher_student_ratio 2.0 \
    --max_iterations 100 \
    --headless
```

### 3. 完整规模训练

```bash
# 使用 8192 个环境（需要足够的 GPU 内存）
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Flat-EVA02-v0 \
    --num_envs 8192 \
    --teacher_student_ratio 3.0 \
    --max_iterations 2000 \
    --headless
```

## 训练参数说明

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--task` | 任务名称 | `Isaac-Velocity-Flat-EVA02-v0` |
| `--num_envs` | 总环境数量 | 512-8192 (根据 GPU) |
| `--teacher_student_ratio` | Teacher:Student 比例 | 2.0-3.0 |
| `--max_iterations` | 最大训练迭代 | 2000-4000 |
| `--headless` | 无头模式（节省显存） | 建议启用 |

### 环境分配示例

```python
--num_envs 1536 --teacher_student_ratio 2.0
→ Teacher: 1024 envs, Student: 512 envs

--num_envs 8192 --teacher_student_ratio 3.0
→ Teacher: 6144 envs, Student: 2048 envs
```

## CTS 网络结构

根据你的代码 (`source/isaaclab_rl/isaaclab_rl/rsl_rl/`)：

### Encoders
- **Privileged Encoder** (Teacher):
  - Input: policy obs (48) + privileged obs
  - Hidden: [512, 256]
  - Latent: 24 dims
  - No history

- **Proprioceptive Encoder** (Student):
  - Input: policy obs (48) × history_length (5)
  - Hidden: [512, 256]
  - Latent: 24 dims
  - With history buffer

### Actor/Critic (Shared)
- **Actor**: [512, 256, 128] → 12 actions
- **Critic**: [512, 256, 128] → 1 value

### Loss Functions
1. **PPO Loss** (Teacher & Student)
2. **Reconstruction Loss** (Student 重建 Teacher latent)
   - Weight: `reconstruction_loss_coef = 1.0`

## 观测空间

EVA02 的观测定义在 `eva02_env.py`:

```python
obs = {
    "policy": [
        base_lin_vel (3),      # 基座线速度
        base_ang_vel (3),      # 基座角速度
        projected_gravity (3), # 投影重力
        commands (3),          # 速度命令
        joint_pos (12),        # 关节位置
        joint_vel (12),        # 关节速度
        actions (12),          # 上一步动作
    ],  # Total: 48 dims
    
    "privileged": [
        # 可选：地形高度图、接触力等
        # 目前 EVA02 可能没有额外的 privileged obs
    ]
}
```

## 训练日志

训练日志保存在：
```
logs/cts/cts_Isaac-Velocity-Flat-EVA02-v0/YYYY-MM-DD_HH-MM-SS/
├── params/
│   └── cts_config.yaml      # CTS 配置
├── model_*.pt               # 保存的模型
└── tensorboard/             # TensorBoard 日志
```

### 查看训练进度

```bash
tensorboard --logdir logs/cts
# 打开 http://localhost:6006
```

## GPU 内存建议

基于你的 RTX 3060 Laptop (6GB)：

| 环境数 | 预估显存 | 训练速度 | 推荐场景 |
|--------|----------|----------|----------|
| 512 | ~2-3 GB | 慢 | 快速测试 |
| 1024 | ~4-5 GB | 中等 | 平衡训练 |
| 1536 | ~5-6 GB | 较快 | 推荐配置 |
| 2048+ | >6 GB | 快 | 需要更大 GPU |

**建议**：从 512 envs 开始测试，逐步增加到 1536 envs。

## 训练流程

1. **初始化** (5-10 分钟)
   - 加载 Isaac Sim
   - 创建环境
   - 初始化网络

2. **训练循环** (每 100 iters ~10-20 分钟)
   - Teacher 收集 rollouts
   - Student 收集 rollouts
   - Teacher 更新 (PPO)
   - Student 更新 (PPO + Reconstruction)

3. **保存模型** (每 100 iters)
   - `model_100.pt`, `model_200.pt`, ...

## 预期训练时间

基于 RTX 3060 + 1536 envs：

- **2000 iterations**: 约 3-4 小时
- **4000 iterations**: 约 6-8 小时

## 故障排除

### 1. GPU 内存不足
```bash
# 减少环境数量
--num_envs 512
```

### 2. 训练不稳定
```bash
# 增加 Student learning epochs
# 修改 train_cts.py 中的 student_num_learning_epochs
```

### 3. Student 性能差
```bash
# 增加 reconstruction loss 权重
# 修改 reconstruction_loss_coef 从 1.0 到 1.5
```

## 下一步

1. **运行小规模测试**:
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train_cts.py \
       --task Isaac-Velocity-Flat-EVA02-v0 \
       --num_envs 512 \
       --max_iterations 100 \
       --headless
   ```

2. **监控训练**:
   ```bash
   tensorboard --logdir logs/cts
   ```

3. **完整训练**:
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train_cts.py \
       --task Isaac-Velocity-Flat-EVA02-v0 \
       --num_envs 1536 \
       --teacher_student_ratio 2.0 \
       --max_iterations 2000 \
       --headless
   ```

---

**配置完成时间**: 2026-07-03  
**状态**: ✅ 可以开始 CTS 训练
