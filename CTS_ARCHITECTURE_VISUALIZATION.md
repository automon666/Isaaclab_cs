# CTS 架构可视化详解

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CTS Training Architecture                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│      TEACHER PATH (6144 envs)    │  │     STUDENT PATH (2048 envs)     │
└───────────────────────────────────┘  └───────────────────────────────────┘

┌──────────────────────────┐          ┌──────────────────────────┐
│  Privileged Observations │          │  Observation History     │
│  - Base velocity (GT)    │          │  - t, t-1, t-2, t-3, t-4│
│  - Terrain heights       │          │  - Only proprioceptive   │
│  - Contact forces        │          │  - IMU + Joint encoders  │
│  - Friction coefficients │          │                          │
│  + Policy observations   │          │                          │
└──────────┬───────────────┘          └──────────┬───────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│  Privileged Encoder      │          │  Proprioceptive Encoder  │
│  Input: 120 dim          │          │  Input: 240 dim (48×5)   │
│  ├─ FC: 512 + ELU        │          │  ├─ FC: 512 + ELU        │
│  ├─ FC: 256 + ELU        │          │  ├─ FC: 256 + ELU        │
│  └─ FC: 24 (latent)      │          │  └─ FC: 24 (latent)      │
│  L2 Normalize            │          │  L2 Normalize            │
└──────────┬───────────────┘          └──────────┬───────────────┘
           │                                     │
           │ z_privileged [24]                   │ z_proprio [24]
           │                                     │
           │                                     │
           └────────────┬────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  Policy Network (SHARED)          │
        │  Input: obs [48] + latent [24]    │
        │  ├─ FC: 512 + ELU                 │
        │  ├─ FC: 256 + ELU                 │
        │  ├─ FC: 128 + ELU                 │
        │  └─ FC: num_actions (12)          │
        │  Gaussian Distribution            │
        └───────────────┬───────────────────┘
                        │
                        ▼
                   ┌─────────┐
                   │ Actions │
                   └─────────┘

═══════════════════════════════════════════════════════════════════════════
                            TRAINING UPDATES
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Teacher PPO Update (RL)                                       │
│  ───────────────────────────────────────────────────────────────────────│
│  Optimize: [Privileged Encoder] + [Policy] + [Critic]                  │
│                                                                          │
│  Loss = L_PPO + λ_value × L_value - λ_entropy × H(π)                   │
│                                                                          │
│  L_PPO = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)                     │
│  where ratio = π_new / π_old                                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 2: Student Encoder Update (Supervised Learning)                  │
│  ───────────────────────────────────────────────────────────────────────│
│  Optimize: [Proprioceptive Encoder] ONLY                               │
│                                                                          │
│  L_rec = MSE(z_proprio, z_privileged.detach())                          │
│        = ||E_s(o_{t-H:t}) - E_t(s_t)||²                                │
│                                                                          │
│  Key: z_privileged is DETACHED (no gradient to teacher)                │
└─────────────────────────────────────────────────────────────────────────┘

```

## 详细组件说明

### 1. Encoder 内部结构

```python
class EncoderModel(nn.Module):
    """
    ┌──────────────────────────────────────────┐
    │         Encoder Architecture              │
    ├──────────────────────────────────────────┤
    │                                           │
    │  Input: [batch, input_dim]               │
    │    ↓                                      │
    │  BatchNorm (if enabled)                  │
    │    ↓                                      │
    │  Linear(input_dim → 512)                 │
    │    ↓                                      │
    │  ELU()                                    │
    │    ↓                                      │
    │  Linear(512 → 256)                       │
    │    ↓                                      │
    │  ELU()                                    │
    │    ↓                                      │
    │  Linear(256 → 24)                        │
    │    ↓                                      │
    │  L2 Normalize (||z|| = 1)                │
    │    ↓                                      │
    │  Output: [batch, 24]                     │
    │                                           │
    └──────────────────────────────────────────┘
    """
    
    def forward(self, obs):
        # Normalize input
        x = self.obs_normalizer(obs)
        
        # MLP encoding
        x = self.encoder(x)
        
        # L2 normalization (critical!)
        z = F.normalize(x, p=2, dim=-1)
        
        return z
```

### 2. 训练循环时间线

```
Iteration i:
│
├─ t=0: Collect Teacher Rollouts (24 steps × 6144 envs)
│   │
│   ├─ Step 0: obs_t → z_t → action_t → env → obs_{t+1}, r_t
│   ├─ Step 1: obs_{t+1} → z_{t+1} → action_{t+1} → ...
│   └─ ...
│   └─ Step 23: collect 6144×24 = 147,456 transitions
│
├─ t=24: Collect Student Rollouts (24 steps × 2048 envs)
│   │
│   ├─ Step 0: history_t → z_s → action_t → env → obs_{t+1}, r_t
│   ├─ Step 1: update history → z_s → ...
│   └─ ...
│   └─ Step 23: collect 2048×24 = 49,152 transitions
│
├─ t=48: Compute GAE advantages and returns
│   │
│   └─ A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
│       where δ_t = r_t + γV(s_{t+1}) - V(s_t)
│
├─ t=49: Teacher PPO Update (5 epochs × 4 mini-batches)
│   │
│   ├─ Epoch 0:
│   │   ├─ Mini-batch 0: Update privileged_encoder + policy + critic
│   │   ├─ Mini-batch 1: ...
│   │   ├─ Mini-batch 2: ...
│   │   └─ Mini-batch 3: ...
│   └─ ...
│   └─ Epoch 4: Final updates
│
├─ t=50: Student Encoder Update (5 epochs × 4 mini-batches)
│   │
│   ├─ Epoch 0:
│   │   ├─ Mini-batch 0: z_s = E_s(history), z_t = E_t(obs).detach()
│   │   │                 loss = MSE(z_s, z_t)
│   │   │                 backward → update E_s only
│   │   └─ ...
│   └─ Epoch 4: Final updates
│
└─ t=51: Save metrics, checkpoint, repeat
```

### 3. 观测历史缓冲详解

```python
"""
Student 需要维护 5 步观测历史

时刻 t: buffer = [o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t]
               ↓ concat
        history = [48×5 = 240 维向量]
               ↓
        z_s = E_s(history)  # [batch, 24]

环境重置时的处理：
─────────────────────────────────────
Episode 开始（step 0-4）:
  step 0: buffer = [0, 0, 0, 0, o_0]  # 零填充
  step 1: buffer = [0, 0, 0, o_0, o_1]
  step 2: buffer = [0, 0, o_0, o_1, o_2]
  step 3: buffer = [0, o_0, o_1, o_2, o_3]
  step 4: buffer = [o_0, o_1, o_2, o_3, o_4]  # 完整

Episode 进行中（step 5+）:
  step 5: buffer = [o_1, o_2, o_3, o_4, o_5]  # 滑动窗口
  step 6: buffer = [o_2, o_3, o_4, o_5, o_6]
  ...

环境终止时:
  - 清空该环境的历史
  - 用零填充重新开始
"""
```

### 4. 梯度流向图

```
Forward Pass (Teacher):
═══════════════════════════════════════════════════════════════
obs → Privileged_Encoder → z_t → [obs, z_t] → Policy → action
                                 ↓
                              Critic → V(s)

Backward Pass (Teacher PPO):
═══════════════════════════════════════════════════════════════
L_PPO + L_value ← Policy ← [obs, z_t] ← z_t ← Privileged_Encoder
                   ↓
                Critic

✓ 所有组件都接收梯度
✓ 教师端到端训练


Forward Pass (Student):
═══════════════════════════════════════════════════════════════
history → Proprioceptive_Encoder → z_s → [obs, z_s] → Policy → action

Reconstruction Loss:
═══════════════════════════════════════════════════════════════
obs → Privileged_Encoder → z_t (DETACHED)
                             ↓
history → Proprioceptive_Encoder → z_s
                                    ↓
                                L_rec = MSE(z_s, z_t)

Backward Pass (Student):
═══════════════════════════════════════════════════════════════
L_rec → Proprioceptive_Encoder (ONLY)

✗ 梯度不回传到 Privileged_Encoder (detach)
✓ 只更新学生编码器
```

### 5. 内存和计算开销

```
Memory Breakdown (8192 envs total):
══════════════════════════════════════════════════════════════

Model Parameters:
├─ Privileged Encoder:    ~670K params
├─ Proprioceptive Encoder: ~670K params
├─ Policy Network:        ~680K params
└─ Critic Network:        ~680K params
   Total:                 ~2.7M params × 4 bytes = ~11 MB

Rollout Storage (per iteration):
├─ Teacher (6144 envs × 24 steps):
│  ├─ Observations: 120 × 147,456 × 4 bytes ≈ 71 MB
│  ├─ Actions: 12 × 147,456 × 4 bytes ≈ 7 MB
│  └─ Other: ~20 MB
│  Subtotal: ~98 MB
│
└─ Student (2048 envs × 24 steps):
   ├─ Observations: 48 × 49,152 × 4 bytes ≈ 9 MB
   ├─ History: 240 × 49,152 × 4 bytes ≈ 47 MB
   ├─ Actions: 12 × 49,152 × 4 bytes ≈ 2 MB
   └─ Other: ~10 MB
   Subtotal: ~68 MB

Total per iteration: ~177 MB (manageable)

GPU Memory Peak:
├─ Model: ~11 MB
├─ Optimizer states: ~33 MB (3× params for Adam)
├─ Rollout buffer: ~177 MB
├─ Mini-batch activations: ~50 MB
└─ Temporary tensors: ~100 MB
   Peak Total: ~371 MB (非常高效！)
```

### 6. 超参数敏感度分析

```python
"""
Critical Hyperparameters:
─────────────────────────────────────────────

1. latent_dim = 24
   - 太小(8-16): 信息瓶颈，学生无法重建
   - 太大(48-64): 过拟合，计算开销大
   - 推荐: 24 (论文验证)

2. history_length = 5
   - 太短(1-2): 无法推断动态信息
   - 太长(10+): 计算慢，可能过时信息
   - 推荐: 5 (平衡)

3. teacher_student_ratio = 3.0
   - 比例太低(<2): 教师探索不足
   - 比例太高(>5): 学生样本少，学习慢
   - 推荐: 3.0 (论文验证)

4. reconstruction_loss_coef = 1.0
   - 太小(0.1): 学生学不到教师知识
   - 太大(10): 可能过度拟合教师
   - 推荐: 1.0 (平衡)

5. learning_rate = 1e-3
   - 太小(1e-4): 收敛慢
   - 太大(1e-2): 不稳定
   - 推荐: 1e-3 with adaptive schedule
"""
```

## 关键实现技巧总结

### ✅ 必须做对的事

1. **L2 归一化**: 确保 `F.normalize(latent, p=2, dim=-1)`
2. **Detach 教师**: `z_privileged.detach()` 在重建损失中
3. **历史管理**: 正确维护和重置观测历史
4. **环境分割**: 确保教师/学生使用不同环境
5. **共享策略**: Teacher 和 Student 使用同一个 policy 网络

### ⚠️ 常见陷阱

1. **忘记归一化**: 导致潜在向量尺度不一致
2. **梯度泄漏**: 没有 detach 导致教师被学生影响
3. **历史混淆**: Episode 重置时没有清空历史
4. **批量大小**: Teacher 和 Student 批量不匹配
5. **观测对齐**: 确保教师和学生看到的 policy obs 一致

### 🎯 性能优化建议

1. **使用 GPU**: 所有张量移到 CUDA
2. **批量归一化**: 而非逐样本归一化
3. **混合精度**: 使用 `torch.cuda.amp` 加速
4. **异步环境**: Rollout 期间 overlap 计算
5. **梯度累积**: 对于大批量训练

这就是 CTS 实现的完整技术细节！
