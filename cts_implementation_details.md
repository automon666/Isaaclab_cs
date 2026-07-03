# CTS 实现的具体技术细节

## 1. EncoderModel（编码器）的实现原理

### 核心设计思想
```python
# 两种编码器：
# Teacher Encoder: 完整状态 → 潜在向量
# Student Encoder: 观测历史序列 → 潜在向量
```

### 关键实现细节

#### 1.1 输入处理
```python
# Teacher Encoder (privileged)
输入维度 = policy_obs_dim + privileged_obs_dim
例如: 48 (policy) + 72 (privileged) = 120 维

# Student Encoder (proprioceptive)  
输入维度 = policy_obs_dim × history_length
例如: 48 × 5 = 240 维（5步历史）
```

#### 1.2 网络结构
```python
self.encoder = self._build_mlp(input_dim, latent_dim, hidden_dims, activation)

# 具体展开：
Input Layer: [input_dim] 
  ↓
Hidden Layer 1: [512] + ELU
  ↓
Hidden Layer 2: [256] + ELU
  ↓
Output Layer: [24] (latent_dim)
  ↓
L2 Normalization (单位超球面)
```

#### 1.3 L2 归一化（关键技术）
```python
def forward(self, obs):
    latent = self.encoder(obs_normalized)
    # 映射到单位超球面（论文要求）
    latent_normalized = F.normalize(latent, p=2, dim=-1)
    return latent_normalized

# 为什么需要 L2 归一化？
# 1. 确保教师和学生的潜在向量在同一尺度
# 2. 使用 MSE 损失时更稳定
# 3. 限制潜在空间的范围，避免爆炸
```

#### 1.4 观测历史处理
```python
# Student Encoder 使用历史序列
if use_history:
    self.input_dim = self.obs_dim * history_length
    
# 实际使用：
obs_buffer = deque(maxlen=5)  # 维护5步历史
for step in range(episode_length):
    current_obs = env.get_obs()
    obs_buffer.append(current_obs)
    
    # 拼接历史
    obs_history = torch.cat(list(obs_buffer), dim=-1)
    latent = student_encoder(obs_history)
```

---

## 2. CTS Algorithm（算法）的实现原理

### 2.1 训练流程图

```
每次迭代:
┌─────────────────────────────────────────────────┐
│ Phase 1: Teacher Rollout                        │
│ ├─ 使用 privileged_encoder 编码完整状态         │
│ ├─ 策略网络生成动作                              │
│ └─ 收集轨迹数据                                  │
├─────────────────────────────────────────────────┤
│ Phase 2: Student Rollout                        │
│ ├─ 使用 proprioceptive_encoder 编码历史观测     │
│ ├─ 策略网络生成动作（共享网络）                  │
│ └─ 收集轨迹数据                                  │
├─────────────────────────────────────────────────┤
│ Phase 3: Teacher PPO Update                     │
│ ├─ 计算 PPO 策略损失                             │
│ ├─ 计算价值函数损失                              │
│ └─ 更新: privileged_encoder + actor + critic    │
├─────────────────────────────────────────────────┤
│ Phase 4: Student Encoder Update (监督学习)      │
│ ├─ z_teacher = privileged_encoder(full_obs)    │
│ ├─ z_student = proprioceptive_encoder(history) │
│ ├─ L_rec = MSE(z_student, z_teacher.detach()) │
│ └─ 更新: proprioceptive_encoder                 │
└─────────────────────────────────────────────────┘
```

### 2.2 关键代码实现

#### 2.2.1 Teacher 动作选择
```python
def act_teacher(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. 编码特权信息
    z_privileged = self.privileged_encoder(obs)  # [batch, 24]
    
    # 2. 获取本体感觉观测
    obs_proprio = obs["policy"]  # [batch, 48]
    
    # 3. 拼接输入策略网络
    policy_input = torch.cat([obs_proprio, z_privileged], dim=-1)  # [batch, 72]
    
    # 4. 生成动作
    actions = self.actor({"policy": policy_input}, stochastic_output=True)
    
    return actions, z_privileged
```

#### 2.2.2 Student 动作选择
```python
def act_student(self, obs: TensorDict, obs_history: torch.Tensor) -> torch.Tensor:
    # 1. 编码观测历史（5步）
    z_proprio = self.proprioceptive_encoder(obs_history)  # [batch, 24]
    
    # 2. 获取当前本体感觉观测
    obs_proprio = obs["policy"]  # [batch, 48]
    
    # 3. 拼接输入策略网络（与 teacher 相同的输入维度）
    policy_input = torch.cat([obs_proprio, z_proprio], dim=-1)  # [batch, 72]
    
    # 4. 生成动作（使用共享的策略网络）
    actions = self.actor({"policy": policy_input}, stochastic_output=True)
    
    return actions
```

#### 2.2.3 重建损失（核心创新）
```python
def _update_student_encoder(self) -> float:
    """监督学习：让学生编码器学习教师的潜在表示"""
    
    for batch in self.teacher_storage.mini_batch_generator():
        obs = batch["obs"]
        obs_history = batch["obs_history"]
        
        # Forward pass through both encoders
        with torch.no_grad():
            # 教师编码（目标）
            z_privileged = self.privileged_encoder(obs)  # [batch, 24]
        
        # 学生编码（预测）
        z_proprio = self.proprioceptive_encoder(obs_history)  # [batch, 24]
        
        # 重建损失（论文 Equation 8）
        # L^rec = ||E^s(o_{t-H:t}) - E^t(s_t)||^2
        reconstruction_loss = F.mse_loss(z_proprio, z_privileged.detach())
        
        # 反向传播（只更新学生编码器）
        self.student_optimizer.zero_grad()
        reconstruction_loss.backward()
        nn.utils.clip_grad_norm_(
            self.proprioceptive_encoder.parameters(), 
            self.max_grad_norm
        )
        self.student_optimizer.step()
    
    return reconstruction_loss.item()
```

### 2.3 优化器设计

```python
# Teacher 优化器（联合训练）
self.teacher_params = (
    list(self.privileged_encoder.parameters()) +  # 教师编码器
    list(self.actor.parameters()) +               # 策略网络（共享）
    list(self.critic.parameters())                # 价值网络
)
self.teacher_optimizer = torch.optim.Adam(self.teacher_params, lr=1e-3)

# Student 优化器（独立监督学习）
self.student_optimizer = torch.optim.Adam(
    self.proprioceptive_encoder.parameters(),  # 只更新学生编码器
    lr=1e-3
)
```

---

## 3. CTSRunner（训练管理器）的实现原理

### 3.1 环境分割策略

```python
# 根据比例分配环境
teacher_student_ratio = 3.0  # 论文推荐 3:1

total_envs = 8192
num_teacher = int(total_envs * 3.0 / 4.0)  # 6144 个教师环境
num_student = total_envs - num_teacher      # 2048 个学生环境

# 环境索引分割
teacher_indices = [0, 6144)
student_indices = [6144, 8192)
```

### 3.2 观测历史缓冲管理

```python
from collections import deque

class ObservationHistoryBuffer:
    def __init__(self, history_length=5, obs_dim=48):
        self.buffer = deque(maxlen=history_length)
        self.history_length = history_length
        self.obs_dim = obs_dim
    
    def push(self, obs):
        """添加新观测"""
        self.buffer.append(obs)
    
    def get_sequence(self):
        """获取历史序列 [batch, history_length * obs_dim]"""
        if len(self.buffer) < self.history_length:
            # 初始化阶段：用零填充
            padding = [torch.zeros_like(self.buffer[0])] * (self.history_length - len(self.buffer))
            sequence = list(padding) + list(self.buffer)
        else:
            sequence = list(self.buffer)
        
        return torch.cat(sequence, dim=-1)
    
    def reset(self, env_ids):
        """重置指定环境的历史"""
        # 对于终止的环境，清空对应的历史
        pass
```

### 3.3 并发 Rollout 收集

```python
def _collect_rollouts(self, obs: TensorDict):
    """并发收集教师和学生的轨迹"""
    
    # 分割观测
    obs_teacher = obs[0:self.num_envs_teacher]      # 前 6144 个环境
    obs_student = obs[self.num_envs_teacher:]       # 后 2048 个环境
    
    teacher_data = []
    student_data = []
    
    # ===== Teacher Rollout =====
    for step in range(self.teacher_steps_per_env):  # 24 步
        # 1. Teacher 采样动作
        actions, z_teacher = self.alg.act_teacher(obs_teacher)
        
        # 2. 环境交互
        next_obs, rewards, dones, infos = self.env.step_partial(
            actions, 
            env_indices=range(self.num_envs_teacher)
        )
        
        # 3. 存储数据
        teacher_data.append({
            "obs": obs_teacher,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "z_latent": z_teacher,  # 保存用于 reconstruction loss
        })
        
        obs_teacher = next_obs
    
    # ===== Student Rollout =====
    for step in range(self.student_steps_per_env):  # 24 步
        # 1. 获取观测历史
        obs_history = self.obs_history_buffer.get_sequence()
        
        # 2. Student 采样动作
        actions = self.alg.act_student(obs_student, obs_history)
        
        # 3. 环境交互
        next_obs, rewards, dones, infos = self.env.step_partial(
            actions,
            env_indices=range(self.num_envs_teacher, self.num_envs)
        )
        
        # 4. 存储数据
        student_data.append({
            "obs": obs_student,
            "obs_history": obs_history,  # 关键：保存历史用于训练
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        })
        
        # 5. 更新历史缓冲
        self.obs_history_buffer.push(next_obs["policy"])
        
        obs_student = next_obs
    
    return teacher_data, student_data
```

---

## 4. 关键技术细节

### 4.1 为什么教师和学生共享策略网络？

```python
# 策略网络的输入维度相同：
teacher_input = [obs_proprio(48), z_privileged(24)] = 72 维
student_input = [obs_proprio(48), z_proprio(24)]    = 72 维

# 优势：
# 1. 学生可以直接利用教师学到的策略知识
# 2. 减少参数量
# 3. 学生的 RL 更新也会改进教师（互相促进）
```

### 4.2 重建损失 vs 动作模仿

```python
# 传统两阶段师生（动作模仿）：
loss = MSE(student_action, teacher_action)
# 问题：学生无法完美复制教师（信息不对称）

# CTS（潜在空间重建）：
loss = MSE(z_student, z_teacher)
# 优势：
# 1. 学生学习"表示"而非"动作"
# 2. 允许学生用不同方式达到相同目标
# 3. 更灵活，性能更好
```

### 4.3 批量大小的设计

```python
# 论文设置：
teacher_batch_size = 8192 × 3 = 24576  # 3 个 mini-batch
student_batch_size = 2048 × 6 = 12288  # 6 个 mini-batch

# 为什么不同？
# 1. 教师环境多，需要更大的批量
# 2. 学生批量小但更新次数多（更细致的学习）
# 3. 总的样本利用率相近
```

### 4.4 训练稳定性技巧

```python
# 1. 梯度裁剪
nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# 2. 学习率自适应（基于 KL 散度）
if kl_divergence > desired_kl * 2:
    learning_rate *= 0.5
elif kl_divergence < desired_kl / 2:
    learning_rate *= 1.5

# 3. 价值函数裁剪
value_pred_clipped = old_values + torch.clamp(
    values - old_values, 
    -clip_param, 
    clip_param
)

# 4. Detach 教师潜在向量（避免反向传播到教师）
reconstruction_loss = MSE(z_student, z_teacher.detach())
```

---

## 5. 数据流图

```
环境交互:
  ┌─────────────────────────────────────────────┐
  │                                             │
  │  Teacher Envs (6144)                       │
  │    obs → privileged_encoder → z_t          │
  │    [obs, z_t] → policy → actions           │
  │    actions → env → next_obs, rewards       │
  │                                             │
  │  Student Envs (2048)                       │
  │    history → proprioceptive_encoder → z_s  │
  │    [obs, z_s] → policy → actions           │
  │    actions → env → next_obs, rewards       │
  │                                             │
  └─────────────────────────────────────────────┘
                      ↓
训练更新:
  ┌─────────────────────────────────────────────┐
  │                                             │
  │  Teacher PPO:                              │
  │    L_policy = PPO_loss(teacher_data)       │
  │    L_value = MSE(V(s), returns)            │
  │    ∇[privileged_enc, policy, critic]       │
  │                                             │
  │  Student Reconstruction:                    │
  │    L_rec = MSE(z_s, z_t.detach())          │
  │    ∇[proprioceptive_encoder]               │
  │                                             │
  │  Student PPO (可选):                        │
  │    L_policy = PPO_loss(student_data)       │
  │    ∇[policy]                                │
  │                                             │
  └─────────────────────────────────────────────┘
```

---

## 6. 与论文对应的实现

| 论文算法步骤 | 代码实现 |
|------------|---------|
| Algorithm 1, Line 3 | `_collect_rollouts()` |
| Algorithm 1, Line 4 | GAE 计算（待集成） |
| Algorithm 1, Line 7 | `_update_teacher()` |
| Algorithm 1, Line 8 | `critic.update()` |
| Algorithm 1, Line 11 | `_update_student_encoder()` |
| Equation 3 (Teacher PPO) | `_update_teacher()` 中的 PPO 损失 |
| Equation 8 (Reconstruction) | `F.mse_loss(z_proprio, z_privileged)` |

---

这就是 CTS 实现的核心技术细节。整个设计非常巧妙地平衡了：
1. 教师的探索能力（使用特权信息）
2. 学生的泛化能力（仅用本体感觉）
3. 知识迁移的效率（潜在空间蒸馏）
