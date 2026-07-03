# CTS 框架搭建全过程（从零到完整）

## 🎯 总览：需要构建的 4 个核心模块

```
1. EncoderModel (encoder_model.py)      - 编码器网络
2. CTSAlgorithm (cts_algorithm.py)      - 训练算法
3. CTSRunner (cts_runner.py)            - 训练循环管理
4. Configuration (cts_cfg.py)           - 配置系统
```

---

## 第一步：搭建 EncoderModel（编码器）

### 为什么从这里开始？
因为编码器是整个 CTS 的"核心发明"！它负责：
- Teacher: 把完整状态（特权信息）压缩成 24 维向量
- Student: 把 5 步历史观测压缩成 24 维向量
- **关键**：这两个 24 维向量要在同一个空间！

### 设计决策 1：统一的 Encoder 类

```python
# 我选择用一个类，而不是两个类，因为：
# ✅ Teacher 和 Student 结构完全一样（只是输入不同）
# ✅ 减少代码重复
# ✅ 更容易保证它们在同一空间

class EncoderModel(nn.Module):
    def __init__(self, 
                 obs_set: str,          # "privileged" 或 "proprioceptive"
                 use_history: bool):     # Teacher=False, Student=True
```

### 设计决策 2：输入维度计算

```python
# Teacher Encoder:
# 输入 = policy_obs (48) + privileged_obs (72) = 120 维
input_dim = self._get_obs_dim(obs, obs_groups, "privileged")

# Student Encoder:
# 输入 = policy_obs (48) × history_length (5) = 240 维
if use_history:
    input_dim = self.obs_dim * history_length
```

**为什么这样设计？**
- Teacher 看到"现在的完整状态"
- Student 看到"过去 5 步的部分观测"
- 通过时间维度（历史）来弥补信息缺失！

### 设计决策 3：网络结构

```python
# 从论文 Table 1 得出：
Input: [120 or 240]
   ↓
Linear(input_dim → 512) + ELU
   ↓
Linear(512 → 256) + ELU
   ↓
Linear(256 → 24)  # latent_dim
   ↓
L2 Normalize  ⭐ 关键！
   ↓
Output: [24] (单位向量)
```

**为什么需要 L2 归一化？**
```python
def forward(self, obs):
    x = self.encoder(obs)
    z = F.normalize(x, p=2, dim=-1)  # ||z|| = 1
    return z

# 原因：
# 1. Teacher 和 Student 的输出必须在同一尺度
# 2. MSE(z_teacher, z_student) 才有意义
# 3. 限制潜在空间范围，训练更稳定
```

### 设计决策 4：观测组管理

```python
# 问题：Isaac Lab 的观测是分组的
obs = {
    "policy": [48],           # 本体感觉
    "critic": [72],           # 额外的 critic 信息
    "height_scan": [187],     # 地形高度
    "contact_forces": [12],   # 接触力
    ...
}

# 解决方案：obs_groups 配置
obs_groups = {
    "privileged": ["policy", "critic", "height_scan", "contact_forces"],
    "proprioceptive": ["policy"]  # Student 只能用这个
}

# 灵活性：用户可以自定义哪些信息是"特权"的
```

### 完整的 EncoderModel 类结构

```python
class EncoderModel(nn.Module):
    """
    职责：
    1. 接收不同类型的观测
    2. 计算正确的输入维度
    3. 构建 MLP 编码器
    4. L2 归一化输出
    """
    
    def __init__(self, obs, obs_groups, obs_set, ...):
        # 1. 解析观测维度
        self.obs_groups, self.obs_dim = self._get_obs_dim(...)
        
        # 2. 计算输入维度（是否使用历史）
        self.input_dim = self.obs_dim * history_length if use_history else self.obs_dim
        
        # 3. 观测归一化（BatchNorm）
        self.obs_normalizer = nn.BatchNorm1d(self.input_dim)
        
        # 4. 构建编码器
        self.encoder = self._build_mlp(...)
    
    def forward(self, obs):
        # 1. 提取并拼接观测组
        obs_tensor = self._extract_obs(obs)
        
        # 2. 归一化
        obs_normalized = self.obs_normalizer(obs_tensor)
        
        # 3. 编码
        latent = self.encoder(obs_normalized)
        
        # 4. L2 归一化（关键！）
        latent_normalized = F.normalize(latent, p=2, dim=-1)
        
        return latent_normalized
    
    def _extract_obs(self, obs):
        """从 TensorDict 中提取指定的观测组"""
        obs_list = []
        for group in self.obs_groups:
            obs_list.append(obs[group])
        return torch.cat(obs_list, dim=-1)
```

### 设计挑战与解决方案

**挑战 1：如何处理历史序列？**
```python
# 问题：Student 需要 5 步历史，但 forward() 只接收一个 obs
# 解决：在外部（Runner）中管理历史，拼接后传入

# 外部代码：
history_buffer = deque(maxlen=5)
for step in range(episode):
    obs = env.get_obs()
    history_buffer.append(obs["policy"])
    
    # 拼接成 [batch, 48*5]
    obs_history = torch.cat(list(history_buffer), dim=-1)
    
    # 传入 encoder
    z = student_encoder(obs_history)
```

**挑战 2：Teacher 和 Student 的观测不同怎么处理？**
```python
# Teacher:
obs_teacher = {
    "policy": [48],
    "critic": [72],
    "height_scan": [187],
    ...
}
teacher_input = extract_obs(obs_teacher, ["policy", "critic", "height_scan"])
# -> [48+72+187] = 307 维（示例）

# Student:
obs_history = obs["policy"]  # [48]
# 重复 5 次：[48, 48, 48, 48, 48] -> [240]
student_input = torch.cat([obs_t0, obs_t1, ..., obs_t4], dim=-1)
```

---

## 第二步：搭建 CTSAlgorithm（训练算法）

### 为什么需要这个模块？
EncoderModel 只是"网络结构"，Algorithm 负责"如何训练"：
1. Teacher PPO 更新
2. Student 监督学习
3. 两个优化器的协调

### 设计决策 1：算法类的职责划分

```python
class CTSAlgorithm:
    """
    职责：
    1. 管理所有网络（2 个 encoder + 1 个 policy + 1 个 critic）
    2. 实现 act_teacher() 和 act_student()
    3. 实现 update() 训练逻辑
    """
    
    def __init__(self, actor, critic, ...):
        # 复用 rsl_rl 的 Actor/Critic
        self.actor = actor
        self.critic = critic
        
        # 创建两个 encoder
        self.privileged_encoder = EncoderModel(
            obs_set="privileged",
            use_history=False
        )
        
        self.proprioceptive_encoder = EncoderModel(
            obs_set="proprioceptive", 
            use_history=True
        )
```

### 设计决策 2：双优化器设计

```python
# 关键设计：两个独立的优化器

# Teacher 优化器：联合训练 3 个模块
self.teacher_optimizer = torch.optim.Adam([
    *self.privileged_encoder.parameters(),  # 教师编码器
    *self.actor.parameters(),                # 策略网络（共享）
    *self.critic.parameters()                # 价值网络
], lr=1e-3)

# Student 优化器：只训练学生编码器
self.student_optimizer = torch.optim.Adam([
    *self.proprioceptive_encoder.parameters()  # 学生编码器（独立）
], lr=1e-3)
```

**为什么分开？**
```python
# Teacher PPO：强化学习，更新整个系统
loss_teacher = PPO_loss + value_loss - entropy_loss
loss_teacher.backward()
teacher_optimizer.step()
# 影响：privileged_encoder ✓, actor ✓, critic ✓

# Student 监督学习：只学习重建
loss_student = MSE(z_student, z_teacher.detach())
loss_student.backward()
student_optimizer.step()
# 影响：proprioceptive_encoder ✓
# 不影响：privileged_encoder ✗ (detach 隔离)
```

### 设计决策 3：动作选择方法

```python
def act_teacher(self, obs: TensorDict):
    """Teacher 动作选择"""
    # 1. 编码特权信息
    z_privileged = self.privileged_encoder(obs)  # [batch, 24]
    
    # 2. 拼接 policy obs
    obs_policy = obs["policy"]  # [batch, 48]
    policy_input = torch.cat([obs_policy, z_privileged], dim=-1)  # [batch, 72]
    
    # 3. 策略网络
    action_dist = self.actor({"policy": policy_input})
    actions = action_dist.sample()
    
    return actions, z_privileged  # 返回 z 用于后续训练

def act_student(self, obs: TensorDict, obs_history: Tensor):
    """Student 动作选择"""
    # 1. 编码历史序列
    z_proprio = self.proprioceptive_encoder(obs_history)  # [batch, 24]
    
    # 2. 拼接当前 policy obs
    obs_policy = obs["policy"]  # [batch, 48]
    policy_input = torch.cat([obs_policy, z_proprio], dim=-1)  # [batch, 72]
    
    # 3. 共享策略网络
    action_dist = self.actor({"policy": policy_input})
    actions = action_dist.sample()
    
    return actions
```

**关键点：输入维度一致！**
```python
teacher_input = [obs_policy(48), z_privileged(24)] = 72 维
student_input = [obs_policy(48), z_proprio(24)]    = 72 维

# 这样它们才能共享同一个策略网络！
```

### 设计决策 4：训练更新逻辑

```python
def update(self):
    """论文 Algorithm 1 的核心"""
    
    # ===== Phase 1: Teacher PPO Update =====
    for epoch in range(num_epochs):
        for mini_batch in teacher_storage:
            # 1. 重新前向传播
            z = self.privileged_encoder(mini_batch.obs)
            policy_input = torch.cat([mini_batch.obs["policy"], z], dim=-1)
            
            # 2. 计算 PPO 损失
            action_dist = self.actor({"policy": policy_input})
            log_prob = action_dist.log_prob(mini_batch.actions)
            ratio = torch.exp(log_prob - mini_batch.old_log_probs)
            
            # PPO clip
            surr1 = ratio * mini_batch.advantages
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * mini_batch.advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 3. 价值函数损失
            values = self.critic({"policy": policy_input})
            value_loss = F.mse_loss(values, mini_batch.returns)
            
            # 4. 熵正则
            entropy_loss = -action_dist.entropy().mean()
            
            # 5. 总损失
            loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
            
            # 6. 反向传播（更新 teacher_encoder + actor + critic）
            self.teacher_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher_params, max_grad_norm)
            self.teacher_optimizer.step()
    
    # ===== Phase 2: Student Encoder Update =====
    for epoch in range(num_epochs):
        for mini_batch in teacher_storage:  # 复用 teacher 数据
            # 1. Teacher 编码（目标）
            with torch.no_grad():
                z_teacher = self.privileged_encoder(mini_batch.obs)
            
            # 2. Student 编码（预测）
            z_student = self.proprioceptive_encoder(mini_batch.obs_history)
            
            # 3. 重建损失（论文 Equation 8）
            reconstruction_loss = F.mse_loss(z_student, z_teacher)
            
            # 4. 反向传播（只更新 student_encoder）
            self.student_optimizer.zero_grad()
            reconstruction_loss.backward()
            nn.utils.clip_grad_norm_(self.proprioceptive_encoder.parameters(), max_grad_norm)
            self.student_optimizer.step()
    
    return mean_policy_loss, mean_reconstruction_loss
```

### 设计挑战与解决方案

**挑战 1：如何防止 Student 影响 Teacher？**
```python
# 问题：如果 reconstruction_loss 反向传播到 privileged_encoder，
# 会导致 teacher 被 student "拉偏"

# 解决：detach()
z_teacher = self.privileged_encoder(obs).detach()  # 停止梯度

reconstruction_loss = MSE(z_student, z_teacher)
# 梯度只流向 z_student → proprioceptive_encoder
# 不流向 z_teacher → privileged_encoder
```

**挑战 2：如何存储历史序列用于训练？**
```python
# 问题：Student 更新需要 obs_history，但 rollout 时只存了当前 obs

# 解决：在 rollout 时同时存储
teacher_storage.add(
    obs=obs,
    actions=actions,
    z_latent=z_privileged  # 存储 teacher 的潜在向量
)

student_storage.add(
    obs=obs,
    obs_history=obs_history,  # ⭐ 存储历史序列
    actions=actions
)

# 训练时：
z_teacher = teacher_storage["z_latent"]  # 直接使用存储的
z_student = self.proprioceptive_encoder(student_storage["obs_history"])
```

---

## 第三步：搭建 CTSRunner（训练管理器）

### 为什么需要这个模块？
Algorithm 只管"怎么更新"，Runner 管理"整个训练流程"：
1. 环境交互（Rollout）
2. 环境分割（Teacher 6144 / Student 2048）
3. 观测历史管理
4. 训练循环控制

### 设计决策 1：环境分割策略

```python
class CTSRunner:
    def __init__(self, env, cfg, ...):
        total_envs = env.num_envs  # 8192
        
        # 根据论文推荐的 3:1 比例
        self.teacher_student_ratio = 3.0
        self.num_envs_teacher = int(total_envs * 3.0 / 4.0)  # 6144
        self.num_envs_student = total_envs - self.num_envs_teacher  # 2048
        
        # 环境索引分割
        self.teacher_env_ids = list(range(0, self.num_envs_teacher))
        self.student_env_ids = list(range(self.num_envs_teacher, total_envs))
```

**为什么 3:1？**
```python
# 论文实验发现：
# - Teacher 需要更多环境来探索（使用特权信息）
# - Student 环境少一点，但训练次数多（监督学习）
# - 总体样本效率：teacher (6144×24) vs student (2048×24)
# - 3:1 是最优平衡点
```

### 设计决策 2：观测历史缓冲

```python
from collections import deque

class CTSRunner:
    def __init__(self, ...):
        # 为每个 student 环境维护一个历史缓冲
        self.obs_history_buffers = [
            deque(maxlen=5) 
            for _ in range(self.num_envs_student)
        ]
    
    def _update_history_buffer(self, obs, env_ids):
        """更新历史缓冲"""
        for i, env_id in enumerate(env_ids):
            local_id = env_id - self.num_envs_teacher
            self.obs_history_buffers[local_id].append(obs[i])
    
    def _get_history_sequence(self, env_ids):
        """获取历史序列 [batch, 240]"""
        sequences = []
        for env_id in env_ids:
            local_id = env_id - self.num_envs_teacher
            buffer = self.obs_history_buffers[local_id]
            
            # 零填充（episode 初期）
            if len(buffer) < 5:
                padding = [torch.zeros_like(buffer[0])] * (5 - len(buffer))
                seq = list(padding) + list(buffer)
            else:
                seq = list(buffer)
            
            sequences.append(torch.cat(seq, dim=-1))
        
        return torch.stack(sequences)
    
    def _reset_history_buffers(self, env_ids):
        """环境重置时清空历史"""
        for env_id in env_ids:
            if env_id >= self.num_envs_teacher:
                local_id = env_id - self.num_envs_teacher
                self.obs_history_buffers[local_id].clear()
```

### 设计决策 3：并发 Rollout 收集

```python
def _collect_rollouts(self):
    """论文 Algorithm 1, Line 3"""
    
    obs = self.env.get_observations()
    
    # 分割观测
    obs_teacher = self._slice_obs(obs, self.teacher_env_ids)
    obs_student = self._slice_obs(obs, self.student_env_ids)
    
    # ===== Teacher Rollout =====
    for step in range(self.num_steps_per_env):  # 24 步
        # 1. Teacher 采样动作
        with torch.no_grad():
            actions_teacher, z_teacher = self.alg.act_teacher(obs_teacher)
        
        # 2. 执行动作（只在 teacher 环境）
        obs_teacher, rewards, dones, infos = self.env.step_partial(
            actions_teacher,
            env_ids=self.teacher_env_ids
        )
        
        # 3. 存储数据
        self.teacher_storage.add(
            obs=obs_teacher,
            actions=actions_teacher,
            rewards=rewards,
            dones=dones,
            z_latent=z_teacher  # 存储用于 student 训练
        )
        
        # 4. 处理 episode 终止
        if dones.any():
            reset_ids = torch.where(dones)[0]
            # Isaac Lab 会自动重置，我们只需记录
    
    # ===== Student Rollout =====
    for step in range(self.num_steps_per_env):  # 24 步
        # 1. 获取历史序列
        obs_history = self._get_history_sequence(self.student_env_ids)
        
        # 2. Student 采样动作
        with torch.no_grad():
            actions_student = self.alg.act_student(obs_student, obs_history)
        
        # 3. 执行动作（只在 student 环境）
        obs_student, rewards, dones, infos = self.env.step_partial(
            actions_student,
            env_ids=self.student_env_ids
        )
        
        # 4. 更新历史缓冲
        self._update_history_buffer(obs_student["policy"], self.student_env_ids)
        
        # 5. 存储数据
        self.student_storage.add(
            obs=obs_student,
            obs_history=obs_history,  # ⭐ 关键
            actions=actions_student,
            rewards=rewards,
            dones=dones
        )
        
        # 6. 处理 episode 终止
        if dones.any():
            reset_ids = torch.where(dones)[0]
            self._reset_history_buffers(reset_ids)
```

### 设计决策 4：训练循环

```python
def learn(self, num_learning_iterations):
    """主训练循环"""
    
    obs = self.env.get_observations()
    
    for iteration in range(num_learning_iterations):
        # 1. 收集 Rollouts
        self._collect_rollouts()
        
        # 2. 计算 GAE advantages
        self._compute_advantages()
        
        # 3. 训练更新
        mean_policy_loss, mean_value_loss, mean_reconstruction_loss = self.alg.update()
        
        # 4. 记录指标
        self.writer.add_scalar("Loss/policy", mean_policy_loss, iteration)
        self.writer.add_scalar("Loss/reconstruction", mean_reconstruction_loss, iteration)
        
        # 5. 保存检查点
        if iteration % save_interval == 0:
            self.save(f"model_{iteration}.pt")
        
        # 6. 清空存储
        self.teacher_storage.clear()
        self.student_storage.clear()
```

---

## 第四步：配置系统（cts_cfg.py）

### 为什么需要配置？
让用户可以调整超参数而不修改代码！

```python
@configclass
class RslRlEncoderCfg:
    """Encoder 配置"""
    latent_dim: int = 24
    hidden_dims: tuple[int, ...] = (512, 256)
    activation: str = "elu"
    obs_normalization: bool = True
    history_length: int = 5

@configclass
class RslRlCtsAlgorithmCfg:
    """Algorithm 配置"""
    # PPO 参数
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    
    # CTS 特定
    reconstruction_loss_coef: float = 1.0
    teacher_batch_size: int = 24576
    student_batch_size: int = 12288

@configclass
class RslRlCtsRunnerCfg:
    """Runner 配置"""
    teacher_student_ratio: float = 3.0
    num_steps_per_env: int = 24
    
    # 观测组定义
    obs_groups: dict = {
        "privileged": ["policy", "critic", "height_scan"],
        "proprioceptive": ["policy"]
    }
```

---

## 完整的数据流总结

```
Iteration i:
├─ 初始化
│  ├─ 分割环境：[0:6144] teacher, [6144:8192] student
│  └─ 初始化历史缓冲（2048 个 deque）
│
├─ Rollout (Teacher)
│  ├─ obs → privileged_encoder → z_t [24]
│  ├─ [obs_policy, z_t] → actor → actions
│  ├─ actions → env → next_obs, rewards
│  └─ 存储 → teacher_storage (6144×24 transitions)
│
├─ Rollout (Student)
│  ├─ history → proprioceptive_encoder → z_s [24]
│  ├─ [obs_policy, z_s] → actor → actions
│  ├─ actions → env → next_obs, rewards
│  ├─ 更新历史缓冲
│  └─ 存储 → student_storage (2048×24 transitions)
│
├─ Update (Teacher PPO)
│  ├─ 计算 PPO 损失
│  └─ 更新：privileged_encoder + actor + critic
│
└─ Update (Student Supervised)
   ├─ z_t = teacher_storage["z_latent"]
   ├─ z_s = proprioceptive_encoder(student_storage["obs_history"])
   ├─ L_rec = MSE(z_s, z_t.detach())
   └─ 更新：proprioceptive_encoder
```

这就是 CTS 框架从零到完整的搭建过程！每一步都有明确的设计决策和理由。
