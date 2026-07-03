# CTS (Concurrent Teacher-Student) Implementation for Isaac Lab

This directory contains the implementation of CTS (Concurrent Teacher-Student) reinforcement learning architecture for legged locomotion in Isaac Lab.

## Overview

CTS is a novel training paradigm that trains teacher and student policies **concurrently** using reinforcement learning, unlike traditional two-stage teacher-student approaches. This implementation is based on the paper:

> Wang et al. "CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion" (2024)
> [arXiv:2405.10830](https://arxiv.org/abs/2405.10830)

### Key Features

- **Concurrent Training**: Teacher and student are trained simultaneously, not sequentially
- **Shared Policy Network**: Both teacher and student use the same policy network
- **Different Encoders**: 
  - Teacher encoder processes privileged information (terrain, contact forces, etc.)
  - Student encoder processes observation history (only proprioceptive sensors)
- **Reconstruction Loss**: Student encoder learns to mimic teacher encoder's latent representation

## Architecture

```
Teacher Path:
  Privileged Obs → Privileged Encoder → Latent (24D) → Policy → Actions
                                                      → Critic → Value

Student Path:
  Obs History → Proprioceptive Encoder → Latent (24D) → Policy → Actions
  (5 steps)                                            → Critic → Value

Training:
  1. Teacher PPO Update (using privileged info)
  2. Student Encoder Supervised Learning (reconstruction loss)
  3. Student PPO Update (optional, using proprioceptive only)
```

## Files Structure

```
isaaclab_rl/rsl_rl/
├── encoder_model.py              # Encoder network for CTS
├── cts_algorithm.py              # CTS training algorithm
├── cts_runner.py                 # Training loop manager
├── cts_cfg.py                    # Configuration classes
└── cts_observation_config.py     # Observation setup guide

scripts/reinforcement_learning/rsl_rl/
└── train_cts.py                  # Training script
```

## Installation

CTS is integrated into Isaac Lab's rsl_rl wrapper. No additional installation required beyond Isaac Lab dependencies.

## Quick Start

### 1. Prepare Your Environment

Your environment must provide two observation groups:
- `policy`: Proprioceptive observations (IMU, joint encoders, commands)
- `privileged`: Full state including terrain, contact forces, etc.

Example:
```python
from isaaclab_rl.rsl_rl import configure_cts_observations

# Configure an existing environment for CTS
cfg = configure_cts_observations(your_env_cfg)
```

### 2. Train with CTS

```bash
# Basic training
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 8192 \
    --headless

# With custom teacher/student ratio
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 8192 \
    --teacher_student_ratio 3.0 \
    --max_iterations 5000 \
    --headless

# With video recording
python scripts/reinforcement_learning/rsl_rl/train_cts.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 8192 \
    --video \
    --video_interval 2000
```

### 3. Deploy Student Policy

After training, deploy only the student policy (proprioceptive encoder + actor):

```python
from isaaclab_rl.rsl_rl import CTSRunner

# Load trained model
runner = CTSRunner.load(checkpoint_path)

# Get student policy for deployment
student_encoder, student_policy = runner.alg.get_student_policy()

# Use in your robot
obs_history = collect_observation_history(history_length=5)
latent = student_encoder(obs_history)
action = student_policy(torch.cat([current_obs, latent], dim=-1))
```

## Configuration

### Default CTS Parameters (from paper)

```python
CTS Configuration:
├── Latent Dimension: 24
├── Encoder Hidden Layers: [512, 256]
├── Policy Hidden Layers: [512, 256, 128]
├── History Length: 5 steps
├── Teacher/Student Ratio: 3:1 (6144/2048 environments)
├── Reconstruction Loss Weight: 1.0
└── Learning Rate: 1e-3
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | 24 | Dimension of encoder output |
| `history_length` | 5 | Observation history for student |
| `teacher_student_ratio` | 3.0 | Ratio of teacher to student envs |
| `reconstruction_loss_coef` | 1.0 | Weight for L^rec |
| `num_learning_epochs` | 5 | PPO epochs per iteration |
| `clip_param` | 0.2 | PPO clipping parameter |

## Observation Setup

### Policy Observations (Proprioceptive - Student Accessible)

```python
- Base angular velocity (IMU)
- Projected gravity (IMU)
- Velocity commands
- Joint positions
- Joint velocities
- Previous actions
```

### Privileged Observations (Teacher Only)

```python
- All policy observations
+ Base linear velocity (ground truth)
+ Terrain heights (3D scan)
+ Contact forces
+ External disturbances
+ Friction coefficients
```

## Performance Comparison

According to the paper, CTS achieves:
- **17.85% lower** tracking error on slopes vs. two-stage teacher-student
- **19.12% lower** tracking error on rough slopes
- **21.85% lower** tracking error on discrete obstacles
- **Better robustness** to external disturbances

## Training Tips

1. **Environment Split**: Use 3:1 teacher/student ratio (paper recommendation)
2. **Batch Sizes**: Teacher 24576, Student 12288 (paper values)
3. **Curriculum**: Start with easy terrain, gradually increase difficulty
4. **Domain Randomization**: Essential for sim-to-real transfer
5. **Training Time**: ~105 minutes for 3000 iterations on RTX 4090

## Troubleshooting

### Issue: Reconstruction loss not decreasing
- Check that observation history is properly maintained
- Verify privileged observations contain useful information
- Try increasing student learning rate

### Issue: Student performs poorly
- Increase teacher/student environment ratio
- Check observation normalization
- Verify history buffer is correctly implemented

### Issue: Training unstable
- Reduce learning rate
- Increase number of environments
- Check reward function scaling

## Advanced Usage

### Custom Encoder Architecture

```python
from isaaclab_rl.rsl_rl import RslRlEncoderCfg

encoder_cfg = RslRlEncoderCfg(
    latent_dim=32,  # Custom latent dimension
    hidden_dims=[1024, 512, 256],  # Deeper network
    activation="relu",  # Different activation
    use_history=True,
    history_length=10,  # Longer history
)
```

### Multi-Modal Observations

```python
# Add visual observations to privileged encoder
privileged_encoder_cfg = RslRlEncoderCfg(
    latent_dim=48,  # Larger for vision
    hidden_dims=[1024, 512, 256],
    # Add CNN layers for image processing
)
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{wang2024cts,
  title={CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion},
  author={Wang, Hongxi and Luo, Haoxiang and Zhang, Wei and Chen, Hua},
  journal={IEEE Robotics and Automation Letters},
  year={2024}
}
```

## References

- Paper: https://arxiv.org/abs/2405.10830
- Project Page: https://clearlab-sustech.github.io/concurrentTS/
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- RSL-RL: https://github.com/leggedrobotics/rsl_rl

## License

This implementation follows Isaac Lab's BSD-3-Clause License.
