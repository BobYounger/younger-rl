# Younger RL

A collection of reinforcement learning algorithm implementations.

## Algorithms

### DDPG (Deep Deterministic Policy Gradient)

Implementation of DDPG for continuous action spaces.

**Usage**:
```bash
./.venv/bin/python scripts/train.py --algo ddpg --env Pendulum-v1
```

### SAC (Soft Actor-Critic)

Implementation of SAC for continuous action spaces in the unified `rl/` training framework.

**Usage**:
```bash
./.venv/bin/python scripts/train.py --algo sac --env Pendulum-v1
```

### SAC (Soft Actor-Critic) - Discrete Version

Implementation of SAC algorithm adapted for discrete action spaces.

📖 **Documentation**: [sac-design.md](./sac/sac-design.md)

**Usage**:
```bash
cd sac/
python train.py
```

### SAC (Soft Actor-Critic) - Continuous Version

Implementation of SAC algorithm for continuous action spaces.

📖 **Documentation**: [sac-design.md](./sac2/sac2-design.md)

**Usage**:
```bash
cd sac2/
python train.py
```
