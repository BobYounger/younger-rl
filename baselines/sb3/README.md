# SB3 Baselines

Third-party baselines live here so they stay separate from the hand-written implementations under `rl/`.

## PPO on Pendulum

Install dependencies first:

```bash
./.venv/bin/pip install stable-baselines3
```

Run:

```bash
./.venv/bin/python baselines/sb3/ppo_pendulum.py --env Pendulum-v1 --device cpu --run-name sb3-ppo-pendulum
```

## SAC on Pendulum

Run:

```bash
./.venv/bin/python baselines/sb3/sac_pendulum.py --env Pendulum-v1 --device cpu --run-name sb3-sac-pendulum
```

## DDPG on Pendulum

Run:

```bash
./.venv/bin/python baselines/sb3/ddpg_pendulum.py --env Pendulum-v1 --device cpu --run-name sb3-ddpg-pendulum
```

Outputs are written to:

```bash
runs/sb3/<run-name>/
```
