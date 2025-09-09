# train.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import gymnasium as gym
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from replay_buffer import ReplayBuffer, RolloutCollector
from algorithm import SACDiscreteVAlgorithm
from config import SACConfig

def set_seed(seed: int) -> None:
    """设置随机种子（numpy/torch/环境等）。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def infer_env_dims(env_id: str, seed: int = 0) -> Tuple[int, int]:
    """推断环境的 state_dim 与 act_dim。"""
    env = gym.make(env_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Get state dimension
    obs, _ = env.reset(seed=seed)
    if isinstance(obs, np.ndarray):
        state_dim = obs.shape[0] if obs.ndim == 1 else np.prod(obs.shape)
    else:
        state_dim = 1
    
    # Get action dimension (assume discrete)
    if hasattr(env.action_space, 'n'):
        act_dim = env.action_space.n
    else:
        raise ValueError(f"Environment {env_id} does not have discrete action space")
    
    env.close()
    return int(state_dim), int(act_dim)

def save_policy_gif(env_id: str, algo: SACDiscreteVAlgorithm, device: str = "cpu", seed: int = 0) -> None:
    """保存训练后策略的演示GIF"""
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    
    with torch.no_grad():
        state, _ = env.reset(seed=seed)
        for _ in range(500):
            frames.append(env.render())
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=torch.device(device))
            action = algo.select_action(state_tensor, eval_mode=True)
            action = int(action.item()) if hasattr(action, 'item') else int(action)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    
    env.close()
    
    # 简单保存为GIF（需要imageio）
    try:
        import imageio
        imageio.mimsave(f"{env_id}_policy.gif", frames, fps=30)
        print(f"GIF saved: {env_id}_policy.gif ({len(frames)} frames)")
    except ImportError:
        print("Install imageio to save GIF: pip install imageio")

def evaluate(env_id: str, algo: SACDiscreteVAlgorithm, episodes: int = 5,
             device: str = "cpu", seed: int = 0) -> Tuple[float, float]:
    """eval_mode=True 下评估若干回合，返回 (avg_return, avg_length)。"""
    # 创建评估环境
    env = gym.make(env_id)
    device = torch.device(device)
    episode_returns, episode_lengths = [], []
    
    # 关闭梯度计算 - 整个评估过程都不需要梯度
    with torch.no_grad():
        # 循环运行指定次数的episodes
        for ep in range(episodes):
            # 重置环境，每个episode使用不同的种子
            state, _ = env.reset(seed=seed + ep)
            done = False
            episode_return, episode_length = 0.0, 0
            while not done:
                state = torch.as_tensor(state, dtype=torch.float32, device=device)  # (1, state_dim)
                action = algo.select_action(state, eval_mode=True)
                action = int(action.item()) if hasattr(action, 'item') else int(action)
                state, reward, terminated, truncated, _ = env.step(action)
                # 判断终止条件
                done = bool(terminated or truncated)
                # 累加统计
                episode_return += float(reward)
                episode_length += 1
            # 记录本回合的结果
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
    env.close()
    avg_return = float(np.mean(episode_returns))
    avg_length = float(np.mean(episode_lengths))
    
    return avg_return, avg_length

def build_components(env_id: str, cfg: SACConfig):
    """根据 cfg 构造 ReplayBuffer / SACDiscreteVAlgorithm / RolloutCollector 等组件。"""
    # 推断环境维度
    state_dim, act_dim = infer_env_dims(env_id, seed=cfg.seed)
    print(f"Environment: {env_id}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {act_dim}")
    
    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(
        state_dim=state_dim, 
        capacity=cfg.buffer_capacity, 
        device=cfg.device
    )
    
    # 创建SAC算法
    algorithm = SACDiscreteVAlgorithm(
        state_dim=state_dim,
        act_dim=act_dim, 
        cfg=cfg
    )
    
    # 创建经验采集器
    collector = RolloutCollector(
        env_id=env_id,
        algorithm=algorithm,
        buffer=replay_buffer,
        seed=cfg.seed,
        device=cfg.device
    )
    
    return replay_buffer, algorithm, collector

def warmup_random(collector: RolloutCollector, n_steps: int) -> None:
    """用随机策略进行 warmup 采样以填充经验池。"""
    logs = collector.collect(n_steps=n_steps,mode="random")
    steps = int(logs.get("steps",n_steps))
    eps   = int(logs.get("episodes_finished", 0))
    print(f"[warmup] collected {steps} steps, finished {eps} episodes.")


def train_loop(env_id: str,
                algo: SACDiscreteVAlgorithm,
                buffer: ReplayBuffer,
                collector: RolloutCollector,
                cfg: SACConfig) -> None:
    """主训练循环：交互→采样→update→评估/日志。"""
    # 读取配置
    total_steps            = cfg.train_total_steps
    batch_size             = cfg.buffer_batch_size
    warmup_steps           = cfg.collector_warmup_steps
    update_every           = cfg.train_update_every
    eval_every             = cfg.train_eval_every      # 修正拼写
    eval_episodes          = cfg.collector_eval_episodes
    device                 = cfg.device
    seed                   = cfg.seed

    # warmup
    warmup_random(collector=collector, n_steps=warmup_steps)

    last_logs = {}
    try:
        for t in range(warmup_steps, total_steps):
            # 1) 交互一步（策略采样）
            collector.collect(n_steps=1, mode="stochastic")

            # 2) 条件更新
            if len(buffer) >= batch_size:
                for _ in range(update_every):
                    batch = buffer.sample_batch(batch_size)
                    last_logs = algo.update(batch)

            # 3) 定期评估
            if (t + 1) % eval_every == 0:
                avg_return, avg_length = evaluate(
                    env_id=env_id, algo=algo,
                    episodes=eval_episodes, device=device, seed=seed + 42
                )
                # 选几个关键指标打印
                if last_logs:
                    print(f"[step {t+1:>6}] return={avg_return:.1f} len={avg_length:.1f} | "
                        f"pi_loss_pi={last_logs.get('pi_loss_pi', float('nan')):.3f} "
                        f"v_loss_v={last_logs.get('v_loss_v', float('nan')):.3f} "
                        f"q_loss_q0={last_logs.get('q_loss_q0', float('nan')):.3f} "
                        f"q_loss_q1={last_logs.get('q_loss_q1', float('nan')):.3f} "
                        f"H={last_logs.get('pi_entropy', float('nan')):.3f}| alpha={float(algo.alpha):.4f}")
                else:
                    print(f"[step {t+1:>6}] return={avg_return:.1f} len={avg_length:.1f}")
    finally:
        # 4) 最终评估 + 关闭环境
        end_return, end_length = evaluate(
            env_id=env_id, algo=algo,
            episodes=max(10, eval_episodes), device=device, seed=seed + 100
        )
        print(f"\n[final] avg_return={end_return:.1f}, avg_length={end_length:.1f}, pool_size={len(buffer)}| alpha={float(algo.alpha):.4f}")
        collector.close()




def main() -> None:
    """入口：解析/设置配置，构建组件并启动训练。"""
    # 0) 配置与种子
    cfg = SACConfig(
        env_id="LunarLander-v3",  # 默认环境
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
    )
    
    set_seed(cfg.seed)
    
    # 1) 组件构建
    replay_buffer, algorithm, collector = build_components(cfg.env_id, cfg)

    train_loop(cfg.env_id,algorithm,replay_buffer,collector,cfg)
    
    # 2) 训练完成后保存策略演示GIF
    print("\n=== Saving trained policy GIF ===")
    save_policy_gif(cfg.env_id, algorithm, cfg.device, cfg.seed)
    

if __name__ == "__main__":
    main()
