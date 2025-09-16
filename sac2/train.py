# sac2/train.py  (Gymnasium version)
import gymnasium as gym
import numpy as np
import torch
from collections import deque
import imageio
import os

from algorithm import SACContinuous
from config import SACContinuousConfig


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs  = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0; self.size = 0; self.max = size; self.device = device

    def push(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i] = o; self.acts[i] = a; self.rews[i] = r
        self.obs2[i] = o2; self.done[i] = d
        self.ptr = (i + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states":      torch.tensor(self.obs[idx],  device=self.device),
            "actions":     torch.tensor(self.acts[idx], device=self.device),
            "rewards":     torch.tensor(self.rews[idx], device=self.device),
            "next_states": torch.tensor(self.obs2[idx], device=self.device),
            "dones":       torch.tensor(self.done[idx], device=self.device),
        }


def evaluate(env, agent, episodes=5, render=False, max_steps=200):
    """Evaluate agent and optionally collect frames for GIF"""
    returns = []
    frames = []

    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        total = 0.0
        step = 0
        ep_frames = []

        while not done and step < max_steps:
            if render and ep == 0:  # Only record first episode
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)

            a = agent.take_action(o, deterministic=True)
            o, r, term, trunc, _ = env.step(np.array(a, dtype=np.float32))
            total += r
            done = term or trunc
            step += 1

        returns.append(total)
        if render and ep == 0:
            frames = ep_frames

    return float(np.mean(returns)), frames


def main(config=None):
    if config is None:
        config = SACContinuousConfig.lunarlander()

    print(f"🚀 Training SAC on {config.env_id}")

    # Set device
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # Create environments
    env = gym.make(config.env_id)
    test_env = gym.make(config.env_id, render_mode="rgb_array")

    # Environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = float(env.action_space.high[0])

    print(f"📊 Environment: {obs_dim} states, {act_dim} actions, bound: ±{act_high}")

    # Create agent
    target_entropy = config.algo_target_entropy if config.algo_target_entropy is not None else -act_dim
    agent = SACContinuous(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=config.net_hidden_dim,
        action_bound=act_high,
        actor_lr=config.algo_lr_actor,
        critic_lr=config.algo_lr_critic,
        alpha_lr=config.algo_lr_alpha,
        target_entropy=target_entropy,
        tau=config.algo_tau,
        gamma=config.algo_gamma,
        device=device
    )

    # Create replay buffer
    buf = ReplayBuffer(obs_dim, act_dim, size=config.buffer_capacity, device=device)

    # Training loop
    o, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    recent = deque(maxlen=10)

    print(f"🏃 Training for {config.train_total_steps:,} steps...")

    for t in range(1, config.train_total_steps + 1):
        # Action selection: random exploration vs policy
        if t < config.train_start_steps:
            a = env.action_space.sample()
        else:
            a = agent.take_action(o, deterministic=False)

        # Environment step
        o2, r, term, trunc, _ = env.step(np.array(a, dtype=np.float32))
        d = float(term or trunc)

        # Store transition
        buf.push(o, a, r, o2, d)

        # Update state and episode stats
        o = o2 if not d else env.reset()[0]
        ep_ret += r
        ep_len += 1

        # Reset episode stats when done
        if d:
            recent.append(ep_ret)
            ep_ret, ep_len = 0.0, 0

        # Training updates
        if (t >= config.train_update_after and
            t % config.train_update_every == 0 and
            buf.size >= config.train_batch_size):
            batch = buf.sample(config.train_batch_size)
            _ = agent.update(batch)

        # Periodic evaluation
        if t % config.train_eval_every == 0:
            eval_ret, _ = evaluate(test_env, agent, episodes=config.train_eval_episodes)
            avg_recent = float(np.mean(recent)) if recent else float("nan")
            alpha = agent.log_alpha.exp().item()
            print(f"Step {t:6d} | Eval: {eval_ret:6.1f} | Recent: {avg_recent:6.1f} | Alpha: {alpha:.3f}")

    # Final evaluation and GIF generation
    if config.save_gif:
        print("\n🎬 Recording final performance...")
        final_ret, frames = evaluate(
            test_env, agent,
            episodes=config.gif_episodes,
            render=True,
            max_steps=config.gif_max_steps
        )
        print(f"✅ Final performance: {final_ret:.1f}")

        # Save GIF
        if frames:
            gif_path = f"{config.env_id}_sac_final.gif"
            imageio.mimsave(gif_path, frames, fps=30, loop=0)
            print(f"🎥 Saved performance GIF: {gif_path}")
        else:
            print("⚠️  No frames captured for GIF")

    env.close()
    test_env.close()


def demo_configs():
    """Demonstrate different configurations"""
    print("🎮 Available configurations:")
    print("1. LunarLander (default)")
    print("2. Pendulum")
    print("3. Custom config")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "2":
        config = SACContinuousConfig.pendulum()
        print("🎯 Using Pendulum configuration")
    elif choice == "3":
        # Custom config example
        config = SACContinuousConfig(
            env_id="BipedalWalker-v3",
            train_total_steps=200_000,
            net_hidden_dim=512,
            train_start_steps=10_000
        )
        print("🔧 Using custom configuration")
    else:
        config = SACContinuousConfig.lunarlander()
        print("🚀 Using LunarLander configuration")

    return config

if __name__ == "__main__":
    # Use demo_configs() for interactive mode, or main() for default
    # config = demo_configs()
    # main(config)
    main()
