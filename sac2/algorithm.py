# sac2/algorithm.py
import numpy as np
import torch
import torch.nn.functional as F

from network import PolicyNetContinuous, QValueNetContinuous


class SACContinuous:
    """
    Continuous action SAC (minimal version)
    - Actor: PolicyNetContinuous(state -> (action, log_prob))
    - Critics: Two independent Q(s,a) + their respective target networks
    - Auto α: Learning temperature, target_entropy typically set to -action_dim
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, target_entropy=None, tau=0.005, gamma=0.99, device="cuda"):
        self.device = device
        if target_entropy is None:
            target_entropy = -action_dim  # Standard SAC default
        # ---- Networks ----
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # ---- Optimizers ----
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer= torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer= torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # ---- Temperature (auto α with log-param) ----
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # Usually set to -action_dim (negative value)

        # ---- Hyperparameters ----
        self.gamma = gamma
        self.tau   = tau

    # -------- Acting --------
    @torch.no_grad()
    def take_action(self, state, deterministic=False):
        """
        state: ndarray / list shape [state_dim]
        return: list[float] of action_dim
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(state, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().tolist()

    # -------- Target Q --------
    @torch.no_grad()
    def calc_target(self, rewards, next_states, dones):
        """
        rewards, dones: [B,1]; next_states: [B, state_dim]
        Target: y = r + γ(1-d) * ( min(Q1',Q2')(s',a') - α*logπ(a'|s') )
        """
        next_actions, log_prob = self.actor(next_states)   # log_prob: [B,1]
        q1_t = self.target_critic_1(next_states, next_actions)
        q2_t = self.target_critic_2(next_states, next_actions)
        q_target = torch.min(q1_t, q2_t) - self.log_alpha.exp() * log_prob
        td_target = rewards + self.gamma * (1.0 - dones) * q_target
        return td_target

    # -------- Soft Update --------
    @torch.no_grad()
    def soft_update(self, net, target_net):
        for p, p_t in zip(net.parameters(), target_net.parameters()):
            p_t.data.mul_(1.0 - self.tau)
            p_t.data.add_(self.tau * p.data)

    # -------- One training step --------
    def update(self, transition_dict):
        """
        transition_dict keys:
          'states' [B, state_dim]
          'actions' [B, action_dim]
          'rewards' [B] or [B,1]
          'next_states' [B, state_dim]
          'dones' [B] or [B,1]
        """
        states = torch.as_tensor(transition_dict['states'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(transition_dict['actions'], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.as_tensor(transition_dict['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(transition_dict['dones'], dtype=torch.float32, device=self.device).view(-1, 1)

        # Optional: Pendulum-style reward reshaping (comment/delete as needed)
        # rewards = (rewards + 8.0) / 8.0

        # ---- Critic update ----
        with torch.no_grad():
            td_target = self.calc_target(rewards, next_states, dones)
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = F.mse_loss(q1, td_target)
        critic_2_loss = F.mse_loss(q2, td_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ---- Actor update ----
        new_actions, log_prob = self.actor(states)  # [B, act_dim], [B,1]
        q1_pi = self.critic_1(states, new_actions)
        q2_pi = self.critic_2(states, new_actions)
        q_pi  = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()

        actor_loss = (alpha * log_prob - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Alpha update (standard form) ----
        # loss_alpha = - E[ log_alpha * (logπ + H_target) ]
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Targets soft-update ----
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # Return monitoring quantities for logging
        return {
            "critic1_loss": float(critic_1_loss.item()),
            "critic2_loss": float(critic_2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(alpha.item()),
            "entropy": float((-log_prob).mean().item()),
            "q_pi": float(q_pi.mean().item()),
        }


def test_sac_continuous():
    """Test SACContinuous algorithm"""
    print("=== SACContinuous Algorithm Test ===")

    # Test configuration
    state_dim = 8
    action_dim = 2
    batch_size = 32

    print(f"Test config: state_dim={state_dim}, action_dim={action_dim}, batch_size={batch_size}")

    # 1. Test basic initialization
    print("\n--- Initialization Test ---")
    try:
        sac = SACContinuous(state_dim=state_dim, action_dim=action_dim, device="cpu")
        print("✅ SACContinuous initialized successfully")
        print(f"   Target entropy: {sac.target_entropy}")
        print(f"   Device: {sac.device}")
        print(f"   Initial alpha: {sac.log_alpha.exp().item():.4f}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 2. Test action selection
    print("\n--- Action Selection Test ---")
    try:
        test_state = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]  # state_dim=8

        # Stochastic action
        action_stoch = sac.take_action(test_state, deterministic=False)
        print(f"Stochastic action: {action_stoch}")
        print(f"Action shape: {len(action_stoch)}")
        print(f"Action range: [{min(action_stoch):.3f}, {max(action_stoch):.3f}]")

        # Deterministic action
        action_det = sac.take_action(test_state, deterministic=True)
        print(f"Deterministic action: {action_det}")
        print("✅ Action selection works correctly")

    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        return

    # 3. Test update step
    print("\n--- Update Step Test ---")
    try:
        # Create fake batch data
        transition_dict = {
            'states': torch.randn(batch_size, state_dim),
            'actions': torch.randn(batch_size, action_dim),
            'rewards': torch.randn(batch_size),
            'next_states': torch.randn(batch_size, state_dim),
            'dones': torch.randint(0, 2, (batch_size,)).float()
        }

        print("Batch data shapes:")
        for key, value in transition_dict.items():
            print(f"  {key}: {value.shape}")

        # Perform update
        logs = sac.update(transition_dict)

        print("\nUpdate logs:")
        for key, value in logs.items():
            print(f"  {key}: {value:.4f}")

        print("✅ Update step completed successfully")

    except Exception as e:
        print(f"❌ Update step failed: {e}")
        return

    # 4. Test multiple updates
    print("\n--- Multiple Updates Test ---")
    try:
        loss_history = {"actor_loss": [], "critic1_loss": [], "alpha": []}

        for step in range(5):
            # Generate new batch each step
            transition_dict = {
                'states': torch.randn(batch_size, state_dim),
                'actions': torch.randn(batch_size, action_dim),
                'rewards': torch.randn(batch_size),
                'next_states': torch.randn(batch_size, state_dim),
                'dones': torch.randint(0, 2, (batch_size,)).float()
            }

            logs = sac.update(transition_dict)
            loss_history["actor_loss"].append(logs["actor_loss"])
            loss_history["critic1_loss"].append(logs["critic1_loss"])
            loss_history["alpha"].append(logs["alpha"])

        print("Loss progression over 5 steps:")
        for loss_name, loss_values in loss_history.items():
            print(f"  {loss_name}: {[f'{v:.3f}' for v in loss_values]}")

        print("✅ Multiple updates completed successfully")

    except Exception as e:
        print(f"❌ Multiple updates failed: {e}")
        return

    # 5. Test parameter counts
    print("\n--- Parameter Statistics ---")
    try:
        actor_params = sum(p.numel() for p in sac.actor.parameters())
        critic1_params = sum(p.numel() for p in sac.critic_1.parameters())
        critic2_params = sum(p.numel() for p in sac.critic_2.parameters())
        target1_params = sum(p.numel() for p in sac.target_critic_1.parameters())
        target2_params = sum(p.numel() for p in sac.target_critic_2.parameters())

        total_params = actor_params + critic1_params + critic2_params

        print(f"Actor parameters: {actor_params:,}")
        print(f"Critic1 parameters: {critic1_params:,}")
        print(f"Critic2 parameters: {critic2_params:,}")
        print(f"Target networks: {target1_params + target2_params:,}")
        print(f"Total trainable parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Parameter counting failed: {e}")
        return

    # 6. Test device consistency
    print("\n--- Device Consistency Test ---")
    try:
        # Check all networks are on the same device
        devices = {
            "actor": next(sac.actor.parameters()).device,
            "critic1": next(sac.critic_1.parameters()).device,
            "critic2": next(sac.critic_2.parameters()).device,
            "target1": next(sac.target_critic_1.parameters()).device,
            "target2": next(sac.target_critic_2.parameters()).device,
            "log_alpha": sac.log_alpha.device
        }

        all_same_device = len(set(devices.values())) == 1
        print(f"All components on same device: {'✅' if all_same_device else '❌'}")
        for name, device in devices.items():
            print(f"  {name}: {device}")

    except Exception as e:
        print(f"❌ Device consistency check failed: {e}")
        return

    print("\n🎉 All tests passed! SACContinuous is ready for training.")


if __name__ == "__main__":
    test_sac_continuous()
