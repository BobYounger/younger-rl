# sac2/networks.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
LOG_2PI = math.log(2.0 * math.pi)
LOG_2 = math.log(2.0)

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound=1.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 标量边界，满足大多数环境

    def forward(self, x, deterministic=False):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)

        if deterministic:
            # 确定性输出：直接使用均值
            action = torch.tanh(mu) * self.action_bound
            return action, None

        log_std = torch.clamp(self.fc_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # 采样 pre-tanh 变量 u 并做 tanh 压缩
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)

        # 正态项：sum over action dim
        logp_normal = -0.5 * (((u - mu) / (std + 1e-8))**2 + 2.0 * log_std + LOG_2PI)
        logp_normal = logp_normal.sum(dim=-1, keepdim=True)

        # tanh 修正（稳定写法，等价于 log(1 - tanh(u)^2)）
        correction = 2.0 * (LOG_2 - u - F.softplus(-2.0 * u))
        correction = correction.sum(dim=-1, keepdim=True)

        log_prob = logp_normal - correction
        action = a * self.action_bound  # 映射到环境动作范围（标量）

        return action, log_prob


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        z = torch.cat([x, a], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.fc_out(z)


class TwinQContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.q1 = QValueNetContinuous(state_dim, hidden_dim, action_dim)
        self.q2 = QValueNetContinuous(state_dim, hidden_dim, action_dim)

    def forward(self, x, a):
        return self.q1(x, a), self.q2(x, a)

def main():
    """测试连续动作网络的输入输出"""
    print("=== 连续SAC网络测试 ===")

    # 设置测试参数
    batch_size = 4
    state_dim = 8  # 例如 LunarLanderContinuous
    action_dim = 2  # 连续动作维度
    hidden_dim = 64
    action_bound = 1.0

    # 创建测试数据
    test_states = torch.randn(batch_size, state_dim)
    test_actions = torch.randn(batch_size, action_dim) * action_bound

    print(f"测试配置: batch_size={batch_size}, state_dim={state_dim}, action_dim={action_dim}")
    print(f"输入状态形状: {test_states.shape}")
    print(f"输入动作形状: {test_actions.shape}")

    # 1. 测试策略网络
    print("\n--- PolicyNetContinuous 测试 ---")
    policy = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound)

    with torch.no_grad():
        actions, log_probs = policy(test_states)
        print(f"输出动作形状: {actions.shape}")
        print(f"输出log_prob形状: {log_probs.shape}")
        print(f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        print(f"log_prob范围: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")
        print(f"动作边界检查: 是否在[-{action_bound}, {action_bound}]内: {(actions.abs() <= action_bound).all().item()}")

    # 2. 测试单个Q网络
    print("\n--- QValueNetContinuous 测试 ---")
    q_net = QValueNetContinuous(state_dim, hidden_dim, action_dim)

    with torch.no_grad():
        q_values = q_net(test_states, test_actions)
        print(f"Q值形状: {q_values.shape}")
        print(f"Q值范围: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
        print(f"Q值均值: {q_values.mean().item():.3f}")

    # 3. 测试双Q网络
    print("\n--- TwinQContinuous 测试 ---")
    twin_q = TwinQContinuous(state_dim, hidden_dim, action_dim)

    with torch.no_grad():
        q1_vals, q2_vals = twin_q(test_states, test_actions)
        print(f"Q1值形状: {q1_vals.shape}")
        print(f"Q2值形状: {q2_vals.shape}")
        print(f"Q1范围: [{q1_vals.min().item():.3f}, {q1_vals.max().item():.3f}]")
        print(f"Q2范围: [{q2_vals.min().item():.3f}, {q2_vals.max().item():.3f}]")
        min_q = torch.min(q1_vals, q2_vals)
        print(f"min(Q1,Q2)均值: {min_q.mean().item():.3f}")

    # 4. 测试策略网络与Q网络的配合
    print("\n--- 策略-Q网络配合测试 ---")
    with torch.no_grad():
        # 策略生成动作
        policy_actions, policy_log_probs = policy(test_states)
        # Q网络评估策略动作
        q1_policy, q2_policy = twin_q(test_states, policy_actions)
        min_q_policy = torch.min(q1_policy, q2_policy)

        print(f"策略动作的Q值评估:")
        print(f"  Q1均值: {q1_policy.mean().item():.3f}")
        print(f"  Q2均值: {q2_policy.mean().item():.3f}")
        print(f"  min(Q1,Q2)均值: {min_q_policy.mean().item():.3f}")
        print(f"  策略log_prob均值: {policy_log_probs.mean().item():.3f}")

    # 5. 测试梯度流
    print("\n--- 梯度流测试 ---")
    # 测试策略网络梯度
    policy_actions, policy_log_probs = policy(test_states)
    policy_loss = policy_log_probs.mean()  # 简单的loss
    policy_loss.backward()

    policy_grad_norm = 0
    for param in policy.parameters():
        if param.grad is not None:
            policy_grad_norm += param.grad.data.norm(2).item() ** 2
    policy_grad_norm = policy_grad_norm ** 0.5
    print(f"策略网络梯度范数: {policy_grad_norm:.6f}")

    # 测试Q网络梯度
    policy.zero_grad()  # 清除策略网络梯度
    q_values = q_net(test_states, test_actions)
    q_loss = q_values.mean()
    q_loss.backward()

    q_grad_norm = 0
    for param in q_net.parameters():
        if param.grad is not None:
            q_grad_norm += param.grad.data.norm(2).item() ** 2
    q_grad_norm = q_grad_norm ** 0.5
    print(f"Q网络梯度范数: {q_grad_norm:.6f}")

    # 6. 网络参数统计
    print("\n--- 网络参数统计 ---")
    policy_params = sum(p.numel() for p in policy.parameters())
    q_params = sum(p.numel() for p in q_net.parameters())
    twin_q_params = sum(p.numel() for p in twin_q.parameters())

    print(f"PolicyNet参数量: {policy_params:,}")
    print(f"QValueNet参数量: {q_params:,}")
    print(f"TwinQ参数量: {twin_q_params:,}")
    print(f"总参数量: {policy_params + twin_q_params:,}")

    print("\n✅ 所有网络测试完成！网络结构正确，可以用于SAC算法。")

if __name__ == "__main__":
    main()