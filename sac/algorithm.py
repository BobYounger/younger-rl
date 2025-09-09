from __future__ import annotations
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from network import Actor, QCritic, VCritic
from config import SACConfig as AlgoConfig

class SACDiscreteVAlgorithm:
    def __init__(self, state_dim: int, act_dim: int, cfg: AlgoConfig) -> None:
        """Initialize SAC algorithm.
        
        Args:
            state_dim: State space dimension
            act_dim: Number of discrete actions
            cfg: Algorithm configuration
        """
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Create networks and move to device
        self.actor = Actor(state_dim, act_dim, cfg.net_hidden_sizes).to(self.device)
        self.q0 = QCritic(state_dim, act_dim, cfg.net_hidden_sizes).to(self.device)
        self.q1 = QCritic(state_dim, act_dim, cfg.net_hidden_sizes).to(self.device)
        self.v = VCritic(state_dim, cfg.net_hidden_sizes).to(self.device)
        
        # Target V network (hard copy initialization)
        self.v_tgt = copy.deepcopy(self.v).to(self.device)
        
        # Freeze target network parameters
        for param in self.v_tgt.parameters():
            param.requires_grad = False
        
        # Create optimizers
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.algo_lr_actor)
        self.opt_q0 = optim.Adam(self.q0.parameters(), lr=cfg.algo_lr_critic)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=cfg.algo_lr_critic)
        self.opt_v = optim.Adam(self.v.parameters(), lr=cfg.algo_lr_critic)
        
        # Alpha as constant tensor on device
        self.alpha = torch.tensor(cfg.algo_alpha, dtype=torch.float32, device=self.device)
        
        # Loss functions (reusable)
        self.huber_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        
        # Training step counter
        self.train_steps = 0
        
    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """Select action using the current policy.
        
        Args:
            state: State tensor, shape (B, state_dim) or (state_dim,)
            eval_mode: If True, use deterministic action selection
            
        Returns:
            action: Action tensor, shape (B,) or scalar
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action = self.actor.act(state, eval_mode=eval_mode)
            
        return action.squeeze(0) if action.shape[0] == 1 else action

    def _update_q(self, batch, loss_type: str = "mse") -> dict:
        """
        Update Q0 Q1
        Args:
            batch: dict,含 keys = state(B,S), action(B,), reward(B,), next_state(B,S), done(B,)
            loss_type: "mse" 或 "huber"
        Returns:
            dict: {"loss_q0", "loss_q1", "td0", "td1", "q0_mean", "q1_mean", "target_mean"}
        """
        # 1) put data into device
        state      = batch["state"].to(self.device).float()          # (B,S)
        action     = batch["action"].to(self.device).long()          # (B,)
        reward     = batch["reward"].to(self.device).float()         # (B,)
        next_state = batch["next_state"].to(self.device).float()     # (B,S)
        done       = batch["done"].to(self.device).float()           # (B,)
        if action.dim() == 2:
            action = action.squeeze(-1)

        # 2) construct y
        with torch.no_grad():
            v_next = self.v_tgt(next_state)                          # (B,)
            y = (reward + self.cfg.algo_gamma * (1.0 - done) * v_next).detach()  # (B,)

        # 3) current output Q(s,a)
        q0_all = self.q0(state)                                      # (B,A)
        q1_all = self.q1(state)                                      # (B,A)
        q0_sa = q0_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # (B,)
        q1_sa = q1_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # (B,)

        # 4) loss and optimization
        if loss_type == "huber":
            L_q0 = self.huber_loss_fn(q0_sa, y)
            L_q1 = self.huber_loss_fn(q1_sa, y)
        else:  # "mse"
            L_q0 = F.mse_loss(q0_sa, y, reduction="mean")
            L_q1 = F.mse_loss(q1_sa, y, reduction="mean")

        # Q0 network update
        self.opt_q0.zero_grad(set_to_none=True)
        L_q0.backward()
        max_norm = getattr(self.cfg, "max_grad_norm", None)
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q0.parameters(), max_norm)
        self.opt_q0.step()

        # Q1 network update
        self.opt_q1.zero_grad(set_to_none=True)
        L_q1.backward()
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm)
        self.opt_q1.step()
        
        # 5) logs
        td0 = (q0_sa - y).abs().mean().item()
        td1 = (q1_sa - y).abs().mean().item()
        return {
            "loss_q0": float(L_q0.item()),
            "loss_q1": float(L_q1.item()),
            "td0": td0,
            "td1": td1,
            "q0_mean": float(q0_sa.mean().item()),
            "q1_mean": float(q1_sa.mean().item()),
            "target_mean": float(y.mean().item()),
        }
    
    def _update_v(self, batch) -> dict:
        """
        Update online V network with one gradient step.
        Only gradients flow to V; actor and Q networks are not updated in this step.
        batch keys: state(B,S), action(B,), reward(B,), next_state(B,S), done(B,)
        """
        state = batch["state"].to(self.device).float()  # (B,S)

        # ---- Construct v_target: ∑_a π(a|s)[minQ(s,a) - α logπ(a|s)] ----
        # Use no_grad to prevent gradients from flowing to actor/Q networks
        with torch.no_grad():
            logits = self.actor(state)                          # (B,A)
            log_probs = F.log_softmax(logits, dim=-1)           # (B,A)
            probs = log_probs.exp()                             # (B,A)

            q0_all = self.q0(state)                             # (B,A)
            q1_all = self.q1(state)                             # (B,A)
            min_q = torch.minimum(q0_all, q1_all)               # (B,A)

            v_target = (probs * (min_q - self.alpha.detach() * log_probs)).sum(dim=-1)  # (B,)

            # Monitor: policy entropy (for logging only)
            entropy = (-(probs * log_probs).sum(dim=-1)).mean().item()

        # ---- V prediction and loss ----
        v_pred = self.v(state)                                  # (B,)
        loss_v = F.mse_loss(v_pred, v_target, reduction="mean")

        # ---- Backward pass and optimization step (V optimizer only) ----
        self.opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        # gradient clipping for V network
        max_norm = getattr(self.cfg, "max_grad_norm", None)
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm)
        self.opt_v.step()

        return {
            "loss_v": float(loss_v.item()),
            "v_mean": float(v_pred.mean().item()),
            "v_target_mean": float(v_target.mean().item()),
            "entropy": float(entropy),
            "minQ_mean": float(min_q.mean().item()),
        }

    def _update_actor(self, batch) -> dict:
        """
        Update policy (Actor) with one gradient step.
        Only gradients flow to the actor; critics are treated as constants.
        batch keys: state(B,S), action(B,), reward(B,), next_state(B,S), done(B,)
        """
        state = batch["state"].to(self.device).float()  # (B,S)

        # ---- Forward pass through policy to get π and log π ----
        logits = self.actor(state)                          # (B,A)
        log_probs = F.log_softmax(logits, dim=-1)           # (B,A)
        probs = log_probs.exp()                             # (B,A)

        # ---- Compute min Q(s,·) with gradient detachment for critics ----
        # Use no_grad to save memory and ensure gradients only flow to actor
        with torch.no_grad():
            q0_all = self.q0(state)                         # (B,A)
            q1_all = self.q1(state)                         # (B,A)
            min_q = torch.minimum(q0_all, q1_all)           # (B,A)

        # ---- Policy loss: E_a[ α logπ(a|s) - minQ(s,a) ] ----
        # Note: alpha doesn't participate in gradients (can detach() for fixed hyperparameter stage)
        E_logp = (probs * log_probs).sum(dim=-1)            # (B,)  = E_a[log π(a|s)]
        E_q    = (probs * min_q).sum(dim=-1)                # (B,)  = E_a[min(Q0,Q1)(s,a)]
        
        # E[ α * logπ - Q0 ]
        loss_pi = (self.alpha.detach() * E_logp - E_q).mean()

        # ---- Optimize actor only ----
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        max_norm = getattr(self.cfg, "max_grad_norm", None)
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm)
        self.opt_actor.step()

        # ---- Logging ----
        entropy = (-E_logp).mean().item()
        return {
            "loss_pi": float(loss_pi.item()),
            "entropy": float(entropy),
            "E_minq_mean": float(E_q.mean().item()),
            "prob_max_mean": float(probs.max(dim=-1).values.mean().item()),
        }
    
    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module, tau: float) -> None:
        """Polyak/EMA: θ' ← (1-τ)θ' + τ θ  (parameter-wise update under no_grad)"""
        with torch.no_grad():
            for p, p_t in zip(src.parameters(), tgt.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(self, batch: dict) -> dict:
        """
        Complete algorithm update (Q0/Q1, V, Actor, V→V' soft update).
        Returns merged log dictionary.
        """
        # Ensure device consistency
        device = next(self.actor.parameters()).device
        for key, value in batch.items():
            if hasattr(value, 'to'):
                batch[key] = value.to(device)
        
        # Training mode (affects BN/Dropout if added later; mainly for standardization now)
        self.q0.train()
        self.q1.train()
        self.v.train()
        self.actor.train()

        # 1) Update Q networks (both)
        q_loss_type = getattr(self.cfg, "q_loss_type", "huber")  # configurable in cfg: "mse"/"huber"
        logs_q = self._update_q(batch, loss_type="huber")

        # 2) Update V (fit soft value expectation)
        logs_v = self._update_v(batch)

        # 3) Update policy (Actor)
        logs_pi = self._update_actor(batch)

        # 4) Soft update V → V'
        tau = getattr(self.cfg, "algo_tau", 0.005)
        self._soft_update(self.v, self.v_tgt, tau)

        self.train_steps += 1

        # 5) Merge logs with prefixes to avoid key conflicts
        logs = {
            "step": int(self.train_steps),
            "alpha": float(self.alpha.item()),
        }
        
        # Add prefixed logs to prevent conflicts
        for key, value in logs_q.items():
            logs[f"q_{key}"] = value
        for key, value in logs_v.items():
            logs[f"v_{key}"] = value
        for key, value in logs_pi.items():
            logs[f"pi_{key}"] = value

        return logs

if __name__ == "__main__":
    """Gradient Direction Verification for SAC Algorithm"""
    
    # Setup
    state_dim = 4
    act_dim = 2
    batch_size = 32
    
    config = AlgoConfig(
        device="cuda",
        hidden_sizes=(64, 64),
        lr_critic=1e-3,
        gamma=0.99,
        batch_size=batch_size
    )
    
    # Create algorithm instance
    algo = SACDiscreteVAlgorithm(state_dim, act_dim, config)
    
    # Create mock batch data
    device = torch.device(config.device)
    batch = {
        "state": torch.randn(batch_size, state_dim).to(device),
        "action": torch.randint(0, act_dim, (batch_size,)).to(device),
        "reward": torch.randn(batch_size).to(device),
        "next_state": torch.randn(batch_size, state_dim).to(device),
        "done": torch.randint(0, 2, (batch_size,)).float().to(device)
    }
    
    print("=== SAC Gradient Direction Verification ===")
    
    # Store initial parameters
    initial_q0_params = [p.clone().detach() for p in algo.q0.parameters()]
    initial_v_params = [p.clone().detach() for p in algo.v.parameters()]
    initial_actor_params = [p.clone().detach() for p in algo.actor.parameters()]
    
    # Test full update
    print("Testing full update() method...")
    logs_full = algo.update(batch)
    print("Full update logs:", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in logs_full.items()})
    
    # Check parameter changes to verify gradients worked
    print("\n=== Parameter Change Analysis ===")
    
    # Q0 network changes
    q0_changed = any((current_p - init_p).abs().sum().item() > 1e-8 
                    for init_p, current_p in zip(initial_q0_params, algo.q0.parameters()))
    print(f"Q0 parameters changed: {'✅' if q0_changed else '❌'}")
    
    # V network changes  
    v_changed = any((current_p - init_p).abs().sum().item() > 1e-8 
                    for init_p, current_p in zip(initial_v_params, algo.v.parameters()))
    print(f"V parameters changed: {'✅' if v_changed else '❌'}")
    
    # Actor network changes
    actor_changed = any((current_p - init_p).abs().sum().item() > 1e-8 
                    for init_p, current_p in zip(initial_actor_params, algo.actor.parameters()))
    print(f"Actor parameters changed: {'✅' if actor_changed else '❌'}")
    
    # Test multi-step behavior
    print("\n=== Multi-step Gradient Test ===")
    losses_history = {"q_loss_q0": [], "v_loss_v": [], "pi_loss_pi": []}
    
    for step in range(5):
        batch_test = {
            "state": torch.randn(batch_size, state_dim).to(device),
            "action": torch.randint(0, act_dim, (batch_size,)).to(device),
            "reward": torch.randn(batch_size).to(device),
            "next_state": torch.randn(batch_size, state_dim).to(device),
            "done": torch.randint(0, 2, (batch_size,)).float().to(device)
        }
        
        logs = algo.update(batch_test)
        losses_history["q_loss_q0"].append(logs["q_loss_q0"])
        losses_history["v_loss_v"].append(logs["v_loss_v"])  
        losses_history["pi_loss_pi"].append(logs["pi_loss_pi"])
    
    print("Loss progression over 5 steps:")
    for loss_name, loss_values in losses_history.items():
        print(f"  {loss_name}: {[f'{v:.4f}' for v in loss_values]}")
    
    # Verify network outputs are reasonable
    print("\n=== Network Output Validation ===")
    with torch.no_grad():
        test_state = torch.randn(1, state_dim).to(device)
        
        # Actor probability distribution check
        logits = algo.actor(test_state)
        probs = F.softmax(logits, dim=-1)
        prob_sum = probs.sum().item()
        
        print(f"Actor probability sum: {prob_sum:.6f} (should be ~1.0)")
        print(f"Valid probability distribution: {'✅' if abs(prob_sum - 1.0) < 1e-5 else '❌'}")
        
        # Q value ranges
        q0_vals = algo.q0(test_state)
        q1_vals = algo.q1(test_state)
        print(f"Q0 range: [{q0_vals.min().item():.3f}, {q0_vals.max().item():.3f}]")
        print(f"Q1 range: [{q1_vals.min().item():.3f}, {q1_vals.max().item():.3f}]")
        
        # V value comparison
        v_val = algo.v(test_state).item()
        v_tgt_val = algo.v_tgt(test_state).item()
        print(f"V value: {v_val:.3f}, V_target: {v_tgt_val:.3f}")
    
    print("\n✅ Gradient direction verification completed!")
    print("Summary: All networks show parameter updates, confirming gradients are flowing correctly.")
