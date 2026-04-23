import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, num_features, num_layers=4):
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers

        self.dense = nn.ModuleList([
            nn.Linear(num_features, num_features)
            for _ in range(num_layers)
        ])
        self.normalize = nn.ModuleList([
            nn.LayerNorm(num_features, eps=1e-6)
            for _ in range(num_layers)
        ])

        # initialize weights
        sd = np.sqrt(1 / (3 * num_features)) # see https://github.com/wang-kevin3290/scaling-crl/blob/main/train.py#L108
        for d in self.dense:
            torch.nn.init.uniform_(d.weight, -sd, sd) # type: ignore
            torch.nn.init.zeros_(d.bias) # type: ignore
            
    def forward(self, x):
        for i in range(self.num_layers):
            output = self.dense[i](x)
            output = self.normalize[i](x)
            output = F.silu(output)
        return x + output
    
class ResidualNetwork(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=1024, num_blocks=4):
        super(ResidualNetwork, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.final = nn.Linear(hidden_dim, out_features)

        # initialize weights
        sd = np.sqrt(1 / (3 * in_features))
        torch.nn.init.uniform_(self.initial[0].weight, -sd, sd) # type: ignore
        torch.nn.init.zeros_(self.initial[0].bias) # type: ignore
        sd = np.sqrt(1 / (3 * hidden_dim))
        torch.nn.init.uniform_(self.final.weight, -sd, sd) # type: ignore
        torch.nn.init.zeros_(self.final.bias) # type: ignore

    def forward(self, x):
        # stop gap measure to prevent nans from crashing training, need to investigate source of nans later
        output = torch.nan_to_num(self.initial(x))
        for block in self.blocks:
            output = torch.nan_to_num(block(output))
        return torch.nan_to_num(self.final(output))
    
    
class Critic(nn.Module):
    def __init__(self, state_encoder, out_features=64, depth=4):
        super(Critic, self).__init__()
        self.s_encoder = state_encoder
        # the action dimension is 10 (8 one-hot variables, 2 continuous)
        self.sa_encoder = ResidualNetwork(state_encoder.embedding_dim + 10, out_features, num_blocks=depth)
        self.g_encoder = ResidualNetwork(state_encoder.embedding_dim, out_features, num_blocks=depth)

    def forward(self, s, a, g, for_info_nce=True):
        # expect s,g to be (B, 3, 210, 160) and a to be (B, 10)
        sa = torch.cat([self.s_encoder(s), a], dim=-1)
        sa = self.sa_encoder(sa)
        g = self.g_encoder(self.s_encoder(g))

        if for_info_nce:
            # returns logits from https://github.com/wang-kevin3290/scaling-crl/blob/main/train.py#L803
            # can interpret BxB matrix as distances between sa and g
            return -torch.linalg.norm(sa[:, None, :] - g[None, :, :], dim=-1)
        return -torch.linalg.norm(sa - g, dim=-1)
    

class _TanhDiagGaussian:
    """
    (From 10703) Lightweight distribution wrapper for a diagonal Gaussian in pre-tanh space.

    Exposes:
      - sample()    : non-reparameterized sample (good for PPO env stepping)
      - rsample()   : reparameterized sample (required for SAC actor update)
      - log_prob(a) : log π(a|s) with tanh + scale correction; returns [B]
      - entropy()   : entropy of base Gaussian (sum over dims); returns [B]
      - mean_action : deterministic action tanh(μ) mapped to env bounds
    """
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor,
                 scale: torch.Tensor, bias: torch.Tensor):
        """
        Args:
            mu:      [B, A]
            log_std: [B, A] or [A]
            scale:   [A]  (maps [-1,1] to env bounds)
            bias:    [A]
        """
        self.mu = mu
        self.log_std = log_std
        self.std = log_std.exp()
        self.scale = scale
        self.bias = bias
        self._base = torch.distributions.Normal(self.mu, self.std)  # broadcast over [B, A]

    # --- sampling ---

    def sample(self) -> torch.Tensor:
        """Non-reparameterized sample (no grads through noise). Returns [B, A]."""
        u = self._base.sample()
        return torch.tanh(u) * self.scale + self.bias

    def rsample(self) -> torch.Tensor:
        """Reparameterized sample (pathwise grads). Returns [B, A]."""
        u = self._base.rsample()
        return torch.tanh(u) * self.scale + self.bias

    # --- densities ---

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        """
        Log π(a|s) with tanh change-of-variables and affine logdet.
        Accepts actions in env bounds; returns [B].
        """
        if a.ndim == 1:  # [A] -> [1, A]
            a = a.unsqueeze(0)

        # Map action back to pre-tanh space: y in (-1,1), u = atanh(y)
        y = (a - self.bias) / (self.scale + 1e-8)
        y = torch.clamp(y, -0.999999, 0.999999)
        u = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

        logp_u = self._base.log_prob(u).sum(dim=-1)   # sum over action dims

        # Jacobian of tanh: diag(1 - tanh(u)^2)
        correction = torch.log(1 - torch.tanh(u).pow(2) + 1e-8).sum(dim=-1)

        # Affine scale (bias adds no volume)
        scale_logdet = torch.log(self.scale + 1e-8).sum(dim=-1)

        return logp_u - correction - scale_logdet

    def entropy(self) -> torch.Tensor:
        """
        Proxy entropy: entropy of the base Gaussian (sum over dims). Returns [B].
        (True tanh entropy has no simple closed form; this proxy is standard in PPO.)
        """
        return self._base.entropy().sum(dim=-1)

    # --- eval convenience ---

    @property
    def mean_action(self) -> torch.Tensor:
        """Deterministic action for evaluation, [B, A]."""
        return torch.tanh(self.mu) * self.scale + self.bias

class Actor(nn.Module):
    def __init__(self, state_encoder, depth=4):
        super(Actor, self).__init__()
        self.s_encoder = state_encoder
        # 8 action probabilities, 4 for mean and log std of ACTION6(x,y)
        self.resnet = ResidualNetwork(2 * state_encoder.embedding_dim, 12, num_blocks=depth)

        self.scale = None
        self.bias = None

    def forward(self, s, g):
        # expect s,g to be (B, 3, 210, 160)
        sg = torch.cat([self.s_encoder(s), self.s_encoder(g)], dim=-1)
        logits, mean, log_std = torch.tensor_split(self.resnet(sg), [8, 10], dim=-1)
        if self.scale is None or self.bias is None:
            low = torch.zeros(2, device=sg.device)
            high = torch.tensor([s.shape[-1], s.shape[-2]], device=sg.device)
            self.scale = (high - low) / 2
            self.bias = (high + low) / 2
        return F.softmax(logits, dim=-1), _TanhDiagGaussian(mean, log_std, self.scale, self.bias)
    

if __name__ == "__main__":
    from trainer import PolicyTrainer, load_environments, RolloutReplayBuffer
    from state_encoder import StateEncoder
    import copy, os

    envs = load_environments()

    state_encoder = StateEncoder((3, 4, 6, 3), 512)
    state_encoder.load_state_dict(torch.load("./checkpoints/arc3-state-encoder_final.pt")["encoder"])

    critic_embed_dim = [32, 64, 128]
    critic_depth = [2, 4, 8]
    actor_depth = [2, 4, 8]

    bs = 2 # this is number of rollouts, so actual dimensions is x1000 worst case
    utd = 40 # do 40 update steps, then sample data once (currently, "one" sample gathers 10 rollouts)
    lr = 3e-4 # currently planned to be constant, but logging just in case

    print("Loading buffer... ", end="")
    buffer = RolloutReplayBuffer(2000)
    try:
        buffer.load_data("./checkpoints", "arc3-policy_init", load_default=True)
        print("Done.")
    except: print("No buffer data found, starting with empty buffer")

    for e_dim in critic_embed_dim:
        for c_depth in critic_depth:
            for a_depth in actor_depth:
                if os.path.exists(f"./checkpoints/ablations/contrastive-rl/arc3-policy_dim{e_dim}_critic{c_depth}_actor{a_depth}.pt"):
                    continue
                actor = Actor(state_encoder, depth=a_depth)
                critic = Critic(state_encoder, out_features=e_dim, depth=c_depth)

                trainer = PolicyTrainer(
                    f"dim{e_dim}_critic{c_depth}_actor{a_depth}",
                    state_encoder, actor, critic, copy.deepcopy(buffer), envs,
                    wandb_logging=True, wandb_project="arc3-policy"
                )
                trainer.init_wandb_run({
                    'critic_embedding_dim': e_dim,
                    'critic_depth': c_depth,
                    'actor_depth': a_depth,
                    'batch_size': bs,
                    'learning_rate': lr,
                    'utd_ratio': utd
                })
                trainer.train(10, bs=bs, utd=utd, directory="./checkpoints/ablations/contrastive-rl")