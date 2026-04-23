import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from abc import ABCMeta, abstractmethod
from collections import deque
from typing import NamedTuple
import os
from tqdm import tqdm
try:
    import wandb
    WANDB_KEY = "your_wandb_key_here"
except ImportError:
    wandb = None

from datasets import Environment


class ReplayBuffer():
    def __init__(self, size, metadata, device):
        '''
        A generic replay buffer accepting arbitrary named tensors as input.
        The metadata argument should be a dict mapping from the name of the tensor to a tuple of (shape, dtype).
        '''
        self.capacity = size
        self.metadata = metadata
        self.device = device
        self.reset()
    @property
    def size(self):
        return self.size_
        
    def reset(self):
        self.size_ = 0
        self.ptr = 0
        for name, (shape, dtype) in self.metadata.items():
            setattr(self, name, torch.zeros((self.capacity, *shape), device=self.device, dtype=dtype))

    def add(self, *args, **kwargs):
        #NOTE: no checking that kwargs don't override args
        names = list(self.metadata.keys())
        for i in range(len(args)):
            getattr(self, names[i])[self.ptr] = args[i]
        for name, value in kwargs.items():
            getattr(self, name)[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)

    def add_batch(self, *args, **kwargs):
        names = list(self.metadata.keys())
        bs = None
        if len(args) > 0: bs = args[0].shape[0]
        elif len(kwargs) > 0: bs = list(kwargs.values())[0].shape[0]
        else: return

        remaining_size = self.capacity - self.size_
        remainder = 0;
        if bs >= remaining_size:
            remainder = bs - remaining_size
            bs = remaining_size
        for i in range(len(args)):
            getattr(self, names[i])[self.ptr:self.ptr+bs] = args[i][:bs]
        for name, value in kwargs.items():
            getattr(self, name)[self.ptr:self.ptr+bs] = value[:bs]
        self.ptr = (self.ptr + bs) % self.capacity
        self.size_ = min(self.size_ + bs, self.capacity)
        if remainder > 0:
            for i in range(len(args)):
                getattr(self, names[i])[:remainder] = args[i][bs:]
            for name, value in kwargs.items():
                getattr(self, name)[:remainder] = value[bs:]
            self.ptr = remainder

    def sample(self, num_samples=None):
        if self.size_ == 0: raise ValueError("Cannot sample from empty buffer")
        
        idx = torch.arange(self.size_, device=self.device)
        if num_samples is not None and num_samples < idx.numel():
            idx = idx[torch.randperm(idx.numel(), device=self.device)[:num_samples]]
        return { name: getattr(self, name)[idx] for name in self.metadata.keys() }

    def get_save_data(self):
        data = { name: getattr(self, name) for name in self.metadata.keys() }
        data['size'] = self.size_
        data['ptr'] = self.ptr
        return data
    def load_save_data(self, data):
        self.size_ = data['size']
        self.ptr = data['ptr']
        for name in self.metadata.keys():
            getattr(self, name)[:self.size_] = data[name]

class Rollout(NamedTuple):
    obs: np.ndarray # (T, 3, 210, 160)
    action: np.ndarray # (T,)
    xy: np.ndarray # (T, 2) only used for ACTION6, can be ignored otherwise
    done: np.ndarray # (T,)
    extras: dict

class RolloutReplayBuffer:
    def __init__(self, size):
        self.capacity = size
        self.buffer = deque(maxlen=size)
    @property
    def size(self):
        return len(self.buffer)

    def add(self, rollout: Rollout) -> None:
        self.buffer.append(rollout)
    def add_batch(self, rollouts: list[Rollout]):
        self.buffer.extend(rollouts)

    def sample(self, num_samples=None) -> list[Rollout]:
        if len(self.buffer) == 0: raise ValueError("Cannot sample from empty buffer")
        
        idx = torch.arange(len(self.buffer))
        if num_samples is not None and num_samples < idx.numel():
            idx = idx[torch.randperm(idx.numel())[:num_samples]]
        return [self.buffer[i] for i in idx]

    def save_data(self, directory, name, step=None):
        path = f"{directory}/{name}_{step}_buffer" if step is not None else f"{directory}/{name}_buffer"
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(f"{path}/obs.npz", *[self.buffer[i].obs for i in range(len(self.buffer))])
        np.savez_compressed(f"{path}/action.npz", *[self.buffer[i].action for i in range(len(self.buffer))])
        np.savez_compressed(f"{path}/xy.npz", *[self.buffer[i].xy for i in range(len(self.buffer))])
        np.savez_compressed(f"{path}/done.npz", *[self.buffer[i].done for i in range(len(self.buffer))])
    def load_data(self, directory, name, step=None, load_default=False):
        path = f"{directory}/{name}_{step}_buffer" if step is not None else f"{directory}/{name}_buffer"
        try:
            obs = np.load(f"{path}/obs.npz")
        except Exception as e:
            if not load_default: raise e
            path = "./checkpoints/arc3-policy_init_buffer"
            obs = np.load(f"{path}/obs.npz")
        obs_keys = obs.files
        action = np.load(f"{path}/action.npz")
        action_keys = action.files
        xy = np.load(f"{path}/xy.npz")
        xy_keys = xy.files
        done = np.load(f"{path}/done.npz")
        done_keys = done.files

        rollouts = [Rollout(
            obs=obs[obs_keys[i]],
            action=action[action_keys[i]],
            xy=xy[xy_keys[i]],
            done=done[done_keys[i]], extras={})
            for i in range(len(obs_keys))
        ]

        self.buffer = deque(rollouts, maxlen=self.capacity)


class Trainer(metaclass=ABCMeta):
    def __init__(self, name, wandb_logging=False, wandb_project=None):
        self.name = name
        self.wandb_logging = wandb_logging
        self.wandb_project = wandb_project

        self.accelerator = Accelerator() # can configure arguments but default works

        if self.wandb_logging and wandb is not None:
            wandb.login(key=WANDB_KEY)
            
    @property
    def device(self):
        return self.accelerator.device
    @property
    def num_gpus(self):
        if torch.cuda.is_available(): return torch.cuda.device_count()
        elif torch.backends.mps.is_available(): return 1
        else: return 0

    def init_wandb_run(self, config):
        if self.wandb_logging and wandb is not None:
            wandb.init(project=self.wandb_project, name=self.name, config=config)
    
    @abstractmethod
    def save(self, directory, step=None): pass
    @abstractmethod
    def load(self, directory, step=None): pass
    @abstractmethod
    def train(self, epochs, bs): pass

@Trainer.register
class PolicyTrainer(Trainer):
    def __init__(self, name,
        state_encoder, actor, critic,
        buffer: RolloutReplayBuffer,
        envs : list[Environment], rollout_length=1000,
        np_seed=None,
        wandb_logging=False, wandb_project=None
    ):
        super().__init__(name, wandb_logging, wandb_project)
        self.rollout_length = rollout_length

        for param in state_encoder.parameters(): param.requires_grad = False
        state_encoder.eval()
        self.state_encoder, self.critic, self.actor, self.critic_opt, self.actor_opt = self.accelerator.prepare(
            state_encoder, critic, actor,
            torch.optim.Adam(critic.parameters(), lr=3e-4),
            torch.optim.Adam(actor.parameters(), lr=3e-4)
        )

        self.envs = envs

        self.buffer = buffer

        self.env_ptr = 0 # tracks env currently being used to collect data
        self.env_step = 0 # tracks rollouts taken in current env, used to determine when to switch to next env
        self.epoch = 0 # track epochs across checkpoints

        self.rng = np.random.default_rng(np_seed)

    def fill_buffer(self, n=1000, show_progress=False):
        if show_progress: pbar = tqdm(total=n)

        rollouts = []
        step = 0
        while step < n:
            env = self.envs[self.env_ptr]
            state, done = env.reset(), False
            obs, actions, xys, dones = [], [], [], []
            t = 0
            while t < self.rollout_length:
                action = self.rng.choice(env.available_actions) #NOTE: no way to be better than random for now without goal predictor
                xy = np.zeros(2) if action != 6 else self.rng.integers(0, (210, 160)) # will be converted to 64x64 coordinates in env.step()

                obs.append(state)
                actions.append(action)
                xys.append(xy)
                dones.append(done)
                if done: break

                state, done, _ = env.step(action, xy)
                t += 1

            rollouts.append(Rollout(
                obs=np.array(obs, dtype=np.float16),
                action=np.array(actions, dtype=np.uint8),
                xy=np.array(xys, dtype=np.uint8),
                done=np.array(dones, dtype=bool),
                extras={}
            ))

            self.env_step += 1
            step += 1

            if self.env_step >= 40: #NOTE: hardcoded swap interval. This number fills the initial buffer evenly with all 25 available arc games
                self.env_ptr = (self.env_ptr + 1) % len(self.envs)
                self.env_step = 0

            if show_progress:
                pbar.update(1)
                pbar.set_description(f"Rollout length: {rollouts[-1].obs.shape[0]}")

        self.buffer.add_batch(rollouts)
        if show_progress: pbar.close()

    def sample_buffer(self, bs):
        '''
        Process buffer sample to ready it for contrastive learning with InfoNCE loss.
        '''
        batch = self.buffer.sample(bs)
        # print(f"obs:    {batch[0].obs.shape}")
        # print(f"action: {batch[0].action.shape}")
        # print(f"xy:     {batch[0].xy.shape}")
        # print(f"done:   {batch[0].done.shape}")

        # concatenate rollouts in batch dimension
        obs = np.concatenate([rollout.obs for rollout in batch], axis=0)
        actions = np.concatenate([rollout.action for rollout in batch], axis=0)
        xys = np.concatenate([rollout.xy for rollout in batch], axis=0)
        dones = np.concatenate([rollout.done for rollout in batch], axis=0)

        # see https://github.com/wang-kevin3290/scaling-crl/blob/main/buffer.py#L186:
        # basically, for each state in rollouts randomly pick a future state as goal
        N = obs.shape[0]
        arrangement = np.arange(N)
        probs = np.triu(0.99 ** (arrangement[None] - arrangement[:, None]), k=1)
        rollouts_idx = np.repeat(np.arange(bs), [rollout.obs.shape[0] for rollout in batch])
        rollouts_mask = np.equal(rollouts_idx[:, None], rollouts_idx[None, :])
        probs = probs * rollouts_mask + np.eye(N) * 1e-5

        goal_idx = torch.distributions.Categorical(probs=torch.from_numpy(probs)).sample().numpy()
        future_obs = obs[goal_idx[:-1]]

        return Rollout(
            obs=obs[:-1],
            action=actions[:-1],
            xy=xys[:-1],
            done=dones[:-1],
            extras={'goals': future_obs}
        )

    def critic_loss(self, batch : Rollout):
        obs = torch.from_numpy(batch.obs).to(dtype=torch.float32, device=self.device)
        goals = torch.from_numpy(batch.extras['goals']).to(dtype=torch.float32, device=self.device)

        actions = torch.from_numpy(batch.action).to(dtype=torch.long, device=self.device)
        xys = torch.from_numpy(batch.xy).to(dtype=torch.float32, device=self.device)
        actions = torch.cat([F.one_hot(actions, num_classes=8), xys], dim=-1)

        # refer to https://github.com/wang-kevin3290/scaling-crl/blob/main/train.py#L805
        logits = self.critic(obs, actions, goals)
        loss = -torch.mean(torch.diagonal(logits) - torch.logsumexp(logits, dim=-1))
        logsumexp = torch.logsumexp(logits + 1e-6, dim=-1)
        loss += 0.1 * logsumexp.square().mean()

        return loss
        
    def actor_loss(self, batch):
        obs = torch.from_numpy(batch.obs).to(dtype=torch.float32, device=self.device)
        goals = torch.from_numpy(batch.extras['goals']).to(dtype=torch.float32, device=self.device)

        probs, xy_dist = self.actor(obs, goals)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1) # (B,)
        xy = torch.where(actions.unsqueeze(-1) == 6, xy_dist.rsample(), torch.zeros(2, device=self.device))
        log_probs = torch.log(probs[torch.arange(probs.size(0)), actions] + 1e-6) + torch.where(actions == 6, xy_dist.log_prob(xy), torch.zeros(1, device=self.device))

        actions = torch.cat([F.one_hot(actions, num_classes=8), xy], dim=-1)
        critic_score = self.critic(obs, actions, goals, for_info_nce=False)

        # maximize critic score with entropy to encourage exploration, hardcode alpha = 0.2
        return torch.mean(0.2 * log_probs - critic_score)

    def train(self, epochs, bs, utd=40, save_every=None, directory="./checkpoints"):
        if self.buffer.size == 0: self.fill_buffer(show_progress=True)

        pbar = tqdm(range(epochs))
        for _ in pbar:
            self.fill_buffer(10)
            c_loss, a_loss = 0.0, 0.0
            for _ in range(utd):
                batch = self.sample_buffer(bs)

                self.critic_opt.zero_grad()
                critic_loss = self.critic_loss(batch)
                self.accelerator.backward(critic_loss)
                self.critic_opt.step()

                self.actor_opt.zero_grad()
                actor_loss = self.actor_loss(batch)
                self.accelerator.backward(actor_loss)
                self.actor_opt.step()

                c_loss += critic_loss.item()
                a_loss += actor_loss.item()
                pbar.set_description(f"Training {self.name} | Critic Loss: {critic_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f}")
            if self.wandb_logging and wandb is not None:
                wandb.log({
                    'critic_loss': c_loss / utd,
                    'actor_loss': a_loss / utd
                }, step=self.epoch)
            
            self.epoch += 1
            if save_every is not None and self.epoch % save_every == 0:
                self.save(directory, step=self.epoch)

        if self.wandb_logging and wandb is not None:
            wandb.finish()
        self.save(directory)
            
    def save(self, directory, step=None, save_buffer=False):
        data = {
            'epoch': self.epoch,
            'env_ptr': self.env_ptr,
            'critic': self.accelerator.get_state_dict(self.critic),
            'actor': self.accelerator.get_state_dict(self.actor),
            'critic_opt': self.critic_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
        }
        name = f"{'' if self.wandb_project is None else self.wandb_project + '_'}{self.name}"
        path = f"{directory}/{name}_{step}.pt" if step is not None else f"{directory}/{name}.pt"
        torch.save(data, path)
        if save_buffer: self.buffer.save_data(directory, name, step)

    def load(self, directory, step=None, load_default_buffer=False):
        name = f"{'' if self.wandb_project is None else self.wandb_project + '_'}{self.name}"
        path = f"{directory}/{name}_{step}.pt" if step is not None else f"{directory}/{name}.pt"
        data = torch.load(path, map_location=self.device)

        self.epoch = data['epoch']
        self.env_ptr = data['env_ptr']
        self.buffer.load_data(directory, name, step, load_default=load_default_buffer)

        critic = self.accelerator.unwrap_model(self.critic)
        actor = self.accelerator.unwrap_model(self.actor)
        critic.load_state_dict(data['critic'])
        actor.load_state_dict(data['actor'])
        self.critic_opt.load_state_dict(data['critic_opt'])
        self.actor_opt.load_state_dict(data['actor_opt'])

        if self.accelerator.scaler is not None and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])


@Trainer.register
class StateEncoderTrainer(Trainer):
    def __init__(self, name,
        encoder_model, decoder_model, # models
        envs : list[Environment],
        np_seed=None,
        wandb_logging=False, wandb_project=None
    ):
        super().__init__(name, wandb_logging, wandb_project)

        encoder = encoder_model
        decoder = decoder_model
        optim = torch.optim.AdamW([
            { 'params': encoder.parameters() },
            { 'params': decoder.parameters() }
        ])
        self.encoder, self.decoder, self.optim = self.accelerator.prepare(encoder, decoder, optim)

        self.envs = envs
        self.buffer = ReplayBuffer(10000, {
            'obs': ((3, 210, 160), torch.float32)
        }, self.device)

        self.env_ptr = 0 # tracks env currently being used to collect data
        self.env_step = 0 # tracks steps taken in current env, used to determine when to switch to next env
        self.epoch = 0 # track epochs across checkpoints

        self.rng = np.random.default_rng(np_seed)

    def fill_buffer(self, n=1000):
        obs = []
        step = 0
        while step < n:
            env = self.envs[self.env_ptr]

            action = self.rng.choice(env.available_actions)
            xy = None if action != 6 else self.rng.integers(0, (210, 160)) # will be converted to 64x64 coordinates in env.step()
            
            state, done, _ = env.step(action, xy)
            if done: env.reset()

            obs.append(state)
            self.env_step += 1
            step += 1

            if self.env_step >= 10: #NOTE: hardcoded swap interval
                self.env_ptr = (self.env_ptr + 1) % len(self.envs)
                self.env_step = 0

        self.buffer.add_batch(torch.tensor(np.array(obs), device=self.device, dtype=torch.float32))

    def train(self, epochs, bs, lr=0.001, save_every=None, directory="./checkpoints"):    
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        if self.buffer.size == 0: self.fill_buffer()

        for _ in tqdm(range(epochs), desc=f"Training {self.name}"):
            self.fill_buffer(10)
            
            self.optim.zero_grad()
            
            batch = self.buffer.sample(bs)['obs']
            result = self.decoder(self.encoder(batch))
            loss = 0.5 * F.mse_loss(result, batch, reduction='sum') / bs

            self.accelerator.backward(loss)
            self.optim.step()

            if self.wandb_logging and wandb is not None:
                wandb.log({ 'loss': loss.item() }, step=self.epoch)

            self.epoch += 1
            if save_every is not None and self.epoch % save_every == 0:
                self.save(directory, step=self.epoch)

        if self.wandb_logging and wandb is not None:
            wandb.finish()
        self.save(directory)

    def save(self, directory, step=None):
        # dunno how to save env state, but as long as we don't interrupt training too often effect should be negligible
        data = {
            'epoch': self.epoch,
            'env_ptr': self.env_ptr,
            'buffer': self.buffer.get_save_data(),
            'encoder': self.accelerator.get_state_dict(self.encoder),
            'decoder': self.accelerator.get_state_dict(self.decoder),
            'optim': self.optim.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
        }
        name = f"{'' if self.wandb_project is None else self.wandb_project + '_'}{self.name}"
        path = f"{directory}/{name}_{step}.pt" if step is not None else f"{directory}/{name}.pt"
        torch.save(data, path)

    def load(self, directory, step=None):
        name = f"{'' if self.wandb_project is None else self.wandb_project + '_'}{self.name}"
        path = f"{directory}/{name}_{step}.pt" if step is not None else f"{directory}/{name}.pt"
        data = torch.load(path, map_location=self.device)

        self.epoch = data['epoch']
        self.env_ptr = data['env_ptr']
        self.buffer.load_save_data(data['buffer'])

        encoder = self.accelerator.unwrap_model(self.encoder)
        decoder = self.accelerator.unwrap_model(self.decoder)
        encoder.load_state_dict(data['encoder'])
        decoder.load_state_dict(data['decoder'])
        self.optim.load_state_dict(data['optim'])

        if self.accelerator.scaler is not None and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])


def load_environments() -> list[Environment]:
    '''
    Get all atari and arc environments
    '''
    import arc_agi
    import gymnasium as gym
    import ale_py

    arc = arc_agi.Arcade()
    arc_env_ids = [game.game_id for game in arc.get_environments()]
    arc_envs_ = [arc.make(game_id) for game_id in arc_env_ids]
    arc_envs = [Environment("arc", env) for env in arc_envs_ if env is not None]
    for env in arc_envs: env.reset()

    gym.register_envs(ale_py)
    # these environments are frameskip=4 and repeat_action_prob=0.25
    atari_env_ids = [eid for eid in gym.envs.registry.keys() if eid.startswith("ALE/")] # type: ignore
    atari_envs = [Environment("atari", gym.make(eid, full_action_space=True)) for eid in atari_env_ids]
    for env in atari_envs: env.reset()

    envs = arc_envs + atari_envs
    return envs