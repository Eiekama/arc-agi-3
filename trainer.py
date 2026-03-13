import torch
from accelerate import Accelerator
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
try:
    import wandb
    WANDB_KEY = "your_wandb_key_here"
except ImportError:
    wandb = None


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
    
    @abstractmethod
    def save(self, directory, step=None): pass
    @abstractmethod
    def load(self, directory, step=None): pass
    @abstractmethod
    def train(self, epochs, bs): pass

@Trainer.register
class StateEncoderTrainer(Trainer):
    def __init__(self, name,
        encoder_model, decoder_model, # models
        envs, # list of environments to train on
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
            'obs': ((3, 210, 160), None) # use default dtype
        }, self.device)

        self.env_ptr = 0 # tracks env currently being used to collect data
        self.epoch = 0 # track epochs across checkpoints
    
    def prefill_buffer(self, n=1000):
        obs = []
        step = 0
        while step < n:
            #TODO: start generating trajectory from self.envs[self.env_ptr]
            # add obs to list and inc step
            # if done or timeout, reset env then inc self.env_ptr to move on to the next env
            pass
        self.buffer.add_batch(torch.tensor(obs, device=self.device))

    def train(self, epochs, bs, lr=0.001, save_every=1):
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        if self.buffer.size == 0: self.prefill_buffer()

        for _ in tqdm(range(epochs)): #TODO
            # add a trajectory to buffer
            # sample batch from buffer and train encoder-decoder to reconstruct the state
            # log loss to wandb
            self.epoch += 1
    
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
        path = f"{directory}/{self.name}_{step}.pt" if step is not None else f"{directory}/{self.name}.pt"
        torch.save(data, path)

    def load(self, directory, step=None):
        path = f"{directory}/{self.name}_{step}.pt" if step is not None else f"{directory}/{self.name}.pt"
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
