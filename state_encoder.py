import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from policy_model import ResidualNetwork

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.normalize1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.normalize2 = nn.BatchNorm2d(channels)

        # initialize weights
        sd = np.sqrt(2 / 3*3*channels)
        nn.init.normal_(self.conv1.weight, 0, sd)
        nn.init.zeros_(self.conv1.bias) # type: ignore
        nn.init.normal_(self.conv2.weight, 0, sd)
        nn.init.zeros_(self.conv2.bias) # type: ignore

    def forward(self, x):
        output = self.conv1(x)
        output = self.normalize1(output)
        output = F.silu(output)
        output = self.conv2(output)
        output = self.normalize2(output)
        return F.silu(x + output)
    
class ResNetDownsizeBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetDownsizeBlock, self).__init__()
        
        out_channels = in_channels * 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.normalize1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.normalize2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

        # initialize weights
        nn.init.normal_(self.conv1.weight, 0, np.sqrt(2 / 3*3*in_channels))
        nn.init.zeros_(self.conv1.bias) # type: ignore
        nn.init.normal_(self.conv2.weight, 0, np.sqrt(2 / 3*3*out_channels))
        nn.init.zeros_(self.conv2.bias) # type: ignore
        nn.init.normal_(self.shortcut.weight, 0, np.sqrt(2 / 1*1*in_channels))

    def forward(self, x):
        output = self.conv1(x)
        output = self.normalize1(output)
        output = F.silu(output)
        output = self.conv2(output)
        output = self.normalize2(output)
        x = self.shortcut(x)
        return F.silu(x + output)

class StateEncoder(nn.Module): # modified ResNet
    def __init__(self, num_blocks : tuple[int, int, int, int], embedding_dim=512):
        assert len(num_blocks) == 4 and all(n > 0 for n in num_blocks), "num_blocks should have 4 elements and all should be positive"
        super(StateEncoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.conv = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)),
                ("normalize", nn.BatchNorm2d(64)),
                ("silu", nn.SiLU()),
                ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])),
            nn.Sequential(*[ResNetBlock(64) for _ in range(num_blocks[0])]),
            nn.Sequential(ResNetDownsizeBlock(64), *[ResNetBlock(128) for _ in range(num_blocks[1]-1)]),
            nn.Sequential(ResNetDownsizeBlock(128), *[ResNetBlock(256) for _ in range(num_blocks[2]-1)]),
            nn.Sequential(ResNetDownsizeBlock(256), *[ResNetBlock(512) for _ in range(num_blocks[3]-1)]),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fn = nn.Linear(960, embedding_dim) # 512+256+128+64 = 960

        # initialize weights
        nn.init.normal_(self.conv[0].conv.weight, 0, np.sqrt(2 / 7*7*3)) #type: ignore
        nn.init.zeros_(self.conv[0].conv.bias) #type: ignore

    def forward(self, x): # input shape (B, 3, 210, 160)
        x = self.conv[0](x)   # shape (B, 64, 53, 40)
        x1 = self.conv[1](x)  # shape (B, 64, 53, 40)
        x2 = self.conv[2](x1) # shape (B, 128, 27, 20)
        x3 = self.conv[3](x2) # shape (B, 256, 14, 10)
        x4 = self.conv[4](x3) # shape (B, 512, 7, 5)
        output = torch.cat((self.pool(x1), self.pool(x2), self.pool(x3), self.pool(x4)), dim=1).reshape(-1, 960)
        output = self.fn(output)
        return output
    
class StateDecoder(nn.Module): # use a similar architecture as the policy model
    def __init__(self, embedding_dim=512, image_size : tuple[int, int]=(210, 160), decoder_depth=4):
        super(StateDecoder, self).__init__()
        self.decoder = ResidualNetwork(embedding_dim, 3*image_size[0]*image_size[1], num_blocks=decoder_depth)

    def forward(self, x):
        output = self.decoder(x)
        return output.reshape(-1, 3, 210, 160)


if __name__ == "__main__":
    from trainer import StateEncoderTrainer, load_environments

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-l", "--load", action='store_true', help="Whether to load from checkpoint")
    # parser.add_argument("-d", "--dir", type=str, default="./checkpoints", help="Directory to load from or save to")
    # parser.add_argument("-s", "--step", type=int, default=None, help="Step to load from, if loading from checkpoint")
    # args = parser.parse_args()

    envs = load_environments()

    encoder_size = (3, 4, 6, 3) # ResNet34
    embedding_dim = 512
    decoder_depth = 2
    bs = 128
    lr = 0.001
    encoder = StateEncoder(encoder_size, embedding_dim)
    decoder = StateDecoder(embedding_dim=embedding_dim, decoder_depth=decoder_depth)
    trainer = StateEncoderTrainer(
        f"final",
        encoder, decoder, envs,
        wandb_logging=True, wandb_project="arc3-state-encoder"
    )
    trainer.init_wandb_run({
        'encoder_size': encoder_size,
        'embedding_dim': embedding_dim,
        'decoder_depth': decoder_depth,
        'batch_size': bs,
        'learning_rate': lr
    })
    trainer.train(epochs=20000, bs=bs, lr=lr, save_every=2000)
