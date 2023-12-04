import torch 
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(len(dilations)):
            block = nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channels, channels, 
                                               kernel_size=kernel_size, dilation=dilations[i], 
                                               padding="same")),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channels, channels, 
                                               kernel_size=kernel_size, dilation=1, 
                                               padding="same"))
            )
            self.blocks.add_module(f"block_{i}", block)
    
    def forward(self, x):
        out = 0
        for block in self.blocks:
            out = out + block(out)
        return out
        

class MRF(nn.Module):
    def __init__(self, channels, resblock_size, resblock_dilations):
        super().__init__()

        self.blocks = nn.ModuleList([
            ResBlock(channels, resblock_size[i], resblock_dilations[i])
            for i in range(len(resblock_size))
        ])

    def forward(self, x):
        out = 0
        for block in self.blocks:
            out += block(x)
        return out


class GeneratorBlock(nn.Module):
    def __init__(self, channels, upsample_size, upsample_stride, resblock_size, resblock_dilations):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=channels,
                out_channels=channels // 2,
                kernel_size=upsample_size,
                stride=upsample_stride,
                padding=(upsample_size - upsample_stride) // 2,
            ),
            MRF(channels // 2, resblock_size, resblock_dilations)
        )
    
    def forward(self, x):
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_chanels, upsample_size, upsample_stride, resblock_size, resblock_dilations):
        super().__init__()
        
        self.head = nn.utils.weight_norm(nn.Conv1d(input_channels, hidden_chanels, kernel_size=7, stride=1, padding=3))

        self.blocks = nn.ModuleList([
            GeneratorBlock(
                channels= hidden_chanels // (2 ** i),
                upsample_size=upsample_size[i],
                upsample_stride=upsample_stride[i],
                resblock_size=resblock_size,
                resblock_dilations=resblock_dilations
            )
            for i in range(len(upsample_size))
        ])

        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.weight_norm(nn.Conv1d(hidden_chanels // (2 ** len(upsample_size)), 1, kernel_size=7, stride=1, padding=3)),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)