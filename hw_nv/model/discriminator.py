import torch
from torch import nn

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
    
        self.period = period

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0))),
                nn.LeakyReLU(),
            ),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))),
        ])
    
    def forward(self, x):
        features = []
        if x.shape[-1] % self.period:
            x = nn.functional.pad(x, (0, self.period - x.shape[-1] % self.period), "reflect")
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            features.append(x)
        return x.reshape(x.shape[0], -1), features              
        
        

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p)
            for p in [2, 3, 5, 7, 11]
        ])
    
    def forward(self, x):
        outs = []
        features = []
        for d in self.discriminators:
            out, feature = d(x)
            outs.append(out)
            features.append(feature)
        return outs, features


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])
    
    def forward(self, x):
        features = []
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            features.append(x)
        return x.reshape(x.shape[0], -1), features
    

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator()
            for i in range(3)
        ])

        self.pooling = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        ])
    
    def forward(self, x):
        outs = []
        features = []
        for i in range(3):
            x_in = x.clone()
            for j in range(i):
                x_in = self.pooling[j](x_in)
            out, feature = self.discriminators[i](x_in)
            outs.append(out)
            features.extend(feature)
        return outs, features
            

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mdp = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
    
    def forward(self, gen, real):
        print(gen.shape, real.shape)
        if gen.shape[-1] > real.shape[-1]:
            real = nn.ConstantPad3d(padding=(0, 0, 0, 0, 0, gen.shape[-1] - real.shape[-1]), value=0)(real)
        
        period_real, period_real_features = self.mdp(real)
        period_gen, period_gen_features = self.mdp(gen)
        scale_real, scale_real_features = self.msd(real)
        scale_gen, scale_gen_features = self.msd(gen)

        return {
            "period_real": period_real,
            "period_gen": period_gen,
            "scale_real": scale_real,
            "scale_gen": scale_gen,
            "period_real_features": period_real_features,
            "period_gen_features": period_gen_features,
            "scale_real_features": scale_real_features,
            "scale_gen_features": scale_gen_features,
        }
