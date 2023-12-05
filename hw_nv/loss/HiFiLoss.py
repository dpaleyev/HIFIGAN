import torch
from torch import nn
from hw_nv.utils import MelSpectrogram, MelSpectrogramConfig

class HiFiLoss(nn.Module):
    def __init__(self, fm_lambda, mel_lambda) -> None:
        super().__init__()
        self.fm_lambda = fm_lambda
        self.mel_lambda = mel_lambda
        self.mel_spectrogram = MelSpectrogram(MelSpectrogramConfig)
    
    def forward(self, spectrogram, gen_wav,
                period_gen,
                period_gen_features,
                period_real,
                period_real_features,
                scale_gen,
                scale_gen_features,
                scale_real,
                scale_real_features, **kwargs):
        
        gen_spectrogram = self.mel_spectrogram(gen_wav.squeeze(1))
        spectrogram = nn.ConstantPad3d(padding=(0, 0, 0, 0, 0, gen_spectrogram.shape[-1] - spectrogram.shape[-1]), value=0)(spectrogram)

        adv_loss = 0
        for periods in period_gen:
            adv_loss = adv_loss + torch.mean((periods - 1) ** 2)

        for scales in scale_gen:
            adv_loss = adv_loss + torch.mean((scales - 1) ** 2)
        

        fm_loss = 0
        for real, gen in zip(period_real_features, period_gen_features):
            fm_loss = fm_loss + torch.nn.functional.l1_loss(gen, real)

        for real, gen in zip(scale_real_features, scale_gen_features):
            fm_loss = fm_loss + torch.nn.functional.l1_loss(gen, real)
        
        mel_loss = torch.nn.functional.l1_loss(gen_spectrogram, spectrogram)

        gen_loss = adv_loss + self.fm_lambda * fm_loss + self.mel_lambda * mel_loss

        disc_loss = 0
        for real, gen in zip(period_real, period_gen):
            disc_loss = disc_loss + torch.mean((real - 1) ** 2) + torch.mean(gen ** 2)
        for real, gen in zip(scale_real, scale_gen):
            disc_loss = disc_loss + torch.mean((real - 1) ** 2) + torch.mean(gen ** 2)
        
        return disc_loss, gen_loss, adv_loss, fm_loss, mel_loss

        
