import torch
from torch import nn

from hw_nv.base.base_model import BaseModel
from hw_nv.model.generator import Generator
from hw_nv.model.discriminator import Discriminator

class HiFiGAN(BaseModel):
    def __init__(self, input_channels, hidden_channels, upsample_size, upsample_stride, resblock_size, resblock_dilations, *args, **kwargs):
        super().__init__()

        self.generator = Generator(input_channels, hidden_channels, upsample_size, upsample_stride, resblock_size, resblock_dilations)
        self.discriminator = Discriminator()
    
    def discriminate(self, gen, real, **batch):
        return self.discriminator(gen, real)
    
    def forward(self, spectrogram, **batch):
        return self.generator(spectrogram)

    def generate(self, **batch):
        return self.forward(**batch)

        
    

        



