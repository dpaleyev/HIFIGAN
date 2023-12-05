import torch

from hw_nv.utils import MelSpectrogram, MelSpectrogramConfig

mel_spec_transf = MelSpectrogram(MelSpectrogramConfig())

def collate_fn(batch):
    """ 
    Args:
        batch: list of dicts with keys "wav_gt"
    """

    wav_gt = [item["wav_gt"] for item in batch]
    wav_gt = torch.stack(wav_gt)

    mel_gt = mel_spec_transf(wav_gt).squeeze(1)

    return {
        "wav_gt": wav_gt,
        "spectrogram": mel_gt,
    }