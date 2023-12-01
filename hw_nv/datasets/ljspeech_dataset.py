from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import random

from hw_nv.utils import MelSpectrogramConfig, MelSpectrogram


class LJSpeechDataset(Dataset):
    def __init__(self, root_dir, wav_max_len=None, limit=None, **kwargs):
        self.root_dir = root_dir
        self.wav_max_len = wav_max_len

        self.files = []
        for path in Path(root_dir).rglob("*.wav"):
            self.files.append(path)

        if limit:
            self.files = self.files[:limit]

        self.mel_spec_transf = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, index):
        wav_gt, _ = torchaudio.load(self.files[index])
        if self.wav_max_len:
            begin = random.randint(0, wav_gt.shape[-1] - self.wav_max_len)
            wav_gt = wav_gt[:, begin:begin + self.wav_max_len]
        mel_gt = self.mel_spec_transf(wav_gt.detach()).squeeze(0)
        return {
            "wav_gt": wav_gt,
            "mel_gt": mel_gt
        }

    def __len__(self): 
        return len(self.files)
        