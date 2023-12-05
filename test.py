import argparse
import json
import torch
from pathlib import Path
import os
import torchaudio

import hw_nv.model as module_model
from hw_nv.utils.parse_config import ConfigParser
from hw_nv.logger import get_visualizer
from hw_nv.utils.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


def synthesize(model, input_dir, output_dir, device, writer):
    mel_transf = MelSpectrogram(MelSpectrogramConfig())
    input_dir = Path(input_dir)

    for path in input_dir.glob("*.wav"):
        filename = os.path.split(path)[-1]
        wav, sr = torchaudio.load(str(path))
        wav = wav[0:1, :]

        mel_spec = mel_transf(wav).squeeze(1)

        pred = model(mel_spec.to(device))["gen_wav"].squeeze(0)
        pred = pred.detach().cpu()
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        torchaudio.save(str(output_dir_path / filename), pred, 22050)


        writer.wandb.log({
            f"genrated_{filename}": writer.wandb.Audio(pred.numpy().T, sample_rate=22050)
        })
        writer.wandb.log({
            f"original_{filename}": writer.wandb.Audio(wav.numpy().T, sample_rate=22050)
        })

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="Path to checkpoint"
    )

    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="Path to config"
    )

    args.add_argument(
        "-i",
        "--input",
        default="./test_data",
        type=str,
        help="Path to input directory"
    )

    args.add_argument(
        "-o",
        "--output",
        default="./synth_result",
        type=str,
        help="Path to output directory"
    )

    args = args.parse_args()

    assert args.resume is not None, "Please specify checkpoint path"

    with open(args.config) as f:
        config = ConfigParser(json.load(f))
    
    logger = config.get_logger("test")
    writer = get_visualizer(
            config, logger, "wandb"
        )
    
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(args.resume, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    synthesize(model, args.input, args.output, device, writer)
