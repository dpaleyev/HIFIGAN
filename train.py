import argparse
import collections
import warnings

import numpy as np
import torch

import hw_nv.loss as module_loss
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch, n_class=len(text_encoder))
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    gen_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    gen_optimizer = config.init_obj(config["gen_optimizer"], torch.optim, gen_trainable_params)

    disc_trainable_params = filter(lambda p: p.requires_grad, model.discriminator.parameters())
    disc_optimizer = config.init_obj(config["disc_optimizer"], torch.optim, disc_trainable_params)

    gen_lr_scheduler = config.init_obj(config["gen_lr_scheduler"], torch.optim.lr_scheduler, gen_optimizer)
    disc_lr_scheduler = config.init_obj(config["disc_lr_scheduler"], torch.optim.lr_scheduler, disc_optimizer)


    trainer = Trainer(
        model,
        loss_module,
        gen_optimizer,
        disc_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        gen_lr_scheduler=gen_lr_scheduler,
        desc_lr_scheduler=disc_lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(discription="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
