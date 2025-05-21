# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset_rf(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.folders = sorted(os.listdir(data_dir))
        self.data_files = []
        for folder in self.folders:
            folder_path = os.path.join(data_dir, folder)
            data_files = sorted(glob(f"{folder_path}/*.npz"))
            self.data_files.extend(data_files)
        print('len:', self.data_files[0:10], len(self.data_files))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data = np.load(data_file)
        features = data['x']
        labels = data['y']
        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_index = 2
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)

    first = False

    if first:
        checkpoint = torch.load("./results/001-DiT-XL-2/checkpoints/0040000.pt", map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load("./results/002-DiT-XL-2/checkpoints/model.pt", map_location=lambda storage, loc: storage)
    
    if "ema" in checkpoint:  # supports checkpoints from train.py
        state_dict = checkpoint["ema"]
    model.load_state_dict(state_dict)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # newly added
    # teacher_model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    # teacher_model.load_state_dict(torch.load("./results/001-DiT-XL-2/checkpoints/0040000.pt", map_location=lambda storage, loc: storage)["ema"])
    # teacher_model.eval()
    diffusion_ode_solver = create_diffusion(str(50))

    # return

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if first:
        pass
    else:
        opt.load_state_dict(checkpoint["opt"])

    print('===> Loaded checkpoint')

    # Setup data:
    # features_dir = f"{args.feature_path}/imagenet256_features"
    # labels_dir = f"{args.feature_path}/imagenet256_labels"

    # data_dir = 'samples/DiT-XL-2-0040000-size-256-vae-ema-cfg-4.0_rf1'
    data_dir = 'samples/DiT-XL-2-model-size-256-vae-ema-cfg-4.0_rf2'
    dataset = CustomDataset_rf(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)
    # teacher_model, model, opt, loader = accelerator.prepare(teacher_model, model, opt, loader)

    # newly added
    # teacher_model = accelerator.unwrap_model(teacher_model).eval()  # important!
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in tqdm(range(args.epochs)):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        
        # num_iters = 500#len(dataset) // args.global_batch_size
    
        # for _ in tqdm(range(num_iters)):
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            
            # y = torch.randint(0, args.num_classes, (args.global_batch_size, 1), device=device)
            
            # print('xy:', x.shape, x.min(), x.max(), y.shape, y.min(), y.max(), diffusion.num_timesteps)
            
            # x = x.squeeze(dim=1)
            # y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (y.shape[0],), device=device)
            model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses(model, x, t, dict(y=y))


            # newly added
            if False:
                z = torch.randn(y.shape[0], 4, latent_size, latent_size, device=device)
                z_test = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * y.shape[0], device=device)
                y_test = torch.cat([y, y_null], 0)
                with torch.no_grad():
                    ode_samples = diffusion_ode_solver.ddim_sample_loop(teacher_model.forward_with_cfg, z_test.shape, z_test, clip_denoised=False, model_kwargs=dict(y=y_test, cfg_scale=args.cfg_scale), progress=False, device=device)
                    ode_samples, _ = ode_samples.chunk(2, dim=0)  # Remove null class samples
                    # print('ode_samples:', ode_samples.shape)
                x = ode_samples * 0.18215

            # print('x:', x.shape, x.min(), x.max())
            # print('z:', z.shape, z.min(), z.max())

            else:
                z = torch.randn(y.shape[0], 4, latent_size, latent_size, device=device)
                alpha_t = t / diffusion.num_timesteps
                x_t = alpha_t.view(-1,1,1,1) * z + (1 - alpha_t).view(-1,1,1,1) * x     # be careful: x is data, z is noise
                model_output = model(x_t, t, **model_kwargs)[:, :4, ...]    # output mean and variance, only need mean for now
                # print('model_output:', model_output.shape, model_output.min(), model_output.max())
                loss = torch.mean((model_output - (x - z))**2)



            # loss_dict = diffusion.training_losses(model, x, t, dict(y=y))
            # loss = loss_dict["loss"].mean()

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    # checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    checkpoint_path = f"{checkpoint_dir}/model.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # return

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)   # 50_000, 10_000
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    args = parser.parse_args()
    main(args)
