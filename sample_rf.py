# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    rf_version = ckpt_path.split('_')[-1].split('.')[0][1:]
    
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)


    start_time = time.time()

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = [387]
    

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )

    num_ddpm_steps = 1000
    skip = int(num_ddpm_steps // args.num_sampling_steps)
    seq = list(range(0, num_ddpm_steps, skip))
    seq_next = [-1] + list(seq[:-1])
    # seq = reversed(seq)

    x_alpha = z
    for i, j in zip(reversed(seq), reversed(seq_next)):
        # z = torch.cat([x_alpha, x_alpha], 0)
        tt = torch.randint(low=i+1, high=i+2, size=(n, )).to(device)
        tt_next = torch.randint(low=j+1, high=j+2, size=(n, )).to(device)
        
        # alpha_start = (tt + 1).float() / 1000
        # alpha_end = tt.float() / 1000

        at = tt.float() / num_ddpm_steps
        at_next = tt_next.float() / num_ddpm_steps


        d = model.forward_with_cfg(torch.cat([x_alpha, x_alpha], 0), torch.cat([tt, tt], 0), **model_kwargs)
        d, _ = d.chunk(2, dim=0)
        # print('d:', d.shape, x_alpha.shape)
        d = d[:, :4, ...]   # use only mean, no need to use variance

        # print('shape:', d.shape, alpha_start.shape, x_alpha.shape)
        # print('at:', at, at_next)
        x_alpha = x_alpha + (at - at_next).view(-1, 1, 1, 1) * d
        # print('x_alpha:', x_alpha.shape)
    samples = x_alpha
    

    # samples = diffusion.ddim_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    
    samples = vae.decode(samples / 0.18215).sample
    
    end_time = time.time()

    # Save and display images:
    save_image(samples, f"sample_rf{rf_version}_{args.num_sampling_steps}steps.png", nrow=4, normalize=True, value_range=(-1, 1))
    logging.info('===> Test done in {:.2f}s.'.format(end_time - start_time))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
