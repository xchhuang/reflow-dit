import gradio as gr
import numpy as np
import torch
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"


num_sampling_steps = 5
image_size = 256
num_classes = 1000
cfg_scale = 4.0
ckpt_path = './results/002-DiT-XL-2/checkpoints/model_v1.pt'
latent_size = image_size // 8
model = DiT_models['DiT-XL/2'](input_size=latent_size, num_classes=num_classes).to(device)
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict)
model.eval()
diffusion = create_diffusion(str(num_sampling_steps))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)    # ema, mse


# python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v2.pt --num-sampling-steps=50 --cfg-scale=4.0

# Placeholder function for your diffusion model
def diffusion_model(class_label):

    
    class_labels = [class_label]    # 12, 13, 567, 444, 666, 778, 23, 44, 49, 10
    
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )

    num_ddpm_steps = 1000
    skip = int(num_ddpm_steps // num_sampling_steps)
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
    # print(samples.shape, samples.min(), samples.max())
    samples = samples.detach().cpu().numpy()[0].transpose(1, 2, 0)
    samples = (samples + 1) / 2
    samples = np.clip(samples, 0, 1)
    return samples


def generate_image(seed):
    return diffusion_model(int(seed))

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Number(label="ImageNet class label (0-999)", value=42),
    outputs="image",
    title="reflow-dit",
    description="interative image generation using reflow-dit",
    # layout="vertical",
)

interface.css = """
body {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
}
input[type='number'] {
  width: 200px;  /* Match width to your preferences */
  margin: 0 auto 20px;  /* Center input and add space below */
  display: block;
}
.output_image {
  border: 2px solid #ccc;  /* Adding a border to mimic container */
  padding: 20px;  /* Padding around the image */
  margin-bottom: 10px;  /* Space between image container and next element */
}
.output_image img {
  width: 100%;
  height: auto;
  max-width: 500px;
  display: block;  /* Ensures the image is block level for proper alignment */
  margin: 0 auto;  /* Center the image */
}
"""


# Launch the Gradio app
interface.launch()
