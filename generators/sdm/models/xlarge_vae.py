from diffusers import DiffusionPipeline, AutoencoderKL
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


pipe_vae_fp16 = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to(DEVICE)

pipe_xl_vae = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=pipe_vae_fp16,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to(DEVICE)

pipe_xl_vae_refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    vae=pipe_vae_fp16,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(DEVICE)


def generate(prompt):
    n_steps = 40
    high_noise_frac = 0.73
    image = pipe_xl_vae(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = pipe_xl_vae_refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    return image
