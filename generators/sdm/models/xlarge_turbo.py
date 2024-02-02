from diffusers import AutoPipelineForText2Image
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


pipe_xl_turbo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to(DEVICE)


def generate(prompt):
    image = pipe_xl_turbo(
        prompt=prompt, guidance_scale=0.0, num_inference_steps=1
    ).images[0]
    return image
