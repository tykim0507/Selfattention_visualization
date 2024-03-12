import torch
from PIL import Image
from diffusers import StableDiffusionSAGPipeline

pipe = StableDiffusionSAGPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of a horse running on a street"
image = pipe(prompt, sag_scale=0.75).images[0]