import torch
from PIL import Image
from diffusers import StableDiffusionSAGPipeline

pipe = StableDiffusionSAGPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, sag_scale=0.75).images[0]

down_attention_maps_0 = pipe.down_attention_maps[::3]
down_attention_maps_1 = pipe.down_attention_maps[1::3]
down_attention_maps_2 = pipe.down_attention_maps[2::3]

attention_maps = pipe.attention_maps

up_attention_maps_0 = pipe.up_attention_maps[::3]
up_attention_maps_1 = pipe.up_attention_maps[1::3]
up_attention_maps_2 = pipe.up_attention_maps[2::3]

down_attention_maps_0 = torch.stack(down_attention_maps_0).to(device='cpu')
down_attention_maps_1 = torch.stack(down_attention_maps_1).to(device='cpu')
down_attention_maps_2 = torch.stack(down_attention_maps_2).to(device='cpu')
attention_maps = torch.stack(attention_maps).to(device='cpu')
up_attention_maps_0 = torch.stack(up_attention_maps_0).to(device='cpu')
up_attention_maps_1 = torch.stack(up_attention_maps_1).to(device='cpu')
up_attention_maps_2 = torch.stack(up_attention_maps_2).to(device='cpu')

torch.save(down_attention_maps_0, './down_attention_maps_0')
torch.save(down_attention_maps_1, './down_attention_maps_1')
torch.save(down_attention_maps_2, './down_attention_maps_2')
torch.save(attention_maps, './mid_attention_maps')
torch.save(up_attention_maps_0, './up_attention_maps_0')
torch.save(up_attention_maps_1, './up_attention_maps_1')
torch.save(up_attention_maps_2, './up_attention_maps_2')

image.save("./astronaut.png")