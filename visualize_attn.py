import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import math
from IPython.display import display

image_path = './astronaut.png'
image = Image.open(image_path)

image_np = np.array(image)

H, W, C = image_np.shape #(512, 512, 3)

attention_ls = ["down_attention_maps_0", "down_attention_maps_1", "down_attention_maps_2", "mid_attention_maps", "up_attention_maps_0", "up_attention_maps_1", "up_attention_maps_2"]

for attention in attention_ls:
    down_attention_maps_0 = torch.load(attention)
    print(down_attention_maps_0.shape)
    folder_name = attention.replace("attention_", "") + "_visualization"
    try:   
        os.mkdir(folder_name)
    except:
        print("not made..?")
        pass
    
    for t, down_attention_map in enumerate(down_attention_maps_0):
        cond_attn = down_attention_map[0].numpy()
        uncond_attn = down_attention_map[1].numpy()
        #conduct global average pooling to change HW by HW to HW
        
        hw_1, hw_2 = cond_attn.shape
        assert hw_1 == hw_2
        
        # cond_attn = cond_attn.mean(axis=0)

        cond_attn = np.sqrt(np.sum(cond_attn**2, axis=0)) / hw_1
        hw = cond_attn.shape[0]
        h, w = int(math.sqrt(hw)), int(math.sqrt(hw))

        # Reshape the attention map to its original 2D shape
        cond_attn = cond_attn.reshape(h, w)

        # Normalize the attention map to range [0, 1]
        cond_attn = (cond_attn - cond_attn.min()) / (cond_attn.max() - cond_attn.min())

        # Convert the normalized attention map to a PyTorch tensor and add batch and channel dimensions
        cond_attn_tensor = torch.tensor(cond_attn).unsqueeze(0).unsqueeze(0).float()

        # Upsample the attention map to the size of the image
        upsampled_cond_attn = F.interpolate(cond_attn_tensor, size=(H, W), mode='nearest')

        # Convert the upsampled attention map to a numpy array
        upsampled_cond_attn_np = upsampled_cond_attn.squeeze(0).squeeze(0).cpu().numpy()

        grayscale_cond_attn = np.uint8(upsampled_cond_attn_np * 255)
        
        grayscale_cond_attn_img = Image.fromarray(grayscale_cond_attn)
        grayscale_cond_attn_img.save(f'./{folder_name}/cond_attn_{t}_grayscale.png')
        
        threshold = 0.1
        thresholded_cond_attn = (upsampled_cond_attn_np > threshold).astype(np.uint8)
        thresholded_cond_attn_img = Image.fromarray(thresholded_cond_attn * 255, 'L')
        thresholded_cond_attn_img.save(f'./{folder_name}/cond_attn_{t}_thresholded.png')
        
        
        attention_colormap = np.zeros((upsampled_cond_attn_np.shape[0], upsampled_cond_attn_np.shape[1], 4), dtype=np.uint8) # black background
        
        # Set the red channel to the attention values (scaled to 255)
        attention_colormap[:, :, 0] = (upsampled_cond_attn_np * 255).astype(np.uint8)

        # Set the alpha channel to 128 (or another value to adjust transparency)
        attention_colormap[:, :, 3] = 128  # 50% transparency; adjust as needed, black background

        # Convert to PIL Image
        attention_colormap_img = Image.fromarray(attention_colormap, 'RGBA')

        # Open the original image and ensure it's in RGBA format for alpha compositing
        original_img = Image.open(image_path).convert('RGBA')
        original_img_array = np.array(original_img)
        original_img_array[:,:,3] = 128
        original_img = Image.fromarray(original_img_array, 'RGBA')
        
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        # Composite the attention colormap over the original image
        combined_img = Image.alpha_composite(original_img, attention_colormap_img)
                
        original_img.save(f'./{folder_name}/cond_attn_{t}_original.png')
        combined_img.save(f'./{folder_name}/cond_attn_{t}.png')
        
        
        attention_colormap_img = Image.alpha_composite(background, attention_colormap_img)
        attention_colormap_img.save(f'./{folder_name}/cond_attn_{t}_attention.png')




    

# down_attention_maps_1 = torch.load('./down_attention_maps_1')
# print(down_attention_maps_1.shape)

# down_attention_maps_2 = torch.load('./down_attention_maps_2')
# print(down_attention_maps_2.shape)

# attention_maps = torch.load('./mid_attention_maps')
# print(attention_maps.shape)

# up_attention_maps_0 = torch.load('./up_attention_maps_0')
# print(up_attention_maps_0.shape)

# up_attention_maps_1 = torch.load('./up_attention_maps_1')
# print(up_attention_maps_1.shape)

# up_attention_maps_2 = torch.load('./up_attention_maps_2')
# print(up_attention_maps_2.shape)


