import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_url, hf_hub_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image


HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=4
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            hf_hub_download(repo_id="repository_id", filename=local_filename, cache_dir=cache_dir)
            print('Weights downloaded to:', os.path.join(cache_dir, local_filename))
        
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            print(" Do 'params'")
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            print(" Do 'params_ema'")
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            print(" Do 'No'")
            self.model.load_state_dict(loadnet, strict=True)
        # self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
    @torch.amp.autocast('cuda')
    def predict(self, lr_image, batch_size=4, patches_size=64, padding=4, pad_size=4):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        print(f"Input image shape: {lr_image.shape}")
        lr_image = pad_reflect(lr_image, pad_size)
        print(f"Padded image shape: {lr_image.shape}")
    
        # Ensure patches are created with correct size
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        print(f"Patches shape: {patches.shape}, p_shape: {p_shape}")
        expected_patch_size = patches_size + 2 * padding
        if patches.shape[1] != expected_patch_size:
            raise ValueError(
                f"Patch size mismatch: expected {expected_patch_size}, got {patches.shape[1]}"
            )
    
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()
    
        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)
    
        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()
        print(f"Scaled patches shape: {np_sr_image.shape}")
    
        # Adjust expected patch size based on observed behavior
        expected_scaled_patch_size = patches_size * scale
        if np_sr_image.shape[1] == (patches_size + 2 * padding) * scale:
            print(f"Warning: Model produced patches of size {(patches_size + 2 * padding) * scale}. Adjusting expected size.")
            expected_scaled_patch_size = (patches_size + 2 * padding) * scale
    
        if np_sr_image.shape[1] != expected_scaled_patch_size:
            raise ValueError(
                f"Scaled patch size mismatch: expected {expected_scaled_patch_size}, got {np_sr_image.shape[1]}"
            )
    
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        print(f"Padded size scaled: {padded_size_scaled}, Target shape: {scaled_image_shape}")
    
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        print(f"Final image shape: {sr_img.shape}")
        sr_img = Image.fromarray(sr_img)
        return sr_img
