import torch
import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image

# Helper function to convert ComfyUI tensor (B=1, H, W, C) to PIL Image (RGB)
def tensor2pil(image_tensor):
    if image_tensor is None or image_tensor.shape[0] == 0:
        return None
    i = 255. * image_tensor[0].cpu().numpy()  # (H, W, C)
    image = np.clip(i, 0, 255).astype(np.uint8)
    
    c = image.shape[-1]
    if c == 1:
        image = np.repeat(image, 3, axis=-1)
    elif c == 3:
        pass
    elif c == 4:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported channels: {c}. Expected 1, 3, or 4.")
    
    return Image.fromarray(image, mode='RGB')

# Helper function to convert PIL Image (RGB) back to ComfyUI tensor (B=1, H, W, C)
def pil2tensor(pil_image):
    if pil_image is None:
        return None
    arr = np.array(pil_image).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    return torch.from_numpy(arr)

# Main node class
class NanoSeedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": (["nano_banana", "seedream", "flux_kontext_pro", "qwen_edit_plus"],),
                "fal_key": ("STRING", {"default": "your_fal_key_here"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "display": "number"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "display": "number"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    OUTPUT_NODE = True

    def edit_image(self, prompt, model, fal_key, image1=None, image2=None, image3=None, image4=None, image5=None,
                   width=0, height=0, num_images=1, seed=0):
        if fal_key == "your_fal_key_here":
            raise ValueError("Please set your fal.ai API key in the node.")
        
        # Collect all non-None images
        images = [img for img in [image1, image2, image3, image4, image5] if img is not None]
        if not images:
            raise ValueError("At least one image input must be connected.")
        
        # Convert each to PIL and encode
        img_data_uris = []
        custom_size = (width > 0 and height > 0)
        
        for img_tensor in images:
            pil_image = tensor2pil(img_tensor)
            if pil_image is None:
                continue
            
            if custom_size and model == "nano_banana":
                pil_image = pil_image.resize((width, height), Image.LANCZOS)
            
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            img_data_uris.append(f"data:image/png;base64,{img_str}")
        
        # Enforce single image for Flux Kontext Pro
        if model == "flux_kontext_pro" and len(img_data_uris) > 1:
            raise ValueError("Flux Kontext Pro supports only a single input image. Use only image1.")
        
        # Prepare payload based on model
        if model == "nano_banana":
            url = "https://fal.run/fal-ai/nano-banana/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": num_images,
                "output_format": "png",
                "sync_mode": True,
            }
        elif model == "seedream":
            url = "https://fal.run/fal-ai/bytedance/seedream/v4/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": 1,
                "max_images": num_images,
                "seed": seed,
                "enable_safety_checker": True,
                "sync_mode": True,
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
        elif model == "flux_kontext_pro":
            url = "https://fal.run/fal-ai/flux-pro/kontext"
            payload = {
                "prompt": prompt,
                "image_url": img_data_uris[0],
                "num_images": num_images,
                "seed": seed,
                "output_format": "png",
                "sync_mode": True,
                "guidance_scale": 3.5,
                "safety_tolerance": "2",
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
        elif model == "qwen_edit_plus":
            url = "https://fal.run/fal-ai/qwen-image-edit-plus"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": num_images,
                "seed": seed,
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "enable_safety_checker": True,
                "output_format": "png",
                "sync_mode": True,
                "acceleration": "regular",
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}

        # API call
        headers = {
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")

        api_result = response.json()
        if "images" not in api_result or len(api_result["images"]) == 0:
            raise ValueError("No images returned from API")

        all_edited_tensors = []

        # Process up to num_images
        for img_info in api_result["images"][:num_images]:
            img_data = img_info.get("data_uri") or img_info.get("url")
            if not img_data:
                continue

            if img_data.startswith("data:"):
                _, encoded = img_data.split(",", 1)
                pil_edited = Image.open(BytesIO(base64.b64decode(encoded)))
            else:
                img_resp = requests.get(img_data)
                if img_resp.status_code != 200:
                    raise ValueError("Failed to download generated image")
                pil_edited = Image.open(BytesIO(img_resp.content))

            tensor_edited = pil2tensor(pil_edited)
            if tensor_edited is not None:
                all_edited_tensors.append(tensor_edited)

        # Stack output
        if all_edited_tensors:
            batched_output = torch.cat(all_edited_tensors, dim=0)
        else:
            batched_output = torch.zeros((1, 512, 512, 3))

        return (batched_output,)