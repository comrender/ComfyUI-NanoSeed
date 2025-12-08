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
                "model": (["nano_banana", "nano_banana_pro", "seedream_4.5", "qwen_edit_plus", "flux_2_edit"],),
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
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "aspect_ratio": (["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"], {"default": "auto"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    OUTPUT_NODE = True

    def edit_image(self, prompt, model, fal_key, image1=None, image2=None, image3=None, image4=None, image5=None,
                   width=0, height=0, num_images=1, seed=0, aspect_ratio="auto", resolution="1K",
                   acceleration="none"):  # Hardcoded to none, kept for compatibility
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
            
            # Resize if custom size (model-specific)
            if custom_size:
                if model in ["nano_banana", "nano_banana_pro"]:
                    # Nano models don't support custom size; ignore and use aspect/resolution
                    pass
                else:
                    pil_image = pil_image.resize((width, height), Image.LANCZOS)
            
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            img_data_uris.append(f"data:image/png;base64,{img_str}")
        
        # Enforce limits
        if model == "flux_2_edit" and len(img_data_uris) > 1:
            raise ValueError("Flux 2 Edit supports only a single input image. Use only image1.")
        if model == "seedream_4.5" and len(img_data_uris) + num_images > 15:
            raise ValueError("Seedream 4.5: Total inputs + outputs must <=15.")
        
        # Model-specific payloads
        if model == "nano_banana":
            url = "https://fal.run/fal-ai/nano-banana/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 4),
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "sync_mode": True,
            }
        elif model == "nano_banana_pro":
            url = "https://fal.run/fal-ai/nano-banana-pro/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 4),
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "output_format": "png",
                "sync_mode": True,
            }
        elif model == "seedream_4.5":
            url = "https://fal.run/fal-ai/bytedance/seedream/v4.5/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 6),
                "seed": seed,
                "enable_safety_checker": True,
                "sync_mode": True,
            }
            if custom_size:
                if not (1920 <= width <= 4096 and 1920 <= height <= 4096):
                    raise ValueError("Seedream 4.5: Width/height must be 1920-4096px.")
                area = width * height
                if not (3686400 <= area <= 16777216):
                    raise ValueError(f"Seedream 4.5: Image area must be 3,686,400-16,777,216px. Got {area}.")
                payload["image_size"] = {"width": width, "height": height}
        elif model == "qwen_edit_plus":
            url = "https://fal.run/fal-ai/qwen-image-edit-plus"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 4),
                "seed": seed,
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "enable_safety_checker": True,
                "output_format": "png",
                "sync_mode": True,
                "acceleration": acceleration,
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
        elif model == "flux_2_edit":
            url = "https://fal.run/fal-ai/flux-2/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris[:1],  # Single image
                "num_images": min(num_images, 4),
                "seed": seed,
                "guidance_scale": 2.5,
                "num_inference_steps": 28,
                "enable_prompt_expansion": False,
                "enable_safety_checker": True,
                "output_format": "png",
                "sync_mode": True,
                "acceleration": acceleration,
            }
            if custom_size:
                if not (512 <= width <= 2048 and 512 <= height <= 2048):
                    raise ValueError("Flux 2: Size must be 512-2048px.")
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
