import torch
import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image

# Helper function to convert ComfyUI tensor (B=1, H, W, C) to PIL Image (RGB)
def tensor2pil(image_tensor):
    # image_tensor is always (1, H, W, C) from the loop
    i = 255. * image_tensor[0].cpu().numpy()  # (H, W, C)
    image = np.clip(i, 0, 255).astype(np.uint8)
    
    # Handle channels for PIL (always output RGB)
    c = image.shape[-1]
    if c == 1:
        image = np.repeat(image, 3, axis=-1)  # (H, W, 3)
    elif c == 3:
        pass  # Already good
    elif c == 4:
        image = image[..., :3]  # Drop alpha
    else:
        raise ValueError(f"Unsupported channels: {c}. Expected 1, 3, or 4.")
    
    return Image.fromarray(image, mode='RGB')

# Helper function to convert PIL Image (RGB) back to ComfyUI tensor (B=1, H, W, C)
def pil2tensor(pil_image):
    arr = np.array(pil_image).astype(np.float32) / 255.0  # (H, W, 3)
    arr = arr[np.newaxis, ...]  # (1, H, W, 3) - No transpose needed for BHWC
    return torch.from_numpy(arr)

# Main node class
class NanoSeedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Supports batch/multi-image input
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": (["nano_banana", "seedream", "flux_kontext_pro", "qwen_edit_plus"],),
                "fal_key": ("STRING", {"default": "your_fal_key_here"}),
            },
            "optional": {
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

    def edit_image(self, image, prompt, model, fal_key, width=0, height=0, num_images=1, seed=0):
        if fal_key == "your_fal_key_here":
            raise ValueError("Please set your fal.ai API key in the node.")
        
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        # Limit to 5 images max for multi-support models
        batch_size = min(batch_size, 5)
        
        img_data_uris = []
        custom_size = (width > 0 and height > 0)
        
        for b in range(batch_size):
            single_image = image[b:b + 1] if batch_size > 1 else image
            pil_image = tensor2pil(single_image)
            
            if custom_size and model == "nano_banana":
                # Resize each input for NanoBanana
                pil_image = pil_image.resize((width, height), Image.LANCZOS)
            
            # Encode to base64 data URI
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            img_data_uris.append(f"data:image/png;base64,{img_str}")
        
        if model == "flux_kontext_pro" and len(img_data_uris) > 1:
            raise ValueError("Flux Kontext Pro supports only a single input image. Use one image or select another model.")
        
        # Prepare API endpoint and payload
        if model == "nano_banana":
            url = "https://fal.run/fal-ai/nano-banana/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,  # Multiple supported
                "num_images": num_images,
                "output_format": "png",
                "sync_mode": True,
            }
        elif model == "seedream":
            url = "https://fal.run/fal-ai/bytedance/seedream/v4/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,  # Multiple supported (up to 10)
                "num_images": 1,  # One generation
                "max_images": num_images,  # Variations
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
                "image_url": img_data_uris[0],  # Single
                "num_images": num_images,
                "seed": seed,
                "output_format": "png",
                "sync_mode": True,
                "guidance_scale": 3.5,  # Default
                "safety_tolerance": "2",  # Default
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
        elif model == "qwen_edit_plus":
            url = "https://fal.run/fal-ai/qwen-image-edit-plus"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,  # Multiple supported
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

        # Make API call
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

        # Process each generated image (limit to num_images)
        for img_info in api_result["images"][:num_images]:
            # Handle data_uri (sync=True) or url
            img_data = img_info.get("data_uri") or img_info.get("url")
            if not img_data:
                continue

            if img_data.startswith("data:"):
                # Parse base64 data URI
                header, encoded = img_data.split(",", 1)
                pil_edited = Image.open(BytesIO(base64.b64decode(encoded)))
            else:
                # Download from URL
                img_resp = requests.get(img_data)
                if img_resp.status_code != 200:
                    raise ValueError("Failed to download generated image")
                pil_edited = Image.open(BytesIO(img_resp.content))

            tensor_edited = pil2tensor(pil_edited)
            all_edited_tensors.append(tensor_edited)

        # Stack all edited tensors into a batch
        if all_edited_tensors:
            batched_output = torch.cat(all_edited_tensors, dim=0)
        else:
            # Fallback: empty batch of 1 dummy image
            batched_output = torch.zeros((1, 512, 512, 3))

        return (batched_output,)
