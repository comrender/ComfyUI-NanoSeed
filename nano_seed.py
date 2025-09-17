import torch
import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import os

# Helper function to convert ComfyUI tensor (B, H, W, C) to PIL Image (single or first from batch)
def tensor2pil(image_tensor):
    # Assume single image or take first from batch
    if image_tensor.shape[0] > 1:
        image_tensor = image_tensor[0:1]  # Process one at a time if batched here
    i = 255. * image_tensor.cpu().numpy()
    image = np.clip(i, 0, 255).astype(np.uint8)
    image = np.transpose(image[0], (1, 2, 0))  # HWC
    return Image.fromarray(image)

# Helper function to convert PIL Image back to ComfyUI tensor (B=1, H, W, C)
def pil2tensor(pil_image):
    arr = np.array(pil_image).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]  # Add batch dim
    arr = np.transpose(arr, (0, 3, 1, 2))  # BHWC
    return torch.from_numpy(arr)

# Main node class
class NanoSeedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Supports batch/multi-image input
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": (["nano_banana", "seedream"],),
                "fal_key": ("STRING", {"default": "your_fal_key_here"}),
            },
            "optional": {
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64, "display": "number"}),
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
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        all_edited_tensors = []

        for b in range(batch_size):
            # Extract single image tensor from batch
            single_image = image[b:b+1] if batch_size > 1 else image

            # Convert to PIL
            pil_image = tensor2pil(single_image)

            # Handle custom resolution
            custom_size = (width > 0 and height > 0)
            if custom_size and model == "nano_banana":
                # For NanoBanana, resize input image
                pil_image = pil_image.resize((width, height), Image.LANCZOS)
            # For Seedream, resolution is handled in API payload

            # Encode to base64 data URI
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            img_data_uri = f"data:image/png;base64,{img_str}"

            # Prepare API endpoint and payload
            if model == "nano_banana":
                url = "https://fal.run/fal-ai/nano-banana/edit"
                payload = {
                    "prompt": prompt,
                    "image_urls": [img_data_uri],
                    "num_images": num_images,
                    "output_format": "png",
                    "sync_mode": True,
                }
            elif model == "seedream":
                url = "https://fal.run/fal-ai/bytedance/seedream/v4/edit"
                payload = {
                    "prompt": prompt,
                    "image_urls": [img_data_uri],
                    "num_images": num_images,
                    "max_images": 1,  # Keep simple; can extend for more variations
                    "seed": seed,
                    "enable_safety_checker": True,
                    "sync_mode": True,
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

            # Download and convert each generated image
            for img_info in api_result["images"][:num_images]:  # Limit to requested num
                img_url = img_info.get("url")
                if not img_url:
                    continue

                img_resp = requests.get(img_url)
                if img_resp.status_code != 200:
                    raise ValueError("Failed to download generated image")

                pil_edited = Image.open(BytesIO(img_resp.content))
                tensor_edited = pil2tensor(pil_edited)
                all_edited_tensors.append(tensor_edited)

        # Stack all edited tensors into a batch
        if all_edited_tensors:
            batched_output = torch.cat(all_edited_tensors, dim=0)
        else:
            batched_output = torch.empty((1, 1, 1, 3))  # Fallback empty

        return (batched_output,)

# For batch processing in workflows, ComfyUI handles it via the IMAGE type