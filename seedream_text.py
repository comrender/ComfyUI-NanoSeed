import torch
import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image


# Helper function to convert PIL Image (RGB/RGBA/etc.) back to ComfyUI tensor (B=1, H, W, C)
def pil2tensor(pil_image):
    if pil_image is None:
        return None

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    arr = np.array(pil_image).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    return torch.from_numpy(arr)


class SeedreamTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "A beautiful cinematic image", "multiline": True}),
                "model": (["seedream_5_lite", "seedream_4.5"],),
                "fal_key": ("STRING", {"default": "your_fal_key_here"}),
            },
            "optional": {
                "image_size": ([
                    "auto_2K",
                    "auto_3K",
                    "auto_4K",
                    "square_hd",
                    "square",
                    "portrait_4_3",
                    "portrait_16_9",
                    "landscape_4_3",
                    "landscape_16_9",
                    "custom",
                ], {"default": "auto_2K"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "display": "number"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "display": "number"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "return_byteplus_urls": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_image"
    CATEGORY = "image/generate"
    OUTPUT_NODE = True

    def generate_image(
        self,
        prompt,
        model,
        fal_key,
        image_size="auto_2K",
        width=0,
        height=0,
        num_images=1,
        max_images=1,
        seed=0,
        enable_safety_checker=True,
        return_byteplus_urls=False,
    ):
        if fal_key == "your_fal_key_here" or not fal_key.strip():
            raise ValueError("Please set your fal.ai API key in the node.")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if model == "seedream_5_lite":
            url = "https://fal.run/fal-ai/bytedance/seedream/v5/lite/text-to-image"
        elif model == "seedream_4.5":
            url = "https://fal.run/fal-ai/bytedance/seedream/v4.5/text-to-image"
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Seedream 4.5 schema does not list auto_3K, so fall back safely.
        if model == "seedream_4.5" and image_size == "auto_3K":
            image_size = "auto_2K"

        payload = {
            "prompt": prompt,
            "num_images": min(num_images, 6),
            "max_images": min(max_images, 6),
            "enable_safety_checker": bool(enable_safety_checker),
            "sync_mode": True,
        }

        # Use custom dimensions only when both width and height are provided.
        # Otherwise use fal's image_size enum.
        if image_size == "custom" or (width > 0 and height > 0):
            if width <= 0 or height <= 0:
                raise ValueError("For custom image_size, both width and height must be greater than 0.")
            if width > 4096 or height > 4096:
                raise ValueError("Seedream text-to-image: width and height must be <= 4096px.")
            payload["image_size"] = {"width": width, "height": height}
        else:
            payload["image_size"] = image_size

        # Seedream 4.5 supports seed in the public schema. V5 Lite docs currently do not list seed as an input.
        if model == "seedream_4.5" and seed > 0:
            payload["seed"] = seed

        # V5 Lite schema exposes this option. Keeping it conditional avoids sending unknown params to 4.5.
        if model == "seedream_5_lite":
            payload["return_byteplus_urls"] = bool(return_byteplus_urls)

        headers = {
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=300)
        if response.status_code != 200:
            raise ValueError(f"API error {response.status_code}: {response.text}")

        api_result = response.json()
        if "images" not in api_result or len(api_result["images"]) == 0:
            raise ValueError(f"No images returned from API. Response: {api_result}")

        output_tensors = []
        max_outputs = min(len(api_result["images"]), max(1, num_images * max_images))

        for img_info in api_result["images"][:max_outputs]:
            img_data = img_info.get("data_uri") or img_info.get("url")
            if not img_data:
                continue

            if img_data.startswith("data:"):
                _, encoded = img_data.split(",", 1)
                pil_image = Image.open(BytesIO(base64.b64decode(encoded)))
            else:
                img_resp = requests.get(img_data, timeout=120)
                if img_resp.status_code != 200:
                    raise ValueError(f"Failed to download generated image: {img_resp.status_code}")
                pil_image = Image.open(BytesIO(img_resp.content))

            tensor_image = pil2tensor(pil_image)
            if tensor_image is not None:
                output_tensors.append(tensor_image)

        if output_tensors:
            batched_output = torch.cat(output_tensors, dim=0)
        else:
            batched_output = torch.zeros((1, 512, 512, 3))

        return (batched_output,)
