import torch
import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image

SEEDREAM_5_AUTO_IMAGE_SIZE_BY_RESOLUTION = {
    "seedream_5": {
        "0.5K": "auto_1K",
        "1K": "auto_1K",
        "2K": "auto_2K",
        "4K": "auto_2K",
    },
    "seedream_5_pro": {
        "0.5K": "auto_1K",
        "1K": "auto_1K",
        "2K": "auto_2K",
        "4K": "auto_2K",
    },
    "seedream_5_lite": {
        "0.5K": "auto_2K",
        "1K": "auto_2K",
        "2K": "auto_2K",
        "4K": "auto_4K",
    },
}

SEEDREAM_5_RESOLUTION_PIXELS = {
    "seedream_5": {
        "0.5K": 1024 * 1024,
        "1K": 1024 * 1024,
        "2K": 2048 * 2048,
        "4K": 2048 * 2048,
    },
    "seedream_5_pro": {
        "0.5K": 1024 * 1024,
        "1K": 1024 * 1024,
        "2K": 2048 * 2048,
        "4K": 2048 * 2048,
    },
    "seedream_5_lite": {
        "0.5K": 2560 * 1440,
        "1K": 2560 * 1440,
        "2K": 2048 * 2048,
        "4K": 4096 * 4096,
    },
}


def _round_to_multiple(value, multiple=8):
    return max(multiple, int(round(value / multiple)) * multiple)


def seedream_5_image_size(aspect_ratio, resolution, model="seedream_5"):
    auto_sizes = SEEDREAM_5_AUTO_IMAGE_SIZE_BY_RESOLUTION.get(
        model,
        SEEDREAM_5_AUTO_IMAGE_SIZE_BY_RESOLUTION["seedream_5"],
    )
    resolution_pixels = SEEDREAM_5_RESOLUTION_PIXELS.get(
        model,
        SEEDREAM_5_RESOLUTION_PIXELS["seedream_5"],
    )

    if aspect_ratio == "auto":
        return auto_sizes.get(resolution, auto_sizes["2K"])

    width_ratio, height_ratio = [int(part) for part in aspect_ratio.split(":", 1)]
    ratio = width_ratio / height_ratio
    target_pixels = resolution_pixels.get(resolution, resolution_pixels["2K"])

    width = (target_pixels * ratio) ** 0.5
    height = target_pixels / width

    max_dimension = 4096
    if width > max_dimension:
        scale = max_dimension / width
        width *= scale
        height *= scale
    if height > max_dimension:
        scale = max_dimension / height
        width *= scale
        height *= scale

    return {
        "width": _round_to_multiple(width),
        "height": _round_to_multiple(height),
    }


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

def tensor2data_uri(image_tensor):
    pil_image = tensor2pil(image_tensor)
    if pil_image is None:
        return None

    return pil2data_uri(pil_image)

def pil2data_uri(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Main node class
class NanoSeedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": (["nano_banana", "nano_banana_pro", "nano_banana_2", "gpt_image_2_edit", "grok_imagine_edit", "seedream_4.5", "seedream_5", "seedream_5_lite", "qwen_edit_plus", "flux_2_edit", "flux_2_pro", "flux_2_flex", "flux_2_klein_9b_edit"],),
                "fal_key": ("STRING", {"default": "your_fal_key_here"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "mask": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "display": "number"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "display": "number"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "aspect_ratio": (["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "4:1", "1:4", "8:1", "1:8"], {"default": "auto"}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "quality": (["low", "medium", "high"], {"default": "high"}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "thinking_level": (["off", "minimal", "high"], {"default": "off"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    OUTPUT_NODE = True

    def edit_image(self, prompt, model, fal_key, image1=None, image2=None, image3=None, image4=None, image5=None,
                   image6=None, image7=None, image8=None, image9=None, image10=None, mask=None,
                   width=0, height=0, num_images=1, num_inference_steps=28, seed=0, aspect_ratio="auto", resolution="1K",
                   quality="high", enable_web_search=False, thinking_level="off", acceleration="none"):  # Hardcoded to none, kept for compatibility
        if fal_key == "your_fal_key_here":
            raise ValueError("Please set your fal.ai API key in the node.")
        
        # Collect all non-None images
        image_inputs = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        images = [img for img in image_inputs if img is not None]
        if not images:
            raise ValueError("At least one image input must be connected.")
        
        # Convert each to PIL and encode
        img_data_uris = []
        custom_size = (width > 0 and height > 0)
        
        for img_tensor in images:
            pil_image = tensor2pil(img_tensor)
            if pil_image is None:
                continue
            
            # Resize if custom size (model-specific behavior)
            # For Flux and Qwen, we usually send image_size in payload, but resizing here 
            # ensures consistent aspect ratio calculation before sending if needed.
            # Nano models ignore this as per original code.
            if custom_size:
                if model in ["nano_banana", "nano_banana_pro", "nano_banana_2"]:
                    pass
                else:
                    pil_image = pil_image.resize((width, height), Image.LANCZOS)
            
            img_data_uri = pil2data_uri(pil_image)
            if img_data_uri is not None:
                img_data_uris.append(img_data_uri)
        
        # Enforce limits (Updated: Removed Flux 2 single image limit)
        if model in ["seedream_4.5", "seedream_5", "seedream_5_pro", "seedream_5_lite"] and len(img_data_uris) + num_images > 15:
            raise ValueError("Seedream: Total inputs + outputs must <=15.")
        
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
        elif model == "nano_banana_2":
            url = "https://fal.run/fal-ai/nano-banana-2/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 4),
                "seed": seed,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "output_format": "png",
                "enable_web_search": enable_web_search,
                "limit_generations": True,
                "safety_tolerance": "6",
                "sync_mode": True,
            }
            if thinking_level != "off":
                payload["thinking_level"] = thinking_level
        elif model == "gpt_image_2_edit":
            url = "https://fal.run/openai/gpt-image-2/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "quality": quality,
                "num_images": num_images,
                "output_format": "png",
                "sync_mode": True,
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
            if mask is not None:
                mask_data_uri = tensor2data_uri(mask)
                if mask_data_uri is not None:
                    payload["mask_image_url"] = mask_data_uri
        elif model == "grok_imagine_edit":
            url = "https://fal.run/xai/grok-imagine-image/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris[:3],
                "num_images": min(num_images, 4),
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
                "enable_safety_checker": False,
                "sync_mode": True,
            }
            if custom_size:
                if not (1920 <= width <= 4096 and 1920 <= height <= 4096):
                    raise ValueError("Seedream 4.5: Width/height must be 1920-4096px.")
                area = width * height
                if not (3686400 <= area <= 16777216):
                    raise ValueError(f"Seedream 4.5: Image area must be 3,686,400-16,777,216px. Got {area}.")
                payload["image_size"] = {"width": width, "height": height}
        elif model in ["seedream_5", "seedream_5_pro", "seedream_5_lite"]:
            if model in ["seedream_5", "seedream_5_pro"]:
                url = "https://fal.run/bytedance/seedream/v5/pro/edit"
            else:
                url = "https://fal.run/fal-ai/bytedance/seedream/v5/lite/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris[:10],
                "image_size": seedream_5_image_size(aspect_ratio, resolution, model),
                "num_images": min(num_images, 6),
                "enable_safety_checker": False,
                "sync_mode": True,
            }
            if model in ["seedream_5", "seedream_5_pro"]:
                payload["output_format"] = "png"
            else:
                payload["max_images"] = 1
        elif model == "qwen_edit_plus":
            url = "https://fal.run/fal-ai/qwen-image-edit-plus"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 4),
                "seed": seed,
                "guidance_scale": 4.0,
                "num_inference_steps": num_inference_steps,
                "enable_safety_checker": False,
                "output_format": "png",
                "sync_mode": True,
                "acceleration": acceleration,
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
        
        # Combined logic for Flux 2 Edit, Pro, Flex, and Klein 9B
        elif model in ["flux_2_edit", "flux_2_pro", "flux_2_flex", "flux_2_klein_9b_edit"]:
            if model == "flux_2_edit":
                url = "https://fal.run/fal-ai/flux-2/edit"
            elif model == "flux_2_pro":
                url = "https://fal.run/fal-ai/flux-2-pro/edit"
            elif model == "flux_2_flex":
                url = "https://fal.run/fal-ai/flux-2-flex/edit"
            elif model == "flux_2_klein_9b_edit":
                url = "https://fal.run/fal-ai/flux-2/klein/9b/edit"

            inference_steps = num_inference_steps
            if model == "flux_2_klein_9b_edit":
                inference_steps = min(max(num_inference_steps, 4), 8)

            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris[:4] if model == "flux_2_klein_9b_edit" else img_data_uris,
                "num_images": min(num_images, 4),
                "seed": seed,
                "num_inference_steps": inference_steps,
                "enable_safety_checker": False,
                "output_format": "png",
                "sync_mode": True,
            }
            if model != "flux_2_klein_9b_edit":
                payload["guidance_scale"] = 2.5
                payload["enable_prompt_expansion"] = False
                payload["acceleration"] = acceleration
            
            if custom_size:
                # Standard validation for Flux Edit, relaxed for Pro/Flex as they might handle more
                if model == "flux_2_edit":
                    if not (512 <= width <= 2048 and 512 <= height <= 2048):
                        raise ValueError("Flux 2 Edit: Size must be 512-2048px.")
                
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
            # Fallback (should be covered by error check above)
            batched_output = torch.zeros((1, 512, 512, 3))

        return (batched_output,)
