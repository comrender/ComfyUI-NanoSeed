import asyncio
import os
import re
import time

import torch
import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image

FAL_RUN_URL_PREFIX = "https://fal.run/"
FAL_QUEUE_URL_PREFIX = "https://queue.fal.run/"
FAL_HTTP_TIMEOUT_SECONDS = 60
FAL_QUEUE_POLL_INTERVAL_SECONDS = 1.0
FAL_MAX_CONCURRENT_RUNS = 8

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


def _prepare_image_data_uris(images, resize_to=None):
    data_uris = []
    for image_tensor in images:
        pil_image = tensor2pil(image_tensor)
        if pil_image is None:
            continue
        if resize_to is not None:
            pil_image = pil_image.resize(resize_to, Image.LANCZOS)
        data_uris.append(pil2data_uri(pil_image))
    return data_uris


def _safe_error_detail(detail):
    detail = re.sub(
        r"data:[^,\s\"']+,[A-Za-z0-9+/=]+",
        "data:[redacted]",
        str(detail or ""),
    ).strip()
    if len(detail) > 1000:
        return f"{detail[:1000]}..."
    return detail


def _response_json(response, action):
    if not 200 <= response.status_code < 300:
        detail = _safe_error_detail(response.text)
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"{action} failed with HTTP {response.status_code}{suffix}")

    try:
        return response.json()
    except ValueError as error:
        raise ValueError(f"{action} returned invalid JSON.") from error


def _queue_endpoint(run_url):
    if not run_url.startswith(FAL_RUN_URL_PREFIX):
        raise ValueError(f"Unsupported fal.ai endpoint: {run_url}")
    return f"{FAL_QUEUE_URL_PREFIX}{run_url[len(FAL_RUN_URL_PREFIX):]}"


def _queue_operation_url(value, fallback):
    if not value:
        return fallback
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"{FAL_QUEUE_URL_PREFIX.rstrip('/')}/{value.lstrip('/')}"


async def _cancel_queue_request(cancel_url, headers):
    try:
        await asyncio.to_thread(
            requests.put,
            cancel_url,
            headers=headers,
            timeout=10,
        )
    except requests.RequestException:
        pass


async def _submit_fal_queue_request(run_url, payload, headers, queue_timeout, run_index, run_count):
    queue_url = _queue_endpoint(run_url)
    try:
        submit_response = await asyncio.to_thread(
            requests.post,
            queue_url,
            json=payload,
            headers=headers,
            timeout=FAL_HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise ValueError(f"fal.ai queue submission failed: {error}") from error

    submit_data = _response_json(submit_response, "fal.ai queue submission")
    request_id = submit_data.get("request_id")
    if not request_id:
        raise ValueError("fal.ai queue submission returned no request_id.")

    request_base_url = f"{queue_url}/requests/{request_id}"
    status_url = _queue_operation_url(
        submit_data.get("status_url"),
        f"{request_base_url}/status",
    )
    response_url = _queue_operation_url(
        submit_data.get("response_url"),
        request_base_url,
    )
    cancel_url = _queue_operation_url(
        submit_data.get("cancel_url"),
        f"{request_base_url}/cancel",
    )
    deadline = time.monotonic() + queue_timeout
    run_label = f"{run_index + 1}/{run_count}"
    print(f"NanoSeed: queued async run {run_label} ({request_id}).")

    try:
        while True:
            if time.monotonic() >= deadline:
                await _cancel_queue_request(cancel_url, headers)
                raise TimeoutError(
                    f"fal.ai queue run {run_label} timed out after {queue_timeout} seconds "
                    f"(request_id: {request_id})."
                )

            try:
                status_response = await asyncio.to_thread(
                    requests.get,
                    status_url,
                    headers=headers,
                    timeout=FAL_HTTP_TIMEOUT_SECONDS,
                )
            except requests.RequestException:
                await asyncio.sleep(FAL_QUEUE_POLL_INTERVAL_SECONDS)
                continue

            if status_response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                await asyncio.sleep(FAL_QUEUE_POLL_INTERVAL_SECONDS)
                continue

            status_data = _response_json(status_response, "fal.ai queue status")
            status = status_data.get("status")
            if status == "COMPLETED":
                error_message = _safe_error_detail(status_data.get("error"))
                if error_message:
                    error_type = status_data.get("error_type")
                    error_suffix = f" ({error_type})" if error_type else ""
                    raise ValueError(
                        f"fal.ai queue run {run_label} failed{error_suffix}: {error_message}"
                    )
                response_url = _queue_operation_url(
                    status_data.get("response_url"),
                    response_url,
                )
                break
            if status in {"FAILED", "CANCELLED", "CANCELED"}:
                error_message = _safe_error_detail(status_data.get("error")) or "Unknown queue error"
                raise ValueError(f"fal.ai queue run {run_label} failed: {error_message}")
            if status not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise ValueError(
                    f"fal.ai queue run {run_label} returned unknown status: {status!r}"
                )

            await asyncio.sleep(FAL_QUEUE_POLL_INTERVAL_SECONDS)

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"fal.ai result retrieval timed out for run {run_label} "
                    f"(request_id: {request_id})."
                )

            try:
                result_response = await asyncio.to_thread(
                    requests.get,
                    response_url,
                    headers=headers,
                    timeout=FAL_HTTP_TIMEOUT_SECONDS,
                )
            except requests.RequestException:
                await asyncio.sleep(FAL_QUEUE_POLL_INTERVAL_SECONDS)
                continue

            if result_response.status_code in {202, 408, 425, 429, 500, 502, 503, 504}:
                await asyncio.sleep(FAL_QUEUE_POLL_INTERVAL_SECONDS)
                continue

            result = _response_json(result_response, "fal.ai queue result")
            if result.get("error") and not result.get("images"):
                error_message = _safe_error_detail(result["error"])
                raise ValueError(f"fal.ai queue run {run_label} failed: {error_message}")
            print(f"NanoSeed: completed async run {run_label} ({request_id}).")
            return result
    except asyncio.CancelledError:
        await asyncio.shield(_cancel_queue_request(cancel_url, headers))
        raise


async def _run_fal_queue_requests(run_url, payload, headers, concurrent_runs, queue_timeout):
    if not 1 <= concurrent_runs <= FAL_MAX_CONCURRENT_RUNS:
        raise ValueError(
            f"concurrent_runs must be between 1 and {FAL_MAX_CONCURRENT_RUNS}."
        )
    if queue_timeout < 1:
        raise ValueError("queue_timeout must be at least 1 second.")

    async def run_request(run_index):
        run_payload = dict(payload)
        run_payload["sync_mode"] = False
        run_headers = dict(headers)
        run_headers["X-Fal-Request-Timeout"] = str(queue_timeout)
        if run_index and isinstance(run_payload.get("seed"), int):
            run_payload["seed"] = (run_payload["seed"] + run_index) % (2**32)
        return await _submit_fal_queue_request(
            run_url,
            run_payload,
            run_headers,
            queue_timeout,
            run_index,
            concurrent_runs,
        )

    ordered_results = await asyncio.gather(
        *(run_request(run_index) for run_index in range(concurrent_runs)),
        return_exceptions=True,
    )
    failures = [
        (run_index, result)
        for run_index, result in enumerate(ordered_results)
        if isinstance(result, Exception)
    ]

    if failures:
        details = "; ".join(
            f"run {run_index + 1}: {error}"
            for run_index, error in sorted(failures)
        )
        raise ValueError(
            f"{len(failures)} of {concurrent_runs} fal.ai async runs failed: {details}"
        )

    return ordered_results

# Main node class
class NanoSeedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": (["nano_banana", "nano_banana_pro", "nano_banana_2", "gpt_image_2_edit", "grok_imagine_edit", "seedream_4.5", "seedream_5", "seedream_5_lite", "qwen_edit_plus", "flux_2_edit", "flux_2_pro", "flux_2_flex", "flux_2_klein_9b_edit"],),
                "fal_key": ("STRING", {"default": "", "multiline": False}),
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
                "concurrent_runs": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "queue_timeout": ("INT", {"default": 900, "min": 60, "max": 3600, "step": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/edit"
    OUTPUT_NODE = True

    async def edit_image(self, prompt, model, fal_key, image1=None, image2=None, image3=None, image4=None, image5=None,
                   image6=None, image7=None, image8=None, image9=None, image10=None, mask=None,
                   width=0, height=0, num_images=1, num_inference_steps=28, seed=0, aspect_ratio="auto", resolution="1K",
                   quality="high", enable_web_search=False, thinking_level="off", acceleration="none",
                   concurrent_runs=1, queue_timeout=900):  # Hardcoded to none, kept for compatibility
        env_fal_key = (os.environ.get("FAL_KEY") or "").strip()
        ui_fal_key = (fal_key or "").strip()
        if ui_fal_key == "your_fal_key_here":
            ui_fal_key = ""

        if env_fal_key:
            fal_key = env_fal_key
            if ui_fal_key and ui_fal_key != env_fal_key:
                print("NanoSeed Info: Using FAL_KEY from environment (UI key ignored).")
        elif ui_fal_key:
            fal_key = ui_fal_key
        else:
            raise ValueError(
                "Missing API key. Set FAL_KEY in your environment or provide a key in the node UI."
            )
        
        # Collect all non-None images
        image_inputs = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]
        images = [img for img in image_inputs if img is not None]
        if not images:
            raise ValueError("At least one image input must be connected.")
        
        custom_size = (width > 0 and height > 0)
        resize_to = None
        if custom_size and model not in ["nano_banana", "nano_banana_pro", "nano_banana_2"]:
            resize_to = (width, height)
        img_data_uris = await asyncio.to_thread(
            _prepare_image_data_uris,
            images,
            resize_to,
        )
        
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
                "sync_mode": False,
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
                "sync_mode": False,
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
                "sync_mode": False,
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
                "sync_mode": False,
            }
            if custom_size:
                payload["image_size"] = {"width": width, "height": height}
            if mask is not None:
                mask_data_uri = await asyncio.to_thread(tensor2data_uri, mask)
                if mask_data_uri is not None:
                    payload["mask_url"] = mask_data_uri
        elif model == "grok_imagine_edit":
            url = "https://fal.run/xai/grok-imagine-image/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris[:3],
                "num_images": min(num_images, 4),
                "output_format": "png",
                "sync_mode": False,
            }
        elif model == "seedream_4.5":
            url = "https://fal.run/fal-ai/bytedance/seedream/v4.5/edit"
            payload = {
                "prompt": prompt,
                "image_urls": img_data_uris,
                "num_images": min(num_images, 6),
                "seed": seed,
                "enable_safety_checker": False,
                "sync_mode": False,
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
                "sync_mode": False,
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
                "sync_mode": False,
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
                "sync_mode": False,
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

        headers = {
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        }
        api_results = await _run_fal_queue_requests(
            url,
            payload,
            headers,
            concurrent_runs,
            queue_timeout,
        )

        all_edited_tensors = []

        for run_index, api_result in enumerate(api_results):
            output_images = api_result.get("images") or []
            if not output_images and api_result.get("image"):
                output_images = [api_result["image"]]
            if not output_images:
                raise ValueError(
                    f"No images returned from fal.ai async run {run_index + 1}."
                )

            for img_info in output_images[:num_images]:
                img_data = img_info.get("data_uri") or img_info.get("url")
                if not img_data:
                    continue

                if img_data.startswith("data:"):
                    _, encoded = img_data.split(",", 1)
                    pil_edited = Image.open(BytesIO(base64.b64decode(encoded)))
                else:
                    try:
                        img_resp = await asyncio.to_thread(
                            requests.get,
                            img_data,
                            timeout=FAL_HTTP_TIMEOUT_SECONDS,
                        )
                    except requests.RequestException as error:
                        raise ValueError(
                            f"Failed to download generated image: {error}"
                        ) from error
                    if img_resp.status_code != 200:
                        raise ValueError(
                            f"Failed to download generated image (HTTP {img_resp.status_code})."
                        )
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
