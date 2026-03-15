# ComfyUI NanoSeed Edit

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue) ![Fal.ai](https://img.shields.io/badge/Backend-Fal.ai-purple)

A Custom Node that integrates [Fal.ai's](https://fal.ai) powerful image editing APIs. This node allows you to edit images using the latest Image Editing Models available via Fal.ai API.
<img width="925" height="464" alt="image" src="https://github.com/user-attachments/assets/e8cce9fc-dacb-4027-a1cf-4e29ace68329" />
## ✨ Features

* **Multi-Model Support:** Access multiple editing models from a single node.
* **Multi-Image Input:** Support for up to 5 input images for context-aware editing (model dependent).
* **Flexible Output:** Control aspect ratios, resolution, and batch size.
* **Cloud Inference:** Offloads heavy GPU processing to Fal.ai's cloud infrastructure.

## 🚀 Supported Models

| Model | ID in Node | Description |
| :--- | :--- | :--- |
| **Nano Banana** | `nano_banana` | Fast, efficient editing. |
| **Nano Banana Pro** | `nano_banana_pro` | Enhanced quality version of Nano Banana. |
| **Nano Banana 2** | `nano_banana_2` | New generation model with web search and thinking options. |
| **Grok Imagine Edit** | `grok_imagine_edit` | xAI image edit model, up to 3 input images. |
| **Seedream 4.5** | `seedream_4.5` | ByteDance's model. Optimized for high-res editing. |
| **Qwen Edit Plus** | `qwen_edit_plus` | Powerful instruction-based editing with guidance control. |
| **Flux 2 (Standard, Pro, Flex)** | `flux_2_edit` | (BFL) High-fidelity editing capabilities. |

## 📦 Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd /path/to/ComfyUI/custom_nodes/
    ```
2.  Clone this repository (or unzip the folder containing these files):
    ```bash
    git clone [https://github.com/yourusername/ComfyUI-NanoSeed.git](https://github.com/yourusername/ComfyUI-NanoSeed.git)
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🔑 Configuration

You must have a **Fal.ai API Key** to use this node.

1.  Go to [fal.ai](https://fal.ai/dashboard/keys) and generate an API Key.
2.  Enter this key into the `fal_key` widget on the node.

## 🎛️ Usage

### Inputs
* **prompt**: (Required) A text description of how to edit the image.
* **model**: Select the specific AI model to use.
* **fal_key**: Your API key.
* **image1 - image5**: Connect the images you want to edit.

### Settings
* **width / height**: Output resolution.
    * *Note:* `nano_banana` models ignore this and use `aspect_ratio`.
    * *Note:* `seedream_4.5` requires high resolutions (see constraints below).
* **num_images**: How many variations to generate (Max 6).
* **seed**: Seed for reproducibility.
* **aspect_ratio**: Set target aspect ratio (e.g., 16:9, 1:1).
* **enable_web_search**: Enables web search context for supported models.
* **thinking_level**: `off`, `minimal`, or `high` for supported models.

### ⚠️ Model Specific Constraints

Different models have different requirements implemented in the node:

1.  **Flux 2 Edit (`flux_2_edit`)**:
    * **Single Image Only:** Do not connect `image2` through `image5`.
    * **Resolution:** Width and Height must be between **512px and 2048px**.

2.  **Seedream 4.5 (`seedream_4.5`)**:
    * **High Res Only:** Width and Height must be between **1920px and 4096px**.
    * **Pixel Count:** Total pixels (W x H) must be between ~3.6MP and 16.7MP.
    * **Batch Limit:** The sum of input images and output `num_images` cannot exceed 15.

3.  **Nano Banana / Pro / 2**:
    * These models rely on `aspect_ratio` and `resolution` settings rather than custom width/height integers.
    * `nano_banana_2` supports `0.5K`, `1K`, `2K`, and `4K` resolution presets.
    * `nano_banana_2` also supports extended aspect ratios: `4:1`, `1:4`, `8:1`, and `1:8`.

4.  **Grok Imagine Edit (`grok_imagine_edit`)**:
    * Supports up to **3 input images**.
    * Uses up to **4 output images**.

## 🛠️ Troubleshooting

* **"No images returned from API"**: Check your `fal_key` balance or ensure your prompt is not triggering safety filters.
* **Resolution Errors**: If using Seedream, ensure your resolution is set high enough (min 1920x1920). If using Flux, ensure it is within 512-2048.

*Star this repo if it helps your workflow! 🚀*
