# ComfyUI-NanoSeed

A custom ComfyUI node for seamless image editing using fal.ai's NanoBanana and Seedream (v4) models. Edit images with AI-powered prompts, supporting multi-image batches, custom resolutions, and easy fal.ai API key integration directly in the node interface.
<img width="958" height="452" alt="image" src="https://github.com/user-attachments/assets/3e773be5-cb7b-4767-bb0e-f0ccc7a6f5d2" />

## Updates
- **15112025 Multi Image Inputs**: added support for direct multiple image input without using extra image batch nodes.
- **19092025 Multi Model Support**: added support for Flux Kontext Pro and Qwen Edit Plus.

## Features

- **Multi Model Support**: Switch between NanoBanana, Seedream, or Flux Kontext Pro.
- **Batch Processing**: Handle multiple input images in one go.
- **Flexible Inputs**: Custom width/height, number of outputs (up to 4), and seed control.
- **User-Friendly**: No hardcoding—enter your fal.ai key right in ComfyUI.

## Installation

1. Clone this repo into `ComfyUI/custom_nodes/`.

   ```bash
   git clone https://github.com/yourusername/ComfyUI-NanoSeed.git
   ```

2. Install dependencies: `pip install -r requirements.txt`.

3. Restart ComfyUI and find the node under **image/edit** as "Edit image NanoBanana + Seededit by ComRender".

Get your fal.ai API key at [fal.ai](https://fal.ai) and drop it into the node for instant edits!

## Usage

### Node Inputs

- **image** (IMAGE): Input image tensor (supports batches).
- **prompt** (STRING): Descriptive prompt for editing the image.
- **model** (COMBO): Select "nano_banana" or "seedream".
- **fal_key** (STRING): Your fal.ai API key.
- **width/height** (INT, optional): Custom output resolution (0 to use input size).
- **num_images** (INT, optional): Number of variations to generate (1-4).
- **seed** (INT, optional): Random seed for reproducibility.

### Outputs

- **edited_image** (IMAGE): Batch of edited image tensors.

### Example Workflow

1. Load an image using `Load Image`.
2. Connect to `NanoSeedEdit`.
3. Set prompt: "Add a cyberpunk neon glow to the cityscape".
4. Select model and enter your API key.
5. Queue prompt—outputs will be ready for preview or further nodes.

## API Details

This node uses the fal.ai endpoints:

- [NanoBanana Edit](https://fal.ai/models/fal-ai/nano-banana/edit/api)
- [Seedream v4 Edit](https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit/api)
- [Flux Kontext Pro](https://fal.ai/models/fal-ai/flux-pro/kontext/api)
- Qwen Edit Plus
Images are encoded as base64 PNGs for upload. Sync mode ensures immediate results.

## Troubleshooting

- **API Errors**: Check your fal.ai key and quota. Ensure internet access.
- **Resolution Issues**: For NanoBanana, custom sizes resize the input; for Seedream, they set output dims.
- **Batch Limits**: ComfyUI handles batches, but API calls are per-image to avoid rate limits.
- **Dependencies**: Only `requests` is required; no Torch extras needed.

If issues persist, open a GitHub issue with logs.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

ComRender

## Built With

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [fal.ai APIs](https://fal.ai)

---

*Star this repo if it helps your workflow! 🚀*
