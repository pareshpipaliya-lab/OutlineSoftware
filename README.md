# OutlineSoftware

AI-powered desktop software that converts uploaded images into engraving-ready black-outline PNGs using **Stable Diffusion Image-to-Image + ControlNet Lineart**.

## Features
- Upload JPG/PNG/JPEG input image.
- Generate engraving-style outline via AI redraw (not Canny edge detect).
- Pure black/white post-processing for laser engraving compatibility.
- Preview generated result in-app.
- Download output PNG.
- Optional line thickness slider.

## Project Structure
```
OutlineSoftware/
├── main.py
├── ui.py
├── processor.py
├── input/
└── output/
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## AI Pipeline
1. Load uploaded image.
2. Resize/prepare image for Stable Diffusion img2img.
3. Build ControlNet guidance image.
4. Run `StableDiffusionControlNetImg2ImgPipeline` with lineart ControlNet.
5. Post-process output into strict black/white binary image.
6. Save PNG in `output/` using same base filename.

## Notes
- First run downloads model weights:
  - `runwayml/stable-diffusion-v1-5`
  - `lllyasviel/control_v11p_sd15_lineart`
- GPU is strongly recommended for speed.
