from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)


@dataclass
class ProcessorConfig:
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_lineart"
    guidance_scale: float = 8.0
    strength: float = 0.7
    num_inference_steps: int = 25
    prompt: str = (
        "high quality engraving outline, single clean black contour lines, "
        "laser engraving ready, no shading, no grayscale, white background"
    )
    negative_prompt: str = (
        "color, grey, grayscale, shadows, texture, blur, double lines, "
        "broken lines, hatching, crosshatch, fill"
    )


class EngravingProcessor:
    """Stable Diffusion + ControlNet engraving outline generator."""

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        self.config = config or ProcessorConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> StableDiffusionControlNetImg2ImgPipeline:
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model_id,
            torch_dtype=self.dtype,
        )

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.config.base_model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if self.device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(self.device)
        return pipe

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    @staticmethod
    def _resize_for_model(image: Image.Image, max_side: int = 768) -> Image.Image:
        width, height = image.size
        scale = min(max_side / max(width, height), 1.0)
        new_w = int((width * scale) // 8 * 8)
        new_h = int((height * scale) // 8 * 8)
        new_w = max(new_w, 512 if width >= height else 384)
        new_h = max(new_h, 512 if height > width else 384)
        new_w = int(new_w // 8 * 8)
        new_h = int(new_h // 8 * 8)
        return image.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _build_control_image(image: Image.Image, line_thickness: int = 1) -> Image.Image:
        np_img = np.array(image)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (3, 3), 0)
        control = cv2.divide(gray, 255 - blur, scale=256)
        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            control = cv2.erode(control, kernel, iterations=1)
        control = cv2.cvtColor(control, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(control)

    @staticmethod
    def _to_binary(image: Image.Image, threshold: int = 180) -> Image.Image:
        np_img = np.array(image.convert("L"))
        _, bw = cv2.threshold(np_img, threshold, 255, cv2.THRESH_BINARY)
        bw = cv2.medianBlur(bw, 3)
        return Image.fromarray(bw).convert("RGB")

    def generate_outline(self, input_path: str | Path, output_dir: str | Path, line_thickness: int = 1) -> Path:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_image = Image.open(input_path)
        source_image = self._to_rgb(source_image)
        model_input = self._resize_for_model(source_image)
        control_image = self._build_control_image(model_input, line_thickness=line_thickness)

        with torch.inference_mode():
            result = self.pipeline(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=model_input,
                control_image=control_image,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.strength,
                num_inference_steps=self.config.num_inference_steps,
            ).images[0]

        binary_image = self._to_binary(result)
        final_image = binary_image.resize(source_image.size, Image.LANCZOS)

        output_path = output_dir / f"{input_path.stem}.png"
        final_image.save(output_path, format="PNG")
        return output_path
