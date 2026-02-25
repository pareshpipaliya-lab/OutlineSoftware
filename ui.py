from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from PIL import Image, ImageTk

from processor import EngravingProcessor


class OutlineApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Engraving Outline Generator")
        self.root.geometry("900x700")

        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.preview_image_ref = None

        self.processor: EngravingProcessor | None = None

        self.status_var = tk.StringVar(value="Ready")
        self.thickness_var = tk.IntVar(value=1)

        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill=tk.X)

        self.upload_btn = ttk.Button(controls, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=6)

        self.generate_btn = ttk.Button(controls, text="Generate Outline", command=self.generate_outline)
        self.generate_btn.pack(side=tk.LEFT, padx=6)

        self.download_btn = ttk.Button(controls, text="Download", command=self.download_output)
        self.download_btn.pack(side=tk.LEFT, padx=6)

        ttk.Label(controls, text="Thickness").pack(side=tk.LEFT, padx=(20, 6))
        self.thickness_scale = ttk.Scale(
            controls,
            from_=1,
            to=4,
            variable=self.thickness_var,
            orient=tk.HORIZONTAL,
            length=160,
        )
        self.thickness_scale.pack(side=tk.LEFT)

        preview_frame = ttk.Frame(self.root, padding=12)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_frame, text="Preview will appear here", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

    def _ensure_processor(self) -> EngravingProcessor:
        if self.processor is None:
            self.status_var.set("Loading Stable Diffusion pipeline...")
            self.root.update_idletasks()
            self.processor = EngravingProcessor()
        return self.processor

    def upload_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")],
        )
        if not path:
            return

        src = Path(path)
        dest = self.input_dir / src.name
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())

        self.input_path = dest
        self.output_path = None
        self.status_var.set(f"Loaded: {self.input_path.name}")
        self._show_preview(self.input_path)

    def _show_preview(self, image_path: Path) -> None:
        image = Image.open(image_path)
        image.thumbnail((850, 560), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(image)
        self.preview_image_ref = tk_img
        self.preview_label.configure(image=tk_img, text="")

    def generate_outline(self) -> None:
        if not self.input_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        try:
            self.generate_btn.configure(state=tk.DISABLED)
            self.status_var.set("Generating AI engraving outline...")
            self.root.update_idletasks()

            processor = self._ensure_processor()
            thickness = int(round(self.thickness_var.get()))
            self.output_path = processor.generate_outline(
                input_path=self.input_path,
                output_dir=self.output_dir,
                line_thickness=thickness,
            )

            self._show_preview(self.output_path)
            self.status_var.set(f"Done: {self.output_path.name}")
        except Exception as exc:
            messagebox.showerror("Generation error", str(exc))
            self.status_var.set("Failed")
        finally:
            self.generate_btn.configure(state=tk.NORMAL)

    def download_output(self) -> None:
        if not self.output_path or not self.output_path.exists():
            messagebox.showwarning("No output", "Generate an outline first.")
            return

        target = filedialog.asksaveasfilename(
            title="Save output",
            initialfile=self.output_path.name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not target:
            return

        Path(target).write_bytes(self.output_path.read_bytes())
        self.status_var.set(f"Saved: {Path(target).name}")
