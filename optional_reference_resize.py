"""Resize optional references without manufacturing placeholder images."""
import math
import comfy.utils
from comfy_api.latest import io


class OptionalReferenceResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OptionalReferenceResize", display_name="Vision Prompt — Optional Reference Resize",
            category="image/transform", inputs=[
                io.Image.Input("image", optional=True),
                io.Int.Input("width", default=1440, min=0, max=16384),
                io.Int.Input("height", default=1440, min=0, max=16384),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
                io.Combo.Input("keep_proportion", options=["total_pixels", "resize", "stretch", "crop"], default="total_pixels"),
                io.Int.Input("divisible_by", default=32, min=1, max=512),
            ], outputs=[io.Image.Output("IMAGE"), io.Int.Output("width"), io.Int.Output("height")],
            description="Missing image produces None, not a black placeholder. Use with consumers that support absent references (Qwen Image 2.1, optional H3 references).")

    @classmethod
    def execute(cls, width=1440, height=1440, upscale_method="lanczos", keep_proportion="total_pixels",
                divisible_by=32, image=None):
        if image is None:
            return io.NodeOutput(None, 0, 0)
        h, w = image.shape[1:3]
        width, height = int(width), int(height)
        if not width and not height:
            width, height = w, h
        elif not width:
            width = round(w * height / h)
        elif not height:
            height = round(h * width / w)
        if keep_proportion == "total_pixels":
            scale = math.sqrt(width * height / (w * h))
            width, height = round(w * scale), round(h * scale)
        elif keep_proportion == "resize":
            scale = min(width / w, height / h)
            width, height = round(w * scale), round(h * scale)
        multiple = max(1, int(divisible_by))
        width, height = max(multiple, width // multiple * multiple), max(multiple, height // multiple * multiple)
        resized = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, upscale_method,
                                            "center" if keep_proportion == "crop" else "disabled").movedim(1, -1)
        return io.NodeOutput(resized, width, height)
