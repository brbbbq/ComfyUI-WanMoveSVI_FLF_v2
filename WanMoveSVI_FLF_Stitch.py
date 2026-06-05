from __future__ import annotations
from typing_extensions import override
import torch
from comfy_api.latest import ComfyExtension, io

class WanMoveSVI_FLF_Stitch(io.ComfyNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMoveSVI_FLF_Stitch",
            category="WanMoveSVI_FLF_v2",
            inputs=[
                io.Image.Input("prev_images", tooltip="The first batch of images (chronologically prior)"),
                io.Image.Input("new_images", tooltip="The second batch of images (chronologically next)"),
                io.Int.Input(
                    "stitch_overlap", 
                    default=5, 
                    min=0, 
                    max=4096, 
                    step=1, 
                    tooltip="Number of overlapping frames to blend between prev_images and new_images"
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
            ]
        )

    @classmethod
    @override
    def execute(cls, prev_images, new_images, stitch_overlap) -> io.NodeOutput:
        # Validate that the resolution of both image batches matches
        if prev_images.shape[1:3] != new_images.shape[1:3]:
            raise ValueError(
                f"Previous and new images must have the same spatial dimensions: "
                f"{prev_images.shape[1:3]} vs {new_images.shape[1:3]}"
            )

        # Restrict the overlap to not exceed the length of the smaller batch
        max_overlap = min(len(prev_images), len(new_images))
        overlap = min(stitch_overlap, max_overlap)

        if overlap <= 0:
            # If overlap is 0, perform a standard sequential concatenation
            IMAGE = torch.cat((prev_images, new_images), dim=0)
            return io.NodeOutput(IMAGE)

        # Split batches into non-overlapping sections and overlap sections
        prefix = prev_images[:-overlap] if overlap < len(prev_images) else prev_images[:0]
        suffix = new_images[overlap:] if overlap < len(new_images) else new_images[:0]

        blend_src = prev_images[-overlap:]
        blend_dst = new_images[:overlap]

        # Linear blend calculation over the overlapping frame dimension
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        alpha = alpha.view(-1, 1, 1, 1)  # Reshape for broadcasting over [Frames, Height, Width, Channels]

        blended_images = (1 - alpha) * blend_src + alpha * blend_dst
        
        # Concatenate the preceding frames, the blended frames, and the remaining trailing frames
        IMAGE = torch.cat((prefix, blended_images, suffix), dim=0)

        return io.NodeOutput(IMAGE)