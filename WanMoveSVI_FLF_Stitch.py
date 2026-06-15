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
                io.Image.Input("prev_images", optional=True, tooltip="The first batch of images (chronologically prior)"),
                io.Image.Input("new_images", optional=True, tooltip="The second batch of images (chronologically next)"),
                io.Int.Input(
                    "stitch_overlap", 
                    default=5, 
                    min=0, 
                    max=4096, 
                    step=1, 
                    tooltip="Number of overlapping frames to blend between prev_images and new_images"
                ),
                io.String.Input(
                    "preview_range",
                    default="0:-1",
                    tooltip="Inclusive frame range for the preview output (format 'start:end'). e.g., '0:10' or '-10:-1'"
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Image.Output("preview"),
            ]
        )

    @classmethod
    @override
    def execute(cls, prev_images=None, new_images=None, stitch_overlap=5, preview_range="0:-1") -> io.NodeOutput:
        # Validate that at least one of the inputs exists
        if prev_images is None and new_images is None:
            raise ValueError("At least one image input (prev_images or new_images) must be provided.")

        # Pass through directly if only one image batch is supplied
        if prev_images is not None and new_images is None:
            IMAGE = prev_images
        elif new_images is not None and prev_images is None:
            IMAGE = new_images
        else:
            # Validate that the resolution of both image batches matches when both are present
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
            else:
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

        # Generate the preview slice based on the preview_range string
        preview = IMAGE
        total_frames = len(IMAGE)

        if preview_range and ":" in preview_range and total_frames > 0:
            parts = preview_range.split(":")
            if len(parts) == 2:
                try:
                    start_str, end_str = parts[0].strip(), parts[1].strip()
                    start_idx = int(start_str)
                    end_idx = int(end_str)

                    # Handle negative index offsets
                    real_start = start_idx if start_idx >= 0 else total_frames + start_idx
                    real_end = end_idx if end_idx >= 0 else total_frames + end_idx

                    # Clamp values to valid index bounds
                    real_start = max(0, min(real_start, total_frames - 1))
                    real_end = max(0, min(real_end, total_frames - 1))

                    if real_start <= real_end:
                        # Slice from real_start to real_end inclusive
                        preview = IMAGE[real_start : real_end + 1]
                    else:
                        # Safe fallback to a single frame if bounds are inverted,
                        # avoiding downstream crashes caused by empty tensors
                        preview = IMAGE[real_start : real_start + 1]
                except ValueError:
                    # In case parsing fails (e.g., non-integers), default to the full IMAGE batch
                    pass

        return io.NodeOutput(IMAGE, preview)