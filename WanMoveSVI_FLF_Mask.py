from __future__ import annotations
from typing_extensions import override
import torch
import math
from comfy_api.latest import io

class WanMoveSVI_FLF_Mask(io.ComfyNode):
    """
    Generates a temporal mask sequence and image batch for a custom Wan-Move SVI FLF process.
    Accounts for various inputs, temporal padding, cross-fades, and sequence matching.
    """
    
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMoveSVI_FLF_Mask",
            category="WanMoveSVI_FLF_v2",
            inputs=[
                io.Image.Input("last_image", optional=True, tooltip="Optional batch of images representing the sequence end."),
                io.String.Input("select_last_range", default="", tooltip="0-based index range to slice last_image, e.g. '5:9' or '-3:-1'"),
                io.Boolean.Input("lock_to_range", default=False, tooltip="If true, pad to the lowest possible VAE latent multiple."),
                io.Int.Input("svi_latent_count", default=1, min=0, max=128, step=1),
                io.Int.Input("last_latent_count", default=1, min=0, max=128, step=1),
                io.String.Input("svi_mask_list", default="", tooltip="Comma-separated floats, e.g. '0.00, 0.10, 0.30'"),
                io.String.Input("last_mask_list", default="", tooltip="Comma-separated floats, e.g. '0.30, 0.10, 0.00'"),
                io.Int.Input("width", default=512, min=16, max=8192, step=16),
                io.Int.Input("height", default=768, min=16, max=8192, step=16),
                io.Int.Input("length", default=41, min=1, max=8192, step=4, tooltip="Total length of the video generation."),
            ],
            outputs=[
                io.Image.Output("last_image_out"),
                io.Mask.Output("mask"),
                io.Int.Output("svi_latent_count"),
                io.String.Output("info"),
            ]
        )

    @classmethod
    @override
    def execute(
		cls, 
		last_image: torch.Tensor = None, 
		select_last_range: str = "", 
		lock_to_range: bool = False, 
		svi_latent_count: int = 1, 
		last_latent_count: int = 1, 
		svi_mask_list: str = "", 
		last_mask_list: str = "", 
		width: int = 512, height: int = 768, length: int = 41
	) -> io.NodeOutput:
        
        # Helper to parse strings to floats safely
        def parse_float_list(s: str) -> list[float]:
            if not s or not s.strip():
                return []
            try:
                return [float(x.strip()) for x in s.split(",")]
            except ValueError:
                return []

        # ==========================================================
        # 1. ASSEMBLE LAST IMAGE BATCH SEQUENCE
        # ==========================================================
        last_image_out = None
        last_image_count = 0
        repeat_count = 0

        if last_image is not None:
            select_last_image = last_image
            
            # Selection Schedule (0-Based Start:End)
            if select_last_range and ":" in select_last_range:
                parts = select_last_range.split(":")
                total_frames = len(last_image)
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
                            select_last_image = last_image[real_start : real_end + 1]
                        else:
                            select_last_image = last_image[real_start : real_start + 1]
                    except ValueError:
                        pass # Ignore parsing failure and default to all
                        
            select_image_count = len(select_last_image)

            # Sequence Padding / Trimming
            if not lock_to_range:
                req_images = ((last_latent_count - 1) * 4 + 1) if last_latent_count > 0 else 0
                
                if req_images == 0:
                    last_image_out = None
                elif select_image_count > req_images:
                    last_image_out = select_last_image[:req_images] # Sliced from the beginning
                elif select_image_count < req_images:
                    repeat_count = req_images - select_image_count
                    last_frame_copy = select_last_image[-1:].repeat(repeat_count, 1, 1, 1)
                    last_image_out = torch.cat([select_last_image, last_frame_copy], dim=0)
                else:
                    last_image_out = select_last_image
            else:
                # lock_to_range = True: Calculate nearest upper multiple of 4 plus 1
                if select_image_count > 0:
                    req_images = math.ceil((select_image_count - 1) / 4) * 4 + 1
                    req_images = max(1, req_images)
                    
                    if select_image_count < req_images:
                        repeat_count = req_images - select_image_count
                        last_frame_copy = select_last_image[-1:].repeat(repeat_count, 1, 1, 1)
                        last_image_out = torch.cat([select_last_image, last_frame_copy], dim=0)
                    else:
                        last_image_out = select_last_image
                else:
                    last_image_out = None
            
            if last_image_out is not None:
                last_image_count = len(last_image_out)


        # ==========================================================
        # 2. ASSEMBLE MASK BATCH SEQUENCE
        # ==========================================================
        
        # Determine total_masks using the new formula
        if length <= 1:
            total_masks = 1
        else:
            total_masks = (length - 2) // 4 + 2
        
        # --- SVI Masks ---
        svi_mask_count = svi_latent_count
        svi_parsed = parse_float_list(svi_mask_list)
        
        if not svi_parsed:
            svi_mask_values = [0.0] * svi_mask_count
        elif len(svi_parsed) > svi_mask_count:
            svi_mask_values = svi_parsed[:svi_mask_count] # Ignore remainders at the end
        else:
            diff = svi_mask_count - len(svi_parsed)
            svi_mask_values = ([0.0] * diff) + svi_parsed # Prepend 0.0s
            
        # --- Last Masks ---
        if last_image_count > 0:
            last_mask_count = (last_image_count - 1) // 4 + 1
        else:
            last_mask_count = 0
            
        if last_mask_count > 0:
            last_parsed = parse_float_list(last_mask_list)
            if not last_parsed:
                last_mask_values = [0.0] * last_mask_count
            elif len(last_parsed) > last_mask_count:
                last_mask_values = last_parsed[-last_mask_count:] # Pin to end, ignore beginnings
            else:
                diff = last_mask_count - len(last_parsed)
                last_mask_values = last_parsed + ([0.0] * diff) # Append 0.0s
        else:
            last_mask_values = []
            
        # --- Middle Masks ---
        mid_mask_count = total_masks - svi_mask_count - last_mask_count
        
        if mid_mask_count < 0:
            raise ValueError(
                f"SVI latent count ({svi_mask_count}) + Last latent count ({last_mask_count}) "
                f"is greater than the total number of latents ({total_masks}). "
                "Either increase the length of the project or decrease the SVI and Last latent counts."
            )
            
        mid_mask_values = [1.0] * mid_mask_count
        
        # --- Concat All Masks ---
        final_mask_values = svi_mask_values + mid_mask_values + last_mask_values
        
        # Create full resolution sequence tensor [Frames, Height, Width]
        mask_tensor = torch.zeros((total_masks, height, width), dtype=torch.float32)
        for i, val in enumerate(final_mask_values):
            mask_tensor[i] = float(val)


        # ==========================================================
        # 3. INFO OUTPUT
        # ==========================================================
        info_str = (
            f"Last Image Out Count: {last_image_count}\n"
            f"Repeat Count:         {repeat_count}\n"
            f"Total Mask Count:     {total_masks}\n"
            f"SVI Mask Count:       {svi_mask_count}\n"
            f"SVI Mask Values:      {svi_mask_values}\n"
            f"Last Mask Count:      {last_mask_count}\n"
            f"Last Mask Values:     {last_mask_values}"
        )

        return io.NodeOutput(last_image_out, mask_tensor, svi_latent_count, info_str)