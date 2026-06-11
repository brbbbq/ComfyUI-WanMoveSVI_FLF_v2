from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import io
import torch
import os
import concurrent.futures
import logging

def process_svi_sequence(
    image: torch.Tensor, 
    vae, 
    crop_amount: int, 
    svi_latent_count: int,
    image_ref: torch.Tensor | None = None,
    colormatch_strength: float = 0.0,
    colormatch_blend: float = 0.0
) -> tuple:
    """
    Shared logic for both Encode and Decode nodes:
    1. Crops pixel frames.
    2. Optionally applies ColorMatch and alpha-blends the result.
    3. Extracts SVI sequence and encodes it.
    4. Extracts the first image.
    5. Calculates stitch overlap.
    """
    num_input_frames = image.shape[0]
    
    # 1) Crop Sequence
    if crop_amount > 0:
        end_idx = max(1, num_input_frames - crop_amount)
        cropped_images = image[:end_idx]
    else:
        cropped_images = image
        
    num_cropped_frames = cropped_images.shape[0]
    
    # 2) Color Match & Blending
    # If we have an image_ref and the strength/blend aren't 0, apply the colormatch
    if image_ref is not None and colormatch_strength > 0.0 and colormatch_blend > 0.0:
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise Exception("Can't import color-matcher. Please install it: pip install color-matcher")
            
        N = num_cropped_frames
        blend_frames = round(N * colormatch_blend)
        
        if blend_frames > 0:
            batch_size = N
            ref_batch_size = image_ref.size(0)
            
            def process_cm(i):
                cm = ColorMatcher()
                target_np = cropped_images[i].cpu().numpy()
                ref_np = image_ref[min(i, ref_batch_size - 1)].cpu().numpy()
                try:
                    # Hardcoded to 'mkl' method
                    result = cm.transfer(src=target_np, ref=ref_np, method='mkl')
                    if colormatch_strength != 1.0:
                        result = target_np + colormatch_strength * (result - target_np)
                    return torch.from_numpy(result)
                except Exception as e:
                    logging.error(f"ColorMatch thread {i} error: {e}")
                    return torch.from_numpy(target_np)

            max_threads = min(os.cpu_count() or 1, batch_size)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                cm_out = list(executor.map(process_cm, range(batch_size)))
            
            # Reconstruct colormatched sequence
            cropped_images_cm = torch.stack(cm_out, dim=0).to(
                device=cropped_images.device, 
                dtype=cropped_images.dtype
            ).clamp_(0, 1)
            
            # Apply Alpha Blending
            prefix_frames = N - blend_frames
            
            # Alpha remains 0.0 until the start of the blend sequence
            alpha_prefix = torch.zeros(prefix_frames, device=cropped_images.device, dtype=cropped_images.dtype)
            
            # Linear ramp from 0.0 to 1.0 across the blend frames length
            alpha_ramp = torch.linspace(0, 1, blend_frames, device=cropped_images.device, dtype=cropped_images.dtype)
            
            alpha_full = torch.cat((alpha_prefix, alpha_ramp), dim=0).view(-1, 1, 1, 1)
            
            # Composite cropped_images_cm over cropped_images
            cropped_images = (1 - alpha_full) * cropped_images + alpha_full * cropped_images_cm

    # 3) Calculate stitch_overlap (which is also the required num_svi_frames)
    stitch_overlap = svi_latent_count * 4 + 1
    
    # 4) SVI Samples
    svi_start_idx = max(0, num_cropped_frames - stitch_overlap)
    svi_images = cropped_images[svi_start_idx:]
    
    # Encode with VAE (Extract RGB channels)
    svi_images_rgb = svi_images[:, :, :, :3]
    latent_tensor = vae.encode(svi_images_rgb)
    LATENT = {"samples": latent_tensor}
    
    # 5) Select First Image
    first_image_idx = num_cropped_frames - stitch_overlap
    first_image_index = max(0, min(first_image_idx, num_cropped_frames - 1))
    first_image = cropped_images[first_image_index : first_image_index + 1]
    
    # Get latent count
    if latent_tensor.ndim == 5:
        latent_count = latent_tensor.shape[2]
    else:
        latent_count = latent_tensor.shape[0]
    
    # 6) Info String
    info_str = (
        f"Image input batch size:  {num_input_frames}\n"
        f"Image output batch size: {num_cropped_frames}\n"
        f"Stitch overlap count:    {stitch_overlap}\n"
        f"Latent count:            {latent_count}\n"
        f"First image index:       {first_image_index}"
    )
    
    return LATENT, cropped_images, first_image, svi_latent_count, stitch_overlap, info_str


class WanMoveSVI_FLF_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMoveSVI_FLF_Encode",
            category="WanMoveSVI_FLF_v2",
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Int.Input("crop_amount", default=0, min=0),
                io.Int.Input("svi_latent_count", default=1, min=1),
            ],
            outputs=[
                io.Latent.Output("LATENT"),
                io.Image.Output("IMAGE"),
                io.Image.Output("first_image"),
                io.Int.Output("svi_latent_count"),
                io.Int.Output("stitch_overlap"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        vae,
        crop_amount: int,
        svi_latent_count: int,
    ) -> io.NodeOutput:
        
        # Bypass color matching entirely
        res = process_svi_sequence(image, vae, crop_amount, svi_latent_count)
        return io.NodeOutput(*res)


class WanMoveSVI_FLF_Decode(io.ComfyNode):
    """
    Decodes Wan-Move SVI FLF generation. Automatically applies linear gradient color-matching 
    if an image_ref is provided, and feeds the resulting frames back into the encoder logic.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveSVI_FLF_Decode",
            category="WanMoveSVI_FLF_v2",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Int.Input("crop_amount", default=0, min=0, tooltip="Number of pixel frames to crop from the end of the sequence."),
                io.Int.Input("svi_latent_count", default=1, min=0, max=128, step=1, tooltip="Pass-through of svi_latent_count."),
                io.Image.Input("image_ref", optional=True, tooltip="Optional reference image for color matching MKL."),
                io.Float.Input("colormatch_strength", default=0.66, min=0.0, max=1.0, step=0.01, tooltip="Strength of the MKL Color Match."),
                io.Float.Input("colormatch_blend", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Length of the alpha blend ramp applied backwards from the end of the sequence."),
            ],
            outputs=[
                io.Latent.Output("LATENT"),
                io.Image.Output("IMAGE"),
                io.Image.Output("first_image"),
                io.Int.Output("svi_latent_count"),
                io.Int.Output("stitch_overlap"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(
        cls, 
        latent, 
        vae, 
        crop_amount: int, 
        svi_latent_count: int,
        image_ref=None,
        colormatch_strength: float = 0.66,
        colormatch_blend: float = 1.00
    ) -> io.NodeOutput:
        
        # 1. Decode the latents (using comfy-core VAE Decode logic)
        images = vae.decode(latent["samples"])
        
        # Ensure images are in standard ComfyUI format (F, H, W, C)
        if images.dim() == 5:
            B, T, H, W, C = images.shape
            images = images.reshape(B * T, H, W, C)
        
        # 2. Pass decoded images into shared function along with color-match parameters
        res = process_svi_sequence(
            images, 
            vae, 
            crop_amount, 
            svi_latent_count,
            image_ref=image_ref,
            colormatch_strength=colormatch_strength,
            colormatch_blend=colormatch_blend
        )
        
        return io.NodeOutput(*res)