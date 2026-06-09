from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import io
import torch

def process_svi_sequence(image: torch.Tensor, vae, crop_amount: int, svi_latent_count: int) -> tuple:
    """
    Shared logic for both Encode and Decode nodes:
    1. Crops pixel frames.
    2. Extracts SVI sequence and encodes it.
    3. Extracts the first image.
    4. Calculates stitch overlap.
    """
    num_input_frames = image.shape[0]
    
    # 1) Crop Sequence
    if crop_amount > 0:
        end_idx = max(1, num_input_frames - crop_amount)
        cropped_images = image[:end_idx]
    else:
        cropped_images = image
        
    num_cropped_frames = cropped_images.shape[0]
    
    # Calculate stitch_overlap (which is also the required num_svi_frames)
    stitch_overlap = svi_latent_count * 4 + 1
    
    # 2) SVI Samples
    # Because of Wan VAE's temporal causal downscaling, encoding T frames 
    # results in (T-1)//4 + 1 latent frames.
    svi_start_idx = max(0, num_cropped_frames - stitch_overlap)
    svi_images = cropped_images[svi_start_idx:]
    
    # Encode with VAE
    # Extract RGB channels
    svi_images_rgb = svi_images[:, :, :, :3]
    latent_tensor = vae.encode(svi_images_rgb)
    LATENT = {"samples": latent_tensor}
    
    # 3) Select First Image
    first_image_idx = num_cropped_frames - stitch_overlap
    first_image_index = max(0, min(first_image_idx, num_cropped_frames - 1))
    first_image = cropped_images[first_image_index : first_image_index + 1]
    
    # Get latent count (temporal dimension for 5D video latents, batch dimension for 4D latents)
    if latent_tensor.ndim == 5:
        latent_count = latent_tensor.shape[2]
    else:
        latent_count = latent_tensor.shape[0]
    
    # 4) Info multiline string with matching spacing
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
        
        # Pass directly into shared function
        res = process_svi_sequence(image, vae, crop_amount, svi_latent_count)
        return io.NodeOutput(*res)


class WanMoveSVI_FLF_Decode(io.ComfyNode):
    """
    Decodes Wan-Move SVI FLF generation, then uses standard encoder logic 
    to output the cropped sequence, extracted first image, and encoded LATENT.
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
    def execute(cls, latent, vae, crop_amount: int, svi_latent_count: int) -> io.NodeOutput:
        # 1. Decode the latents first (using comfy-core VAE Decode logic)
        images = vae.decode(latent["samples"])
        
        # Ensure images are in standard ComfyUI format (F, H, W, C)
        # Wan VAE natively handles 3D latents and can sometimes return a 5D pixel tensor
        if images.dim() == 5:
            B, T, H, W, C = images.shape
            images = images.reshape(B * T, H, W, C)
        
        # 2. Pass decoded images into shared function (identically recreating the Encode behaviour)
        res = process_svi_sequence(images, vae, crop_amount, svi_latent_count)
        return io.NodeOutput(*res)