from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import io
import torch

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
                io.Image.Output("IMAGE"),
                io.Latent.Output("prev_samples"),
                io.Image.Output("first_image"),
                io.Int.Output("svi_latent_count"),
                io.Int.Output("stitch_overlap"),
				io.String.Output("debug"),
				
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
        num_input_frames = image.shape[0]
        
        # 1) Crop Sequence
        if crop_amount > 0:
            end_idx = max(1, num_input_frames - crop_amount)
            cropped_images = image[:end_idx]
        else:
            cropped_images = image
            
        num_cropped_frames = cropped_images.shape[0]
        
        # 2) SVI Samples
        # Because of Wan VAE's temporal causal downscaling, encoding T frames 
        # results in (T-1)//4 + 1 latent frames. To get a latent representation 
        # that decodes to 4*svi_latent_count + 1 frames, we must encode 
        # 4*svi_latent_count + 1 frames.
        num_svi_frames = 4 * svi_latent_count + 1
        svi_start_idx = max(0, num_cropped_frames - num_svi_frames)
        svi_images = cropped_images[svi_start_idx:]
        
        # Encode with VAE
        # Extract RGB channels
        svi_images_rgb = svi_images[:, :, :, :3]
        latent_tensor = vae.encode(svi_images_rgb)
        prev_samples = {"samples": latent_tensor}
        
        # 3) Select First Image
        first_image_idx = num_cropped_frames - 4 * svi_latent_count - 1
        first_image_index = max(0, min(first_image_idx, num_cropped_frames - 1))
        first_image = cropped_images[first_image_index : first_image_index + 1]
        
        # Get latent count (temporal dimension for 5D video latents, batch dimension for 4D latents)
        if latent_tensor.ndim == 5:
            latent_count = latent_tensor.shape[2]
        else:
            latent_count = latent_tensor.shape[0]
        
		# Calculate stitch_overlap
        stitch_overlap = svi_latent_count * 4 + 1
		
        # Debug multiline string with matching spacing
        debug_str = (
            f"Image input batch size:  {num_input_frames}\n"
            f"Image output batch size: {num_cropped_frames}\n"
			f"SVI images count:        {svi_images.shape[0]}\n"
            f"Latent count:            {latent_count}\n"
            f"First image index:       {first_image_index}"
        )
        
        return io.NodeOutput(
            cropped_images,
            prev_samples,
            first_image,
            svi_latent_count,
			stitch_overlap,
            debug_str
        )