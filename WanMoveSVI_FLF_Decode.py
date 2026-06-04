from comfy_api.latest import io

class WanMoveSVI_FLF_Decode(io.ComfyNode):
    """
    Decodes Wan-Move SVI FLF generation, crops the end of the sequence to mitigate still frames,
    and outputs the cropped sequence, extracted first image, and pass-through values for next steps.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveSVI_FLF_Decode",
            category="conditioning/video_models",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Int.Input("svi_latent_count", default=1, min=0, max=128, step=1, tooltip="Pass-through of svi_latent_count."),
                io.Int.Input("latent_crop_count", default=1, min=0, max=250, step=1, tooltip="Number of latent frames to crop from the end of the sequence. Each latent frame represents 4 pixel frames."),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Image.Output("image"),
                io.Image.Output("first_image"),
                io.Int.Output("svi_latent_count"),
                io.Int.Output("overlap"),
            ],
        )

    @classmethod
    def execute(cls, latent, vae, svi_latent_count, latent_crop_count) -> io.NodeOutput:
        # 1. Decode the latents (using comfy-core VAE Decode logic)
        images = vae.decode(latent["samples"])
        
        # Ensure images are in standard ComfyUI format (F, H, W, C)
        # Wan VAE natively handles 3D latents and can sometimes return a 5D pixel tensor
        if images.dim() == 5:
            B, T, H, W, C = images.shape
            images = images.reshape(B * T, H, W, C)
        
        # 2. Crop the image sequence
        # Wan uses 4x temporal compression, meaning 1 latent frame = 4 image frames.
        pixel_crop_amount = latent_crop_count * 4
        N = images.shape[0]
        new_N = max(1, N - pixel_crop_amount)
        cropped_images = images[:new_N]
        
        # 3. Extract the first_image
        # If the new sequence has 33 frames (new_N=33), the last index is 32.
        target_index = max(0, new_N - (svi_latent_count * 4) - 1)
        first_image = images[target_index : target_index + 1]
        
        # 4. Crop the latent sequence
        samples = latent["samples"]
        cropped_latent = latent.copy()
        
        if samples.dim() == 5:
            # Wan Video Latent Shape: [B, C, T, H, W]
            T_latent = samples.shape[2]
            new_T = max(1, T_latent - latent_crop_count)
            cropped_latent["samples"] = samples[:, :, :new_T, :, :]
        elif samples.dim() == 4:
            # Fallback for standard 4D latents: [B (Time), C, H, W]
            T_latent = samples.shape[0]
            new_T = max(1, T_latent - latent_crop_count)
            cropped_latent["samples"] = samples[:new_T, :, :, :]
            
        # 5. Calculate overlap
        overlap = svi_latent_count * 4
        
        return io.NodeOutput(cropped_latent, cropped_images, first_image, svi_latent_count, overlap)