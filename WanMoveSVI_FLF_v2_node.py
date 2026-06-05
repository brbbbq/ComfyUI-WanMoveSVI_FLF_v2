import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import node_helpers
from comfy_api.latest import io

# --- WAN-MOVE HELPER FUNCTIONS ---

def create_pos_embeddings(
    pred_tracks: torch.Tensor, 
    pred_visibility: torch.Tensor, 
    downsample_ratios: list[int], 
    height: int, 
    width: int, 
    track_num: int = -1, 
    t_down_strategy: str = "sample"
):
    assert t_down_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."

    t, n, _ = pred_tracks.shape
    t_down, h_down, w_down = downsample_ratios
    track_pos = - torch.ones(n, (t-1) // t_down + 1, 2, dtype=torch.long)

    if track_num == -1:
        track_num = n

    tracks_idx = torch.randperm(n)[:track_num]
    tracks = pred_tracks[:, tracks_idx]
    visibility = pred_visibility[:, tracks_idx]

    for t_idx in range(0, t, t_down):
        if t_down_strategy == "sample" or t_idx == 0:
            cur_tracks = tracks[t_idx] 
            cur_visibility = visibility[t_idx] 
        else:
            cur_tracks = tracks[t_idx:t_idx+t_down].mean(dim=0)
            cur_visibility = torch.any(visibility[t_idx:t_idx+t_down], dim=0)

        for i in range(track_num):
            if not cur_visibility[i] or cur_tracks[i][0] < 0 or cur_tracks[i][1] < 0 or cur_tracks[i][0] >= width or cur_tracks[i][1] >= height:
                continue
            x, y = cur_tracks[i]
            x, y = int(x // w_down), int(y // h_down)
            track_pos[i, t_idx // t_down, 0], track_pos[i, t_idx // t_down, 1] = y, x

    return track_pos 

def replace_feature(
    vae_feature: torch.Tensor,  
    track_pos: torch.Tensor,    
    strength: float = 1.0
) -> torch.Tensor:
    b, _, t, h, w = vae_feature.shape
    assert b == track_pos.shape[0], "Batch size mismatch."
    n = track_pos.shape[1]

    track_pos = track_pos[:, torch.randperm(n), :, :]

    current_pos = track_pos[:, :, 1:, :] 
    mask = (current_pos[..., 0] >= 0) & (current_pos[..., 1] >= 0) 

    valid_indices = mask.nonzero(as_tuple=False) 
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return vae_feature

    batch_idx = valid_indices[:, 0]
    track_idx = valid_indices[:, 1]
    t_rel = valid_indices[:, 2]
    t_target = t_rel + 1  

    h_target = current_pos[batch_idx, track_idx, t_rel, 0].long()  
    w_target = current_pos[batch_idx, track_idx, t_rel, 1].long()
    h_source = track_pos[batch_idx, track_idx, 0, 0].long()
    w_source = track_pos[batch_idx, track_idx, 0, 1].long()

    src_features = vae_feature[batch_idx, :, 0, h_source, w_source]
    dst_features = vae_feature[batch_idx, :, t_target, h_target, w_target]

    vae_feature[batch_idx, :, t_target, h_target, w_target] = dst_features + (src_features - dst_features) * strength

    return vae_feature

# --- MAIN NODE ---

class WanMoveSVI_FLF_v2(io.ComfyNode):
    """
    Integrates Wan-Move trajectory tracking with Wan SVI and First-Last-Frame (FLF) control. Version 2.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveSVI_FLF_v2",
            category="WanMoveSVI_FLF_v2",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.ClipVision.Input("clip_vision", optional=True),
                io.Image.Input("first_image", optional=True, tooltip="The first image to anchor the generation."),
                io.Image.Input("last_image", optional=True, tooltip="Optional target last image(s) to hard-lock the ending of the generation."),
                io.Tracks.Input("tracks", optional=True),
                io.Latent.Input("prev_samples", optional=True, tooltip="Previous frames for motion continuity."),
                io.Float.Input("move_strength", default=2.0, min=0.0, max=100.0, step=0.01),
                io.Int.Input("svi_latent_count", default=1, min=0, max=128, step=1, tooltip="How many previous latent frames SVI injects."),
                io.Int.Input("svi_blend_length", default=1, min=0, max=16, step=1, tooltip="Latent frames taken to crossfade from SVI momentum to Wan-Move tracking."),
                io.Int.Input("width", default=512, min=16, max=8192, step=16),
                io.Int.Input("height", default=512, min=16, max=8192, step=16),
                io.Int.Input("length", default=41, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Boolean.Input("apply_padding_noise", default=False, tooltip="Inject noise into gray padding to prevent VAE artifacts. Try disabling if you notice last-frame color shifts."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, first_image, svi_latent_count, 
                svi_blend_length, move_strength, width, height, length, batch_size,
                apply_padding_noise, clip_vision=None, last_image=None, prev_samples=None, tracks=None) -> io.NodeOutput:
        
        device = comfy.model_management.intermediate_device()
        
        if first_image is None:
            raise ValueError("WanMove SVI Integration requires 'first_image'.")

        clip_vision_output = None
        if clip_vision is not None:
            clip_vision_output = clip_vision.encode_image(first_image)

        # 1. Resolve Anchor Latent (using VAE Encode logic)
        first_image_resized = comfy.utils.common_upscale(first_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        anchor_latent = vae.encode(first_image_resized[:, :, :, :3])

        if anchor_latent.shape[0] == 1 and batch_size > 1:
            anchor_latent = anchor_latent.repeat(batch_size, 1, 1, 1, 1)

        B, C, _, H_latent, W_latent = anchor_latent.shape
        total_latents = ((length - 1) // 4) + 1
        empty_latent = torch.zeros([batch_size, 16, total_latents, H_latent, W_latent], device=device)

        # 2. Resolve Motion Latent (SVI)
        if prev_samples is not None and svi_latent_count > 0:
            svi_latent = prev_samples["samples"][:, :, -svi_latent_count:].clone()
            if svi_latent.shape[0] == 1 and batch_size > 1:
                svi_latent = svi_latent.repeat(batch_size, 1, 1, 1, 1)
            svi_injected_count = svi_latent_count
        else:
            svi_latent = None
            svi_injected_count = 0

        # 3. Generate Padding
        frames_to_pad_svi = total_latents - 1 - svi_injected_count
        frames_to_pad_wan = total_latents - 1

        # VAE Encode a 50% Gray video (Native Wan-Move padding)
        gray_video = torch.ones((length, height, width, 3), device=device, dtype=anchor_latent.dtype) * 0.5
        
        if apply_padding_noise:
            # Add a tiny amount of noise to prevent VAE GroupNorm zero-variance explosions
            gray_video += torch.randn_like(gray_video) * 0.005
            gray_video = torch.clamp(gray_video, 0.0, 1.0)
        
        gray_latent = vae.encode(gray_video)
        
        if gray_latent.shape[0] == 1 and batch_size > 1:
            gray_latent = gray_latent.repeat(batch_size, 1, 1, 1, 1)
        
        svi_padding = gray_latent[:, :, -frames_to_pad_svi:] if frames_to_pad_svi > 0 else None
        wan_padding = gray_latent[:, :, -frames_to_pad_wan:] if frames_to_pad_wan > 0 else None

        # 4. Construct the Environments
        if svi_padding is not None and svi_latent is not None:
            svi_base = torch.cat([anchor_latent, svi_latent, svi_padding], dim=2)
        elif svi_latent is not None:
            svi_base = torch.cat([anchor_latent, svi_latent], dim=2)[:, :, :total_latents]
        else:
            svi_base = torch.cat([anchor_latent, wan_padding], dim=2)

        wan_base = torch.cat([anchor_latent, wan_padding], dim=2)

        # 5. Apply Wan-Move Tracking to the wan_base
        if tracks is not None and move_strength > 0.0:
            tracks_path = tracks["track_path"][:length]
            num_tracks = tracks_path.shape[-2]
            track_visibility = tracks.get("track_visibility", torch.ones((length, num_tracks), dtype=torch.bool, device=device))

            track_pos = create_pos_embeddings(tracks_path, track_visibility, [4, 8, 8], height, width, track_num=num_tracks)
            track_pos = comfy.utils.resize_to_batch_size(track_pos.unsqueeze(0), batch_size)

            wan_tracked = replace_feature(wan_base.clone(), track_pos, move_strength)
        else:
            wan_tracked = wan_base.clone()

        # 6. Apply Time-based Alpha Blend
        blend_weights = torch.zeros(total_latents, device=device, dtype=anchor_latent.dtype)
        blend_weights[0] = 1.0 # Anchor is safe
        
        for t in range(1, total_latents):
            if t <= svi_injected_count:
                blend_weights[t] = 0.0
            else:
                ramp_t = t - svi_injected_count
                alpha = ramp_t / svi_blend_length if svi_blend_length > 0 else 1.0
                blend_weights[t] = min(1.0, alpha)

        blend_weights = blend_weights.view(1, 1, total_latents, 1, 1)

        # Crossfade
        concat_latent_image_pos = svi_base * (1.0 - blend_weights) + wan_tracked * blend_weights
        concat_latent_image_neg = svi_base * (1.0 - blend_weights) + wan_base * blend_weights

        # 7. Apply FLF-style Overwrite to Target End Frames
        last_t_fix = 0
        if last_image is not None:
            last_image_resized = comfy.utils.common_upscale(last_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            last_latent = vae.encode(last_image_resized[:, :, :, :3])

            # Broadcast batch dimension if needed
            if last_latent.shape[0] == 1 and batch_size > 1:
                last_latent = last_latent.repeat(batch_size, 1, 1, 1, 1)

            # Ensure compatible channel count and spatial dimensions
            if (last_latent.shape[1] == C and 
                last_latent.shape[3] == H_latent and 
                last_latent.shape[4] == W_latent):
                
                T_last = last_latent.shape[2]
                last_t_fix = min(T_last, total_latents)

                if last_t_fix > 0:
                    # Overwrite trailing temporal slots of positive and negative structures
                    concat_latent_image_pos[:, :, -last_t_fix:] = last_latent[:, :, -last_t_fix:]
                    concat_latent_image_neg[:, :, -last_t_fix:] = last_latent[:, :, -last_t_fix:]
            else:
                # Fallback path if spatial parameters do not match
                last_t_fix = 0

        # 8. Apply Conditioning Mask
        mask = torch.ones((1, 1, total_latents, H_latent, W_latent), device=anchor_latent.device, dtype=anchor_latent.dtype)
        mask[:, :, :1] = 0.0 # Lock anchor frame (t=0)

        if last_t_fix > 0:
            mask[:, :, -last_t_fix:] = 0.0 # Lock target end frames

        # Set values to Conditioning API
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_pos, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image_neg, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": empty_latent}

        return io.NodeOutput(positive, negative, out_latent)