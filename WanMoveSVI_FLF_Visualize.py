import torch
import torchvision.transforms.functional as TF
import comfy.model_management
import numpy as np
import ast
from PIL import Image, ImageDraw, ImageColor
from comfy_api.latest import io

def parse_colors(color_map_str):
    fallback = [(102, 153, 255), (0, 255, 255), (255, 255, 0), (255, 102, 204), (0, 255, 0)]
    try:
        # First try parsing the input as Python literals
        raw_list = ast.literal_eval(f"[{color_map_str}]")
    except Exception:
        # If it fails (e.g. unquoted hex strings/color names), parse via parenthesis-aware splitting
        raw_list = []
        current = []
        paren_count = 0
        for char in color_map_str:
            if char in ('(', '['):
                paren_count += 1
                current.append(char)
            elif char in (')', ']'):
                paren_count -= 1
                current.append(char)
            elif char == ',' and paren_count == 0:
                raw_list.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            raw_list.append("".join(current).strip())
    
    parsed_colors = []
    for item in raw_list:
        if isinstance(item, (tuple, list)):
            if len(item) >= 3:
                parsed_colors.append((int(item[0]), int(item[1]), int(item[2])))
        elif isinstance(item, str):
            cleaned = item.strip().strip("'\"")
            if not cleaned:
                continue
            # If the string contains a bracketed tuple format
            if (cleaned.startswith('(') and cleaned.endswith(')')) or (cleaned.startswith('[') and cleaned.endswith(']')):
                try:
                    inner_tuple = ast.literal_eval(cleaned)
                    if isinstance(inner_tuple, (tuple, list)) and len(inner_tuple) >= 3:
                        parsed_colors.append((int(inner_tuple[0]), int(inner_tuple[1]), int(inner_tuple[2])))
                        continue
                except Exception:
                    pass
            # Standard string fallback (handles '#ff0000', 'red', 'rgb(255,0,0)', etc.)
            try:
                parsed_colors.append(ImageColor.getrgb(cleaned))
            except Exception:
                pass
    
    if not parsed_colors:
        return fallback
    return parsed_colors

def _draw_gradient_polyline_on_overlay(overlay, line_width, points, start_color, opacity=1.0):
    draw = ImageDraw.Draw(overlay, 'RGBA')
    points = points[::-1]

    # Compute total length
    total_length = 0
    segment_lengths = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        length = (dx * dx + dy * dy) ** 0.5
        segment_lengths.append(length)
        total_length += length

    if total_length == 0:
        return

    accumulated_length = 0

    # Draw the gradient polyline
    for idx, (start_point, end_point) in enumerate(zip(points[:-1], points[1:])):
        segment_length = segment_lengths[idx]
        steps = max(int(segment_length), 1)

        for i in range(steps):
            current_length = accumulated_length + (i / steps) * segment_length
            ratio = current_length / total_length

            alpha = int(255 * (1 - ratio) * opacity)
            color = (*start_color, alpha)

            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)

            dynamic_line_width = max(int(line_width * (1 - ratio)), 1)
            draw.line([(x, y), (x + 1, y)], fill=color, width=dynamic_line_width)

        accumulated_length += segment_length


def draw_tracks_on_video(video, tracks, visibility=None, track_frame=24, circle_size=12, opacity=0.5, line_width=16, color_map=None):
    if color_map is None or len(color_map) == 0:
        color_map = [(102, 153, 255), (0, 255, 255), (255, 255, 0), (255, 102, 204), (0, 255, 0)]

    video = video.byte().cpu().numpy()  # (81, 480, 832, 3)
    tracks = tracks[0].long().detach().cpu().numpy()
    if visibility is not None:
        visibility = visibility[0].detach().cpu().numpy()

    num_frames, height, width = video.shape[:3]
    num_tracks = tracks.shape[1]
    alpha_opacity = int(255 * opacity)

    output_frames = []
    for t in range(num_frames):
        frame_rgb = video[t].astype(np.float32)

        # Create a single RGBA overlay for all tracks in this frame
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        polyline_data = []

        # Draw all circles on a single overlay
        for n in range(num_tracks):
            if visibility is not None and visibility[t, n] == 0:
                continue

            track_coord = tracks[t, n]
            color = color_map[n % len(color_map)]
            circle_color = color + (alpha_opacity,)

            draw_overlay.ellipse(
                (track_coord[0] - circle_size, track_coord[1] - circle_size,
                 track_coord[0] + circle_size, track_coord[1] + circle_size),
                fill=circle_color
            )

            # Store polyline data for batch processing
            tracks_coord = tracks[max(t - track_frame, 0):t + 1, n]
            if len(tracks_coord) > 1:
                polyline_data.append((tracks_coord, color))

        # Blend circles overlay once
        overlay_np = np.array(overlay)
        alpha = overlay_np[:, :, 3:4] / 255.0
        frame_rgb = overlay_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)

        # Draw all polylines on a single overlay
        if polyline_data:
            polyline_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for tracks_coord, color in polyline_data:
                _draw_gradient_polyline_on_overlay(polyline_overlay, line_width, tracks_coord, color, opacity)

            # Blend polylines overlay once
            polyline_np = np.array(polyline_overlay)
            alpha = polyline_np[:, :, 3:4] / 255.0
            frame_rgb = polyline_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)

        output_frames.append(Image.fromarray(frame_rgb.astype(np.uint8)))

    return output_frames


class WanMoveSVI_FLF_Visualize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanMoveSVI_FLF_Visualize",
            category="WanMoveSVI_FLF_v2",
            inputs=[
                io.Image.Input("images"),
                io.Tracks.Input("tracks", optional=True),
                io.Int.Input("line_resolution", default=24, min=1, max=1024),
                io.Int.Input("circle_size", default=12, min=1, max=128, advanced=True),
                io.Float.Input("opacity", default=0.75, min=0.0, max=1.0, step=0.01),
                io.Int.Input("line_width", default=16, min=1, max=128, advanced=True),
                io.String.Input("color_map", default="#69f, #0ff, #ff0, #f6c, #0f0"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, images, line_resolution, circle_size, opacity, line_width, color_map="#69f, #0ff, #ff0, #f6c, #0f0", tracks=None) -> io.NodeOutput:
        if tracks is None:
            return io.NodeOutput(images)

        parsed_colors = parse_colors(color_map)

        track_path = tracks["track_path"].unsqueeze(0)
        track_visibility = tracks["track_visibility"].unsqueeze(0)
        images_in = images * 255.0
        
        if images_in.shape[0] == 1:
            images_in = images_in.repeat(track_path.shape[1], 1, 1, 1)
        else:
            min_len = min(images_in.shape[0], track_path.shape[1])
            images_in = images_in[:min_len]
            track_path = track_path[:, :min_len]
            track_visibility = track_visibility[:, :min_len]

        track_video = draw_tracks_on_video(
            images_in,
            track_path,
            track_visibility,
            track_frame=line_resolution,
            circle_size=circle_size,
            opacity=opacity,
            line_width=line_width,
            color_map=parsed_colors
        )
        track_video = torch.stack([TF.to_tensor(frame) for frame in track_video], dim=0).movedim(1, -1).float()

        return io.NodeOutput(track_video.to(comfy.model_management.intermediate_device()))