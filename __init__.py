from .WanMoveSVI_FLF_v2_node import WanMoveSVI_FLF_v2
from .WanMoveSVI_FLF_Decode import WanMoveSVI_FLF_Decode
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# Comfy API Extension Registration
class WanMoveSVI_FLFExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # Returns both nodes to be populated in the ComfyUI Node menu
        return [
            WanMoveSVI_FLF_v2,
            WanMoveSVI_FLF_Decode
        ]

async def comfy_entrypoint() -> WanMoveSVI_FLFExtension:
    return WanMoveSVI_FLFExtension()

# Only expose the entrypoint
__all__ = ["comfy_entrypoint"]