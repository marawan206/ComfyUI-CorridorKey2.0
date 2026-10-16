"""ComfyUI-CorridorKey2.0: CorridorKey 2.0 green/blue-screen keying nodes for ComfyUI.

CorridorKey 2.0 is by Corridor Digital (model designed and trained by Zachary
Harris with Niko Pueringer). This pack is an unofficial integration and is
distributed under the same Corridor Key Licence; see LICENSE.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import all_nodes


class CorridorKey2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return all_nodes()


async def comfy_entrypoint() -> CorridorKey2Extension:
    return CorridorKey2Extension()
