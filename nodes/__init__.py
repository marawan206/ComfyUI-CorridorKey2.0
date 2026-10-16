"""Node modules of the pack. Each module exposes a ``NODES`` list."""

import importlib
import importlib.util

_MODULES = (
    "model_nodes",
    "keyer_basic",
    "keyer_advanced",
    "refine_nodes",
    "hint_nodes",
    "depth_nodes",
    "semantic_nodes",
    "motion_nodes",
    "temporal_nodes",
    "lora_nodes",
    "train_nodes",
    "demo_nodes",
)


def all_nodes():
    nodes = []
    for name in _MODULES:
        if importlib.util.find_spec(f"{__name__}.{name}") is None:
            continue
        module = importlib.import_module(f"{__name__}.{name}")
        nodes.extend(module.NODES)
    return nodes
