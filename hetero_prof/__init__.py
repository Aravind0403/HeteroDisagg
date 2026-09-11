"""HeteroDisagg Accelerator Capability Descriptor (ACD) & Roofline Profiler."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from hetero_prof.schema import (
    AcceleratorSpec,
    ComputeTflops,
    InterconnectSpec,
    RooflinePoint,
    ChunkSizingRecommendation,
)
from hetero_prof.device_detector import detect_device, load_spec_from_file
from hetero_prof.roofline import calculate_inflection_point, recommend_chunk_size

__all__ = [
    "AcceleratorSpec",
    "ComputeTflops",
    "InterconnectSpec",
    "RooflinePoint",
    "ChunkSizingRecommendation",
    "detect_device",
    "load_spec_from_file",
    "calculate_inflection_point",
    "recommend_chunk_size",
]
