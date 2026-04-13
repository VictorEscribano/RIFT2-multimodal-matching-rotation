"""RIFT2: Radiation-variation Insensitive Feature Transform (v2).

A Python implementation of the multimodal image matching descriptor described in

    J. Li, W. Xu, Q. Hu, Y. Zhang, "RIFT2: Speeding-up RIFT with A New
    Rotation-Invariance Technique," arXiv:2303.00319.

and its predecessor

    J. Li, Q. Hu, M. Ai, "RIFT: Multi-modal image matching based on
    radiation-variation insensitive feature transform," IEEE TIP, 2020.

The public entry point is the :class:`RIFT2` class, which exposes a pipeline
compatible with typical OpenCV feature-matching workflows.
"""

from .rift2 import RIFT2, RIFT2Result
from .phase_congruency import phase_congruency
from .detector import detect_fast_keypoints
from .orientation import compute_dominant_orientations
from .descriptor import build_max_index_map, compute_descriptors
from .matcher import match_descriptors
from .fsc import estimate_similarity_ransac

__all__ = [
    "RIFT2",
    "RIFT2Result",
    "phase_congruency",
    "detect_fast_keypoints",
    "compute_dominant_orientations",
    "build_max_index_map",
    "compute_descriptors",
    "match_descriptors",
    "estimate_similarity_ransac",
]

__version__ = "0.1.0"
