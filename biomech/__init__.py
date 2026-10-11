"""Computer-vision based biomechanical motion analysis.

Marker tracking -> trajectory cleaning -> joint kinematics -> report and annotated video.
"""
__version__ = "1.0.0"

JOINT_SETS = {
    "leg": ("hip", "knee", "ankle"),
    "arm": ("shoulder", "elbow", "wrist"),
}
