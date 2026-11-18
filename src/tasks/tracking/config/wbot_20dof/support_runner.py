"""Export the residual action convention alongside supported get-up policies."""
from pathlib import Path
import json

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class SupportTrackingRunner(MotionTrackingOnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos)
        directory = Path(path).parent
        metadata = {
            "action_offset_mode": "reference_joint_position",
            "action_target_formula": "clip(q_ref + clip(action * action_scale, -0.75, 0.75), joint_lower, joint_upper)",
            "reference_end_behavior": "hold_final_frame_no_teleport",
        }
        for file in (directory / "policy.onnx", directory / (directory.name + ".onnx")):
            if file.exists():
                attach_metadata_to_onnx(str(file), metadata)
        (directory / "action_semantics.json").write_text(json.dumps(metadata, indent=2) + "\n")
