"""GR00T N1.7 modality config for RAPID: UR16e on a linear rail with a Robotiq 2F-85 and a wrist D405.

Matches the `meta/modality.json` the arpa_vla recorder writes:
    state  (8): arm = 6 UR joints [rad], rail [m], gripper [0..1 command scale]
    action (7): arm = the 6 UR joints measured at the next frame (absolute) [rad], gripper target [0..1]
    video:      wrist (D405 colour, 640x480, 10 Hz)
The rail is state only: the pick places it before the episode and holds it.

Registered under NEW_EMBODIMENT, as GR00T requires for a robot it was not pre-trained on.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ur16e_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "rail", "gripper"],
    ),
    # 16 steps at 10 Hz = 1.6 s per chunk; the executor runs the first k <= 16 of them.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            # The dataset stores absolute joint targets; GR00T subtracts the state of the same key
            # ("arm") to train on joint deltas.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(ur16e_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
