from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np

from lerobot_bw_policy_runner.ros_io import BWObservationReader


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


def test_missing_control_source_does_not_block_diagnostic_observation_startup() -> None:
    camera_topics = {"env_cam": "/env", "left_wrist_cam": "/left", "right_wrist_cam": "/right"}
    reader = SimpleNamespace(
        _lock=threading.Lock(),
        _latest_state=object(),
        _latest_control_source=None,
        _latest_images={name: object() for name in camera_topics},
        config=SimpleNamespace(
            robot=SimpleNamespace(
                input_topics=SimpleNamespace(
                    state="/state",
                    control_source="/control_source",
                    cameras=camera_topics,
                )
            ),
            inference=SimpleNamespace(require_all_cameras=True),
        ),
    )

    assert BWObservationReader.missing_inputs(reader) == []


def test_blocked_handover_keeps_model_debug_but_omits_action_final() -> None:
    reader = SimpleNamespace(
        debug_act_publisher=_Publisher(),
        debug_delta_publisher=_Publisher(),
        debug_composed_publisher=_Publisher(),
        debug_final_publisher=_Publisher(),
        debug_gripper_class_publisher=_Publisher(),
    )
    message = object()

    BWObservationReader.publish_debug_actions(
        reader,
        message,
        message,
        message,
        message,
        np.zeros(2, dtype=np.int64),
        publish_final=False,
    )

    assert reader.debug_act_publisher.messages == [message]
    assert reader.debug_delta_publisher.messages == [message]
    assert reader.debug_composed_publisher.messages == [message]
    assert reader.debug_final_publisher.messages == []
    assert len(reader.debug_gripper_class_publisher.messages) == 1
