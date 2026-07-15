from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets

from lerobot_bw_policy_runner.constants import JOINT_NAMES
from lerobot_bw_policy_runner.visualization.buffer import ActionHistory
from lerobot_bw_policy_runner.visualization.dashboard import ActionDashboard


def _populate(history: ActionHistory) -> None:
    now = time.monotonic()
    for sample_index in range(3):
        act = np.arange(len(JOINT_NAMES), dtype=np.float32) * 0.01 + sample_index
        delta = np.full(len(JOINT_NAMES), 0.02, dtype=np.float32)
        composed = act + 0.2 * delta
        final = composed * 0.8
        values = {"act": act, "delta": delta, "composed": composed, "final": final}
        for stream, vector in values.items():
            history.add_message(
                stream,
                1_000_000_000 + sample_index * 33_000_000,
                JOINT_NAMES,
                vector,
                received_monotonic=now,
            )


def test_dashboard_has_four_selectable_panels_and_three_curves_each() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    history = ActionHistory(window_seconds=10.0)
    _populate(history)
    dashboard = ActionDashboard(history=history, robot_sn="BW_TEST", refresh_hz=20.0)
    dashboard.refresh_once()

    assert len(dashboard.panels) == 4
    for panel in dashboard.panels:
        assert panel.joint_selector.count() == len(JOINT_NAMES)
        assert len(panel.plot_widget.plotItem.listDataItems()) == 3
        assert panel.act_curve.getData()[0].size == 3

    dashboard.panels[0].joint_selector.setCurrentText("right_gripper_joint")
    dashboard.refresh_once()
    assert dashboard.panels[0].joint_name == "right_gripper_joint"
    assert "lambda*delta" in dashboard.panels[0].metrics_label.text()
    dashboard.close()
    app.processEvents()
