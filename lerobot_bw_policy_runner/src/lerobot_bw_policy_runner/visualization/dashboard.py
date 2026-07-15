"""PyQtGraph dashboard for selected robot joint action streams."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..constants import JOINT_NAMES
from .buffer import ActionHistory, ActionHistorySnapshot

DEFAULT_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_elbow_joint",
)


class JointActionPanel(QtWidgets.QFrame):
    def __init__(self, *, panel_number: int, default_joint: str, window_seconds: float) -> None:
        super().__init__()
        self.window_seconds = float(window_seconds)
        self.setObjectName("jointPanel")
        self.setMinimumSize(420, 280)

        title = QtWidgets.QLabel(f"Joint {panel_number}")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        self.joint_selector = QtWidgets.QComboBox()
        self.joint_selector.addItems(JOINT_NAMES)
        self.joint_selector.setCurrentText(default_joint)
        self.joint_selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.joint_selector)

        self.metrics_label = QtWidgets.QLabel("No data")
        self.metrics_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        metrics_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        metrics_font.setPointSize(9)
        self.metrics_label.setFont(metrics_font)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Action value")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.addLegend(offset=(8, 8))
        self.plot_widget.setXRange(-self.window_seconds, 0.0, padding=0.0)
        self.plot_widget.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("#9aa0a6", width=1)))
        self.act_curve = self.plot_widget.plot(
            name="ACT", pen=pg.mkPen("#2563eb", width=2), connect="finite"
        )
        self.composed_curve = self.plot_widget.plot(
            name="ACT + lambda*delta", pen=pg.mkPen("#d97706", width=2), connect="finite"
        )
        self.final_curve = self.plot_widget.plot(
            name="Final command", pen=pg.mkPen("#15803d", width=2), connect="finite"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addLayout(header)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.plot_widget, 1)

    @property
    def joint_name(self) -> str:
        return self.joint_selector.currentText()

    def update_data(self, snapshot: ActionHistorySnapshot) -> None:
        if snapshot.sample_count == 0:
            return
        joint_index = JOINT_NAMES.index(self.joint_name)
        times = (snapshot.timestamps_ns - snapshot.timestamps_ns[-1]).astype(np.float64) / 1_000_000_000.0
        act = snapshot.act[:, joint_index]
        delta = snapshot.delta[:, joint_index]
        composed = snapshot.composed[:, joint_index]
        final = snapshot.final[:, joint_index]
        self.act_curve.setData(times, act)
        self.composed_curve.setData(times, composed)
        self.final_curve.setData(times, final)
        self.plot_widget.setXRange(-self.window_seconds, 0.0, padding=0.0)

        effective_delta = float(composed[-1] - act[-1])
        postprocess_delta = float(final[-1] - composed[-1])
        self.metrics_label.setText(
            f"ACT {float(act[-1]):+.5f}    delta {float(delta[-1]):+.5f}    lambda*delta {effective_delta:+.5f}\n"
            f"Composed {float(composed[-1]):+.5f}    Final {float(final[-1]):+.5f}    Post {postprocess_delta:+.5f}"
        )


class ActionDashboard(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        history: ActionHistory,
        robot_sn: str,
        refresh_hz: float = 20.0,
        stale_after_seconds: float = 1.0,
        error_provider: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__()
        self.history = history
        self.stale_after_seconds = float(stale_after_seconds)
        self.error_provider = error_provider
        self.setWindowTitle(f"BW Action Visualizer - {robot_sn}")
        self.setMinimumSize(900, 620)
        self.resize(1280, 820)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 12)
        self.status_label = QtWidgets.QLabel("Waiting for synchronized action messages")
        self.status_label.setObjectName("statusLabel")
        root.addWidget(self.status_label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        self.panels = [
            JointActionPanel(
                panel_number=index + 1,
                default_joint=joint_name,
                window_seconds=history.window_seconds,
            )
            for index, joint_name in enumerate(DEFAULT_JOINTS)
        ]
        for index, panel in enumerate(self.panels):
            grid.addWidget(panel, index // 2, index % 2)
        root.addLayout(grid, 1)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #f7f7f8; color: #202124; }"
            "QFrame#jointPanel { background: white; border: 1px solid #d6d8dc; border-radius: 4px; }"
            "QComboBox { background: white; border: 1px solid #b9bdc5; padding: 4px 8px; min-height: 22px; }"
            "QLabel#statusLabel { padding: 5px 2px; font-weight: 600; }"
        )

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(1, round(1000.0 / float(refresh_hz))))
        self._timer.timeout.connect(self.refresh_once)
        self._timer.start()

    def refresh_once(self) -> None:
        snapshot = self.history.snapshot()
        error = self.error_provider() if self.error_provider is not None else None
        if error:
            self._set_status(f"ROS subscriber stopped: {error}", "#b42318")
            return
        if snapshot.sample_count == 0 or snapshot.last_received_monotonic is None:
            self._set_status("Waiting for synchronized action messages", "#9a6700")
            return
        age = max(0.0, time.monotonic() - snapshot.last_received_monotonic)
        if age > self.stale_after_seconds:
            self._set_status(f"Disconnected - last complete sample {age:.1f}s ago", "#b42318")
            return

        rate = 0.0
        if snapshot.sample_count > 1:
            duration = float(snapshot.timestamps_ns[-1] - snapshot.timestamps_ns[0]) / 1_000_000_000.0
            if duration > 0:
                rate = (snapshot.sample_count - 1) / duration
        self._set_status(
            f"Connected | complete samples {rate:.1f} Hz | display age {age * 1000.0:.0f} ms",
            "#157347",
        )
        for panel in self.panels:
            panel.update_data(snapshot)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")
