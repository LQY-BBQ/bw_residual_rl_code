"""Command-line entry point for the independent action visualization process."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys

from ..config import default_config_path, load_config
from .buffer import ActionHistory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ACT, residual-composed, and final robot actions.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--robot-sn", required=False)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--refresh-hz", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.window_seconds <= 0:
        print("ERROR: --window-seconds must be positive.", file=sys.stderr)
        return 2
    if args.refresh_hz <= 0:
        print("ERROR: --refresh-hz must be positive.", file=sys.stderr)
        return 2

    try:
        from PyQt5 import QtCore, QtWidgets
        import pyqtgraph as pg
    except ModuleNotFoundError as exc:
        print(
            "ERROR: action visualization dependencies are missing. "
            "Install them with: python3 -m pip install -e '.[visualization]'\n"
            f"Missing module: {exc.name}",
            file=sys.stderr,
        )
        return 2

    from .dashboard import ActionDashboard
    from .ros_bridge import VisualizationRosRunner

    config = load_config(args.config, robot_sn=args.robot_sn)
    os.environ["ROS_DOMAIN_ID"] = str(config.ros.domain_id)
    history = ActionHistory(window_seconds=args.window_seconds)

    pg.setConfigOptions(background="w", foreground="#202124", antialias=False)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName("BW Action Visualizer")
    runner = VisualizationRosRunner(
        robot_sn=config.robot.robot_sn,
        topics=config.robot.output_topics,
        history=history,
    )
    try:
        runner.start()
    except Exception as exc:
        print(f"ERROR: failed to start ROS action subscriber: {exc}", file=sys.stderr)
        runner.stop()
        return 3

    window = ActionDashboard(
        history=history,
        robot_sn=config.robot.robot_sn,
        refresh_hz=args.refresh_hz,
        error_provider=lambda: runner.last_error,
    )
    app.aboutToQuit.connect(runner.stop)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    signal_timer = QtCore.QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(250)
    window.show()
    try:
        return int(app.exec_())
    finally:
        runner.stop()


if __name__ == "__main__":
    raise SystemExit(main())
