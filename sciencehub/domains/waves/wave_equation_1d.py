"""
Wave Equation Simulator
Interactive 1D wave equation simulator for strings and sound waves.
"""

from sciencehub.ui.components.tool_base import ScienceHubTool

import math
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyqtgraph as pg


# ============================================================
# Physics constants
# ============================================================

class WaveConstants:
    DEFAULT_LENGTH = 1.0          # meters
    DEFAULT_WAVE_SPEED = 100.0    # m/s
    DEFAULT_DAMPING = 0.0
    DEFAULT_DX = 0.01             # meters
    DEFAULT_DT = 0.0005           # seconds
    DEFAULT_TIME = 2.0            # seconds


# ============================================================
# Core simulation engine
# ============================================================

class WaveEquationSimulator:
    """
    Explicit finite-difference solver for the 1D wave equation.
    """

    def simulate(
        self,
        length: float,
        wave_speed: float,
        damping: float,
        dx: float,
        dt: float,
        total_time: float,
    ) -> np.ndarray:
        nx = int(length / dx)
        nt = int(total_time / dt)

        u_prev = np.zeros(nx)
        u = np.zeros(nx)
        u_next = np.zeros(nx)

        # Initial condition: pluck at center
        midpoint = nx // 2
        u[midpoint] = 1.0

        snapshots = []

        c2 = (wave_speed * dt / dx) ** 2

        for _ in range(nt):
            for i in range(1, nx - 1):
                u_next[i] = (
                    2 * u[i]
                    - u_prev[i]
                    + c2 * (u[i + 1] - 2 * u[i] + u[i - 1])
                )
                u_next[i] *= (1.0 - damping)

            # Fixed boundaries
            u_next[0] = 0.0
            u_next[-1] = 0.0

            snapshots.append(u.copy())

            u_prev, u, u_next = u, u_next, u_prev

        return np.array(snapshots)


# ============================================================
# Main tool UI
# ============================================================

class WaveEquationTool(ScienceHubTool):
    """Wave Equation Simulator Tool."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.simulator = WaveEquationSimulator()
        self.snapshots: Optional[np.ndarray] = None
        self.current_frame = 0

        self.setup_ui()
        self.setup_plot()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.is_playing = False
    # --------------------------------------------------------

    def setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        self.setup_input_panel(main_layout)
        self.setup_results_panel(main_layout)

        self.root_layout.addLayout(main_layout)

    # --------------------------------------------------------
    # Input panel
    # --------------------------------------------------------

    def setup_input_panel(self, parent_layout):
        widget = QWidget()
        widget.setObjectName("toolCard")
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        title = QLabel("Wave Parameters")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)

        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(0.1, 10.0)
        self.length_input.setValue(WaveConstants.DEFAULT_LENGTH)
        self.length_input.setSuffix(" m")

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(1.0, 1000.0)
        self.speed_input.setValue(WaveConstants.DEFAULT_WAVE_SPEED)
        self.speed_input.setSuffix(" m/s")

        self.damping_input = QDoubleSpinBox()
        self.damping_input.setRange(0.0, 0.1)
        self.damping_input.setDecimals(4)
        self.damping_input.setValue(WaveConstants.DEFAULT_DAMPING)

        self.dx_input = QDoubleSpinBox()
        self.dx_input.setRange(0.001, 0.1)
        self.dx_input.setDecimals(4)
        self.dx_input.setValue(WaveConstants.DEFAULT_DX)
        self.dx_input.setSuffix(" m")

        self.dt_input = QDoubleSpinBox()
        self.dt_input.setRange(1e-5, 0.01)
        self.dt_input.setDecimals(5)
        self.dt_input.setValue(WaveConstants.DEFAULT_DT)
        self.dt_input.setSuffix(" s")

        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.1, 10.0)
        self.time_input.setValue(WaveConstants.DEFAULT_TIME)
        self.time_input.setSuffix(" s")

        form.addRow("String Length:", self.length_input)
        form.addRow("Wave Speed:", self.speed_input)
        form.addRow("Damping:", self.damping_input)
        form.addRow("Spatial Step (dx):", self.dx_input)
        form.addRow("Time Step (dt):", self.dt_input)
        form.addRow("Total Time:", self.time_input)

        layout.addLayout(form)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_simulation)

        layout.addWidget(self.run_button)

        parent_layout.addWidget(widget, 1)

    # --------------------------------------------------------
    # Results panel
    # --------------------------------------------------------

    def setup_results_panel(self, parent_layout):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        title = QLabel("Wave Visualization")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")

        self.plot_widget.getAxis('left').setPen('w')
        self.plot_widget.getAxis('bottom').setPen('w')
        self.plot_widget.getAxis('left').setTextPen('w')
        self.plot_widget.getAxis('bottom').setTextPen('w')

        self.plot_widget.setLabel("left", "Displacement", color="w")
        self.plot_widget.setLabel("bottom", "Position (m)", color="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        layout.addWidget(self.plot_widget)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.update_frame)
        self.slider.setEnabled(False)

        layout.addWidget(self.slider)

        controls = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("secondaryButton")
        self.play_button.clicked.connect(self.toggle_play)

        controls.addWidget(self.play_button)

        layout.addLayout(controls)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(140)
        self.info_box.setObjectName("infoText")

        layout.addWidget(self.info_box)

        parent_layout.addWidget(widget, 2)

    # --------------------------------------------------------
    # Plot setup
    # --------------------------------------------------------

    def setup_plot(self):
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#00bfff", width=2)
        )

    # --------------------------------------------------------
    # Simulation logic
    # --------------------------------------------------------

    def run_simulation(self):
        length = self.length_input.value()
        speed = self.speed_input.value()
        damping = self.damping_input.value()
        dx = self.dx_input.value()
        dt = self.dt_input.value()
        total_time = self.time_input.value()

        courant = speed * dt / dx

        if courant > 0.95:
            dt = 0.95 * dx / speed
            self.dt_input.setValue(dt)

            QMessageBox.information(
                self,
                "Time Step Adjusted",
                "Time step was automatically reduced to satisfy\n"
                "the Courant stability condition:\n\n"
                f"c·dt/dx = {speed * dt / dx:.2f}"
            )
        self.snapshots = self.simulator.simulate(
            length, speed, damping, dx, dt, total_time
        )
        self.positions = np.linspace(0, length, self.snapshots.shape[1])

        self.slider.setMaximum(len(self.snapshots) - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)

        self.update_info(length, speed, damping, dx, dt, total_time)
        self.update_frame(0)

    def toggle_play(self):
        if self.snapshots is None:
            return

        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
        else:
            self.timer.start(30)  # ~33 FPS
            self.play_button.setText("Pause")
            self.is_playing = True

    def advance_frame(self):
        if self.snapshots is None:
            return

        self.current_frame += 1

        if self.current_frame >= len(self.snapshots):
            self.current_frame = 0  # loop

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)

        self.update_frame(self.current_frame)

    # --------------------------------------------------------

    def update_frame(self, frame: int):
        if self.snapshots is None:
            return

        self.current_frame = frame
        self.curve.setData(self.positions, self.snapshots[frame])

    # --------------------------------------------------------

    def update_info(self, length, speed, damping, dx, dt, total_time):
        fundamental_freq = speed / (2 * length)
        courant = speed * dt / dx

        info = (
            "Wave Equation Simulation\n\n"
            f"String Length: {length:.2f} m\n"
            f"Wave Speed: {speed:.2f} m/s\n"
            f"Damping: {damping:.4f}\n"
            f"dx: {dx:.4f} m\n"
            f"dt: {dt:.5f} s\n"
            f"Total Time: {total_time:.2f} s\n\n"
            f"Fundamental Frequency: {fundamental_freq:.2f} Hz\n"
            f"Courant Number (c·dt/dx): {courant:.2f}\n\n"
            "Physics:\n"
            "This simulation solves the 1D wave equation using\n"
            "finite differences with fixed boundary conditions.\n"
            "The initial condition is a plucked string."
        )

        self.info_box.setPlainText(info)


# ============================================================
# Tool metadata
# ============================================================

TOOL_META = {
    "name": "Wave Equation Simulator",
    "description": "Interactive 1D wave equation simulator for strings and sound waves",
    "category": "Physics",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "1D wave equation solver",
        "Plucked-string initial condition",
        "Real-time visualization",
        "Damping effects",
        "Courant stability analysis"
    ],
    "educational_value": "Visualize wave propagation, reflection, and standing waves",
    "keywords": ["wave equation", "waves", "string", "sound", "physics"]
}


def create_tool(parent=None):
    return WaveEquationTool(parent)
