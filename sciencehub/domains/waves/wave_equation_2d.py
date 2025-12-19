"""
2D Wave Equation Simulator
Interactive 2D wave equation simulator for membranes (drumheads, water surfaces).
"""

from sciencehub.ui.components.tool_base import ScienceHubTool

import numpy as np
from typing import Optional

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ============================================================
# Constants
# ============================================================

class Wave2DConstants:
    DEFAULT_WIDTH = 1.0        # meters
    DEFAULT_HEIGHT = 1.0       # meters
    DEFAULT_WAVE_SPEED = 50.0  # m/s
    DEFAULT_DAMPING = 0.0
    DEFAULT_DX = 0.008          # meters
    DEFAULT_DT = 0.0003        # seconds
    DEFAULT_TIME = 2.0         # seconds


# ============================================================
# Core solver
# ============================================================

class WaveEquation2DSimulator:
    """Explicit finite-difference solver for the 2D wave equation."""

    def simulate(
        self,
        width: float,
        height: float,
        wave_speed: float,
        damping: float,
        dx: float,
        dt: float,
        total_time: float,
    ) -> np.ndarray:

        nx = int(width / dx)
        ny = int(height / dx)
        nt = int(total_time / dt)

        u_prev = np.zeros((nx, ny))
        u = np.zeros((nx, ny))
        u_next = np.zeros((nx, ny))

        # Initial condition: Gaussian splash at center
        cx, cy = nx // 2, ny // 2
        for i in range(nx):
            for j in range(ny):
                r2 = (i - cx)**2 + (j - cy)**2
                u[i, j] = np.exp(-r2 / (0.02 * nx * ny))

        snapshots = []

        c2 = (wave_speed * dt / dx) ** 2

        for _ in range(nt):
            u_next[1:-1, 1:-1] = (
                2 * u[1:-1, 1:-1]
                - u_prev[1:-1, 1:-1]
                + c2 * (
                    u[2:, 1:-1] + u[:-2, 1:-1]
                    + u[1:-1, 2:] + u[1:-1, :-2]
                    - 4 * u[1:-1, 1:-1]
                )
            )
            u_next[1:-1, 1:-1] *= (1.0 - damping)

            # Fixed boundaries
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0

            snapshots.append(u.copy())
            u_prev, u, u_next = u, u_next, u_prev

        return np.array(snapshots)


# ============================================================
# Tool UI
# ============================================================

class WaveEquation2DTool(ScienceHubTool):
    """2D Wave Equation Simulator Tool."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.simulator = WaveEquation2DSimulator()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.is_playing = False

        self.setup_ui()
        self.setup_plot()

        self.update_view_mode()

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

        title = QLabel("Membrane Parameters")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        form = QFormLayout()

        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(0.2, 5.0)
        self.width_input.setValue(Wave2DConstants.DEFAULT_WIDTH)
        self.width_input.setSuffix(" m")

        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.2, 5.0)
        self.height_input.setValue(Wave2DConstants.DEFAULT_HEIGHT)
        self.height_input.setSuffix(" m")

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(1, 500)
        self.speed_input.setValue(Wave2DConstants.DEFAULT_WAVE_SPEED)
        self.speed_input.setSuffix(" m/s")

        self.damping_input = QDoubleSpinBox()
        self.damping_input.setRange(0.0, 0.1)
        self.damping_input.setDecimals(4)
        self.damping_input.setValue(Wave2DConstants.DEFAULT_DAMPING)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Low (dx = 0.03 m)",
            "Medium (dx = 0.02 m)",
            "High (dx = 0.01 m)",
            "Ultra (dx = 0.008 m)",
            "Ultra + (dx = 0.005 m)",
            "Ultra ++ (dx = 0.003 m)",
            "Extreme (dx = 0.002 m)"
        ])
        self.resolution_combo.setCurrentIndex(1)  # Medium default

        self.dt_input = QDoubleSpinBox()
        self.dt_input.setRange(1e-5, 0.01)
        self.dt_input.setDecimals(5)
        self.dt_input.setValue(Wave2DConstants.DEFAULT_DT)
        self.dt_input.setSuffix(" s")

        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.2, 10.0)
        self.time_input.setValue(Wave2DConstants.DEFAULT_TIME)
        self.time_input.setSuffix(" s")

        form.addRow("Width:", self.width_input)
        form.addRow("Height:", self.height_input)
        form.addRow("Wave Speed:", self.speed_input)
        form.addRow("Damping:", self.damping_input)
        form.addRow("Resolution:", self.resolution_combo)
        form.addRow("dt:", self.dt_input)
        form.addRow("Total Time:", self.time_input)

        layout.addLayout(form)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "2D Field",
            "3D Heightmap"
        ])
        form.addRow("View Mode:", self.view_mode_combo)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button)

        parent_layout.addWidget(widget, 1)

        self.view_mode_combo.currentIndexChanged.connect(self.update_view_mode)

    # --------------------------------------------------------
    # Results panel
    # --------------------------------------------------------

    def setup_results_panel(self, parent_layout):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        title = QLabel("Wave Field")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.view_stack = QStackedWidget()
        layout.addWidget(self.view_stack)

        # --- 2D view ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.view_stack.addWidget(self.plot_widget)

        # --- 3D view ---
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor('#1e1e1e')
        self.gl_view.opts['distance'] = 2.5
        self.view_stack.addWidget(self.gl_view)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(140)
        layout.addWidget(self.info_box)

        parent_layout.addWidget(widget, 2)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    def setup_plot(self):
        self.image = pg.ImageItem()
        self.image.setOpts(
            axisOrder='row-major',
            autoDownsample=True,
            interpolation='bilinear'
        )
        self.plot_widget.addItem(self.image)
        self.plot_widget.getViewBox().setAspectLocked(True)

        cmap = pg.colormap.get("viridis")
        self.image.setLookupTable(cmap.getLookupTable())

    # --------------------------------------------------------

    def update_view_mode(self):
        index = self.view_mode_combo.currentIndex()
        self.view_stack.setCurrentIndex(index)

        # Only create surface AFTER simulation exists
        if index == 1 and hasattr(self, "u"):
            if not hasattr(self, "surface"):
                self.surface = gl.GLSurfacePlotItem(
                    shader='heightColor',
                    smooth=True,
                    computeNormals=False
                )
                self.surface.scale(1.0, 1.0, 0.3)
                self.gl_view.addItem(self.surface)

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    def run_simulation(self):
        self.width = self.width_input.value()
        self.height = self.height_input.value()
        self.speed = self.speed_input.value()
        self.damping = self.damping_input.value()
        self.dx = self.get_dx_from_resolution()
        self.dt = self.dt_input.value()

        cfl = self.speed * self.dt / self.dx
        if cfl > 0.7:
            self.dt = 0.7 * self.dx / self.speed
            self.dt_input.setValue(self.dt)

            QMessageBox.information(
                self,
                "Time Step Adjusted",
                f"dt reduced for 2D stability:\n\nc·dt/dx = {self.speed * self.dt / self.dx:.3f}"
            )

        nx = int(self.width / self.dx)
        ny = int(self.height / self.dx)

        cells = nx * ny
        if cells > 300_000:
            QMessageBox.information(
                self,
                "High Resolution Mode",
                f"Grid size: {nx} × {ny} ≈ {cells:,} cells\n\n"
                "High-resolution mode enabled.\n"
                "Simulation may use significant CPU."
            )
        self.u_prev = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.u_next = np.zeros((nx, ny))

        cx, cy = nx // 2, ny // 2
        sigma = 0.05 * min(nx, ny)
        x = np.arange(nx)[:, None]
        y = np.arange(ny)[None, :]

        self.u = np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
        )

        self.timer.stop()
        self.timer.start(16)  # ~60 FPS

        # Reset 3D surface if needed
        if hasattr(self, "surface"):
            nx, ny = self.u.shape
            x = np.linspace(0, self.width, nx)
            y = np.linspace(0, self.height, ny)
            self.surface.setData(x=x, y=y, z=self.u)

    def step_simulation(self):
        c2 = (self.speed * self.dt / self.dx) ** 2

        u = self.u
        u_prev = self.u_prev
        u_next = self.u_next

        u_next[1:-1, 1:-1] = (
            2 * u[1:-1, 1:-1]
            - u_prev[1:-1, 1:-1]
            + c2 * (
                u[2:, 1:-1] + u[:-2, 1:-1]
                + u[1:-1, 2:] + u[1:-1, :-2]
                - 4 * u[1:-1, 1:-1]
            )
        )

        u_next[1:-1, 1:-1] *= (1.0 - self.damping)
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0

        self.u_prev, self.u, self.u_next = u, u_next, u_prev

    # --------------------------------------------------------

    def get_dx_from_resolution(self):
        mapping = {
            0: 0.03,
            1: 0.02,
            2: 0.01,
            3: 0.008,
            4: 0.005,
            5: 0.003,
            6: 0.002
        }
        return mapping[self.resolution_combo.currentIndex()]

    # --------------------------------------------------------
    def advance_frame(self):
        self.step_simulation()

        data = self.u
        m = np.max(np.abs(data)) + 1e-12

        # --- 2D display (existing) ---
        data_disp = np.sign(data) * np.sqrt(np.abs(data) / m)
        self.image.setImage(data_disp, levels=(-1, 1), autoLevels=False)

        # --- 3D display ---
        if self.view_mode_combo.currentIndex() == 1 and hasattr(self, "surface"):
            nx, ny = data.shape
            x = np.linspace(0, self.width, nx)
            y = np.linspace(0, self.height, ny)
            self.surface.setData(x=x, y=y, z=data)

    # --------------------------------------------------------

    def update_info(self, width, height, speed, damping, dx, dt, total_time):
        info = (
            "2D Wave Equation Simulation\n\n"
            f"Domain: {width:.2f} × {height:.2f} m\n"
            f"Wave Speed: {speed:.2f} m/s\n"
            f"Damping: {damping:.4f}\n"
            f"dx: {dx:.4f} m\n"
            f"dt: {dt:.5f} s\n"
            f"Total Time: {total_time:.2f} s\n\n"
            "Physics:\n"
            "This simulates a vibrating membrane (drumhead)\n"
            "using the 2D wave equation with fixed boundaries."
        )
        self.info_box.setPlainText(info)


# ============================================================
# Metadata
# ============================================================

TOOL_META = {
    "name": "2D Wave Equation Simulator",
    "description": "Interactive 2D wave equation simulator for membranes and surfaces",
    "category": "Physics",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "2D wave equation solver",
        "Gaussian splash excitation",
        "Heatmap visualization",
        "Real-time animation",
        "CFL-stable timestep enforcement"
    ],
    "educational_value": "Visualize wave propagation, interference, and membrane vibration",
    "keywords": ["2D waves", "membrane", "drumhead", "physics", "simulation"]
}


def create_tool(parent=None):
    return WaveEquation2DTool(parent)
