#TODO: Fix light ray animation not working properly, improve performance, add more presets, refine UI. 
#TODO: Add more detailed physics calculations for light bending.
#TODO: Optimize 3D rendering for large black hole masses/distances.

"""
3D Schwarzschild Black Hole Simulator
Advanced 3D visualization of Schwarzschild black holes with spacetime curvature,
light bending, and interactive exploration.
"""

import sys
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from sciencehub.ui.components.tool_base import ScienceHubTool
from sciencehub.domains.astronomy.schwarzschild_black_hole_simulator import (
    SchwarzschildCalculator,
    PhysicsConstants
)


class BlackHole3DCanvas(FigureCanvas):
    """3D matplotlib canvas for black hole visualization."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111, projection='3d')

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data structures
        self.black_hole_data = {}
        self.light_rays = []
        self.observer_pos = np.array([0, 0, 10])  # Default observer position
        self.camera_angle = [30, 45]  # Elevation, azimuth

        # Animation state
        self.animation_timer = None
        self.is_animating = False
        self.animation_angle = 0
        self.animation_speed = 2  # degrees per frame

        # Setup the plot
        self.setup_plot()

    def setup_plot(self):
        """Initialize the 3D plot."""
        self.axes.clear()

        # Set dark theme colors
        self.axes.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        self.axes.grid(True, alpha=0.3)
        self.axes.xaxis.pane.fill = False
        self.axes.yaxis.pane.fill = False
        self.axes.zaxis.pane.fill = False

        # Set labels and title
        self.axes.set_xlabel('X (Rs)', color='white', fontsize=10)
        self.axes.set_ylabel('Y (Rs)', color='white', fontsize=10)
        self.axes.set_zlabel('Z (Rs)', color='white', fontsize=10)
        self.axes.set_title('3D Schwarzschild Black Hole', color='white', fontsize=12)

        # Set tick colors
        self.axes.tick_params(colors='white')

        # Set equal aspect ratio
        self.axes.set_box_aspect([1, 1, 1])

        # Initial view
        self.axes.view_init(elev=self.camera_angle[0], azim=self.camera_angle[1])

    def update_black_hole(self, mass_kg: float, rs: float):
        """Update black hole visualization data."""
        self.black_hole_data = {
            'mass': mass_kg,
            'rs': rs,
            'photon_sphere': 1.5 * rs,
            'isco': 6 * rs
        }

    def update_observer(self, position: np.ndarray):
        """Update observer position."""
        self.observer_pos = position

    def draw_black_hole(self):
        """Draw the black hole components."""
        if not self.black_hole_data:
            return

        rs = self.black_hole_data['rs']
        photon_sphere = self.black_hole_data['photon_sphere']
        isco = self.black_hole_data['isco']

        # Create sphere coordinates (higher resolution for better visibility)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Event horizon (black sphere) - make it more visible
        self.axes.plot_surface(rs * x, rs * y, rs * z,
                             color='black', alpha=0.9, label='Event Horizon')

        # Photon sphere (red wireframe) - more visible
        self.axes.plot_wireframe(photon_sphere * x, photon_sphere * y, photon_sphere * z,
                               color='#ff6b6b', alpha=0.8, linewidth=2, label='Photon Sphere')

        # ISCO (blue wireframe) - more visible
        self.axes.plot_wireframe(isco * x, isco * y, isco * z,
                               color='#4ecdc4', alpha=0.6, linewidth=2, label='ISCO')

    def draw_light_rays(self, impact_parameters: List[float], rs: float):
        """Draw light ray trajectories showing gravitational bending."""
        self.light_rays = []

        for b in impact_parameters:
            if b < rs * 1.01:  # Skip rays that fall in
                continue

            # Calculate light bending trajectory
            trajectory = self.calculate_light_trajectory(b, rs)
            if trajectory:
                x_vals, y_vals = trajectory
                z_vals = np.zeros_like(x_vals)  # Light rays in equatorial plane

                # Plot the trajectory - use consistent scaling
                line, = self.axes.plot(x_vals, y_vals, z_vals,
                                     color='#45b7d1', alpha=0.8, linewidth=2)
                self.light_rays.append(line)

    def calculate_light_trajectory(self, b: float, rs: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Calculate light ray trajectory for given impact parameter."""
        if b <= rs * 1.01:
            return None

        # More accurate light bending calculation
        # For null geodesics in Schwarzschild metric

        # Create trajectory points
        num_points = 200
        phi_max = 2 * np.pi  # Full circle for visualization

        # For visualization, create a trajectory that shows bending
        # This is a simplified approximation
        phi = np.linspace(0, phi_max, num_points)

        # Impact parameter in units of rs
        b_rs = b / rs

        # Calculate radial coordinate for each angle
        # This is an approximation for visualization purposes
        if b_rs > 1.5:  # Rays that don't get captured
            # Straight line approximation for distant rays
            r = b_rs / np.sin(phi + 0.1)  # Slight bending
            r = np.clip(r, rs * 1.1, rs * 50)  # Limit range
        else:
            # Rays that get captured or highly bent
            r = rs * (1.5 + 0.5 * np.sin(phi * 2))

        # Convert to Cartesian coordinates
        x = r * rs * np.cos(phi)
        y = r * rs * np.sin(phi)

        # Limit the trajectory to reasonable bounds
        mask = (np.abs(x) < rs * 20) & (np.abs(y) < rs * 20)
        x = x[mask][:100]  # Limit to 100 points
        y = y[mask][:100]

        if len(x) < 10:
            return None

        return x, y

    def draw_coordinate_grid(self, rs: float, max_radius: float = 20):
        """Draw coordinate grid showing spacetime curvature."""
        # Radial lines
        for phi in np.linspace(0, 2*np.pi, 12):
            r_vals = np.linspace(rs * 1.1, max_radius * rs, 50)
            x = r_vals * np.cos(phi)
            y = r_vals * np.sin(phi)
            z = np.zeros_like(x)

            self.axes.plot(x/rs, y/rs, z/rs, color='#666666', alpha=0.3, linewidth=0.5)

        # Angular circles
        for r in np.linspace(rs * 2, max_radius * rs, 8):
            phi = np.linspace(0, 2*np.pi, 50)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = np.zeros_like(x)

            self.axes.plot(x/rs, y/rs, z/rs, color='#666666', alpha=0.3, linewidth=0.5)

    def draw_observer(self):
        """Draw observer position and field of view."""
        obs_x, obs_y, obs_z = self.observer_pos

        # Draw observer as a small sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v)) * 0.1 + obs_x
        y = np.outer(np.sin(u), np.sin(v)) * 0.1 + obs_y
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.1 + obs_z

        self.axes.plot_surface(x, y, z, color='#ffff00', alpha=0.8)

        # Draw field of view cone (simplified)
        # This would show what the observer can see

    def update_view(self, elev: float, azim: float):
        """Update camera view angle."""
        self.camera_angle = [elev, azim]
        self.axes.view_init(elev=elev, azim=azim)
        self.draw()

    def clear_plot(self):
        """Clear all plot elements."""
        self.axes.clear()
        self.setup_plot()
        self.light_rays = []

    def redraw(self):
        """Redraw the entire visualization."""
        self.clear_plot()

        if self.black_hole_data:
            rs = self.black_hole_data['rs']
            self.draw_black_hole()
            self.draw_coordinate_grid(rs)
            self.draw_light_rays([rs * 2, rs * 3, rs * 5, rs * 10], rs)

        self.draw_observer()
        self.draw()

    def start_animation(self):
        """Start the 3D animation."""
        if self.animation_timer is None:
            self.animation_timer = self.new_timer(50)  # 20 FPS
            self.animation_timer.timeout.connect(self.animate_frame)
            self.animation_timer.start()
            self.is_animating = True

    def stop_animation(self):
        """Stop the 3D animation."""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer = None
            self.is_animating = False

    def animate_frame(self):
        """Animate one frame of the 3D visualization."""
        if not self.is_animating:
            return

        # Rotate camera around the black hole
        self.animation_angle += self.animation_speed
        if self.animation_angle >= 360:
            self.animation_angle = 0

        # Update camera position
        if self.black_hole_data:
            rs = self.black_hole_data['rs']
            radius = rs * 8
            self.axes.view_init(elev=20, azim=self.animation_angle)
            self.axes.set_xlim([-radius*1.5, radius*1.5])
            self.axes.set_ylim([-radius*1.5, radius*1.5])
            self.axes.set_zlim([-radius*1.5, radius*1.5])

        # Redraw the scene
        self.redraw()

    def set_animation_speed(self, speed: int):
        """Set animation speed (1-10)."""
        self.animation_speed = speed * 0.5  # Scale to reasonable rotation speed


class Schwarzschild3DSimulator(ScienceHubTool):
    """3D Schwarzschild Black Hole Simulator Tool Widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculator = SchwarzschildCalculator()
        self.constants = PhysicsConstants()

        # Initialize variables
        self.current_mass = 10.0  # M☉
        self.current_distance = 10.0  # Rs
        self.rs_current = 0

        # Animation state
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animation_step)

        self.camera_angle_anim = 0.0
        self.light_phase = 0

        self.cached_rays = {}
        self.light_ray_lines = []

        self.setup_ui()

    def setup_ui(self):
        """Set up the 3D simulator interface."""
        main_layout = QHBoxLayout()

        # Left panel - controls
        self.setup_control_panel(main_layout)

        # Right panel - 3D visualization
        self.setup_visualization_panel(main_layout)

        self.root_layout.addLayout(main_layout)

        # Initialize with default values
        self.update_visualization()

    def setup_control_panel(self, parent_layout):
        """Set up the control panel."""
        control_widget = QWidget()
        control_widget.setObjectName("toolCard")
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(15)

        # Black hole parameters
        self.setup_black_hole_controls(control_layout)

        # Visualization controls
        self.setup_visualization_controls(control_layout)

        # Camera controls
        self.setup_camera_controls(control_layout)

        # Animation controls
        self.setup_animation_controls(control_layout)

        parent_layout.addWidget(control_widget, 1)

    def setup_black_hole_controls(self, layout):
        """Set up black hole parameter controls."""
        bh_group = QGroupBox("Black Hole Parameters")
        bh_group.setObjectName("toolCard")
        bh_layout = QFormLayout(bh_group)
        bh_layout.setSpacing(10)

        # Mass input
        self.mass_3d_input = QDoubleSpinBox()
        self.mass_3d_input.setRange(0.1, 1e12)
        self.mass_3d_input.setValue(10.0)
        self.mass_3d_input.setDecimals(2)
        self.mass_3d_input.valueChanged.connect(self.on_mass_changed)

        self.mass_3d_unit = QComboBox()
        self.mass_3d_unit.addItems(['M☉', 'M⊕', 'MJ', 'kg'])
        self.mass_3d_unit.setCurrentText('M☉')
        self.mass_3d_unit.currentTextChanged.connect(self.on_mass_changed)

        mass_layout = QHBoxLayout()
        mass_layout.addWidget(self.mass_3d_input)
        mass_layout.addWidget(self.mass_3d_unit)
        bh_layout.addRow("Mass:", mass_layout)

        # Observer distance
        self.distance_3d_input = QDoubleSpinBox()
        self.distance_3d_input.setRange(1.0, 1e6)
        self.distance_3d_input.setValue(10.0)
        self.distance_3d_input.setDecimals(1)
        self.distance_3d_input.valueChanged.connect(self.on_distance_changed)

        self.distance_3d_unit = QComboBox()
        self.distance_3d_unit.addItems(['Rs', 'km', 'AU', 'ly'])
        self.distance_3d_unit.setCurrentText('Rs')
        self.distance_3d_unit.currentTextChanged.connect(self.on_distance_changed)

        distance_layout = QHBoxLayout()
        distance_layout.addWidget(self.distance_3d_input)
        distance_layout.addWidget(self.distance_3d_unit)
        bh_layout.addRow("Observer Distance:", distance_layout)

        layout.addWidget(bh_group)

    def setup_visualization_controls(self, layout):
        """Set up visualization control options."""
        vis_group = QGroupBox("Visualization Options")
        vis_group.setObjectName("toolCard")
        vis_layout = QVBoxLayout(vis_group)

        # Visibility toggles
        self.show_event_horizon = QCheckBox("Show Event Horizon")
        self.show_event_horizon.setChecked(True)
        self.show_event_horizon.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.show_event_horizon)

        self.show_photon_sphere = QCheckBox("Show Photon Sphere")
        self.show_photon_sphere.setChecked(True)
        self.show_photon_sphere.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.show_photon_sphere)

        self.show_isco = QCheckBox("Show ISCO")
        self.show_isco.setChecked(True)
        self.show_isco.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.show_isco)

        self.show_light_rays = QCheckBox("Show Light Rays")
        self.show_light_rays.setChecked(True)
        self.show_light_rays.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.show_light_rays)

        self.show_coordinate_grid = QCheckBox("Show Coordinate Grid")
        self.show_coordinate_grid.setChecked(True)
        self.show_coordinate_grid.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.show_coordinate_grid)

        self.show_observer = QCheckBox("Show Observer")
        self.show_observer.setChecked(True)
        self.show_observer.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.show_observer)

        layout.addWidget(vis_group)

    def setup_camera_controls(self, layout):
        """Set up camera control sliders."""
        camera_group = QGroupBox("Camera Controls")
        camera_group.setObjectName("toolCard")
        camera_layout = QVBoxLayout(camera_group)

        # Elevation slider
        elev_layout = QHBoxLayout()
        elev_layout.addWidget(QLabel("Elevation:"))
        self.elev_slider = QSlider(Qt.Orientation.Horizontal)
        self.elev_slider.setRange(-90, 90)
        self.elev_slider.setValue(30)
        self.elev_slider.valueChanged.connect(self.on_camera_changed)
        elev_layout.addWidget(self.elev_slider)
        self.elev_label = QLabel("30°")
        elev_layout.addWidget(self.elev_label)
        camera_layout.addLayout(elev_layout)

        # Azimuth slider
        azim_layout = QHBoxLayout()
        azim_layout.addWidget(QLabel("Azimuth:"))
        self.azim_slider = QSlider(Qt.Orientation.Horizontal)
        self.azim_slider.setRange(0, 360)
        self.azim_slider.setValue(45)
        self.azim_slider.valueChanged.connect(self.on_camera_changed)
        azim_layout.addWidget(self.azim_slider)
        self.azim_label = QLabel("45°")
        azim_layout.addWidget(self.azim_label)
        camera_layout.addLayout(azim_layout)

        # Preset views
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset Views:"))

        self.view_xy_btn = QPushButton("XY Plane")
        self.view_xy_btn.clicked.connect(lambda: self.set_preset_view(0, 0))
        preset_layout.addWidget(self.view_xy_btn)

        self.view_xz_btn = QPushButton("XZ Plane")
        self.view_xz_btn.clicked.connect(lambda: self.set_preset_view(0, 90))
        preset_layout.addWidget(self.view_xz_btn)

        self.view_3d_btn = QPushButton("3D View")
        self.view_3d_btn.clicked.connect(lambda: self.set_preset_view(30, 45))
        preset_layout.addWidget(self.view_3d_btn)

        camera_layout.addLayout(preset_layout)

        layout.addWidget(camera_group)

    def setup_animation_controls(self, layout):
        """Set up animation controls."""
        anim_group = QGroupBox("Animation")
        anim_group.setObjectName("toolCard")
        anim_layout = QVBoxLayout(anim_group)

        # Animation controls
        self.animate_camera = QCheckBox("Rotate Camera")
        self.animate_camera.stateChanged.connect(self.on_animation_changed)
        anim_layout.addWidget(self.animate_camera)

        self.animate_light_rays = QCheckBox("Animate Light Rays")
        self.animate_light_rays.stateChanged.connect(self.on_animation_changed)
        anim_layout.addWidget(self.animate_light_rays)

        # Animation speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.animation_speed = QSlider(Qt.Orientation.Horizontal)
        self.animation_speed.setRange(1, 10)
        self.animation_speed.setValue(5)
        speed_layout.addWidget(self.animation_speed)
        anim_layout.addLayout(speed_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_animation_btn = QPushButton("Start")
        self.start_animation_btn.clicked.connect(self.start_animation)
        button_layout.addWidget(self.start_animation_btn)

        self.stop_animation_btn = QPushButton("Stop")
        self.stop_animation_btn.clicked.connect(self.stop_animation)
        button_layout.addWidget(self.stop_animation_btn)

        self.reset_animation_btn = QPushButton("Reset")
        self.reset_animation_btn.clicked.connect(self.reset_animation)
        button_layout.addWidget(self.reset_animation_btn)

        anim_layout.addLayout(button_layout)

        layout.addWidget(anim_group)

    def setup_visualization_panel(self, parent_layout):
        """Set up the 3D visualization panel."""
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)

        # Create the 3D canvas
        self.canvas_3d = BlackHole3DCanvas(self, width=8, height=6)
        vis_layout.addWidget(self.canvas_3d)

        # Info panel
        info_widget = QWidget()
        info_widget.setObjectName("toolCard")
        info_layout = QVBoxLayout(info_widget)

        self.info_3d_label = QLabel("3D Visualization Info:")
        self.info_3d_label.setObjectName("sectionTitle")
        info_layout.addWidget(self.info_3d_label)

        self.info_3d_text = QTextEdit()
        self.info_3d_text.setMaximumHeight(100)
        self.info_3d_text.setReadOnly(True)
        self.info_3d_text.setPlainText(
            "• Black sphere: Event horizon\n"
            "• Red wireframe: Photon sphere\n"
            "• Blue wireframe: ISCO\n"
            "• Cyan lines: Light ray trajectories\n"
            "• Yellow sphere: Observer position\n"
            "• Gray grid: Coordinate system"
        )
        info_layout.addWidget(self.info_3d_text)

        vis_layout.addWidget(info_widget)

        parent_layout.addWidget(vis_widget, 2)

    def get_mass_kg(self) -> float:
        """Get black hole mass in kg."""
        mass = self.mass_3d_input.value()
        unit = self.mass_3d_unit.currentText()

        if unit == 'M☉':
            return mass * self.constants.SOLAR_MASS
        elif unit == 'M⊕':
            return mass * self.constants.EARTH_MASS
        elif unit == 'MJ':
            return mass * self.constants.JUPITER_MASS
        else:  # kg
            return mass

    def get_distance_rs(self) -> float:
        """Get observer distance in units of Rs."""
        distance = self.distance_3d_input.value()
        unit = self.distance_3d_unit.currentText()

        mass_kg = self.get_mass_kg()
        rs = self.calculator.schwarzschild_radius(mass_kg)

        if unit == 'Rs':
            return distance
        elif unit == 'km':
            return distance * 1000 / rs
        elif unit == 'AU':
            return distance * self.constants.AU_TO_M / rs
        elif unit == 'ly':
            return distance * self.constants.LY_TO_M / rs
        else:
            return distance

    def on_mass_changed(self):
        """Handle mass input changes."""
        self.update_visualization()

    def on_distance_changed(self):
        """Handle distance input changes."""
        distance_rs = self.get_distance_rs()
        observer_pos = np.array([0, 0, distance_rs])
        self.canvas_3d.update_observer(observer_pos)
        self.update_visualization()

    def on_camera_changed(self):
        """Handle camera slider changes."""
        elev = self.elev_slider.value()
        azim = self.azim_slider.value()

        self.elev_label.setText(f"{elev}°")
        self.azim_label.setText(f"{azim}°")

        self.canvas_3d.update_view(elev, azim)

    def set_preset_view(self, elev: int, azim: int):
        """Set preset camera view."""
        self.elev_slider.setValue(elev)
        self.azim_slider.setValue(azim)

    def update_visualization(self):
        """Update the 3D visualization."""
        mass_kg = self.get_mass_kg()
        rs = self.calculator.schwarzschild_radius(mass_kg)

        self.rs_current = rs
        self.canvas_3d.update_black_hole(mass_kg, rs)

        # Update observer position
        distance_rs = self.get_distance_rs()
        observer_pos = np.array([0, 0, distance_rs])
        self.canvas_3d.update_observer(observer_pos)

        # Redraw
        self.canvas_3d.redraw()

        # Update info
        self.update_info()

        self.init_light_rays()


    def update_info(self):
        """Update the information display."""
        mass_kg = self.get_mass_kg()
        rs = self.rs_current
        distance_rs = self.get_distance_rs()

        info = f"""3D Black Hole Visualization:

Mass: {mass_kg/self.constants.SOLAR_MASS:.2f} M☉
Schwarzschild Radius: {rs/1000:.2f} km
Observer Distance: {distance_rs:.1f} Rs ({distance_rs*rs/1000:.2f} km)

Visible Elements:
• Event Horizon: r = 1 Rs
• Photon Sphere: r = 1.5 Rs
• ISCO: r = 6 Rs
• Light Rays: Various impact parameters
• Observer: Current position
• Coordinate Grid: Spacetime coordinates

Camera: {self.elev_slider.value()}° elev, {self.azim_slider.value()}° azim"""

        self.info_3d_text.setPlainText(info)

    def init_light_rays(self):
        self.light_ray_lines.clear()
        self.cached_rays.clear()

        rs = self.rs_current
        impact_params = [rs * 2, rs * 3, rs * 5, rs * 10]

        for b in impact_params:
            traj = self.canvas_3d.calculate_light_trajectory(b, rs)
            if not traj:
                continue

            x, y = traj
            z = np.zeros_like(x)

            line, = self.canvas_3d.axes.plot(
            [], [], [],
            color='#00ffff',     # brighter cyan
            linewidth=3.5,       # thicker
            alpha=1.0
        )
            self.cached_rays[line] = (x, y, z)
            self.light_ray_lines.append(line)

    def on_animation_changed(self, state):
        """Handle animation toggle changes."""
        # Animation logic would go here
        pass

    def start_animation(self):
        interval = int(100 / self.animation_speed.value())  # ms
        self.anim_timer.start(interval)

    def stop_animation(self):
        self.anim_timer.stop()

    def reset_animation(self):
        self.stop_animation()
        self.camera_angle_anim = 0
        self.light_phase = 0
        self.set_preset_view(30, 45)
        self.update_visualization()

    def animation_step(self):
    # ---- CAMERA ROTATION ----
        if self.animate_camera.isChecked():
            self.camera_angle_anim += 1.0
            azim = self.camera_angle_anim % 360
            elev = self.elev_slider.value()
            self.canvas_3d.update_view(elev, azim)

        # ---- LIGHT RAY ANIMATION ----
        if self.animate_light_rays.isChecked():
            self.animate_light_rays_step()

    def animate_light_rays_step(self):
        phase = self.light_phase
        self.light_phase = (self.light_phase + 2) % 100

        for line, (x, y, z) in self.cached_rays.items():
            end = max(2, int(len(x) * phase / 100))
            line.set_data_3d(x[:end], y[:end], z[:end])

        self.canvas_3d.draw_idle()

TOOL_META_3D = {
    "name": "3D Schwarzschild Black Hole Simulator",
    "description": "Interactive 3D visualization of Schwarzschild black holes with spacetime curvature, light bending, and real-time exploration",
    "category": "Astronomy",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "3D event horizon visualization",
        "Photon sphere and ISCO rendering",
        "Light ray trajectory bending",
        "Interactive camera controls",
        "Observer position tracking",
        "Coordinate grid display",
        "Real-time parameter updates",
        "Animation capabilities",
        "Multiple viewing presets",
        "Spacetime curvature representation"
    ],
    "educational_value": "Explore general relativity in 3D space, understand gravitational lensing, and visualize black hole physics interactively",
    "keywords": ["black hole", "3D visualization", "Schwarzschild", "general relativity", "spacetime", "gravitational lensing", "light bending", "interactive"]
}


def create_tool(parent=None):
    """Create and return a Schwarzschild3DSimulator instance."""
    return Schwarzschild3DSimulator(parent)