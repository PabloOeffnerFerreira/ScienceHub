"""
Schwarzschild Black Hole Simulator
A comprehensive simulator for Schwarzschild black holes with redshift, time dilation, and horizon calculations.
"""

from sciencehub.ui.components.tool_base import ScienceHubTool
import sys
import math
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pyqtgraph as pg


class PhysicsConstants:
    """Physical constants used in calculations."""
    # Fundamental constants
    SPEED_OF_LIGHT = 299792458  # m/s
    GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/kg·s²
    PLANCK_CONSTANT = 6.62607015e-34  # J·s
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
    STEFAN_BOLTZMANN = 5.670367e-8  # W/m²·K⁴

    # Astronomical constants
    SOLAR_MASS = 1.989e30  # kg
    EARTH_MASS = 5.972e24  # kg
    JUPITER_MASS = 1.898e27  # kg

    # Conversion factors
    KM_TO_M = 1000
    AU_TO_M = 1.496e11
    PC_TO_M = 3.086e16
    LY_TO_M = 9.461e15


class SchwarzschildCalculator:
    """Core calculation engine for Schwarzschild black holes."""

    def __init__(self):
        self.c = PhysicsConstants.SPEED_OF_LIGHT
        self.G = PhysicsConstants.GRAVITATIONAL_CONSTANT
        self.constants = PhysicsConstants()

    def schwarzschild_radius(self, mass: float) -> float:
        """Calculate Schwarzschild radius (event horizon)."""
        return (2 * self.G * mass) / (self.c ** 2)

    def photon_sphere_radius(self, mass: float) -> float:
        """Calculate photon sphere radius (unstable photon orbits)."""
        rs = self.schwarzschild_radius(mass)
        return 1.5 * rs

    def isco_radius(self, mass: float) -> float:
        """Calculate Innermost Stable Circular Orbit radius."""
        rs = self.schwarzschild_radius(mass)
        return 6 * rs

    def gravitational_redshift(self, r: float, rs: float) -> float:
        """Calculate gravitational redshift factor."""
        if r <= rs:
            return float('inf')  # Inside event horizon
        return math.sqrt(1 - rs/r)

    def time_dilation_factor(self, r: float, rs: float) -> float:
        """Calculate time dilation factor for stationary observer."""
        if r <= rs:
            return float('inf')
        return math.sqrt(1 - rs/r)

    def orbital_velocity(self, r: float, rs: float) -> float:
        """Calculate orbital velocity at radius r."""
        if r <= rs:
            return 0
        return self.c * math.sqrt(rs / (2 * r))

    def escape_velocity(self, r: float, rs: float) -> float:
        """Calculate escape velocity at radius r."""
        if r <= rs:
            return 0
        return self.c * math.sqrt(rs / r)

    def hawking_temperature(self, mass: float) -> float:
        """Calculate Hawking temperature (simplified)."""
        rs = self.schwarzschild_radius(mass)
        return (self.constants.PLANCK_CONSTANT * self.c**3) / (8 * math.pi * self.G * mass * self.constants.BOLTZMANN_CONSTANT)

    def tidal_force(self, mass: float, r: float, object_size: float) -> float:
        """Calculate tidal acceleration across object."""
        rs = self.schwarzschild_radius(mass)
        if r <= rs:
            return float('inf')
        return (2 * self.G * mass * object_size) / (r**3)

    def effective_potential(self, r: float, rs: float, l: float = 0) -> float:
        """Calculate effective potential for radial motion."""
        if r <= rs:
            return float('inf')
        return -(self.G * rs / (2 * r)) + (l**2 * self.c**2) / (2 * r**2) - (self.G * rs * l**2) / (r**3)

    def light_bending_angle(self, r_min: float, rs: float) -> float:
        """Calculate light bending angle for closest approach r_min."""
        if r_min <= rs:
            return float('inf')
        return 2 * math.acos(1 / (1 + (2 * r_min * (r_min - rs)) / (rs * (r_min - rs))))

    def coordinate_time_to_proper_time(self, r: float, rs: float, coordinate_time: float) -> float:
        """Convert coordinate time to proper time for stationary observer."""
        time_factor = self.time_dilation_factor(r, rs)
        return coordinate_time / time_factor


class SchwarzschildBlackHoleSimulator(ScienceHubTool):
    """Schwarzschild Black Hole Simulator Tool Widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculator = SchwarzschildCalculator()
        self.constants = PhysicsConstants()

        # Initialize variables
        self.current_units = {
            'mass': 'M☉',
            'length': 'km',
            'time': 's',
            'temperature': 'K'
        }
        self.current_display_mode = 'basic'

        self.setup_ui()
        self.setup_plot()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)

        self.setup_input_panel(main_layout)
        self.setup_results_panel(main_layout)

        # Add advanced features controls
        self.setup_advanced_controls(main_layout)

        self.root_layout.addLayout(main_layout)

    def setup_input_panel(self, parent_layout):
        """Set up the input controls panel."""
        input_widget = QWidget()
        input_widget.setObjectName("toolCard")
        input_layout = QVBoxLayout(input_widget)
        input_layout.setSpacing(15)

        # Black hole properties
        self.setup_black_hole_properties(input_layout)

        # Observer/Object properties
        self.setup_observer_properties(input_layout)

        # Calculation parameters
        self.setup_calculation_parameters(input_layout)

        # Buttons
        self.setup_buttons(input_layout)

        parent_layout.addWidget(input_widget, 1)

    def setup_black_hole_properties(self, layout):
        """Set up black hole property inputs."""
        title = QLabel("Black Hole Properties")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        bh_widget = QWidget()
        bh_widget.setObjectName("toolCard")
        bh_layout = QFormLayout(bh_widget)
        bh_layout.setSpacing(10)

        # Black hole mass
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 1e12)
        self.mass_input.setValue(10.0)
        self.mass_input.setDecimals(2)
        self.mass_input.setToolTip("Mass of the black hole")
        self.mass_unit_combo = QComboBox()
        self.mass_unit_combo.addItems(['M☉', 'M⊕', 'MJ', 'kg'])
        self.mass_unit_combo.setCurrentText('M☉')
        self.mass_unit_combo.currentTextChanged.connect(self.update_units)

        mass_layout = QHBoxLayout()
        mass_layout.addWidget(self.mass_input)
        mass_layout.addWidget(self.mass_unit_combo)
        bh_layout.addRow("Black Hole Mass:", mass_layout)

        # Spin parameter (for future extension)
        self.spin_input = QDoubleSpinBox()
        self.spin_input.setRange(0.0, 0.999)
        self.spin_input.setValue(0.0)
        self.spin_input.setDecimals(3)
        self.spin_input.setToolTip("Spin parameter (0 = Schwarzschild, <1 = Kerr)")
        self.spin_input.setEnabled(False)  # Disabled for now
        bh_layout.addRow("Spin Parameter:", self.spin_input)

        layout.addWidget(bh_widget)

    def setup_observer_properties(self, layout):
        """Set up observer/object property inputs."""
        title = QLabel("Observer/Object Properties")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        obs_widget = QWidget()
        obs_widget.setObjectName("toolCard")
        obs_layout = QFormLayout(obs_widget)
        obs_layout.setSpacing(10)

        # Distance from black hole
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(1.0, 1e15)
        self.distance_input.setValue(100.0)
        self.distance_input.setDecimals(1)
        self.distance_input.setToolTip("Distance from black hole center")
        self.distance_unit_combo = QComboBox()
        self.distance_unit_combo.addItems(['km', 'Rs', 'AU', 'ly', 'pc'])
        self.distance_unit_combo.setCurrentText('km')
        self.distance_unit_combo.currentTextChanged.connect(self.update_units)

        distance_layout = QHBoxLayout()
        distance_layout.addWidget(self.distance_input)
        distance_layout.addWidget(self.distance_unit_combo)
        obs_layout.addRow("Distance:", distance_layout)

        # Object type
        self.object_combo = QComboBox()
        self.object_combo.addItems([
            "Stationary Observer",
            "Orbiting Object",
            "Light Ray",
            "Spaceship",
            "Planet",
            "Star"
        ])
        self.object_combo.currentTextChanged.connect(self.on_object_changed)
        obs_layout.addRow("Object Type:", self.object_combo)

        # Orbital parameters (shown when orbiting)
        self.orbital_widget = QWidget()
        orbital_layout = QFormLayout(self.orbital_widget)

        self.orbital_velocity_input = QDoubleSpinBox()
        self.orbital_velocity_input.setRange(0.0, 1.0)
        self.orbital_velocity_input.setValue(0.5)
        self.orbital_velocity_input.setDecimals(3)
        self.orbital_velocity_input.setSuffix(" c")
        orbital_layout.addRow("Orbital Velocity:", self.orbital_velocity_input)

        self.orbital_widget.setVisible(False)
        obs_layout.addRow(self.orbital_widget)

        layout.addWidget(obs_widget)

    def setup_calculation_parameters(self, layout):
        """Set up calculation parameter inputs."""
        title = QLabel("Calculation Parameters")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        calc_widget = QWidget()
        calc_widget.setObjectName("toolCard")
        calc_layout = QFormLayout(calc_widget)
        calc_layout.setSpacing(10)

        # Observer location
        self.observer_combo = QComboBox()
        self.observer_combo.addItems([
            "Distant Observer (∞)",
            "Near Black Hole",
            "ISCO Orbit",
            "Photon Sphere",
            "Event Horizon"
        ])
        self.observer_combo.currentTextChanged.connect(self.on_observer_changed)
        calc_layout.addRow("Observer Location:", self.observer_combo)

        # Calculation type
        self.calculation_combo = QComboBox()
        self.calculation_combo.addItems([
            "Redshift & Time Dilation",
            "Orbital Mechanics",
            "Light Bending",
            "Tidal Forces",
            "Hawking Radiation"
        ])
        self.calculation_combo.currentTextChanged.connect(self.on_calculation_changed)
        calc_layout.addRow("Calculation Type:", self.calculation_combo)

        layout.addWidget(calc_widget)

    def setup_buttons(self, layout):
        """Set up control buttons."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.setObjectName("primaryButton")
        self.calculate_btn.clicked.connect(self.calculate)
        self.calculate_btn.setToolTip("Calculate black hole properties and effects")

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("secondaryButton")
        self.reset_btn.clicked.connect(self.reset_inputs)
        self.reset_btn.setToolTip("Reset all inputs to default values")

        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("secondaryButton")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setToolTip("Save current configuration to file")

        self.load_btn = QPushButton("Load")
        self.load_btn.setObjectName("secondaryButton")
        self.load_btn.clicked.connect(self.load_config)
        self.load_btn.setToolTip("Load configuration from file")

        button_layout.addWidget(self.calculate_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.load_btn)

        layout.addLayout(button_layout)

    def setup_advanced_controls(self, parent_layout):
        """Set up advanced features controls panel."""
        advanced_widget = QWidget()
        advanced_widget.setObjectName("toolCard")
        advanced_layout = QVBoxLayout(advanced_widget)
        advanced_layout.setSpacing(15)

        # Live update mode
        self.live_update_checkbox = QCheckBox("Live Update Mode")
        self.live_update_checkbox.setObjectName("controlCheckbox")
        self.live_update_checkbox.setToolTip("Automatically recalculate when inputs change")
        self.live_update_checkbox.stateChanged.connect(self.on_live_update_changed)
        advanced_layout.addWidget(self.live_update_checkbox)

        # Display mode buttons
        display_group = QGroupBox("Display Options")
        display_group.setObjectName("toolCard")
        display_layout = QVBoxLayout(display_group)

        self.basic_results_btn = QPushButton("Basic Results")
        self.basic_results_btn.setObjectName("secondaryButton")
        self.basic_results_btn.setCheckable(True)
        self.basic_results_btn.setChecked(True)
        self.basic_results_btn.clicked.connect(lambda: self.set_display_mode("basic"))
        display_layout.addWidget(self.basic_results_btn)

        self.detailed_physics_btn = QPushButton("Detailed Physics")
        self.detailed_physics_btn.setObjectName("secondaryButton")
        self.detailed_physics_btn.setCheckable(True)
        self.detailed_physics_btn.clicked.connect(lambda: self.set_display_mode("detailed"))
        display_layout.addWidget(self.detailed_physics_btn)

        self.visualization_btn = QPushButton("Visualization")
        self.visualization_btn.setObjectName("secondaryButton")
        self.visualization_btn.setCheckable(True)
        self.visualization_btn.clicked.connect(lambda: self.set_display_mode("visualization"))
        display_layout.addWidget(self.visualization_btn)

        advanced_layout.addWidget(display_group)

        # Preset scenarios
        preset_group = QGroupBox("Preset Black Holes")
        preset_group.setObjectName("toolCard")
        preset_layout = QVBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Stellar Black Hole (10 M☉)",
            "Sagittarius A* (4.3M M☉)",
            "M87* (6.5B M☉)",
            "TON 618 (70B M☉)",
            "Micro Black Hole (1 M⊕)",
            "Primordial Black Hole (10^15 kg)"
        ])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)

        advanced_layout.addWidget(preset_group)

        # Plot options
        plot_group = QGroupBox("Plot Options")
        plot_group.setObjectName("toolCard")
        plot_layout = QVBoxLayout(plot_group)

        self.plot_redshift_checkbox = QCheckBox("Show Redshift Profile")
        self.plot_redshift_checkbox.setObjectName("controlCheckbox")
        self.plot_redshift_checkbox.setChecked(True)
        plot_layout.addWidget(self.plot_redshift_checkbox)

        self.plot_potential_checkbox = QCheckBox("Show Effective Potential")
        self.plot_potential_checkbox.setObjectName("controlCheckbox")
        plot_layout.addWidget(self.plot_potential_checkbox)

        self.plot_light_bending_checkbox = QCheckBox("Show Light Bending")
        self.plot_light_bending_checkbox.setObjectName("controlCheckbox")
        plot_layout.addWidget(self.plot_light_bending_checkbox)

        advanced_layout.addWidget(plot_group)

        # Export plot button
        export_layout = QHBoxLayout()
        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.setObjectName("secondaryButton")
        self.export_plot_btn.clicked.connect(self.export_plot)
        self.export_plot_btn.setEnabled(False)
        export_layout.addWidget(self.export_plot_btn)
        export_layout.addStretch()
        advanced_layout.addLayout(export_layout)

        parent_layout.addWidget(advanced_widget, 1)

    def setup_results_panel(self, parent_layout):
        """Set up the results and plotting panel."""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setSpacing(15)

        # Results display
        self.setup_results_display(results_layout)

        # Plot widget
        self.setup_plot_widget(results_layout)

        parent_layout.addWidget(results_widget, 2)

    def setup_results_display(self, layout):
        """Set up the results display area."""
        title = QLabel("Black Hole Properties & Effects")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        results_widget = QWidget()
        results_widget.setObjectName("toolCard")
        results_layout = QGridLayout(results_widget)
        results_layout.setSpacing(10)

        # Key radii
        results_layout.addWidget(QLabel("Schwarzschild Radius:"), 0, 0)
        self.rs_result = QLabel("0.00 km")
        self.rs_result.setObjectName("resultValue")
        results_layout.addWidget(self.rs_result, 0, 1)

        results_layout.addWidget(QLabel("Photon Sphere:"), 1, 0)
        self.photon_sphere_result = QLabel("0.00 km")
        results_layout.addWidget(self.photon_sphere_result, 1, 1)

        results_layout.addWidget(QLabel("ISCO Radius:"), 2, 0)
        self.isco_result = QLabel("0.00 km")
        results_layout.addWidget(self.isco_result, 2, 1)

        # Current effects
        results_layout.addWidget(QLabel("Gravitational Redshift:"), 3, 0)
        self.redshift_result = QLabel("1.000")
        results_layout.addWidget(self.redshift_result, 3, 1)

        results_layout.addWidget(QLabel("Time Dilation:"), 4, 0)
        self.time_dilation_result = QLabel("1.000")
        results_layout.addWidget(self.time_dilation_result, 4, 1)

        # Additional info
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Click 'Calculate' to see black hole properties...")
        self.info_text.setObjectName("infoText")

        layout.addWidget(results_widget)
        layout.addWidget(QLabel("Detailed Analysis:"))
        layout.addWidget(self.info_text)

    def setup_plot_widget(self, layout):
        """Set up the plotting widget."""
        title = QLabel("Visualization")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        plot_container = QWidget()
        plot_container.setObjectName("toolPlot")
        plot_layout = QVBoxLayout(plot_container)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")

        # Style the axes for dark theme
        self.plot_widget.getAxis('left').setPen('w')
        self.plot_widget.getAxis('bottom').setPen('w')
        self.plot_widget.getAxis('left').setTextPen('w')
        self.plot_widget.getAxis('bottom').setTextPen('w')

        self.plot_widget.setTitle("Black Hole Visualization", color='w', size='14pt')
        self.plot_widget.setLabel('left', 'Value', color='w')
        self.plot_widget.setLabel('bottom', 'Radius (Rs)', color='w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        plot_layout.addWidget(self.plot_widget)
        layout.addWidget(plot_container)

    def setup_plot(self):
        """Initialize the plot."""
        self.redshift_curve = self.plot_widget.plot(pen=pg.mkPen(color='#ff6b6b', width=2))
        self.potential_curve = self.plot_widget.plot(pen=pg.mkPen(color='#4ecdc4', width=2))
        self.light_bending_curve = self.plot_widget.plot(pen=pg.mkPen(color='#45b7d1', width=2))

    # Event handlers
    def on_object_changed(self, object_type):
        """Handle object type selection."""
        self.orbital_widget.setVisible(object_type == "Orbiting Object")
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_observer_changed(self, observer_location):
        """Handle observer location selection."""
        # Auto-set distance based on observer location
        mass = self.get_mass_kg()
        rs = self.calculator.schwarzschild_radius(mass)

        if observer_location == "Event Horizon":
            distance = rs * 1.01  # Just outside horizon
        elif observer_location == "Photon Sphere":
            distance = self.calculator.photon_sphere_radius(mass)
        elif observer_location == "ISCO Orbit":
            distance = self.calculator.isco_radius(mass)
        elif observer_location == "Near Black Hole":
            distance = rs * 10
        else:  # Distant Observer
            distance = rs * 1000

        # Convert to km for display
        distance_km = distance / 1000
        self.distance_input.setValue(distance_km)
        self.distance_unit_combo.setCurrentText('km')

        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_calculation_changed(self, calculation_type):
        """Handle calculation type selection."""
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_preset_changed(self, preset):
        """Handle preset black hole selection."""
        if preset == "Custom":
            return

        presets = {
            "Stellar Black Hole (10 M☉)": {"mass": 10.0, "unit": "M☉"},
            "Sagittarius A* (4.3M M☉)": {"mass": 4.3e6, "unit": "M☉"},
            "M87* (6.5B M☉)": {"mass": 6.5e9, "unit": "M☉"},
            "TON 618 (70B M☉)": {"mass": 7e10, "unit": "M☉"},
            "Micro Black Hole (1 M⊕)": {"mass": 1.0, "unit": "M⊕"},
            "Primordial Black Hole (10^15 kg)": {"mass": 1e15, "unit": "kg"}
        }

        if preset in presets:
            data = presets[preset]
            self.mass_input.setValue(data["mass"])
            self.mass_unit_combo.setCurrentText(data["unit"])

            if self.live_update_checkbox.isChecked():
                self.calculate()

    def get_mass_kg(self):
        """Get black hole mass in kg."""
        mass = self.mass_input.value()
        unit = self.mass_unit_combo.currentText()

        if unit == 'M☉':
            return mass * self.constants.SOLAR_MASS
        elif unit == 'M⊕':
            return mass * self.constants.EARTH_MASS
        elif unit == 'MJ':
            return mass * self.constants.JUPITER_MASS
        else:  # kg
            return mass

    def get_distance_m(self):
        """Get distance in meters."""
        distance = self.distance_input.value()
        unit = self.distance_unit_combo.currentText()

        if unit == 'km':
            return distance * 1000
        elif unit == 'Rs':
            mass = self.get_mass_kg()
            rs = self.calculator.schwarzschild_radius(mass)
            return distance * rs
        elif unit == 'AU':
            return distance * self.constants.AU_TO_M
        elif unit == 'ly':
            return distance * self.constants.LY_TO_M
        elif unit == 'pc':
            return distance * self.constants.PC_TO_M
        else:
            return distance

    def calculate(self):
        """Perform the black hole calculations."""
        try:
            # Get inputs
            mass_kg = self.get_mass_kg()
            distance_m = self.get_distance_m()
            object_type = self.object_combo.currentText()
            calculation_type = self.calculation_combo.currentText()

            # Calculate basic properties
            rs = self.calculator.schwarzschild_radius(mass_kg)
            photon_sphere = self.calculator.photon_sphere_radius(mass_kg)
            isco = self.calculator.isco_radius(mass_kg)

            # Update basic results
            self.rs_result.setText(f"{rs/1000:.2f} km")
            self.photon_sphere_result.setText(f"{photon_sphere/1000:.2f} km")
            self.isco_result.setText(f"{isco/1000:.2f} km")

            # Calculate effects at current distance
            redshift = self.calculator.gravitational_redshift(distance_m, rs)
            time_dilation = self.calculator.time_dilation_factor(distance_m, rs)

            if redshift == float('inf'):
                self.redshift_result.setText("∞ (inside horizon)")
                self.time_dilation_result.setText("∞ (inside horizon)")
            else:
                self.redshift_result.setText(f"{redshift:.6f}")
                self.time_dilation_result.setText(f"{time_dilation:.6f}")

            # Update detailed info
            self.update_info_text(mass_kg, distance_m, rs, object_type, calculation_type)

            # Update plot
            self.update_plot(mass_kg, rs, distance_m)

            self.export_plot_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", f"An error occurred during calculation:\n{str(e)}")

    def update_info_text(self, mass_kg, distance_m, rs, object_type, calculation_type):
        """Update the detailed analysis text."""
        display_mode = getattr(self, 'current_display_mode', 'basic')

        if display_mode == "basic":
            info = f"""Black Hole Analysis:

Mass: {mass_kg/self.constants.SOLAR_MASS:.2e} M☉
Distance: {distance_m/1000:.2f} km ({distance_m/rs:.1f} Rs)

Key Radii:
• Event Horizon: {rs/1000:.2f} km
• Photon Sphere: {1.5*rs/1000:.2f} km
• ISCO: {6*rs/1000:.2f} km

At current distance:
• Gravitational Redshift: {self.calculator.gravitational_redshift(distance_m, rs):.6f}
• Time Dilation: {self.calculator.time_dilation_factor(distance_m, rs):.6f}
• Escape Velocity: {self.calculator.escape_velocity(distance_m, rs)/self.constants.SPEED_OF_LIGHT:.3f} c"""

        elif display_mode == "detailed":
            hawking_temp = self.calculator.hawking_temperature(mass_kg)
            tidal_force = self.calculator.tidal_force(mass_kg, distance_m, 1.0)  # 1m object

            info = f"""Advanced Black Hole Physics:

MASS & DIMENSIONS:
Mass: {mass_kg/self.constants.SOLAR_MASS:.2e} M☉ ({mass_kg:.2e} kg)
Schwarzschild Radius: {rs:.2e} m ({rs/1000:.2f} km)
Surface Area: {4*math.pi*rs**2:.2e} m²
Volume: {(4/3)*math.pi*rs**3:.2e} m³

GRAVITATIONAL EFFECTS:
Current Distance: {distance_m:.2e} m ({distance_m/rs:.1f} Rs)
Gravitational Redshift: {self.calculator.gravitational_redshift(distance_m, rs):.6f}
Time Dilation Factor: {self.calculator.time_dilation_factor(distance_m, rs):.6f}
Escape Velocity: {self.calculator.escape_velocity(distance_m, rs)/1000:.2f} km/s ({self.calculator.escape_velocity(distance_m, rs)/self.constants.SPEED_OF_LIGHT:.3f} c)

QUANTUM EFFECTS:
Hawking Temperature: {hawking_temp:.2e} K
Tidal Acceleration (1m object): {tidal_force:.2e} m/s²

ORBITAL MECHANICS:
ISCO Radius: {6*rs/1000:.2f} km
Photon Sphere: {1.5*rs/1000:.2f} km
Orbital Velocity at ISCO: {self.calculator.orbital_velocity(6*rs, rs)/1000:.2f} km/s"""

        elif display_mode == "visualization":
            info = f"""Visualization Data:

Black Hole Mass: {mass_kg/self.constants.SOLAR_MASS:.2e} M☉
Schwarzschild Radius: {rs/1000:.2f} km

Current Position: {distance_m/rs:.1f} Rs ({distance_m/1000:.2f} km)

PLOTTED QUANTITIES:
• Gravitational Redshift vs Radius
• Effective Potential for Orbital Motion
• Light Deflection Angles

INTERPRETATION:
- Redshift increases dramatically near the event horizon
- Effective potential shows stable and unstable orbits
- Light bending becomes extreme close to the photon sphere"""

        else:
            info = "Unknown display mode"

        self.info_text.setPlainText(info)

    def update_plot(self, mass_kg, rs, current_distance):
        """Update the visualization plot."""
        # Clear existing plots
        self.redshift_curve.setData([], [])
        self.potential_curve.setData([], [])
        self.light_bending_curve.setData([], [])

        # Create radius array (from just outside horizon to 100 Rs)
        r_min = rs * 1.01
        r_max = rs * 100
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 1000)
        radii_rs = radii / rs

        # Plot redshift if enabled
        if self.plot_redshift_checkbox.isChecked():
            redshifts = [self.calculator.gravitational_redshift(r, rs) for r in radii]
            # Convert to percentage redshift
            redshift_percent = [(1/z - 1) * 100 if z > 0 else 0 for z in redshifts]
            self.redshift_curve.setData(radii_rs, redshift_percent)

        # Plot effective potential if enabled
        if self.plot_potential_checkbox.isChecked():
            # Use angular momentum corresponding to ISCO
            l_isco = math.sqrt(12) * rs * self.constants.SPEED_OF_LIGHT
            potentials = [self.calculator.effective_potential(r, rs, l_isco) for r in radii]
            # Normalize to make it visible
            max_pot = max(abs(p) for p in potentials if not math.isinf(p))
            if max_pot > 0:
                potentials_normalized = [p / max_pot if not math.isinf(p) else 0 for p in potentials]
                self.potential_curve.setData(radii_rs, potentials_normalized)

        # Plot light bending if enabled
        if self.plot_light_bending_checkbox.isChecked():
            # Calculate deflection angles for different impact parameters
            impact_params = np.linspace(rs * 1.1, rs * 10, 50)
            deflection_angles = []
            valid_params = []

            for b in impact_params:
                try:
                    angle = self.calculator.light_bending_angle(b, rs)
                    if not math.isinf(angle) and not math.isnan(angle):
                        deflection_angles.append(angle * 180 / math.pi)  # Convert to degrees
                        valid_params.append(b / rs)
                except:
                    continue

            if deflection_angles:
                self.light_bending_curve.setData(valid_params, deflection_angles)

        # Update plot labels based on what's shown
        if self.plot_redshift_checkbox.isChecked():
            self.plot_widget.setLabel('left', 'Gravitational Redshift (%)', color='w')
        elif self.plot_potential_checkbox.isChecked():
            self.plot_widget.setLabel('left', 'Effective Potential (normalized)', color='w')
        elif self.plot_light_bending_checkbox.isChecked():
            self.plot_widget.setLabel('left', 'Light Deflection (°)', color='w')

    def update_units(self):
        """Update unit conversions."""
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_live_update_changed(self, state):
        """Handle live update mode toggle."""
        signals = [
            self.mass_input.valueChanged,
            self.distance_input.valueChanged,
            self.mass_unit_combo.currentTextChanged,
            self.distance_unit_combo.currentTextChanged,
            self.object_combo.currentTextChanged,
            self.observer_combo.currentTextChanged,
            self.calculation_combo.currentTextChanged,
        ]

        if state == Qt.CheckState.Checked:
            for sig in signals:
                sig.connect(self.calculate)
        else:
            for sig in signals:
                try:
                    sig.disconnect(self.calculate)
                except TypeError:
                    pass  # already disconnected

    def set_display_mode(self, mode):
        """Set the display mode for results."""
        # Uncheck all buttons
        self.basic_results_btn.setChecked(False)
        self.detailed_physics_btn.setChecked(False)
        self.visualization_btn.setChecked(False)

        # Check the selected button
        if mode == "basic":
            self.basic_results_btn.setChecked(True)
        elif mode == "detailed":
            self.detailed_physics_btn.setChecked(True)
        elif mode == "visualization":
            self.visualization_btn.setChecked(True)

        # Update display
        self.current_display_mode = mode
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def reset_inputs(self):
        """Reset all inputs to default values."""
        self.mass_input.setValue(10.0)
        self.mass_unit_combo.setCurrentText('M☉')
        self.distance_input.setValue(100.0)
        self.distance_unit_combo.setCurrentText('km')
        self.spin_input.setValue(0.0)
        self.object_combo.setCurrentText("Stationary Observer")
        self.observer_combo.setCurrentText("Distant Observer (∞)")
        self.calculation_combo.setCurrentText("Redshift & Time Dilation")
        self.preset_combo.setCurrentText("Custom")

        # Clear results
        self.rs_result.setText("0.00 km")
        self.photon_sphere_result.setText("0.00 km")
        self.isco_result.setText("0.00 km")
        self.redshift_result.setText("1.000")
        self.time_dilation_result.setText("1.000")
        self.info_text.setPlainText("Click 'Calculate' to see black hole properties...")

        # Clear plot
        self.redshift_curve.setData([], [])
        self.potential_curve.setData([], [])
        self.light_bending_curve.setData([], [])

    def save_config(self):
        """Save current configuration to file."""
        config = {
            'mass': self.mass_input.value(),
            'mass_unit': self.mass_unit_combo.currentText(),
            'distance': self.distance_input.value(),
            'distance_unit': self.distance_unit_combo.currentText(),
            'spin': self.spin_input.value(),
            'object_type': self.object_combo.currentText(),
            'observer_location': self.observer_combo.currentText(),
            'calculation_type': self.calculation_combo.currentText(),
            'preset': self.preset_combo.currentText(),
            'display_mode': self.current_display_mode,
            'plot_redshift': self.plot_redshift_checkbox.isChecked(),
            'plot_potential': self.plot_potential_checkbox.isChecked(),
            'plot_light_bending': self.plot_light_bending_checkbox.isChecked()
        }

        filename, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON files (*.json)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def load_config(self):
        """Load configuration from file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)

                # Apply loaded values
                self.mass_input.setValue(config.get('mass', 10.0))
                self.mass_unit_combo.setCurrentText(config.get('mass_unit', 'M☉'))
                self.distance_input.setValue(config.get('distance', 100.0))
                self.distance_unit_combo.setCurrentText(config.get('distance_unit', 'km'))
                self.spin_input.setValue(config.get('spin', 0.0))
                self.object_combo.setCurrentText(config.get('object_type', 'Stationary Observer'))
                self.observer_combo.setCurrentText(config.get('observer_location', 'Distant Observer (∞)'))
                self.calculation_combo.setCurrentText(config.get('calculation_type', 'Redshift & Time Dilation'))
                self.preset_combo.setCurrentText(config.get('preset', 'Custom'))

                # Load display settings
                display_mode = config.get('display_mode', 'basic')
                self.set_display_mode(display_mode)

                self.plot_redshift_checkbox.setChecked(config.get('plot_redshift', True))
                self.plot_potential_checkbox.setChecked(config.get('plot_potential', False))
                self.plot_light_bending_checkbox.setChecked(config.get('plot_light_bending', False))

                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def export_plot(self):
        """Export the current plot to image file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG files (*.png);;SVG files (*.svg)"
        )
        if filename:
            try:
                if filename.endswith('.png'):
                    exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
                    exporter.export(filename)
                elif filename.endswith('.svg'):
                    exporter = pg.exporters.SVGExporter(self.plot_widget.plotItem)
                    exporter.export(filename)
                QMessageBox.information(self, "Success", "Plot exported successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export plot:\n{str(e)}")


TOOL_META = {
    "name": "Schwarzschild Black Hole Simulator",
    "description": "Advanced simulator for Schwarzschild black holes with redshift, time dilation, horizons, and comprehensive visualizations",
    "category": "Astronomy",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "Schwarzschild radius calculations",
        "Gravitational redshift analysis",
        "Time dilation effects",
        "Event horizon and photon sphere",
        "Innermost stable circular orbit (ISCO)",
        "Orbital mechanics",
        "Light bending simulations",
        "Hawking radiation temperature",
        "Tidal force calculations",
        "Effective potential plots",
        "Multiple display modes",
        "Preset black hole scenarios",
        "Interactive visualizations",
        "Live update mode",
        "Plot export functionality"
    ],
    "educational_value": "Learn about general relativity, black hole physics, gravitational effects, and spacetime curvature",
    "keywords": ["black hole", "Schwarzschild", "general relativity", "gravitational redshift", "time dilation", "event horizon", "photon sphere", "ISCO"]
}


def create_tool(parent=None):
    """Create and return a SchwarzschildBlackHoleSimulator instance."""
    return SchwarzschildBlackHoleSimulator(parent)
