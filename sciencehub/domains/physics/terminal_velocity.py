"""
Terminal Velocity Calculator
A comprehensive physics calculator for terminal velocity with advanced GUI features.
"""

TOOL_META = {
    "name": "Terminal Velocity",
    "domain": "Physics",
    "icon": "📉",
    "description": "Advanced terminal velocity simulator with fluid dynamics",
    "difficulty": "Intermediate"
}

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
    G_EARTH = 9.81  # m/s²
    G_MOON = 1.62   # m/s²
    G_MARS = 3.71   # m/s²

    # Fluid densities (kg/m³)
    DENSITY_AIR_SEA_LEVEL = 1.225
    DENSITY_AIR_10000M = 0.4135
    DENSITY_WATER = 1000.0
    DENSITY_SEAWATER = 1025.0
    DENSITY_MERCURY = 13534.0
    DENSITY_OIL = 800.0

    # Dynamic viscosities (Pa·s)
    VISCOSITY_AIR = 1.81e-5
    VISCOSITY_WATER = 1.002e-3
    VISCOSITY_SEAWATER = 1.05e-3
    VISCOSITY_MERCURY = 1.526e-3
    VISCOSITY_OIL = 0.1

    # Drag coefficients for common shapes
    DRAG_SPHERE = 0.47
    DRAG_CUBE = 1.05
    DRAG_CYLINDER = 0.82
    DRAG_STREAMLINED = 0.04
    DRAG_PARACHUTE = 1.4

    # Reynolds number thresholds
    RE_LAMINAR_MAX = 2000
    RE_TRANSITIONAL_MAX = 4000


class UnitConverter:
    """Handles unit conversions for the calculator."""

    @staticmethod
    def mass_units() -> Dict[str, float]:
        return {
            "kg": 1.0,
            "g": 0.001,
            "lb": 0.453592,
            "oz": 0.0283495,
            "ton (metric)": 1000.0,
            "ton (US)": 907.185,
            "stone": 6.350294
        }

    @staticmethod
    def length_units() -> Dict[str, float]:
        return {
            "m": 1.0,
            "cm": 0.01,
            "mm": 0.001,
            "ft": 0.3048,
            "in": 0.0254,
            "yd": 0.9144,
            "km": 1000.0,
            "mi": 1609.34
        }

    @staticmethod
    def area_units() -> Dict[str, float]:
        return {
            "m²": 1.0,
            "cm²": 0.0001,
            "mm²": 1e-6,
            "ft²": 0.092903,
            "in²": 0.00064516,
            "yd²": 0.836127,
            "km²": 1e6,
            "mi²": 2.59e6
        }

    @staticmethod
    def velocity_units() -> Dict[str, float]:
        return {
            "m/s": 1.0,
            "km/h": 1/3.6,
            "mph": 0.44704,
            "ft/s": 0.3048,
            "knots": 0.514444,
            "in/s": 0.0254,
            "ft/min": 0.00508,
            "cm/s": 0.01,
            "mm/s": 0.001,
        }

    @staticmethod
    def density_units() -> Dict[str, float]:
        return {
            "kg/m³": 1.0,
            "g/cm³": 1000.0,
            "lb/ft³": 16.0185,
            "oz/in³": 1729.99404386,
            "ton (US)/yd³": 1198.29,
            "ton (metric)/m³": 1000.0,
            "slug/ft³": 515.378818
        }


class TerminalVelocityCalculator:
    """Core calculation engine for terminal velocity."""

    def __init__(self):
        self.constants = PhysicsConstants()

    def calculate_terminal_velocity(self, mass: float, area: float, drag_coeff: float,
                                  fluid_density: float, gravity: float) -> float:
        """
        Calculate terminal velocity using v_t = sqrt(2mg / (ρAC_d))
        """
        try:
            numerator = 2 * mass * gravity
            denominator = fluid_density * area * drag_coeff
            return math.sqrt(numerator / denominator)
        except (ZeroDivisionError, ValueError):
            return 0.0

    def calculate_velocity_vs_time(self, mass: float, area: float, drag_coeff: float,
                                 fluid_density: float, gravity: float,
                                 time_points: np.ndarray) -> np.ndarray:
        """
        Calculate velocity as a function of time during free fall.
        """
        try:
            # v(t) = v_t * tanh(gt/v_t)
            v_t = self.calculate_terminal_velocity(mass, area, drag_coeff, fluid_density, gravity)
            if v_t == 0:
                return np.zeros_like(time_points)

            gt_vt = gravity * time_points / v_t
            return v_t * np.tanh(gt_vt)
        except:
            return np.zeros_like(time_points)

    def calculate_distance_vs_time(self, mass: float, area: float, drag_coeff: float,
                                 fluid_density: float, gravity: float,
                                 time_points: np.ndarray) -> np.ndarray:
        """
        Calculate distance fallen as a function of time.
        """
        try:
            v_t = self.calculate_terminal_velocity(mass, area, drag_coeff, fluid_density, gravity)
            if v_t == 0:
                return 0.5 * gravity * time_points**2

            # d(t) = (v_t²/g) * ln(cosh(gt/v_t))
            gt_vt = gravity * time_points / v_t
            return (v_t**2 / gravity) * np.log(np.cosh(gt_vt))
        except:
            return 0.5 * gravity * time_points**2


class TerminalVelocityTool(ScienceHubTool):
    """Terminal Velocity Calculator Tool Widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calculator = TerminalVelocityCalculator()
        self.converter = UnitConverter()

        # Initialize variables
        self.current_units = {
            'mass': 'kg',
            'length': 'm',
            'area': 'm²',
            'velocity': 'm/s',
            'density': 'kg/m³'
        }
        self.current_display_mode = 'basic'  # Initialize display mode

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

        # Object properties
        self.setup_object_properties(input_layout)

        # Environmental properties
        self.setup_environment_properties(input_layout)

        # Shape selection
        self.setup_shape_selection(input_layout)

        # Buttons
        self.setup_buttons(input_layout)

        parent_layout.addWidget(input_widget, 1)

    def setup_object_properties(self, layout):
        """Set up object property inputs."""
        # Section title
        title = QLabel("Object Properties")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        obj_widget = QWidget()
        obj_widget.setObjectName("toolCard")
        obj_layout = QFormLayout(obj_widget)
        obj_layout.setSpacing(10)

        # Mass input
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.001, 1000000)
        self.mass_input.setValue(1.0)
        self.mass_input.setDecimals(3)
        self.mass_input.setToolTip("Mass of the falling object in selected units")
        self.mass_unit_combo = QComboBox()
        self.mass_unit_combo.addItems(list(self.converter.mass_units().keys()))
        self.mass_unit_combo.currentTextChanged.connect(self.update_units)

        mass_layout = QHBoxLayout()
        mass_layout.addWidget(self.mass_input)
        mass_layout.addWidget(self.mass_unit_combo)
        obj_layout.addRow("Mass:", mass_layout)

        # Cross-sectional area
        self.area_input = QDoubleSpinBox()
        self.area_input.setRange(0.000001, 10000)
        self.area_input.setValue(0.01)
        self.area_input.setDecimals(6)
        self.area_input.setToolTip("Cross-sectional area perpendicular to motion")
        self.area_unit_combo = QComboBox()
        self.area_unit_combo.addItems(list(self.converter.area_units().keys()))
        self.area_unit_combo.currentTextChanged.connect(self.update_units)

        area_layout = QHBoxLayout()
        area_layout.addWidget(self.area_input)
        area_layout.addWidget(self.area_unit_combo)
        obj_layout.addRow("Cross-sectional Area:", area_layout)

        # Drag coefficient
        self.drag_input = QDoubleSpinBox()
        self.drag_input.setRange(0.01, 10.0)
        self.drag_input.setValue(0.47)
        self.drag_input.setSingleStep(0.01)
        self.drag_input.setDecimals(3)
        self.drag_input.setToolTip("Drag coefficient (Cd) - dimensionless measure of aerodynamic drag")
        obj_layout.addRow("Drag Coefficient:", self.drag_input)

        layout.addWidget(obj_widget)

    def setup_environment_properties(self, layout):
        """Set up environmental property inputs."""
        # Section title
        title = QLabel("Environmental Properties")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        env_widget = QWidget()
        env_widget.setObjectName("toolCard")
        env_layout = QFormLayout(env_widget)
        env_layout.setSpacing(10)

        # Gravity
        self.gravity_combo = QComboBox()
        self.gravity_combo.addItems(["Earth (9.81 m/s²)", "Moon (1.62 m/s²)", "Mars (3.71 m/s²)", "Custom"])
        self.gravity_combo.currentTextChanged.connect(self.on_gravity_changed)
        self.gravity_combo.setToolTip("Gravitational acceleration for different celestial bodies")

        self.gravity_input = QDoubleSpinBox()
        self.gravity_input.setRange(0.1, 50.0)
        self.gravity_input.setValue(9.81)
        self.gravity_input.setDecimals(2)
        self.gravity_input.setEnabled(False)
        self.gravity_input.setToolTip("Custom gravitational acceleration in m/s²")

        gravity_layout = QHBoxLayout()
        gravity_layout.addWidget(self.gravity_combo)
        gravity_layout.addWidget(self.gravity_input)
        env_layout.addRow("Gravity:", gravity_layout)

        # Fluid density
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems([
            "Air (Sea Level)", "Air (10,000m)", "Water", "Seawater", "Mercury", "Oil", "Custom"
        ])
        self.fluid_combo.currentTextChanged.connect(self.on_fluid_changed)
        self.fluid_combo.setToolTip("Predefined fluid densities or custom value")

        self.fluid_density_input = QDoubleSpinBox()
        self.fluid_density_input.setRange(0.1, 20000)
        self.fluid_density_input.setValue(1.225)
        self.fluid_density_input.setDecimals(3)
        self.fluid_density_input.setToolTip("Fluid density in selected units")
        self.fluid_density_unit_combo = QComboBox()
        self.fluid_density_unit_combo.addItems(list(self.converter.density_units().keys()))
        self.fluid_density_unit_combo.currentTextChanged.connect(self.update_units)

        fluid_layout = QHBoxLayout()
        fluid_layout.addWidget(self.fluid_combo)
        fluid_layout.addWidget(self.fluid_density_input)
        fluid_layout.addWidget(self.fluid_density_unit_combo)
        env_layout.addRow("Fluid Density:", fluid_layout)

        layout.addWidget(env_widget)

    def setup_shape_selection(self, layout):
        """Set up shape selection with presets."""
        # Section title
        title = QLabel("Shape Presets")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        shape_widget = QWidget()
        shape_widget.setObjectName("toolCard")
        shape_layout = QVBoxLayout(shape_widget)
        shape_layout.setSpacing(10)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems([
            "Custom", "Sphere", "Cube", "Cylinder", "Streamlined Body", "Parachute",
            "Flat Plate", "Half Sphere", "Cone", "Teardrop", "Disc", "Square Plate",
            "Ellipsoid", "Hemisphere", "Circular Disc", "Wedge", "Pyramid", "Prism",
            "Rocket", "Arrow", "Bullet", "Feather", "Shuttlecock", "Baseball",
            "Basketball", "Tennis Ball", "Golf Ball", "Bowling Ball", "Frisbee", "Plate",
            "Airfoil", "Bluff Body", "Square Rod", "Round Rod", "I-Beam", "T-Shape",
            "L-Shape", "Boomerang", "Kite", "Sail", "Sphere with Dimples", "Smooth Sphere",
            "Roughened Sphere", "Cube (Smooth)", "Cube (Rough)", "Triangular Prism",
            "Hexagonal Prism", "Octagonal Prism"
        ])
        self.shape_combo.currentTextChanged.connect(self.on_shape_changed)
        shape_layout.addWidget(self.shape_combo)

        # Shape info label
        self.shape_info = QLabel("Select a shape preset or use custom values above.")
        self.shape_info.setWordWrap(True)
        self.shape_info.setObjectName("infoLabel")
        shape_layout.addWidget(self.shape_info)

        layout.addWidget(shape_widget)

    def setup_buttons(self, layout):
        """Set up control buttons."""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.setObjectName("primaryButton")
        self.calculate_btn.clicked.connect(self.calculate)
        self.calculate_btn.setToolTip("Calculate terminal velocity and update results")

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

        # Velocity unit selector
        velocity_unit_layout = QHBoxLayout()
        velocity_unit_layout.addWidget(QLabel("Display Velocity in:"))
        self.velocity_unit_combo = QComboBox()
        self.velocity_unit_combo.addItems(list(self.converter.velocity_units().keys()))
        self.velocity_unit_combo.setCurrentText("m/s")
        self.velocity_unit_combo.currentTextChanged.connect(self.update_display_units)
        velocity_unit_layout.addWidget(self.velocity_unit_combo)
        advanced_layout.addLayout(velocity_unit_layout)

        # Multi-fluid comparison
        self.multi_fluid_checkbox = QCheckBox("Compare Multiple Fluids")
        self.multi_fluid_checkbox.setObjectName("controlCheckbox")
        self.multi_fluid_checkbox.setToolTip("Show terminal velocity in different fluids")
        self.multi_fluid_checkbox.stateChanged.connect(self.on_multi_fluid_changed)
        advanced_layout.addWidget(self.multi_fluid_checkbox)

        # Altitude effects
        altitude_layout = QHBoxLayout()
        altitude_layout.addWidget(QLabel("Altitude:"))
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(0, 50000)
        self.altitude_input.setValue(0)
        self.altitude_input.setSuffix(" m")
        self.altitude_input.setToolTip("Altitude affects air density")
        self.altitude_input.valueChanged.connect(self.on_altitude_changed)
        altitude_layout.addWidget(self.altitude_input)
        advanced_layout.addLayout(altitude_layout)

        # Preset scenarios
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset Scenarios:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "None",
            "Skydiver (no parachute)",
            "Skydiver (with parachute)",
            "Baseball",
            "Golf Ball",
            "Raindrop",
            "Snowflake",
            "Bullet",
            "Feather",
            "Paper Airplane",
            "Coin",
            "Ping Pong Ball",
            "Tennis Ball",
            "Bowling Ball"
        ])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        advanced_layout.addLayout(preset_layout)

        # Force breakdown display
        self.force_breakdown_checkbox = QCheckBox("Show Force Breakdown")
        self.force_breakdown_checkbox.setObjectName("controlCheckbox")
        self.force_breakdown_checkbox.setToolTip("Display gravitational, drag, and net forces")
        self.force_breakdown_checkbox.stateChanged.connect(self.on_force_breakdown_changed)
        advanced_layout.addWidget(self.force_breakdown_checkbox)

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

        self.detailed_results_btn = QPushButton("Detailed Physics")
        self.detailed_results_btn.setObjectName("secondaryButton")
        self.detailed_results_btn.setCheckable(True)
        self.detailed_results_btn.clicked.connect(lambda: self.set_display_mode("detailed"))
        display_layout.addWidget(self.detailed_results_btn)

        self.reynolds_analysis_btn = QPushButton("Reynolds Analysis")
        self.reynolds_analysis_btn.setObjectName("secondaryButton")
        self.reynolds_analysis_btn.setCheckable(True)
        self.reynolds_analysis_btn.clicked.connect(lambda: self.set_display_mode("reynolds"))
        display_layout.addWidget(self.reynolds_analysis_btn)

        advanced_layout.addWidget(display_group)

        # Fluid selection for comparison
        fluid_group = QGroupBox("Fluids for Comparison")
        fluid_group.setObjectName("toolCard")
        fluid_layout = QVBoxLayout(fluid_group)

        self.fluid_checkboxes = {}
        fluids = [
            ("Air (Sea Level)", 1.225),
            ("Air (10,000m)", 0.4135),
            ("Water", 1000.0),
            ("Seawater", 1025.0),
            ("Oil", 800.0),
            ("Mercury", 13534.0),
            ("Custom", None)
        ]

        for fluid_name, density in fluids:
            checkbox = QCheckBox(fluid_name)
            checkbox.setObjectName("controlCheckbox")
            if fluid_name in ["Air (Sea Level)", "Water", "Seawater", "Oil", "Mercury"]:
                checkbox.setChecked(True)  # Default selection
            checkbox.stateChanged.connect(self.on_fluid_selection_changed)
            self.fluid_checkboxes[fluid_name] = checkbox
            fluid_layout.addWidget(checkbox)

        advanced_layout.addWidget(fluid_group)

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
        # Section title
        title = QLabel("Results")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        results_widget = QWidget()
        results_widget.setObjectName("toolCard")
        results_layout = QGridLayout(results_widget)
        results_layout.setSpacing(10)

        # Terminal velocity result
        results_layout.addWidget(QLabel("Terminal Velocity:"), 0, 0)
        self.velocity_result = QLabel("0.00 m/s")
        self.velocity_result.setObjectName("resultValue")
        results_layout.addWidget(self.velocity_result, 0, 1)

        # Time to reach 95% of terminal velocity
        results_layout.addWidget(QLabel("Time to 95% v_t:"), 1, 0)
        self.time_result = QLabel("0.00 s")
        results_layout.addWidget(self.time_result, 1, 1)

        # Distance to reach 95% of terminal velocity
        results_layout.addWidget(QLabel("Distance to 95% v_t:"), 2, 0)
        self.distance_result = QLabel("0.00 m")
        results_layout.addWidget(self.distance_result, 2, 1)

        # Additional info
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Click 'Calculate' to see results...")
        self.info_text.setObjectName("infoText")

        layout.addWidget(results_widget)
        layout.addWidget(QLabel("Calculation Details:"))
        layout.addWidget(self.info_text)

    def setup_plot_widget(self, layout):
        """Set up the plotting widget."""
        # Section title
        title = QLabel("Velocity vs Time")
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

        self.plot_widget.setTitle("Velocity vs Time", color='w', size='14pt')
        self.plot_widget.setLabel('left', 'Velocity (m/s)', color='w')
        self.plot_widget.setLabel('bottom', 'Time (s)', color='w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        plot_layout.addWidget(self.plot_widget)
        layout.addWidget(plot_container)

    def setup_plot(self):
        """Initialize the plot."""
        self.velocity_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00bfff', width=2))
        self.terminal_line = self.plot_widget.plot(pen=pg.mkPen(color='#ff6b6b', width=2, style=Qt.PenStyle.DashLine))

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&&File")
        save_action = QAction("&&Save Results", self)
        load_action = QAction("&&Load Config", self)
        exit_action = QAction("&&Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("&&Help")
        about_action = QAction("&&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def apply_modern_style(self):
        """Apply modern styling to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #fafafa;
            }
            QLabel {
                color: #333333;
            }
        """)

    def update_units(self):
        self.current_units['mass'] = self.mass_unit_combo.currentText()
        self.current_units['area'] = self.area_unit_combo.currentText()
        self.current_units['density'] = self.fluid_density_unit_combo.currentText()

        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_gravity_changed(self, text):
        """Handle gravity preset selection."""
        gravity_values = {
            "Earth (9.81 m/s²)": 9.81,
            "Moon (1.62 m/s²)": 1.62,
            "Mars (3.71 m/s²)": 3.71,
            "Custom": self.gravity_input.value()
        }

        if text == "Custom":
            self.gravity_input.setEnabled(True)
        else:
            self.gravity_input.setEnabled(False)
            self.gravity_input.setValue(gravity_values[text])

        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_fluid_changed(self, text):
        """Handle fluid preset selection."""
        density_values = {
            "Air (Sea Level)": 1.225,
            "Air (10,000m)": 0.4135,
            "Water": 1000.0,
            "Seawater": 1025.0,
            "Mercury": 13534.0,
            "Oil": 800.0,
            "Custom": self.fluid_density_input.value()
        }

        if text == "Custom":
            self.fluid_density_input.setEnabled(True)
        else:
            self.fluid_density_input.setEnabled(False)
            self.fluid_density_input.setValue(density_values[text])

        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_shape_changed(self, text):
        """Handle shape preset selection."""
        drag_values = {
            "Custom": self.drag_input.value(),
            "Sphere": 0.47,
            "Cube": 1.05,
            "Cylinder": 0.82,
            "Streamlined Body": 0.04,
            "Parachute": 1.4,
            "Flat Plate": 1.28,
            "Half Sphere": 0.42,
            "Cone": 0.50,
            "Teardrop": 0.04,
            "Disc": 1.11,
            "Square Plate": 1.10,
            "Ellipsoid": 0.20,
            "Hemisphere": 0.38,
            "Circular Disc": 1.17,
            "Wedge": 0.80,
            "Pyramid": 0.75,
            "Prism": 0.90,
            "Rocket": 0.25,
            "Arrow": 0.10,
            "Bullet": 0.295,
            "Feather": 1.3,
            "Shuttlecock": 0.65,
            "Baseball": 0.33,
            "Basketball": 0.41,
            "Tennis Ball": 0.55,
            "Golf Ball": 0.47,
            "Bowling Ball": 0.30,
            "Frisbee": 0.10,
            "Plate": 1.28,
            "Airfoil": 0.05,
            "Bluff Body": 1.15,
            "Square Rod": 0.95,
            "Round Rod": 0.47,
            "I-Beam": 1.2,
            "T-Shape": 0.85,
            "L-Shape": 1.0,
            "Boomerang": 0.8,
            "Kite": 0.95,
            "Sail": 1.0,
            "Sphere with Dimples": 0.25,
            "Smooth Sphere": 0.47,
            "Roughened Sphere": 0.51,
            "Cube (Smooth)": 1.05,
            "Cube (Rough)": 1.25,
            "Triangular Prism": 0.75,
            "Hexagonal Prism": 0.65,
            "Octagonal Prism": 0.60
        }

        shape_info = {
            "Custom": "Use custom drag coefficient value above.",
            "Sphere": "Typical drag coefficient for a smooth sphere.",
            "Cube": "Drag coefficient for a cube facing the flow.",
            "Cylinder": "Drag coefficient for a cylinder with flow perpendicular to axis.",
            "Streamlined Body": "Very low drag for aerodynamic shapes.",
            "Parachute": "High drag coefficient for parachute-like shapes.",
            "Flat Plate": "Perpendicular flow across a flat square plate.",
            "Half Sphere": "Hemisphere with curved side facing flow.",
            "Cone": "Pointed cone shape, moderate drag.",
            "Teardrop": "Optimal aerodynamic teardrop shape.",
            "Disc": "Thin circular disc perpendicular to flow.",
            "Square Plate": "Square flat surface perpendicular to flow.",
            "Ellipsoid": "Elongated ellipsoidal body.",
            "Hemisphere": "Half-sphere with flat side facing flow.",
            "Circular Disc": "Circular disk perpendicular to flow direction.",
            "Wedge": "Wedge-shaped body at optimal angle.",
            "Pyramid": "Pyramid shape, pointed end forward.",
            "Prism": "Triangular prism configuration.",
            "Rocket": "Streamlined rocket body with fins.",
            "Arrow": "Fletched arrow with minimal drag.",
            "Bullet": "Spitzer bullet or projectile shape.",
            "Feather": "Bird feather, relatively high drag.",
            "Shuttlecock": "Badminton shuttlecock with high drag.",
            "Baseball": "Stitched baseball with seams.",
            "Basketball": "Inflated basketball.",
            "Tennis Ball": "Fuzzy tennis ball.",
            "Golf Ball": "Golf ball with dimple pattern.",
            "Bowling Ball": "Smooth heavy bowling ball.",
            "Frisbee": "Flying disc with low drag.",
            "Plate": "Flat serving plate.",
            "Airfoil": "Wing-like airfoil section.",
            "Bluff Body": "Bluff body with maximum separation.",
            "Square Rod": "Extruded square rod.",
            "Round Rod": "Cylindrical rod or pole.",
            "I-Beam": "Steel I-beam structural shape.",
            "T-Shape": "T-shaped cross-section.",
            "L-Shape": "L-shaped channel or angle.",
            "Boomerang": "Returning boomerang shape.",
            "Kite": "Flying kite with frame.",
            "Sail": "Fabric sail shape.",
            "Sphere with Dimples": "Dimpled sphere similar to golf ball.",
            "Smooth Sphere": "Perfectly smooth sphere.",
            "Roughened Sphere": "Rough textured sphere.",
            "Cube (Smooth)": "Cube with smooth surfaces.",
            "Cube (Rough)": "Cube with rough surfaces.",
            "Triangular Prism": "Three-sided prism shape.",
            "Hexagonal Prism": "Six-sided prism shape.",
            "Octagonal Prism": "Eight-sided prism shape."
        }

        if text == "Custom":
            self.drag_input.setEnabled(True)
        else:
            self.drag_input.setEnabled(False)
            self.drag_input.setValue(drag_values[text])

        self.shape_info.setText(shape_info[text])

        if self.live_update_checkbox.isChecked():
            self.calculate()

    def calculate(self):
        """Perform the terminal velocity calculation."""
        try:
            # Get input values with unit conversion
            mass = self.mass_input.value() * self.converter.mass_units()[self.current_units['mass']]
            area = self.area_input.value() * self.converter.area_units()[self.current_units['area']]
            drag_coeff = self.drag_input.value()
            fluid_density = self.fluid_density_input.value() * self.converter.density_units()[self.current_units['density']]
            gravity = self.gravity_input.value()

            # Calculate terminal velocity
            v_t = self.calculator.calculate_terminal_velocity(mass, area, drag_coeff, fluid_density, gravity)

            # Convert result to display units
            v_t_display = v_t / self.converter.velocity_units()[self.current_units.get('velocity', 'm/s')]

            # Update results display
            self.velocity_result.setText(f"{v_t_display:.2f} m/s")

            # Calculate time and distance to reach 95% of terminal velocity
            if v_t > 0:
                time_95 = (v_t / gravity) * math.atanh(0.95)
                distance_95 = (v_t**2 / gravity) * math.log(math.cosh(gravity * time_95 / v_t))

                self.time_result.setText(f"{time_95:.2f} s")
                self.distance_result.setText(f"{distance_95:.2f} m")
            else:
                self.time_result.setText("N/A")
                self.distance_result.setText("N/A")

            # Update info text
            self.update_info_text(mass, area, drag_coeff, fluid_density, gravity, v_t)

            # Update plot
            self.update_plot(mass, area, drag_coeff, fluid_density, gravity, v_t)

        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", f"An error occurred during calculation:\n{str(e)}")

    def update_info_text(self, mass, area, drag_coeff, density, gravity, v_t):
        """Update the calculation details text."""
        v_t_kmh = v_t * 3.6
        info = ("Terminal Velocity Calculation Results:\n\n"
                f"Mass: {mass:.3f} kg\n"
                f"Cross-sectional Area: {area:.6f} m²\n"
                f"Drag Coefficient: {drag_coeff:.3f}\n"
                f"Fluid Density: {density:.3f} kg/m³\n"
                f"Gravity: {gravity:.2f} m/s²\n\n"
                f"Terminal Velocity: {v_t:.2f} m/s ({v_t_kmh:.1f} km/h)\n\n"
                "The terminal velocity is reached when the drag force equals the gravitational force.\n"
                "For objects in free fall, this represents the maximum speed achievable.")

        self.info_text.setPlainText(info)

    def update_plot(self, mass, area, drag_coeff, density, gravity, v_t):
        """Update the velocity vs time plot."""
        # If multi-fluid comparison is active, update multi-fluid plot instead
        if hasattr(self, 'multi_fluid_checkbox') and self.multi_fluid_checkbox.isChecked():
            self.update_multi_fluid_plot()
            return

        # Clear any multi-fluid curves if they exist
        if hasattr(self, 'fluid_curves'):
            for curve in self.fluid_curves:
                self.plot_widget.removeItem(curve)
            self.fluid_curves = []

        # Clear legend if it exists
        if hasattr(self, 'legend'):
            self.legend.clear()

        time_points = np.linspace(0, 10, 1000)  # 10 seconds
        velocities = self.calculator.calculate_velocity_vs_time(
            mass, area, drag_coeff, density, gravity, time_points
        )

        self.velocity_curve.setData(time_points, velocities)

        # Add horizontal line at terminal velocity
        if v_t > 0:
            self.terminal_line.setData([0, time_points[-1]], [v_t, v_t])

    def reset_inputs(self):
        """Reset all inputs to default values."""
        self.mass_input.setValue(1.0)
        self.area_input.setValue(0.01)
        self.drag_input.setValue(0.47)
        self.gravity_combo.setCurrentText("Earth (9.81 m/s²)")
        self.fluid_combo.setCurrentText("Air (Sea Level)")
        self.shape_combo.setCurrentText("Custom")

        # Clear results
        self.velocity_result.setText("0.00 m/s")
        self.time_result.setText("0.00 s")
        self.distance_result.setText("0.00 m")
        self.info_text.setPlainText("Click 'Calculate' to see results...")

        # Clear plot
        self.velocity_curve.setData([], [])
        self.terminal_line.setData([], [])

    def save_config(self):
        """Save current configuration to file."""
        config = {
            'mass': self.mass_input.value(),
            'mass_unit': self.mass_unit_combo.currentText(),
            'area': self.area_input.value(),
            'area_unit': self.area_unit_combo.currentText(),
            'drag_coeff': self.drag_input.value(),
            'gravity_preset': self.gravity_combo.currentText(),
            'gravity_value': self.gravity_input.value(),
            'fluid_preset': self.fluid_combo.currentText(),
            'fluid_density': self.fluid_density_input.value(),
            'fluid_density_unit': self.fluid_density_unit_combo.currentText(),
            'shape': self.shape_combo.currentText()
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
                self.mass_input.setValue(config.get('mass', 1.0))
                self.mass_unit_combo.setCurrentText(config.get('mass_unit', 'kg'))
                self.area_input.setValue(config.get('area', 0.01))
                self.area_unit_combo.setCurrentText(config.get('area_unit', 'm²'))
                self.drag_input.setValue(config.get('drag_coeff', 0.47))
                self.gravity_combo.setCurrentText(config.get('gravity_preset', 'Earth (9.81 m/s²)'))
                self.gravity_input.setValue(config.get('gravity_value', 9.81))
                self.fluid_combo.setCurrentText(config.get('fluid_preset', 'Air (Sea Level)'))
                self.fluid_density_input.setValue(config.get('fluid_density', 1.225))
                self.fluid_density_unit_combo.setCurrentText(config.get('fluid_density_unit', 'kg/m³'))
                self.shape_combo.setCurrentText(config.get('shape', 'Custom'))

                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration:\n{str(e)}")

    # Advanced Features Methods

    def on_live_update_changed(self, state):
        signals = [
            self.mass_input.valueChanged,
            self.area_input.valueChanged,
            self.drag_input.valueChanged,
            self.gravity_input.valueChanged,
            self.fluid_density_input.valueChanged,
            self.altitude_input.valueChanged,
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

    def update_display_units(self, unit):
        self.current_units['velocity'] = unit
        if self.live_update_checkbox.isChecked():
            self.calculate()

        if hasattr(self, 'velocity_curve'):
            x, y = self.velocity_curve.getData()
            if x is not None and len(x) > 0:
                self.calculate()

    def on_multi_fluid_changed(self, state):
        """Handle multi-fluid comparison toggle."""
        if state == Qt.CheckState.Checked:
            self.update_multi_fluid_plot()
        else:
            # Return to single fluid plot
            self.calculate()

    def on_altitude_changed(self, altitude):
        """Handle altitude change for air density adjustment."""
        # Calculate air density based on altitude (simplified model)
        # Density decreases exponentially with altitude
        sea_level_density = 1.225  # kg/m³
        scale_height = 8500  # meters
        density = sea_level_density * math.exp(-altitude / scale_height)

        if self.fluid_combo.currentText() == "Air (Sea Level)" or self.fluid_combo.currentText() == "Air (10,000m)":
            self.fluid_density_input.setValue(density)
        
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def on_preset_changed(self, preset):
        """Handle preset scenario selection."""
        if preset == "None":
            return

        presets = {
            "Skydiver (no parachute)": {"mass": 80, "area": 0.7, "drag": 0.8},
            "Skydiver (with parachute)": {"mass": 80, "area": 25, "drag": 1.4},
            "Baseball": {"mass": 0.145, "area": 0.0042, "drag": 0.33},
            "Golf Ball": {"mass": 0.0459, "area": 0.0014, "drag": 0.47},
            "Raindrop": {"mass": 0.000005, "area": 0.0000005, "drag": 0.5},
            "Snowflake": {"mass": 0.0000001, "area": 0.000001, "drag": 1.0},
            "Bullet": {"mass": 0.005, "area": 0.000005, "drag": 0.295},
            "Feather": {"mass": 0.0001, "area": 0.001, "drag": 1.3},
            "Paper Airplane": {"mass": 0.005, "area": 0.01, "drag": 0.8},
            "Coin": {"mass": 0.005, "area": 0.0005, "drag": 1.1},
            "Ping Pong Ball": {"mass": 0.0027, "area": 0.0013, "drag": 0.5},
            "Tennis Ball": {"mass": 0.057, "area": 0.0033, "drag": 0.55},
            "Bowling Ball": {"mass": 7.26, "area": 0.011, "drag": 0.3}
        }

        if preset in presets:
            data = presets[preset]
            self.mass_input.setValue(data["mass"])
            self.area_input.setValue(data["area"])
            self.drag_input.setValue(data["drag"])
            self.shape_combo.setCurrentText("Custom")

            if self.live_update_checkbox.isChecked():
                self.calculate()

    def on_force_breakdown_changed(self, state):
        """Handle force breakdown display toggle."""
        # Force breakdown is now handled in update_info_text
        # Just recalculate to update the display
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def set_display_mode(self, mode):
        """Set the display mode for results."""
        # Uncheck all buttons
        self.basic_results_btn.setChecked(False)
        self.detailed_results_btn.setChecked(False)
        self.reynolds_analysis_btn.setChecked(False)

        # Check the selected button
        if mode == "basic":
            self.basic_results_btn.setChecked(True)
        elif mode == "detailed":
            self.detailed_results_btn.setChecked(True)
        elif mode == "reynolds":
            self.reynolds_analysis_btn.setChecked(True)

        # Update display
        self.update_display_mode(mode)
        if self.live_update_checkbox.isChecked():
            self.calculate()

    def update_display_mode(self, mode):
        """Update the display based on the selected mode."""
        self.current_display_mode = mode
        # The actual display update happens in update_info_text

    def on_fluid_selection_changed(self, state):
        """Handle fluid selection changes for comparison."""
        # Update multi-fluid plot if it's currently active
        if hasattr(self, 'multi_fluid_checkbox') and self.multi_fluid_checkbox.isChecked():
            self.update_multi_fluid_plot()

    def update_multi_fluid_plot(self):
        """Update plot to show terminal velocity in selected fluids."""
        try:
            mass = self.mass_input.value() * self.converter.mass_units()[self.current_units['mass']]
            area = self.area_input.value() * self.converter.area_units()[self.current_units['area']]
            drag_coeff = self.drag_input.value()
            gravity = self.gravity_input.value()

            # Get selected fluids
            selected_fluids = {}
            for fluid_name, checkbox in self.fluid_checkboxes.items():
                if checkbox.isChecked():
                    if fluid_name == "Custom":
                        # Use current fluid density
                        current_density = self.fluid_density_input.value() * self.converter.density_units()[self.current_units['density']]
                        selected_fluids[fluid_name] = current_density
                    else:
                        # Use predefined densities
                        fluid_densities = {
                            "Air (Sea Level)": 1.225,
                            "Air (10,000m)": 0.4135,
                            "Water": 1000.0,
                            "Seawater": 1025.0,
                            "Oil": 800.0,
                            "Mercury": 13534.0
                        }
                        if fluid_name in fluid_densities:
                            selected_fluids[fluid_name] = fluid_densities[fluid_name]

            # Clear existing curves (both single and multi-fluid)
            if hasattr(self, 'fluid_curves'):
                for curve in self.fluid_curves:
                    self.plot_widget.removeItem(curve)
            self.fluid_curves = []

            # Clear single fluid curves
            self.velocity_curve.setData([], [])
            self.terminal_line.setData([], [])

            if not selected_fluids:
                return  # No fluids selected

            colors = ['#00bfff', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#e84393', '#00b894']
            i = 0

            for fluid_name, density in selected_fluids.items():
                v_t = self.calculator.calculate_terminal_velocity(mass, area, drag_coeff, density, gravity)
                time_points = np.linspace(0, 10, 1000)
                velocities = self.calculator.calculate_velocity_vs_time(
                    mass, area, drag_coeff, density, gravity, time_points
                )

                color = colors[i % len(colors)]
                curve = self.plot_widget.plot(time_points, velocities,
                                            pen=pg.mkPen(color=color, width=2),
                                            name=f"{fluid_name}: {v_t:.1f} m/s")
                self.fluid_curves.append(curve)
                i += 1

            # Update legend
            if not hasattr(self, 'legend'):
                self.legend = self.plot_widget.addLegend()
            else:
                self.legend.clear()
                for curve in self.fluid_curves:
                    self.legend.addItem(curve, curve.name())

        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error updating multi-fluid plot:\n{str(e)}")

    def calculate_reynolds_number(self, velocity, characteristic_length, fluid_density, dynamic_viscosity):
        """Calculate Reynolds number for flow regime analysis."""
        return (fluid_density * velocity * characteristic_length) / dynamic_viscosity

    def get_flow_regime(self, reynolds_number):
        """Determine flow regime based on Reynolds number."""
        if reynolds_number < PhysicsConstants.RE_LAMINAR_MAX:
            return "Laminar"
        elif reynolds_number < PhysicsConstants.RE_TRANSITIONAL_MAX:
            return "Transitional"
        else:
            return "Turbulent"

    def export_plot(self):
        """Export the current plot to image file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG files (*.png);;SVG files (*.svg)"
        )
        if filename:
            try:
                if filename.endswith('.png'):
                    # Export as PNG
                    exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
                    exporter.export(filename)
                elif filename.endswith('.svg'):
                    # Export as SVG
                    exporter = pg.exporters.SVGExporter(self.plot_widget.plotItem)
                    exporter.export(filename)
                QMessageBox.information(self, "Success", "Plot exported successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export plot:\n{str(e)}")

    def update_info_text(self, mass, area, drag_coeff, density, gravity, v_t):
        """Update the calculation details text with advanced information."""
        v_t_kmh = v_t * 3.6

        # Calculate Reynolds number (using area as approximation for characteristic length)
        characteristic_length = math.sqrt(area / math.pi) * 2  # Equivalent diameter
        viscosity = self.get_fluid_viscosity(density)
        reynolds = self.calculate_reynolds_number(v_t, characteristic_length, density, viscosity)
        flow_regime = self.get_flow_regime(reynolds)

        # Get current display mode
        display_mode = getattr(self, 'current_display_mode', 'basic')

        if display_mode == "basic":
            info = ("Terminal Velocity Calculation Results:\n\n"
                    f"Mass: {mass:.3f} kg\n"
                    f"Cross-sectional Area: {area:.6f} m²\n"
                    f"Drag Coefficient: {drag_coeff:.3f}\n"
                    f"Fluid Density: {density:.3f} kg/m³\n"
                    f"Gravity: {gravity:.2f} m/s²\n\n"
                    f"Terminal Velocity: {v_t:.2f} m/s ({v_t_kmh:.1f} km/h)\n\n"
                    "The terminal velocity is reached when the drag force equals the gravitational force.\n"
                    "For objects in free fall, this represents the maximum speed achievable.")

        elif display_mode == "detailed":
            info = ("Detailed Physics Analysis:\n\n"
                    f"Object Properties:\n"
                    f"  Mass: {mass:.3f} kg\n"
                    f"  Cross-sectional Area: {area:.6f} m²\n"
                    f"  Drag Coefficient: {drag_coeff:.3f}\n\n"
                    f"Environmental Conditions:\n"
                    f"  Fluid Density: {density:.3f} kg/m³\n"
                    f"  Gravity: {gravity:.2f} m/s²\n"
                    f"  Dynamic Viscosity: {viscosity:.2e} Pa·s\n\n"
                    f"Results:\n"
                    f"  Terminal Velocity: {v_t:.2f} m/s ({v_t_kmh:.1f} km/h)\n"
                    f"  Reynolds Number: {reynolds:.0f}\n"
                    f"  Flow Regime: {flow_regime}\n\n"
                    "Physics Explanation:\n"
                    "Terminal velocity occurs when gravitational force equals drag force.\n"
                    "Reynolds number indicates whether flow is laminar (< 2000) or turbulent (> 4000).")

        elif display_mode == "reynolds":
            info = ("Reynolds Number Analysis:\n\n"
                    f"Terminal Velocity: {v_t:.2f} m/s\n"
                    f"Characteristic Length: {characteristic_length:.6f} m\n"
                    f"Fluid Density: {density:.3f} kg/m³\n"
                    f"Dynamic Viscosity: {viscosity:.2e} Pa·s\n\n"
                    f"Reynolds Number: {reynolds:.0f}\n"
                    f"Flow Regime: {flow_regime}\n\n"
                    "Reynolds Number Regimes:\n"
                    "• < 2000: Laminar flow (smooth, predictable)\n"
                    "• 2000-4000: Transitional flow\n"
                    "• > 4000: Turbulent flow (chaotic, higher drag)\n\n"
                    "For falling objects, higher Reynolds numbers indicate more complex\n"
                    "airflow patterns and potentially higher drag coefficients.")

        else:
            info = "Unknown display mode"

        # Add force breakdown if enabled
        if hasattr(self, 'force_breakdown_checkbox') and self.force_breakdown_checkbox.isChecked():
            # Calculate forces
            gravitational_force = mass * gravity
            drag_force = 0.5 * drag_coeff * density * area * v_t**2
            net_force = gravitational_force - drag_force

            force_breakdown = f"""

Force Breakdown at Terminal Velocity:

Gravitational Force: {gravitational_force:.2f} N (downward)
Drag Force: {drag_force:.2f} N (upward)
Net Force: {net_force:.2f} N

At terminal velocity, gravitational force equals drag force,
resulting in zero net acceleration."""
            info += force_breakdown

        self.info_text.setPlainText(info)

        # Enable export button when we have a plot
        self.export_plot_btn.setEnabled(True)

    def get_fluid_viscosity(self, density):
        """Get viscosity for a given fluid density."""
        # Approximate viscosity based on common fluids
        if abs(density - 1.225) < 0.1:  # Air
            return PhysicsConstants.VISCOSITY_AIR
        elif abs(density - 1000) < 10:  # Water
            return PhysicsConstants.VISCOSITY_WATER
        elif abs(density - 1025) < 10:  # Seawater
            return PhysicsConstants.VISCOSITY_SEAWATER
        elif abs(density - 13534) < 100:  # Mercury
            return PhysicsConstants.VISCOSITY_MERCURY
        elif abs(density - 800) < 10:  # Oil
            return PhysicsConstants.VISCOSITY_OIL
        else:
            return PhysicsConstants.VISCOSITY_AIR  # Default to air


TOOL_META = {
    "name": "Terminal Velocity Calculator",
    "description": "Advanced terminal velocity calculator with physics simulation, multi-fluid comparison, and educational features",
    "category": "Physics",
    "version": "2.0.0",
    "author": "ScienceHub Team",
    "features": [
        "Terminal velocity calculation",
        "Velocity vs time plotting",
        "Multi-fluid comparison",
        "Reynolds number analysis",
        "Altitude effects on air density",
        "Preset scenarios",
        "Force breakdown analysis",
        "Plot export (PNG/SVG)",
        "Live update mode",
        "Unit conversion",
        "Shape presets with drag coefficients"
    ],
    "educational_value": "Learn about terminal velocity, drag forces, fluid dynamics, and atmospheric physics",
    "keywords": ["terminal velocity", "drag force", "fluid dynamics", "physics", "simulation", "education"]
}


def create_tool(parent=None):
    """Create and return a TerminalVelocityTool instance."""
    return TerminalVelocityTool(parent)