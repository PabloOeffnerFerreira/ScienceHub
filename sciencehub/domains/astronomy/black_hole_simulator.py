"""
Integrated Schwarzschild Black Hole Simulator
Combines 2D and 3D visualizations for comprehensive black hole physics exploration.
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from sciencehub.data.functions.schwarzschild_black_hole_simulator import (
    SchwarzschildBlackHoleSimulator,
    TOOL_META
)
from sciencehub.data.functions.schwarzschild_3d_simulator import (
    Schwarzschild3DSimulator,
    TOOL_META_3D
)


class IntegratedBlackHoleSimulator(QMainWindow):
    """Integrated window combining 2D and 3D black hole simulators."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Schwarzschild Black Hole Simulator")
        self.setGeometry(100, 100, 1600, 900)

        # Initialize simulators
        self.simulator_2d = SchwarzschildBlackHoleSimulator()
        self.simulator_3d = Schwarzschild3DSimulator()

        # Setup synchronization
        self.setup_synchronization()

        # Setup UI
        self.setup_ui()

        # Apply dark theme
        self.apply_dark_theme()

    def setup_ui(self):
        """Setup the integrated interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Title and description
        title_layout = QHBoxLayout()
        title_label = QLabel("Integrated Schwarzschild Black Hole Simulator")
        title_label.setObjectName("mainTitle")
        title_layout.addWidget(title_label)

        sync_button = QPushButton("🔄 Sync Parameters")
        sync_button.setObjectName("primaryButton")
        sync_button.clicked.connect(self.sync_parameters)
        title_layout.addWidget(sync_button)

        main_layout.addLayout(title_layout)

        # Description
        desc_label = QLabel(
            "Explore black hole physics with synchronized 2D and 3D visualizations. "
            "Changes in one view automatically update the other for seamless exploration."
        )
        desc_label.setWordWrap(True)
        desc_label.setObjectName("description")
        main_layout.addWidget(desc_label)

        # Main content area as tabs
        tabs = QTabWidget()
        tabs.setObjectName("simTabs")

        # 2D Simulator tab
        tab_2d = QWidget()
        tab_2d_layout = QVBoxLayout(tab_2d)
        tab_2d_layout.addWidget(self.simulator_2d)
        tabs.addTab(tab_2d, "2D Analysis")

        # 3D Simulator tab
        tab_3d = QWidget()
        tab_3d_layout = QVBoxLayout(tab_3d)
        tab_3d_layout.addWidget(self.simulator_3d)
        tabs.addTab(tab_3d, "3D Visualization")

        main_layout.addWidget(tabs)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Parameters synchronized between 2D and 3D views")

    def setup_synchronization(self):
        """Setup parameter synchronization between 2D and 3D simulators."""
        # Connect 2D simulator changes to 3D updates
        self.simulator_2d.mass_input.valueChanged.connect(self.on_2d_mass_changed)
        self.simulator_2d.mass_unit_combo.currentTextChanged.connect(self.on_2d_mass_changed)
        self.simulator_2d.distance_input.valueChanged.connect(self.on_2d_distance_changed)
        self.simulator_2d.distance_unit_combo.currentTextChanged.connect(self.on_2d_distance_changed)

        # Connect 3D simulator changes to 2D updates
        self.simulator_3d.mass_3d_input.valueChanged.connect(self.on_3d_mass_changed)
        self.simulator_3d.mass_3d_unit.currentTextChanged.connect(self.on_3d_mass_changed)
        self.simulator_3d.distance_3d_input.valueChanged.connect(self.on_3d_distance_changed)
        self.simulator_3d.distance_3d_unit.currentTextChanged.connect(self.on_3d_distance_changed)

    def on_2d_mass_changed(self):
        """Handle mass changes from 2D simulator."""
        mass = self.simulator_2d.mass_input.value()
        unit = self.simulator_2d.mass_unit_combo.currentText()

        # Block signals to prevent recursion
        self.simulator_3d.mass_3d_input.blockSignals(True)
        self.simulator_3d.mass_3d_unit.blockSignals(True)

        self.simulator_3d.mass_3d_input.setValue(mass)
        self.simulator_3d.mass_3d_unit.setCurrentText(unit)

        self.simulator_3d.mass_3d_input.blockSignals(False)
        self.simulator_3d.mass_3d_unit.blockSignals(False)

        self.status_bar.showMessage(f"Synchronized mass: {mass} {unit}")

    def on_2d_distance_changed(self):
        """Handle distance changes from 2D simulator."""
        distance = self.simulator_2d.distance_input.value()
        unit = self.simulator_2d.distance_unit_combo.currentText()

        # Convert to Rs for 3D simulator
        mass_kg = self.simulator_2d.get_mass_kg()
        rs = self.simulator_2d.calculator.schwarzschild_radius(mass_kg)

        if unit == 'km':
            distance_rs = distance * 1000 / rs
            unit_3d = 'Rs'
        elif unit == 'Rs':
            distance_rs = distance
            unit_3d = 'Rs'
        elif unit == 'AU':
            distance_rs = distance * 1.496e11 / rs
            unit_3d = 'Rs'
        elif unit == 'ly':
            distance_rs = distance * 9.461e15 / rs
            unit_3d = 'Rs'
        else:
            distance_rs = distance
            unit_3d = unit

        # Block signals to prevent recursion
        self.simulator_3d.distance_3d_input.blockSignals(True)
        self.simulator_3d.distance_3d_unit.blockSignals(True)

        self.simulator_3d.distance_3d_input.setValue(distance_rs)
        self.simulator_3d.distance_3d_unit.setCurrentText(unit_3d)

        self.simulator_3d.distance_3d_input.blockSignals(False)
        self.simulator_3d.distance_3d_unit.blockSignals(False)

        self.status_bar.showMessage(f"Synchronized distance: {distance} {unit}")

    def on_3d_mass_changed(self):
        """Handle mass changes from 3D simulator."""
        mass = self.simulator_3d.mass_3d_input.value()
        unit = self.simulator_3d.mass_3d_unit.currentText()

        # Block signals to prevent recursion
        self.simulator_2d.mass_input.blockSignals(True)
        self.simulator_2d.mass_unit_combo.blockSignals(True)

        self.simulator_2d.mass_input.setValue(mass)
        self.simulator_2d.mass_unit_combo.setCurrentText(unit)

        self.simulator_2d.mass_input.blockSignals(False)
        self.simulator_2d.mass_unit_combo.blockSignals(False)

        self.status_bar.showMessage(f"Synchronized mass: {mass} {unit}")

    def on_3d_distance_changed(self):
        """Handle distance changes from 3D simulator."""
        distance_rs = self.simulator_3d.distance_3d_input.value()
        unit_3d = self.simulator_3d.distance_3d_unit.currentText()

        # Convert to km for 2D simulator
        mass_kg = self.simulator_3d.get_mass_kg()
        rs = self.simulator_3d.calculator.schwarzschild_radius(mass_kg)

        if unit_3d == 'Rs':
            distance_km = distance_rs * rs / 1000
            unit_2d = 'km'
        else:
            distance_km = distance_rs
            unit_2d = unit_3d

        # Block signals to prevent recursion
        self.simulator_2d.distance_input.blockSignals(True)
        self.simulator_2d.distance_unit_combo.blockSignals(True)

        self.simulator_2d.distance_input.setValue(distance_km)
        self.simulator_2d.distance_unit_combo.setCurrentText(unit_2d)

        self.simulator_2d.distance_input.blockSignals(False)
        self.simulator_2d.distance_unit_combo.blockSignals(False)

        self.status_bar.showMessage(f"Synchronized distance: {distance_km} km")

    def sync_parameters(self):
        """Manually sync all parameters between simulators."""
        # Sync from 2D to 3D
        self.on_2d_mass_changed()
        self.on_2d_distance_changed()

        self.status_bar.showMessage("All parameters synchronized")

    def apply_dark_theme(self):
        """Apply dark theme styling."""
        dark_theme = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }

        QLabel {
            color: #ffffff;
        }

        QLabel#mainTitle {
            font-size: 18px;
            font-weight: bold;
            color: #4ecdc4;
            margin: 10px;
        }

        QLabel#panelTitle {
            font-size: 14px;
            font-weight: bold;
            color: #45b7d1;
            margin: 5px;
        }

        QLabel#description {
            color: #cccccc;
            margin: 5px;
            padding: 5px;
        }

        QPushButton#primaryButton {
            background-color: #4ecdc4;
            color: #1e1e1e;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton#primaryButton:hover {
            background-color: #45b7d1;
        }

        QStatusBar {
            background-color: #2d2d2d;
            color: #ffffff;
            border-top: 1px solid #404040;
        }
        """

        self.setStyleSheet(dark_theme)

    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up simulators
        self.simulator_2d.deleteLater()
        self.simulator_3d.deleteLater()
        event.accept()


def create_tool(parent=None):
    """Create and return an IntegratedBlackHoleSimulator instance."""
    return IntegratedBlackHoleSimulator()


def launch_integrated_simulator():
    """Launch the integrated black hole simulator."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Integrated Schwarzschild Simulator")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ScienceHub")

    # Create and show the integrated simulator
    simulator = IntegratedBlackHoleSimulator()
    simulator.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_integrated_simulator())
