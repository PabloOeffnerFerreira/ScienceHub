"""
Bohr Model Viewer - Interactive Atomic Structure Visualization
===========================================================

A comprehensive tool for visualizing Bohr atomic models with 2D and 3D views,
complete atomic data display, and quantum mechanics integration capabilities.

Features:
- Interactive atom selection from periodic table
- Complete atomic data display (properties, electron configuration, etc.)
- 2D Bohr model visualization with electron orbits
- 3D Bohr model visualization with orbital representation
- Real-time parameter adjustment
- Quantum model integration framework
- Beautiful, modern UI design
- Educational information panels
- Export capabilities for images and data

Author: ScienceHub Team
"""
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import Circle, Arc
from matplotlib.collections import PatchCollection

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QTextEdit,
    QSplitter, QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QSlider, QProgressBar, QTabWidget,
    QLineEdit, QMessageBox, QMenuBar, QMenu, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon
from PyQt6 import sip

from sciencehub.ui.components.tool_base import ScienceHubTool

class PhysicsConstants:
    """Physical constants for calculations."""

    # Bohr radius in meters
    BOHR_RADIUS = 5.29177210903e-11

    # Elementary charge in Coulombs
    ELEMENTARY_CHARGE = 1.602176634e-19

    # Vacuum permittivity in F/m
    VACUUM_PERMITTIVITY = 8.854187817e-12

    # Planck constant in J⋅s
    PLANCK_CONSTANT = 6.62607015e-34

    # Reduced Planck constant in J⋅s
    REDUCED_PLANCK = PLANCK_CONSTANT / (2 * 3.14159265359)

    # Speed of light in m/s
    SPEED_OF_LIGHT = 299792458

    # Electron mass in kg
    ELECTRON_MASS = 9.1093837015e-31


class ElementData:
    """Element data management."""

    def __init__(self):
        self.elements = []

    def get_element(self, atomic_number: int) -> dict:
        """Get element data by atomic number."""
        # This would load from the periodic table data
        # For now, return basic structure
        return {
            'number': atomic_number,
            'symbol': 'H',  # Placeholder
            'name': 'Hydrogen',  # Placeholder
            'atomic_mass': 1.008
        }


class BohrModelCalculator:
    """Calculator for Bohr model parameters and quantum properties."""

    def __init__(self):
        self.constants = PhysicsConstants()
        self.element_data = ElementData()

    def calculate_bohr_radius(self, n: int, Z: int = 1) -> float:
        """Calculate Bohr radius for given quantum numbers."""
        a0 = self.constants.BOHR_RADIUS
        return a0 * n**2 / Z

    def calculate_orbital_velocity(self, n: int, Z: int = 1) -> float:
        """Calculate electron orbital velocity."""
        # v_n = Z * e^2 / (2 * ε0 * h * n)
        return (Z * self.constants.ELEMENTARY_CHARGE**2) / (
            2 * self.constants.VACUUM_PERMITTIVITY *
            self.constants.PLANCK_CONSTANT * n
        )

    def calculate_energy_level(self, n: int, Z: int = 1) -> float:
        """Calculate energy of nth energy level."""
        # E_n = -13.6 * Z^2 / n^2 eV
        return -13.6 * Z**2 / n**2

    def calculate_angular_momentum(self, n: int) -> float:
        """Calculate angular momentum for nth level."""
        return n * self.constants.REDUCED_PLANCK

    def get_electron_configuration(self, atomic_number: int) -> Dict[str, Any]:
        """Get electron configuration for given atomic number."""
        # This would use the element data to determine configuration
        # For now, return a basic structure
        shells = []
        remaining_electrons = atomic_number

        # Simplified electron filling
        shell_capacities = [2, 8, 18, 32, 50, 72, 98]  # Maximum electrons per shell

        for n, capacity in enumerate(shell_capacities, 1):
            if remaining_electrons <= 0:
                break
            electrons_in_shell = min(remaining_electrons, capacity)
            shells.append({
                'n': n,
                'electrons': electrons_in_shell,
                'max_electrons': capacity,
                'subshells': self._get_subshells(n, electrons_in_shell)
            })
            remaining_electrons -= electrons_in_shell

        return {
            'atomic_number': atomic_number,
            'total_electrons': atomic_number,
            'shells': shells,
            'configuration_string': self._configuration_to_string(shells)
        }

    def _get_subshells(self, n: int, electrons: int) -> List[Dict]:
        """Get subshell distribution for a shell."""
        subshells = []
        subshell_names = ['s', 'p', 'd', 'f']
        subshell_capacities = [2, 6, 10, 14]

        for i, (name, capacity) in enumerate(zip(subshell_names, subshell_capacities)):
            if i >= n:  # Can't have f subshell in n=1, etc.
                break
            if electrons <= 0:
                break

            electrons_in_subshell = min(electrons, capacity)
            subshells.append({
                'name': f'{n}{name}',
                'electrons': electrons_in_subshell,
                'max_electrons': capacity,
                'l': i  # azimuthal quantum number
            })
            electrons -= electrons_in_subshell

        return subshells

    def _configuration_to_string(self, shells: List[Dict]) -> str:
        """Convert shell configuration to string notation."""
        parts = []
        for shell in shells:
            for subshell in shell['subshells']:
                if subshell['electrons'] > 0:
                    parts.append(f"{subshell['name']}{subshell['electrons']}")
        return ' '.join(parts)


class BohrModel2DCanvas(FigureCanvasQTAgg):
    """2D Canvas for Bohr model visualization."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data
        self.atomic_data = {}
        self.show_quantum_numbers = True
        self.show_electron_cloud = False
        self.animation_angle = 0
        self.is_animating = False
        self.animation_timer = None

        self.setup_plot()

    def setup_plot(self):
        """Initialize the 2D plot."""
        self.axes.clear()
        self.axes.set_facecolor('#0a0a0a')
        self.axes.set_aspect('equal')
        self.axes.set_xlim(-15, 15)
        self.axes.set_ylim(-15, 15)
        self.axes.axis('off')

        # Set title
        self.axes.set_title('Bohr Atomic Model - 2D View',
                          color='white', fontsize=14, fontweight='bold', pad=20)

    def update_atom(self, element_data: Dict, electron_config: Dict):
        """Update the atom being displayed."""
        self.atomic_data = {
            'element': element_data,
            'config': electron_config
        }
        self.redraw()

    def draw_nucleus(self):
        """Draw the atomic nucleus."""
        if not self.atomic_data:
            return

        element = self.atomic_data['element']
        atomic_number = element.get('number', 1)

        # Nucleus size based on atomic number
        nucleus_radius = max(0.3, min(1.0, atomic_number / 20))

        # Draw nucleus
        nucleus = Circle((0, 0), nucleus_radius,
                        facecolor='#ff6b6b', edgecolor='#ff4444',
                        linewidth=2, alpha=0.9)
        self.axes.add_patch(nucleus)

        # Add atomic symbol and number
        self.axes.text(0, 0, element.get('symbol', 'H'),
                      ha='center', va='center',
                      fontsize=12, fontweight='bold', color='white')

        # Add atomic number
        self.axes.text(0, -nucleus_radius-0.5, f'{atomic_number}',
                      ha='center', va='top',
                      fontsize=8, color='#cccccc')

    def draw_electron_orbits(self):
        """Draw electron orbits and electrons."""
        if not self.atomic_data:
            return

        config = self.atomic_data['config']
        shells = config.get('shells', [])

        calculator = BohrModelCalculator()

        for i, shell in enumerate(shells):
            n = shell['n']
            electrons = shell['electrons']
            max_electrons = shell['max_electrons']

            # Calculate orbit radius (simplified Bohr model)
            radius = 2 + i * 2.5  # Scale for visibility

            # Draw orbit
            orbit = Circle((0, 0), radius,
                          fill=False, edgecolor='#4ecdc4',
                          linewidth=2, alpha=0.6, linestyle='--')
            self.axes.add_patch(orbit)

            # Draw energy level label
            energy = calculator.calculate_energy_level(n, config.get('atomic_number', 1))
            self.axes.text(radius + 0.5, 0, f'n={n}\n{energy:.1f} eV',
                          ha='left', va='center',
                          fontsize=8, color='#cccccc')

            # Draw electrons
            if electrons > 0:
                angles = np.linspace(0, 2*np.pi, electrons, endpoint=False)
                electron_radius = 0.15

                for j, angle in enumerate(angles):
                    # Add animation offset
                    animated_angle = angle + np.radians(self.animation_angle * (j + 1) * 10)

                    x = radius * np.cos(animated_angle)
                    y = radius * np.sin(animated_angle)

                    # Draw electron
                    electron = Circle((x, y), electron_radius,
                                    facecolor='#ffff00', edgecolor='#ffaa00',
                                    linewidth=1, alpha=0.9)
                    self.axes.add_patch(electron)

                    # Add quantum numbers if enabled
                    if self.show_quantum_numbers and j == 0:
                        self.axes.text(x + 0.3, y + 0.3, f'n={n}',
                                     fontsize=6, color='#cccccc', alpha=0.8)

    def draw_quantum_probabilities(self):
        """Draw quantum probability clouds (simplified)."""
        if not self.show_electron_cloud or not self.atomic_data:
            return

        config = self.atomic_data['config']
        shells = config.get('shells', [])

        for shell in shells:
            n = shell['n']
            electrons = shell['electrons']

            if electrons > 0:
                radius = 2 + (n-1) * 2.5

                # Draw probability cloud (simplified as shaded ring)
                theta = np.linspace(0, 2*np.pi, 100)
                r_inner = radius - 0.5
                r_outer = radius + 0.5

                # Create shaded ring
                for r in np.linspace(r_inner, r_outer, 10):
                    alpha = 0.1 * (1 - abs(r - radius) / 0.5)
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    self.axes.fill(x, y, color='#4ecdc4', alpha=alpha)

    def start_animation(self):
        """Start electron animation."""
        if self.animation_timer is None:
            self.animation_timer = self.new_timer(interval=50)  # 20 FPS
            self.animation_timer.add_callback(self.animate_frame)
            self.animation_timer.start()
            self.is_animating = True


    def stop_animation(self):
        if self.animation_timer is not None:
            try:
                self.animation_timer.stop()
            except Exception:
                pass
            self.animation_timer = None
            self.is_animating = False
    
    def closeEvent(self, event):
        self.stop_animation()
        super().closeEvent(event)

    def animate_frame(self):
        if sip.isdeleted(self):
            return

        self.animation_angle = (self.animation_angle + 2) % 360
        self.redraw()

    def redraw(self):
        """Redraw the entire visualization."""
        self.setup_plot()

        if self.atomic_data:
            self.draw_nucleus()
            self.draw_electron_orbits()
            self.draw_quantum_probabilities()

        self.draw()


class BohrModel3DCanvas(FigureCanvasQTAgg):
    """3D Canvas for Bohr model visualization."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.figure.add_subplot(111, projection='3d')

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data
        self.atomic_data = {}
        self.show_quantum_numbers = True
        self.show_orbitals = False
        self.animation_angle = 0
        self.is_animating = False
        self.animation_timer = None
        self.view_angle = [30, 45]  # elevation, azimuth

        self.setup_plot()

    def setup_plot(self):
        """Initialize the 3D plot."""
        self.axes.clear()
        self.axes.set_facecolor('#0a0a0a')

        # Set pane colors
        self.axes.xaxis.pane.fill = False
        self.axes.yaxis.pane.fill = False
        self.axes.zaxis.pane.fill = False
        self.axes.xaxis.pane.set_edgecolor('#333333')
        self.axes.yaxis.pane.set_edgecolor('#333333')
        self.axes.zaxis.pane.set_edgecolor('#333333')

        # Set labels
        self.axes.set_xlabel('X', color='white', fontsize=10)
        self.axes.set_ylabel('Y', color='white', fontsize=10)
        self.axes.set_zlabel('Z', color='white', fontsize=10)

        # Set tick colors
        self.axes.tick_params(colors='white')

        # Set title
        self.axes.set_title('Bohr Atomic Model - 3D View',
                          color='white', fontsize=14, fontweight='bold', pad=20)

        # Set initial view
        self.axes.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        # Set equal aspect
        self.axes.set_box_aspect([1, 1, 1])

    def update_atom(self, element_data: Dict, electron_config: Dict):
        """Update the atom being displayed."""
        self.atomic_data = {
            'element': element_data,
            'config': electron_config
        }
        self.redraw()

    def draw_nucleus_3d(self):
        """Draw 3D nucleus."""
        if not self.atomic_data:
            return

        element = self.atomic_data['element']
        atomic_number = element.get('number', 1)

        # Create sphere for nucleus
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v)) * 0.5
        y = np.outer(np.sin(u), np.sin(v)) * 0.5
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.5

        self.axes.plot_surface(x, y, z, color='#ff6b6b', alpha=0.9)

        # Add atomic symbol
        self.axes.text(0, 0, 0.8, element.get('symbol', 'H'),
                      ha='center', va='center',
                      fontsize=12, fontweight='bold', color='white')

    def draw_electron_orbits_3d(self):
        """Draw 3D electron orbits."""
        if not self.atomic_data:
            return

        config = self.atomic_data['config']
        shells = config.get('shells', [])

        calculator = BohrModelCalculator()

        for i, shell in enumerate(shells):
            n = shell['n']
            electrons = shell['electrons']

            if electrons == 0:
                continue

            radius = 2 + i * 2.5

            # Draw orbital rings in different planes
            planes = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]  # xy, xz, yz planes

            for plane_idx, (a, b, c) in enumerate(planes):
                theta = np.linspace(0, 2*np.pi, 100)

                if plane_idx == 0:  # xy plane
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    z = np.zeros_like(theta)
                elif plane_idx == 1:  # xz plane
                    x = radius * np.cos(theta)
                    y = np.zeros_like(theta)
                    z = radius * np.sin(theta)
                else:  # yz plane
                    x = np.zeros_like(theta)
                    y = radius * np.cos(theta)
                    z = radius * np.sin(theta)

                self.axes.plot(x, y, z, color='#4ecdc4', alpha=0.6, linewidth=2)

                # Draw electrons on orbits
                if electrons > 0:
                    electron_angles = np.linspace(0, 2*np.pi, electrons, endpoint=False)

                    for j, angle in enumerate(electron_angles):
                        animated_angle = angle + np.radians(self.animation_angle * (j + 1) * 5)

                        if plane_idx == 0:
                            ex = radius * np.cos(animated_angle)
                            ey = radius * np.sin(animated_angle)
                            ez = 0
                        elif plane_idx == 1:
                            ex = radius * np.cos(animated_angle)
                            ey = 0
                            ez = radius * np.sin(animated_angle)
                        else:
                            ex = 0
                            ey = radius * np.cos(animated_angle)
                            ez = radius * np.sin(animated_angle)

                        # Draw electron as small sphere
                        u = np.linspace(0, 2*np.pi, 10)
                        v = np.linspace(0, np.pi, 10)
                        ex_surf = np.outer(np.cos(u), np.sin(v)) * 0.1 + ex
                        ey_surf = np.outer(np.sin(u), np.sin(v)) * 0.1 + ey
                        ez_surf = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.1 + ez

                        self.axes.plot_surface(ex_surf, ey_surf, ez_surf,
                                             color='#ffff00', alpha=0.9)

    def draw_quantum_orbitals_3d(self):
        """Draw quantum mechanical orbitals (simplified)."""
        if not self.show_orbitals or not self.atomic_data:
            return

        config = self.atomic_data['config']
        shells = config.get('shells', [])

        for shell in shells:
            n = shell['n']
            electrons = shell['electrons']

            if electrons > 0:
                radius = 2 + (n-1) * 2.5

                # Draw simplified s-orbital (spherical cloud)
                if any(s['l'] == 0 for s in shell.get('subshells', [])):
                    u = np.linspace(0, 2*np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    x = np.outer(np.cos(u), np.sin(v)) * (radius + 1)
                    y = np.outer(np.sin(u), np.sin(v)) * (radius + 1)
                    z = np.outer(np.ones(np.size(u)), np.cos(v)) * (radius + 1)

                    self.axes.plot_surface(x, y, z, color='#45b7d1', alpha=0.1)

                # Draw simplified p-orbitals (dumbbell shapes)
                if any(s['l'] == 1 for s in shell.get('subshells', [])):
                    # pz orbital
                    z_vals = np.linspace(-radius*2, radius*2, 50)
                    x = radius * np.cos(z_vals / (radius*2) * np.pi)
                    y = np.zeros_like(z_vals)
                    z = z_vals
                    self.axes.plot(x, y, z, color='#ffaa00', alpha=0.3, linewidth=3)

    def start_animation(self):
        """Start 3D animation."""
        if self.animation_timer is None:
            self.animation_timer = self.new_timer(interval=50)
            self.animation_timer.add_callback(self.animate_frame)
            self.animation_timer.start()
            self.is_animating = True

    def stop_animation(self):
        if self.animation_timer is not None:
            try:
                self.animation_timer.stop()
            except Exception:
                pass
            self.animation_timer = None
            self.is_animating = False

    def closeEvent(self, event):
        self.stop_animation()
        super().closeEvent(event)

    def animate_frame(self):
        if sip.isdeleted(self):
            return

        self.animation_angle = (self.animation_angle + 2) % 360
        self.redraw()

    def set_view_angle(self, elev: float, azim: float):
        """Set camera view angle."""
        self.view_angle = [elev, azim]
        self.axes.view_init(elev=elev, azim=azim)
        self.redraw()

    def redraw(self):
        """Redraw the entire 3D visualization."""
        self.setup_plot()

        if self.atomic_data:
            self.draw_nucleus_3d()
            self.draw_electron_orbits_3d()
            self.draw_quantum_orbitals_3d()

        self.draw()


class BohrModelViewer(ScienceHubTool):
    """Comprehensive Bohr Model Viewer with 2D/3D visualization and atomic data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize calculators and data
        self.calculator = BohrModelCalculator()
        self.constants = PhysicsConstants()

        # Load periodic table data
        self.elements_data = self.load_periodic_table()

        # Current atom data
        self.current_element = None
        self.current_config = None

        self.canvas_2d = None
        self.canvas_3d = None

        # Setup UI
        self.setup_ui()

        QTimer.singleShot(0, lambda: self.load_element_by_number(1))

    def load_periodic_table(self) -> List[Dict]:
        """Load periodic table data."""
        try:
            with open('sciencehub/data/datasets/ptable.json', 'r') as f:
                data = json.load(f)
                return data.get('elements', [])
        except FileNotFoundError:
            # Fallback to basic elements
            return [
                {'number': 1, 'symbol': 'H', 'name': 'Hydrogen', 'atomic_mass': 1.008},
                {'number': 2, 'symbol': 'He', 'name': 'Helium', 'atomic_mass': 4.002},
                {'number': 6, 'symbol': 'C', 'name': 'Carbon', 'atomic_mass': 12.011},
                {'number': 8, 'symbol': 'O', 'name': 'Oxygen', 'atomic_mass': 15.999},
                # Add more as needed
            ]

    def setup_ui(self):
        """Setup the main user interface."""
        # Create main layout
        main_layout = self.layout()
        if main_layout is None: 
            main_layout = QHBoxLayout()
            self.setLayout(main_layout)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls and Data
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Visualizations
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 800])

    def create_left_panel(self) -> QWidget:
        """Create the left control and data panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Element Selection
        element_group = QGroupBox("Element Selection")
        element_group.setObjectName("toolCard")
        element_layout = QVBoxLayout(element_group)

        # Search/Filter
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter element name or symbol...")
        self.search_input.textChanged.connect(self.filter_elements)
        search_layout.addWidget(self.search_input)
        element_layout.addLayout(search_layout)

        # Element selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Element:"))
        self.element_combo = QComboBox()
        selector_layout.addWidget(self.element_combo)
        element_layout.addLayout(selector_layout)

        # Atomic number input
        number_layout = QHBoxLayout()
        number_layout.addWidget(QLabel("Atomic Number:"))
        self.atomic_number_spin = QSpinBox()
        self.atomic_number_spin.setRange(1, 118)
        self.atomic_number_spin.setValue(1)
        self.atomic_number_spin.valueChanged.connect(self.on_atomic_number_changed)
        number_layout.addWidget(self.atomic_number_spin)
        element_layout.addLayout(number_layout)

        layout.addWidget(element_group)

        # Atomic Data Display
        data_tabs = QTabWidget()

        # Basic Properties
        basic_tab = self.create_basic_properties_tab()
        data_tabs.addTab(basic_tab, "Basic Properties")

        # Electron Configuration
        config_tab = self.create_electron_config_tab()
        data_tabs.addTab(config_tab, "Electron Configuration")

        # Quantum Properties
        quantum_tab = self.create_quantum_properties_tab()
        data_tabs.addTab(quantum_tab, "Quantum Properties")

        # Bohr Model Parameters
        bohr_tab = self.create_bohr_parameters_tab()
        data_tabs.addTab(bohr_tab, "Bohr Parameters")

        layout.addWidget(data_tabs)

        # Visualization Controls
        vis_group = QGroupBox("Visualization Controls")
        vis_group.setObjectName("toolCard")
        vis_layout = QVBoxLayout(vis_group)

        # 2D Controls
        self.show_2d_quantum = QCheckBox("Show quantum numbers (2D)")
        self.show_2d_quantum.setChecked(True)
        self.show_2d_quantum.stateChanged.connect(self.update_2d_visualization)
        vis_layout.addWidget(self.show_2d_quantum)

        self.show_2d_cloud = QCheckBox("Show electron cloud (2D)")
        self.show_2d_cloud.stateChanged.connect(self.update_2d_visualization)
        vis_layout.addWidget(self.show_2d_cloud)

        # 3D Controls
        self.show_3d_quantum = QCheckBox("Show quantum numbers (3D)")
        self.show_3d_quantum.setChecked(True)
        self.show_3d_quantum.stateChanged.connect(self.update_3d_visualization)
        vis_layout.addWidget(self.show_3d_quantum)

        self.show_3d_orbitals = QCheckBox("Show quantum orbitals (3D)")
        self.show_3d_orbitals.stateChanged.connect(self.update_3d_visualization)
        vis_layout.addWidget(self.show_3d_orbitals)

        # Animation controls
        anim_layout = QHBoxLayout()
        self.animate_2d_btn = QPushButton("Animate 2D")
        self.animate_2d_btn.setCheckable(True)
        self.animate_2d_btn.clicked.connect(self.toggle_2d_animation)
        anim_layout.addWidget(self.animate_2d_btn)

        self.animate_3d_btn = QPushButton("Animate 3D")
        self.animate_3d_btn.setCheckable(True)
        self.animate_3d_btn.clicked.connect(self.toggle_3d_animation)
        anim_layout.addWidget(self.animate_3d_btn)
        vis_layout.addLayout(anim_layout)

        layout.addWidget(vis_group)

        # Populate element combo
        self.populate_element_combo()

        self.element_combo.currentIndexChanged.connect(self.on_element_changed)

        return panel

    def create_basic_properties_tab(self) -> QWidget:
        """Create basic properties display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create table for properties
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties_table.horizontalHeader().setStretchLastSection(True)
        self.properties_table.setAlternatingRowColors(True)

        layout.addWidget(self.properties_table)

        return widget

    def create_electron_config_tab(self) -> QWidget:
        """Create electron configuration display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Configuration display
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(200)
        layout.addWidget(self.config_text)

        # Shell details table
        self.shell_table = QTableWidget()
        self.shell_table.setColumnCount(4)
        self.shell_table.setHorizontalHeaderLabels(["Shell (n)", "Electrons", "Max Electrons", "Subshells"])
        self.shell_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.shell_table)

        return widget

    def create_quantum_properties_tab(self) -> QWidget:
        """Create quantum properties display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Quantum numbers explanation
        quantum_info = QTextEdit()
        quantum_info.setReadOnly(True)
        quantum_info.setPlainText("""
Quantum Numbers:
• Principal (n): Energy level / shell number
• Azimuthal (l): Orbital shape (s=0, p=1, d=2, f=3)
• Magnetic (m_l): Orbital orientation
• Spin (m_s): Electron spin (±1/2)

Energy Levels (Bohr Model):
E_n = -13.6 × Z² / n² eV

Orbital Velocity:
v_n = Z × e² / (2 × ε₀ × h × n)
        """)
        quantum_info.setMaximumHeight(150)
        layout.addWidget(quantum_info)

        # Quantum calculations table
        self.quantum_table = QTableWidget()
        self.quantum_table.setColumnCount(4)
        self.quantum_table.setHorizontalHeaderLabels(["n", "Energy (eV)", "Radius (a₀)", "Velocity (m/s)"])
        layout.addWidget(self.quantum_table)

        return widget

    def create_bohr_parameters_tab(self) -> QWidget:
        """Create Bohr model parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Bohr model explanation
        bohr_info = QLabel("""
The Bohr model describes atoms as having electrons orbiting the nucleus
in quantized energy levels. Key assumptions:

• Electrons orbit the nucleus in circular paths
• Angular momentum is quantized: L = n × h/(2π)
• Energy is quantized: E_n = -13.6 × Z² / n² eV
• Electrons can jump between energy levels by absorbing/emitting photons

Limitations:
• Doesn't account for electron wave nature
• Can't explain multi-electron atoms properly
• Contradicts Heisenberg uncertainty principle
        """)
        bohr_info.setWordWrap(True)
        layout.addWidget(bohr_info)

        # Bohr parameters table
        self.bohr_table = QTableWidget()
        self.bohr_table.setColumnCount(3)
        self.bohr_table.setHorizontalHeaderLabels(["Parameter", "Formula", "Value"])
        layout.addWidget(self.bohr_table)

        return widget

    def create_right_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Visualization tabs
        vis_tabs = QTabWidget()

        # 2D View
        self.canvas_2d = BohrModel2DCanvas(self)
        vis_tabs.addTab(self.canvas_2d, "2D Bohr Model")

        # 3D View
        self.canvas_3d = BohrModel3DCanvas(self)
        vis_tabs.addTab(self.canvas_3d, "3D Bohr Model")

        layout.addWidget(vis_tabs)

        # 3D Camera controls
        camera_group = QGroupBox("3D Camera Controls")
        camera_group.setObjectName("toolCard")
        camera_layout = QVBoxLayout(camera_group)

        # Elevation
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

        # Azimuth
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

        layout.addWidget(camera_group)

        return panel

    def populate_element_combo(self):
        """Populate the element selection combo box."""
        self.element_combo.clear()
        for element in self.elements_data:
            symbol = element.get('symbol', '')
            name = element.get('name', '')
            atomic_number = element.get('number', '')
            display_text = f"{atomic_number:3d} {symbol:2s} - {name}"
            self.element_combo.addItem(display_text, element)

    def filter_elements(self, text: str):
        """Filter elements based on search text."""
        if not text:
            self.populate_element_combo()
            return

        self.element_combo.clear()
        text = text.lower()
        for element in self.elements_data:
            symbol = element.get('symbol', '').lower()
            name = element.get('name', '').lower()
            if text in symbol or text in name:
                atomic_number = element.get('number', '')
                display_text = f"{atomic_number:3d} {symbol.upper():2s} - {name}"
                self.element_combo.addItem(display_text, element)

    def on_element_changed(self, index: int):
        """Handle element selection change."""
        if index >= 0:
            element = self.element_combo.itemData(index)
            if element:
                self.load_element(element)

    def on_atomic_number_changed(self, value: int):
        """Handle atomic number change."""
        self.load_element_by_number(value)

    def load_element_by_number(self, atomic_number: int):
        """Load element by atomic number."""
        for element in self.elements_data:
            if element.get('number') == atomic_number:
                self.load_element(element)
                break

    def load_element(self, element: Dict):
        """Load and display element data."""
        self.current_element = element
        atomic_number = element.get('number', 1)

        # Update spin box
        self.atomic_number_spin.blockSignals(True)
        self.atomic_number_spin.setValue(atomic_number)
        self.atomic_number_spin.blockSignals(False)

        # Calculate electron configuration
        self.current_config = self.calculator.get_electron_configuration(atomic_number)

        # Update displays
        self.update_basic_properties()
        self.update_electron_config()
        self.update_quantum_properties()
        self.update_bohr_parameters()

        # Update visualizations
        if self.canvas_2d is not None:
            self.canvas_2d.update_atom(element, self.current_config)

        if self.canvas_3d is not None:
            self.canvas_3d.update_atom(element, self.current_config)

    def update_basic_properties(self):
        """Update basic properties table."""
        if not self.current_element:
            return

        element = self.current_element
        properties = [
            ("Name", element.get('name', 'Unknown')),
            ("Symbol", element.get('symbol', '??')),
            ("Atomic Number", str(element.get('number', 0))),
            ("Atomic Mass", f"{element.get('atomic_mass', 0):.3f} u"),
            ("Category", element.get('category', 'Unknown')),
            ("Phase", element.get('phase', 'Unknown')),
            ("Density", f"{element.get('density', 0):.3f} g/cm³" if element.get('density') else 'Unknown'),
            ("Melting Point", f"{element.get('melt', 0):.1f} K" if element.get('melt') else 'Unknown'),
            ("Boiling Point", f"{element.get('boil', 0):.1f} K" if element.get('boil') else 'Unknown'),
            ("Electronegativity", f"{element.get('electronegativity_pauling', 0):.2f}" if element.get('electronegativity_pauling') else 'Unknown'),
            ("Electron Affinity", f"{element.get('electron_affinity', 0):.1f} kJ/mol" if element.get('electron_affinity') else 'Unknown'),
        ]

        self.properties_table.setRowCount(len(properties))
        for i, (prop, value) in enumerate(properties):
            self.properties_table.setItem(i, 0, QTableWidgetItem(prop))
            self.properties_table.setItem(i, 1, QTableWidgetItem(value))

        self.properties_table.resizeColumnsToContents()

    def update_electron_config(self):
        """Update electron configuration display."""
        if not self.current_config:
            return

        config = self.current_config

        # Configuration string
        config_text = f"""
Total Electrons: {config.get('total_electrons', 0)}
Configuration: {config.get('configuration_string', '')}

Shell Structure:
"""
        for shell in config.get('shells', []):
            config_text += f"  n={shell['n']}: {shell['electrons']}/{shell['max_electrons']} electrons\n"

        self.config_text.setPlainText(config_text.strip())

        # Shell table
        shells = config.get('shells', [])
        self.shell_table.setRowCount(len(shells))
        for i, shell in enumerate(shells):
            self.shell_table.setItem(i, 0, QTableWidgetItem(str(shell['n'])))
            self.shell_table.setItem(i, 1, QTableWidgetItem(str(shell['electrons'])))
            self.shell_table.setItem(i, 2, QTableWidgetItem(str(shell['max_electrons'])))

            # Subshells
            subshell_text = ', '.join([f"{s['name']}: {s['electrons']}" for s in shell.get('subshells', [])])
            self.shell_table.setItem(i, 3, QTableWidgetItem(subshell_text))

        self.shell_table.resizeColumnsToContents()

    def update_quantum_properties(self):
        """Update quantum properties table."""
        if not self.current_config:
            return

        atomic_number = self.current_config.get('atomic_number', 1)
        shells = self.current_config.get('shells', [])

        # Calculate for each occupied shell
        quantum_data = []
        for shell in shells:
            n = shell['n']
            if shell['electrons'] > 0:
                energy = self.calculator.calculate_energy_level(n, atomic_number)
                radius = self.calculator.calculate_bohr_radius(n, atomic_number)
                velocity = self.calculator.calculate_orbital_velocity(n, atomic_number)

                quantum_data.append((n, energy, radius, velocity))

        self.quantum_table.setRowCount(len(quantum_data))
        for i, (n, energy, radius, velocity) in enumerate(quantum_data):
            self.quantum_table.setItem(i, 0, QTableWidgetItem(str(n)))
            self.quantum_table.setItem(i, 1, QTableWidgetItem(f"{energy:.2f}"))
            self.quantum_table.setItem(i, 2, QTableWidgetItem(f"{radius:.3f}"))
            self.quantum_table.setItem(i, 3, QTableWidgetItem(f"{velocity:.2e}"))

        self.quantum_table.resizeColumnsToContents()

    def update_bohr_parameters(self):
        """Update Bohr parameters table."""
        if not self.current_element:
            return

        element = self.current_element
        atomic_number = element.get('number', 1)

        parameters = [
            ("Rydberg Constant", "R = 1.097 × 10⁷ m⁻¹", "1.097e7"),
            ("Bohr Radius", "a₀ = 5.292 × 10⁻¹¹ m", "5.292e-11"),
            ("Ground State Energy", "E₁ = -13.6 eV", "-13.6"),
            ("Ionization Energy", f"I = 13.6 × Z² eV", f"{13.6 * atomic_number**2:.1f}"),
            ("Reduced Planck Constant", "ℏ = 1.055 × 10⁻³⁴ J⋅s", "1.055e-34"),
        ]

        self.bohr_table.setRowCount(len(parameters))
        for i, (param, formula, value) in enumerate(parameters):
            self.bohr_table.setItem(i, 0, QTableWidgetItem(param))
            self.bohr_table.setItem(i, 1, QTableWidgetItem(formula))
            self.bohr_table.setItem(i, 2, QTableWidgetItem(value))

        self.bohr_table.resizeColumnsToContents()

    def update_2d_visualization(self):
        """Update 2D visualization settings."""
        self.canvas_2d.show_quantum_numbers = self.show_2d_quantum.isChecked()
        self.canvas_2d.show_electron_cloud = self.show_2d_cloud.isChecked()
        self.canvas_2d.redraw()

    def update_3d_visualization(self):
        """Update 3D visualization settings."""
        self.canvas_3d.show_quantum_numbers = self.show_3d_quantum.isChecked()
        self.canvas_3d.show_orbitals = self.show_3d_orbitals.isChecked()
        self.canvas_3d.redraw()

    def toggle_2d_animation(self):
        """Toggle 2D animation."""
        if self.animate_2d_btn.isChecked():
            self.canvas_2d.start_animation()
        else:
            self.canvas_2d.stop_animation()

    def toggle_3d_animation(self):
        """Toggle 3D animation."""
        if self.animate_3d_btn.isChecked():
            self.canvas_3d.start_animation()
        else:
            self.canvas_3d.stop_animation()

    def on_camera_changed(self):
        """Handle camera angle changes."""
        elev = self.elev_slider.value()
        azim = self.azim_slider.value()

        self.elev_label.setText(f"{elev}°")
        self.azim_label.setText(f"{azim}°")

        self.canvas_3d.set_view_angle(elev, azim)

    def reset_views(self):
        """Reset all views to default."""
        # Reset camera
        self.elev_slider.setValue(30)
        self.azim_slider.setValue(45)

        # Stop animations
        self.animate_2d_btn.setChecked(False)
        self.animate_3d_btn.setChecked(False)
        self.canvas_2d.stop_animation()
        self.canvas_3d.stop_animation()

        # Reset visualizations
        self.canvas_2d.redraw()
        self.canvas_3d.redraw()

    def export_2d_view(self):
        """Export 2D view as image."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export 2D View", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            self.canvas_2d.figure.savefig(filename, dpi=300, bbox_inches='tight')

    def export_3d_view(self):
        """Export 3D view as image."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export 3D View", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            self.canvas_3d.figure.savefig(filename, dpi=300, bbox_inches='tight')

    def export_atomic_data(self):
        """Export atomic data as JSON."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Atomic Data", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump({
                        'element': self.current_element,
                        'configuration': self.current_config
                    }, f, indent=2)
            elif filename.endswith('.csv'):
                # Export as CSV (simplified)
                with open(filename, 'w') as f:
                    f.write("Property,Value\n")
                    if self.current_element:
                        for key, value in self.current_element.items():
                            f.write(f"{key},{value}\n")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Bohr Model Viewer",
            "Bohr Model Viewer v1.0\n\n"
            "An interactive tool for visualizing atomic structure using the Bohr model.\n\n"
            "Features:\n"
            "• 2D and 3D atomic visualizations\n"
            "• Complete atomic data display\n"
            "• Electron configuration analysis\n"
            "• Quantum mechanical properties\n"
            "• Real-time parameter adjustment\n"
            "• Export capabilities\n\n"
            "Ready for quantum model integration."
        )


TOOL_META = {
    "name": "Bohr Model Viewer",
    "description": "Interactive 2D/3D visualization of atomic structure using the Bohr model with complete atomic data display and quantum mechanics integration",
    "category": "Chemistry",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "Interactive atom selection from periodic table",
        "2D Bohr model visualization with electron orbits",
        "3D Bohr model visualization with orbital representation",
        "Complete atomic properties database",
        "Electron configuration analysis",
        "Quantum mechanical calculations",
        "Real-time parameter adjustment",
        "Animation capabilities",
        "Export functionality",
        "Quantum model integration framework",
        "Educational information panels",
        "Beautiful modern UI design"
    ],
    "educational_value": "Explore atomic structure, understand electron configuration, learn quantum numbers, and visualize the Bohr model interactively",
    "keywords": ["bohr model", "atomic structure", "electron configuration", "quantum numbers", "periodic table", "3d visualization", "chemistry education"]
}


def create_tool(parent=None):
    """Create and return a BohrModelViewer instance."""
    return BohrModelViewer(parent)
