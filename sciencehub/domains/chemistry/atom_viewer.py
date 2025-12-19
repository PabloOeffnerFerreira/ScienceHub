"""
Atom Viewer - Interactive Atomic Structure Visualization
===========================================================

A comprehensive tool for visualizing atomic structures with 2D and 3D views,
complete atomic data display, and quantum mechanics integration capabilities.

Features:
- Interactive atom selection from periodic table
- Complete atomic data display (properties, electron configuration, etc.)
- 2D atomic model visualization with electron orbits
- 3D atomic model visualization with orbital representation
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
from scipy.special import sph_harm, genlaguerre, factorial
from scipy.integrate import quad
import cmath

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


class HydrogenLikeOrbitalCalculator:
    """Hydrogen-like atomic orbitals using effective nuclear charge."""

    def __init__(self):
        self.constants = PhysicsConstants()
        self.a0 = self.constants.BOHR_RADIUS  # Bohr radius

    def radial_wavefunction(self, n: int, l: int, r: float, Z: int = 1) -> float:
        """
        Calculate the radial wavefunction R_nl(r) for hydrogen-like atoms.

        Parameters:
        n: Principal quantum number
        l: Azimuthal quantum number
        r: Radial distance (in units of a0)
        Z: Atomic number

        Returns:
        Radial wavefunction value
        """
        rho = 2 * Z * r / (n * self.a0)

        # Associated Laguerre polynomial
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)
        L = laguerre(rho)

        # Normalization factor
        norm_factor = np.sqrt((2 * Z / (n * self.a0))**3 *
                             factorial(n - l - 1) / (2 * n * factorial(n + l)))

        # Radial wavefunction
        R = norm_factor * np.exp(-rho / 2) * rho**l * L

        return float(R)

    def angular_wavefunction(self, l: int, m: int, theta: float, phi: float) -> complex:
        """
        Calculate the angular wavefunction Y_lm(theta, phi).

        Parameters:
        l: Azimuthal quantum number
        m: Magnetic quantum number
        theta: Polar angle
        phi: Azimuthal angle

        Returns:
        Angular wavefunction value (complex)
        """
        # Spherical harmonics
        Y_lm = sph_harm(m, l, phi, theta)
        return complex(Y_lm)

    def wavefunction(self, n: int, l: int, m: int, r: float, theta: float, phi: float, Z: int = 1) -> complex:
        """
        Calculate the complete wavefunction ψ_nlm(r, θ, φ).

        Parameters:
        n, l, m: Quantum numbers
        r, theta, phi: Spherical coordinates
        Z: Atomic number

        Returns:
        Complete wavefunction value
        """
        R = self.radial_wavefunction(n, l, r, Z)
        Y = self.angular_wavefunction(l, m, theta, phi)
        return R * Y

    def probability_density(self, n: int, l: int, m: int, r: float, theta: float, phi: float, Z: int = 1) -> float:
        """
        Calculate the probability density |ψ|².

        Returns:
        Probability density
        """
        psi = self.wavefunction(n, l, m, r, theta, phi, Z)
        return abs(psi)**2

    def radial_probability_density(self, n: int, l: int, r: float, Z: int = 1) -> float:
        """
        Calculate the radial probability density P(r) = r²|R_nl(r)|².

        Returns:
        Radial probability density
        """
        R = self.radial_wavefunction(n, l, r, Z)
        return r**2 * abs(R)**2

    def energy_level(self, n: int, Z: int = 1) -> float:
        """
        Calculate the energy level for quantum state (n).

        Returns:
        Energy in eV
        """
        # E_n = -13.6 * Z² / n² eV
        return -13.6 * (Z ** 2) / (n ** 2)

    def expectation_value_r(self, n: int, l: int, Z: int = 1) -> float:
        """
        Calculate the expectation value of r for state (n,l).

        Returns:
        <r> in units of a0
        """
        # <r> = (n² a0 / 2Z) * [3n² - l(l+1)]
        return (n**2 * self.a0 / (2 * Z)) * (3 * n**2 - l * (l + 1))

    def orbital_names(self) -> Dict[Tuple[int, int], str]:
        """
        Get orbital names for different (l, m) combinations.

        Returns:
        Dictionary mapping (l, m) to orbital names
        """
        orbital_names = {}
        subshell_letters = ['s', 'p', 'd', 'f', 'g', 'h']

        for l in range(6):
            for m in range(-l, l + 1):
                if l < len(subshell_letters):
                    if m == 0:
                        name = subshell_letters[l]
                    elif m > 0:
                        name = f"{subshell_letters[l]}{m}"
                    else:
                        name = f"{subshell_letters[l]}{-m}*"
                    orbital_names[(l, m)] = name

        return orbital_names

    def get_quantum_states(self, max_n: int = 4) -> List[Tuple[int, int, int]]:
        """
        Get all valid quantum states up to max_n.

        Returns:
        List of (n, l, m) tuples
        """
        states = []
        for n in range(1, max_n + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    states.append((n, l, m))
        return states

    def calculate_orbital_data(
        self,
        n: int,
        l: int,
        m: int,
        Z: float = 1.0,
        resolution: int = 50
    ) -> Dict[str, Any]:
        """
        Calculate hydrogen-like orbital data for visualization.

        This implementation:
        • is fully vectorized
        • preserves angular structure (lobes, nodes)
        • separates radial and angular components
        • supports Z_eff transparently

        Returns fields suitable for both
        - textbook orbital shapes
        - probability-density plots
        """

        # -------------------------------
        # Coordinate grids
        # -------------------------------
        r_max = 5.0 * n**2 * self.a0 / Z
        r = np.linspace(0.01 * self.a0, r_max, resolution * 2)
        theta = np.linspace(0.0, np.pi, resolution)
        phi = np.linspace(0.0, 2.0 * np.pi, resolution)

        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing="ij")

        # -------------------------------
        # Angular wavefunction Y_l^m
        # -------------------------------
        Y_lm = sph_harm(m, l, Phi, Theta)

        # -------------------------------
        # Radial wavefunction R_nl(r)
        # -------------------------------
        rho = 2.0 * Z * R / (n * self.a0)

        laguerre = genlaguerre(n - l - 1, 2 * l + 1)
        L = laguerre(rho)

        norm = np.sqrt(
            (2 * Z / (n * self.a0)) ** 3
            * factorial(n - l - 1)
            / (2 * n * factorial(n + l))
        )

        R_nl = norm * np.exp(-rho / 2.0) * rho**l * L

        # -------------------------------
        # Full wavefunction ψ = R · Y
        # -------------------------------
        psi = R_nl * Y_lm

        # -------------------------------
        # Probability density
        # -------------------------------
        prob_density = np.abs(psi) ** 2
        prob_density /= np.max(prob_density)

        # -------------------------------
        # Cartesian coordinates
        # -------------------------------
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Zc = R * np.cos(Theta)

        # -------------------------------
        # Return structured data
        # -------------------------------
        return {
            "quantum_numbers": (n, l, m),
            "energy": self.energy_level(n, Z),

            # geometry
            "X": X,
            "Y": Y,
            "Z": Zc,

            # full fields
            "psi": psi,
            "probability_density": prob_density,

            # separated components (IMPORTANT)
            "radial_part": R_nl,
            "angular_complex": Y_lm,
            "angular_real": np.real(Y_lm),
            "angular_abs": np.abs(Y_lm),

            # metadata
            "radial_coords": r,
            "angular_coords": (theta, phi),
            "expectation_r": self.expectation_value_r(n, l, Z),
        }


class EffectiveNuclearCharge:
    """Compute Z_eff using Slater's rules."""

    @staticmethod
    def compute_Z_eff(Z: int, n: int, l: int, config: Dict) -> float:
        S = 0.0

        for shell in config["shells"]:
            n_shell = shell["n"]
            for subshell in shell["subshells"]:
                electrons = subshell["electrons"]
                l_shell = subshell["l"]

                if n_shell > n:
                    continue

                if n_shell == n:
                    if l == 0:
                        S += 0.30 * max(0, electrons - 1)
                    else:
                        S += 0.35 * max(0, electrons - 1)
                elif n_shell == n - 1:
                    S += 0.85 * electrons
                else:
                    S += 1.00 * electrons

        return max(1.0, Z - S)


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

class QuantumOrbitalCanvas(FigureCanvasQTAgg):
    """3D Canvas for quantum orbital visualization."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.figure.add_subplot(111, projection='3d')

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data
        self.orbital_data = {}
        self.show_probability = True
        self.show_phase = False
        self.opacity = 0.3
        self.isosurface_level = 0.1
        self.view_angle = [30, 45]  # elevation, azimuth

        self.setup_plot()

    def setup_plot(self):
        """Initialize the 3D quantum orbital plot."""
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
        self.axes.set_xlabel('X (a₀)', color='white', fontsize=10)
        self.axes.set_ylabel('Y (a₀)', color='white', fontsize=10)
        self.axes.set_zlabel('Z (a₀)', color='white', fontsize=10)

        # Set tick colors
        self.axes.tick_params(colors='white')

        # Set title
        self.axes.set_title('Quantum Orbital - 3D Probability Density',
                          color='white', fontsize=14, fontweight='bold', pad=20)

        # Set initial view
        self.axes.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        # Set equal aspect
        self.axes.set_box_aspect([1, 1, 1])

    def update_orbital(self, orbital_data: Dict):
        """Update the orbital being displayed."""
        self.orbital_data = orbital_data
        self.redraw()

    def draw_quantum_orbital(self):
        """Draw the quantum orbital probability density."""
        if not self.orbital_data:
            return

        X = self.orbital_data['X']
        Y = self.orbital_data['Y']
        Z = self.orbital_data['Z']
        r = self.orbital_data['radial_coords']
        theta, phi = self.orbital_data['angular_coords']

        n, l, m = self.orbital_data['quantum_numbers']

        if l == 0:
            # s orbitals → probability density
            data = self.orbital_data['probability_density']
            mode = "density"
            title = "Probability Density |ψ|²"
        else:
            # p, d, f orbitals → signed wavefunction
            if l == 0:
                data = self.orbital_data['probability_density']
                mode = "density"
                title = "Probability Density |ψ|²"
            else:
                # PURE ANGULAR SHAPE (this is the key)
                angular = self.orbital_data["angular_real"]
                radial = self.orbital_data["radial_part"]

                # weak radial envelope to localize shape
                data = angular * np.exp(-radial / np.max(radial))

                mode = "angular"
                title = "Angular Wavefunction (orbital shape)"

            mode = "wavefunction"
            title = "Wavefunction ψ (signed)"   


        # Ensure isosurface level is within data range
        if mode == "density":
            level = self.isosurface_level
        else:
            max_amp = np.max(np.abs(data))
            level = 0.2


        # --- Optimized isosurface rendering ---
        try:
            from skimage import measure

            # 1. Downsample data for marching cubes (huge speedup)
            step = max(1, data.shape[0] // 40)
            data_ds = data[::step, ::step, ::step]

            rv_step = r[1] - r[0]
            th_step = theta[1] - theta[0]
            ph_step = phi[1] - phi[0]

            def render_surface(surface_data, level, color):
                verts, faces, _, _ = measure.marching_cubes(surface_data, level=level)
                verts = verts * step

                rv = r[0] + verts[:, 0] * rv_step
                thetav = theta[0] + verts[:, 1] * th_step
                phiv = phi[0] + verts[:, 2] * ph_step

                xv = rv * np.sin(thetav) * np.cos(phiv)
                yv = rv * np.sin(thetav) * np.sin(phiv)
                zv = rv * np.cos(thetav)

                self.axes.plot_trisurf(
                    xv, yv, faces, zv,
                    color=color,
                    alpha=self.opacity,
                    linewidth=0,
                    antialiased=False
                )

            if mode == "density":
                render_surface(data_ds, level, "cyan")
            else:
                # positive lobe
                render_surface(data_ds, +level, "red")
                # negative lobe
                render_surface(data_ds, -level, "blue")

        except Exception as e:
            # --- Fast fallback: stochastic point cloud ---
            print(f"Isosurface fallback: {e}")

            # Select high-probability region
            thresh = np.percentile(data, 97)
            idx = np.argwhere(data > thresh)

            # Cap number of points (critical)
            max_points = 8000
            if len(idx) > max_points:
                idx = idx[np.random.choice(len(idx), max_points, replace=False)]

            self.axes.scatter(
                X[tuple(idx.T)],
                Y[tuple(idx.T)],
                Z[tuple(idx.T)],
                c=data[tuple(idx.T)],
                cmap='viridis',
                alpha=self.opacity,
                s=2
            )

        # ---- Title (single source of truth) ----
        n, l, m = self.orbital_data['quantum_numbers']
        Z_eff = self.orbital_data.get("Z_eff", None)
        element = self.orbital_data.get("element", {})

        symbol = element.get("symbol", "")

        title_lines = [
            f"{symbol} n={n}, l={l}, m={m}",
            title
        ]

        if Z_eff is not None:
            title_lines.append(f"Hydrogen-like approximation (Z_eff ≈ {Z_eff:.2f})")

        self.axes.set_title(
            "\n".join(title_lines),
            color="white",
            fontsize=12,
            fontweight="bold"
        )

    def set_visualization_mode(self, show_probability: bool, show_phase: bool):
        """Set visualization mode."""
        self.show_probability = show_probability
        self.show_phase = show_phase
        self.redraw()

    def set_opacity(self, opacity: float):
        """Set isosurface opacity."""
        self.opacity = opacity
        self.redraw()

    def set_isosurface_level(self, level: float):
        """Set isosurface level."""
        self.isosurface_level = level
        self.redraw()

    def set_view_angle(self, elev: float, azim: float):
        """Set camera view angle."""
        self.view_angle = [elev, azim]
        self.axes.view_init(elev=elev, azim=azim)
        self.redraw()

    def redraw(self):
        """Redraw the entire quantum orbital visualization."""
        self.setup_plot()

        if self.orbital_data:
            self.draw_quantum_orbital()

        self.draw()


class RadialWavefunctionCanvas(FigureCanvasQTAgg):
    """Canvas for radial wavefunction and probability plots."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data
        self.wavefunction_data = {}
        self.show_radial_wavefunction = True
        self.show_radial_probability = True

        self.setup_plot()

    def setup_plot(self):
        """Initialize the radial plot."""
        self.axes.clear()
        self.axes.set_facecolor('#0a0a0a')
        self.axes.grid(True, alpha=0.3)

        # Set labels
        self.axes.set_xlabel('r (a₀)', color='white', fontsize=12)
        self.axes.set_ylabel('Amplitude / Probability', color='white', fontsize=12)

        # Set tick colors
        self.axes.tick_params(colors='white')

        # Set title
        self.axes.set_title('Radial Wavefunction and Probability Density',
                          color='white', fontsize=14, fontweight='bold', pad=20)

    def update_wavefunction(self, n: int, l: int, Z: int = 1):
        """Update the wavefunction being displayed."""
        calculator = HydrogenLikeOrbitalCalculator()
        a0 = PhysicsConstants.BOHR_RADIUS
        r_ao = np.linspace(0.001, 10 * n**2 / Z, 1000)
        r = r_ao * a0  # convert to meters

        R_nl = np.array([
            calculator.radial_wavefunction(n, l, ri, Z)
            for ri in r
        ])

        P_nl = np.array([
            calculator.radial_probability_density(n, l, ri, Z)
            for ri in r
        ])

        self.wavefunction_data = {
            'r': r_ao,
            'R_nl': R_nl,
            'P_nl': P_nl,
            'n': n,
            'l': l,
            'Z': Z
        }
        self.redraw()

    def draw_radial_plots(self):
        """Draw radial wavefunction and probability plots."""
        if not self.wavefunction_data:
            return

        r = self.wavefunction_data['r']
        R_nl = self.wavefunction_data['R_nl']
        P_nl = self.wavefunction_data['P_nl']
        n, l = self.wavefunction_data['n'], self.wavefunction_data['l']

        # Plot radial wavefunction
        if self.show_radial_wavefunction:
            self.axes.plot(r, R_nl, 'b-', linewidth=2, label=f'R_{n}{l}(r)', alpha=0.8)
            self.axes.plot(self.wavefunction_data['r'], R_nl, label=f"R_{n}{l}(r)")

        # Plot radial probability density
        if self.show_radial_probability:
            ax2 = self.axes.twinx()
            ax2.plot(r, P_nl, 'r-', linewidth=2, label=f'P_{n}{l}(r)', alpha=0.8)
            ax2.set_ylabel('Radial Probability Density', color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')

        # Add expectation value line
        calculator = HydrogenLikeOrbitalCalculator()
        expectation_r = calculator.expectation_value_r(n, l, self.wavefunction_data['Z'])
        self.axes.axvline(x=expectation_r, color='green', linestyle='--', alpha=0.7,
                         label=f'<r> = {expectation_r:.2f} a₀')

        # Add legend
        lines1, labels1 = self.axes.get_legend_handles_labels()
        if self.show_radial_probability:
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.axes.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            self.axes.legend(lines1, labels1, loc='upper right')

        # Update title
        self.axes.set_title(f'Radial Functions for n={n}, l={l}',
                          color='white', fontsize=12, fontweight='bold')

    def set_display_options(self, show_wavefunction: bool, show_probability: bool):
        """Set display options."""
        self.show_radial_wavefunction = show_wavefunction
        self.show_radial_probability = show_probability
        self.redraw()

    def redraw(self):
        """Redraw the radial plots."""
        self.setup_plot()

        if self.wavefunction_data:
            self.draw_radial_plots()

        self.draw()


class BohrModelViewer(ScienceHubTool):
    """Comprehensive Bohr Model Viewer with 2D/3D visualization and atomic data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize calculators and data
        self.calculator = BohrModelCalculator()
        self.quantum_calculator = HydrogenLikeOrbitalCalculator()
        self.constants = PhysicsConstants()

        # Load periodic table data
        self.elements_data = self.load_periodic_table()

        # Current atom data
        self.current_element = None
        self.current_config = None

        self.canvas_2d = None
        self.canvas_3d = None
        self.quantum_canvas = None
        self.radial_canvas = None

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
        # Create main layout with scroll area
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

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

        # Create scroll area and set the main widget
        scroll_area = QScrollArea()
        scroll_area.setWidget(main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Set the scroll area as the main layout
        layout = self.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.setLayout(layout)
        layout.addWidget(scroll_area)

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

        # Quantum Orbital Controls
        quantum_group = QGroupBox("Quantum Orbital Controls")
        quantum_group.setObjectName("toolCard")
        quantum_layout = QVBoxLayout(quantum_group)
        self.auto_orbital_checkbox = QCheckBox("Auto orbital (from electron configuration)")
        self.auto_orbital_checkbox.setChecked(True)
        self.auto_orbital_checkbox.stateChanged.connect(self.update_quantum_orbital)
        quantum_layout.addWidget(self.auto_orbital_checkbox)

        # Quantum number inputs
        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Principal (n):"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 6)
        self.n_spin.setValue(1)
        self.n_spin.valueChanged.connect(self.on_quantum_numbers_changed)
        n_layout.addWidget(self.n_spin)
        quantum_layout.addLayout(n_layout)

        l_layout = QHBoxLayout()
        l_layout.addWidget(QLabel("Azimuthal (l):"))
        self.l_spin = QSpinBox()
        self.l_spin.setRange(0, 5)
        self.l_spin.setValue(0)
        self.l_spin.valueChanged.connect(self.on_quantum_numbers_changed)
        l_layout.addWidget(self.l_spin)
        quantum_layout.addLayout(l_layout)

        m_layout = QHBoxLayout()
        m_layout.addWidget(QLabel("Magnetic (m):"))
        self.m_spin = QSpinBox()
        self.m_spin.setRange(-5, 5)
        self.m_spin.setValue(0)
        self.m_spin.valueChanged.connect(self.on_quantum_numbers_changed)
        m_layout.addWidget(self.m_spin)
        quantum_layout.addLayout(m_layout)

        # Orbital preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Orbital Preset:"))
        self.orbital_preset = QComboBox()
        self.orbital_preset.addItem("1s (Ground state)", (1, 0, 0))
        self.orbital_preset.addItem("2s", (2, 0, 0))
        self.orbital_preset.addItem("2p (m=0)", (2, 1, 0))
        self.orbital_preset.addItem("2p (m=±1)", (2, 1, 1))
        self.orbital_preset.addItem("3s", (3, 0, 0))
        self.orbital_preset.addItem("3p (m=0)", (3, 1, 0))
        self.orbital_preset.addItem("3d (m=0)", (3, 2, 0))
        self.orbital_preset.addItem("3d (m=±1)", (3, 2, 1))
        self.orbital_preset.addItem("3d (m=±2)", (3, 2, 2))
        self.orbital_preset.currentIndexChanged.connect(self.on_orbital_preset_changed)
        preset_layout.addWidget(self.orbital_preset)
        quantum_layout.addLayout(preset_layout)

        layout.addWidget(quantum_group)

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

        # Schrödinger Model
        schrodinger_tab = self.create_schrodinger_tab()
        data_tabs.addTab(schrodinger_tab, "Schrödinger Model")

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

        # Quantum Orbital Controls
        self.show_probability_density = QCheckBox("Show probability density")
        self.show_probability_density.setChecked(True)
        self.show_probability_density.stateChanged.connect(self.update_quantum_visualization)
        vis_layout.addWidget(self.show_probability_density)

        self.show_wavefunction_phase = QCheckBox("Show wavefunction phase")
        self.show_wavefunction_phase.stateChanged.connect(self.update_quantum_visualization)
        vis_layout.addWidget(self.show_wavefunction_phase)

        # Radial plot controls
        self.show_radial_wavefunction = QCheckBox("Show radial wavefunction")
        self.show_radial_wavefunction.setChecked(True)
        self.show_radial_wavefunction.stateChanged.connect(self.update_radial_visualization)
        vis_layout.addWidget(self.show_radial_wavefunction)

        self.show_radial_probability = QCheckBox("Show radial probability")
        self.show_radial_probability.setChecked(True)
        self.show_radial_probability.stateChanged.connect(self.update_radial_visualization)
        vis_layout.addWidget(self.show_radial_probability)

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

        self.animate_quantum_btn = QPushButton("Animate Quantum")
        self.animate_quantum_btn.setCheckable(True)
        self.animate_quantum_btn.clicked.connect(self.toggle_quantum_animation)
        anim_layout.addWidget(self.animate_quantum_btn)
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

    def create_schrodinger_tab(self) -> QWidget:
        """Create Schrödinger model tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Schrödinger explanation
        schrodinger_info = QLabel("""
The Schrödinger equation describes electrons as quantum mechanical waves.
Key concepts:

• Wavefunction ψ(r,θ,φ): Probability amplitude
• Probability density |ψ|²: Electron location probability
• Radial wavefunction R_nl(r): Radial dependence
• Angular wavefunction Y_lm(θ,φ): Angular dependence
• Quantum numbers: n (energy), l (shape), m (orientation)

The complete wavefunction: ψ_nlm(r,θ,φ) = R_nl(r) × Y_lm(θ,φ)

This model correctly predicts atomic spectra and electron distributions.
        """)
        schrodinger_info.setWordWrap(True)
        layout.addWidget(schrodinger_info)

        # Quantum orbital data table
        self.schrodinger_table = QTableWidget()
        self.schrodinger_table.setColumnCount(4)
        self.schrodinger_table.setHorizontalHeaderLabels(["Quantum State", "Energy (eV)", "<r> (a₀)", "Orbital Type"])
        layout.addWidget(self.schrodinger_table)

        # Orbital isosurface controls
        iso_layout = QHBoxLayout()
        iso_layout.addWidget(QLabel("Isosurface Level:"))
        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setRange(1, 50)
        self.iso_slider.setValue(10)
        self.iso_slider.valueChanged.connect(self.on_iso_level_changed)
        iso_layout.addWidget(self.iso_slider)
        self.iso_label = QLabel("0.10")
        iso_layout.addWidget(self.iso_label)
        layout.addLayout(iso_layout)

        # Opacity control
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("0.30")
        opacity_layout.addWidget(self.opacity_label)
        layout.addLayout(opacity_layout)

        return widget

    def create_right_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Visualization tabs
        vis_tabs = QTabWidget()

        # 2D Bohr Model
        self.canvas_2d = BohrModel2DCanvas(self)
        vis_tabs.addTab(self.canvas_2d, "2D Bohr Model")

        # 3D Bohr Model
        self.canvas_3d = BohrModel3DCanvas(self)
        vis_tabs.addTab(self.canvas_3d, "3D Bohr Model")

        # Quantum Orbital 3D
        self.quantum_canvas = QuantumOrbitalCanvas(self)
        vis_tabs.addTab(self.quantum_canvas, "Quantum Orbital 3D")

        # Radial Wavefunctions
        self.radial_canvas = RadialWavefunctionCanvas(self)
        vis_tabs.addTab(self.radial_canvas, "Radial Functions")

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
        self.update_schrodinger_data()

        # Update visualizations
        if self.canvas_2d is not None:
            self.canvas_2d.update_atom(element, self.current_config)

        if self.canvas_3d is not None:
            self.canvas_3d.update_atom(element, self.current_config)

        # Update quantum visualizations
        self.update_quantum_orbital()
        self.update_radial_wavefunction()

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

    def update_schrodinger_data(self):
        """Update Schrödinger model data."""
        if not self.current_element:
            return

        atomic_number = self.current_element.get('number', 1)

        # Get quantum states for this element
        states = self.quantum_calculator.get_quantum_states(max_n=4)
        orbital_names = self.quantum_calculator.orbital_names()

        schrodinger_data = []
        for n, l, m in states:
            if n**2 >= atomic_number:  # Only show states that would be occupied
                continue

            energy = self.quantum_calculator.energy_level(n, atomic_number)
            expectation_r = self.quantum_calculator.expectation_value_r(n, l, atomic_number)
            orbital_name = orbital_names.get((l, m), f"{n}{'spdf'[l] if l < 4 else str(l)}{abs(m) if m != 0 else ''}")

            schrodinger_data.append((f"n={n}, l={l}, m={m}", energy, expectation_r, orbital_name))

        self.schrodinger_table.setRowCount(len(schrodinger_data))
        for i, (state, energy, exp_r, orbital) in enumerate(schrodinger_data):
            self.schrodinger_table.setItem(i, 0, QTableWidgetItem(state))
            self.schrodinger_table.setItem(i, 1, QTableWidgetItem(f"{energy:.2f}"))
            self.schrodinger_table.setItem(i, 2, QTableWidgetItem(f"{exp_r:.2f}"))
            self.schrodinger_table.setItem(i, 3, QTableWidgetItem(orbital))

        self.schrodinger_table.resizeColumnsToContents()
    def get_auto_orbital(self):
        """
        Determine (n, l, m) from the highest occupied subshell.
        """
        if not self.current_config:
            return 1, 0, 0

        shells = self.current_config["shells"]

        # Find highest n with electrons
        for shell in reversed(shells):
            for subshell in reversed(shell["subshells"]):
                if subshell["electrons"] > 0:
                    n = shell["n"]
                    l = subshell["l"]
                    m = 0  # default orientation
                    return n, l, m

        return 1, 0, 0
    
    def update_quantum_orbital(self):
        if self.auto_orbital_checkbox.isChecked():
            n, l, m = self.get_auto_orbital()

            # Sync UI (without triggering loops)
            self.n_spin.blockSignals(True)
            self.l_spin.blockSignals(True)
            self.m_spin.blockSignals(True)

            self.n_spin.setValue(n)
            self.l_spin.setValue(l)
            self.m_spin.setValue(m)

            self.n_spin.blockSignals(False)
            self.l_spin.blockSignals(False)
            self.m_spin.blockSignals(False)
        else:
            n = self.n_spin.value()
            l = self.l_spin.value()
            m = self.m_spin.value()

        # --- validation ---
        if l >= n or abs(m) > l:
            return

        Z = self.current_element.get("number", 1)
        Z_eff = EffectiveNuclearCharge.compute_Z_eff(Z, n, l, self.current_config)

        orbital_data = self.quantum_calculator.calculate_orbital_data(
            n, l, m, Z_eff, resolution=30
        )

        orbital_data["Z_eff"] = Z_eff
        orbital_data["element"] = self.current_element

        if self.quantum_canvas:
            self.quantum_canvas.update_orbital(orbital_data)

            if self.quantum_canvas:
                self.quantum_canvas.update_orbital(orbital_data)

    def update_radial_wavefunction(self):
        if self.auto_orbital_checkbox.isChecked():
            n, l, _ = self.get_auto_orbital()
        else:
            n = self.n_spin.value()
            l = self.l_spin.value()

        Z = self.current_element.get("number", 1)

        if self.radial_canvas:
            self.radial_canvas.update_wavefunction(n, l, Z)

    def on_quantum_numbers_changed(self):
        """Handle quantum number changes."""
        # Validate l and m ranges
        n = self.n_spin.value()
        l = self.l_spin.value()
        m = self.m_spin.value()

        # Adjust l if necessary
        if l >= n:
            self.l_spin.blockSignals(True)
            self.l_spin.setValue(n - 1)
            self.l_spin.blockSignals(False)
            l = n - 1

        # Adjust m if necessary
        if abs(m) > l:
            self.m_spin.blockSignals(True)
            self.m_spin.setValue(0)
            self.m_spin.blockSignals(False)
            m = 0

        self.update_quantum_orbital()
        self.update_radial_wavefunction()

    def on_orbital_preset_changed(self, index: int):
        if index < 0:
            return

        n, l, m = self.orbital_preset.itemData(index)

        # Turn OFF auto mode
        self.auto_orbital_checkbox.setChecked(False)

        self.n_spin.setValue(n)
        self.l_spin.setValue(l)
        self.m_spin.setValue(m)

        self.update_quantum_orbital()


    def on_iso_level_changed(self, value: int):
        """Handle isosurface level change."""
        level = value / 100.0
        self.iso_label.setText(f"{level:.2f}")
        if self.quantum_canvas:
            self.quantum_canvas.set_isosurface_level(level)

    def on_opacity_changed(self, value: int):
        """Handle opacity change."""
        opacity = value / 100.0
        self.opacity_label.setText(f"{opacity:.2f}")
        if self.quantum_canvas:
            self.quantum_canvas.set_opacity(opacity)

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

    def update_quantum_visualization(self):
        """Update quantum orbital visualization settings."""
        show_prob = self.show_probability_density.isChecked()
        show_phase = self.show_wavefunction_phase.isChecked()
        if self.quantum_canvas:
            self.quantum_canvas.set_visualization_mode(show_prob, show_phase)

    def update_radial_visualization(self):
        """Update radial wavefunction visualization settings."""
        show_wave = self.show_radial_wavefunction.isChecked()
        show_prob = self.show_radial_probability.isChecked()
        if self.radial_canvas:
            self.radial_canvas.set_display_options(show_wave, show_prob)

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

    def toggle_quantum_animation(self):
        """Toggle quantum orbital animation."""
        if self.animate_quantum_btn.isChecked():
            if self.quantum_canvas:
                self.quantum_canvas.start_animation()
        else:
            if self.quantum_canvas:
                self.quantum_canvas.stop_animation()

    def on_camera_changed(self):
        """Handle camera angle changes."""
        elev = self.elev_slider.value()
        azim = self.azim_slider.value()

        self.elev_label.setText(f"{elev}°")
        self.azim_label.setText(f"{azim}°")

        if self.canvas_3d:
            self.canvas_3d.set_view_angle(elev, azim)
        if self.quantum_canvas:
            self.quantum_canvas.set_view_angle(elev, azim)

    def reset_views(self):
        """Reset all views to default."""
        # Reset camera
        self.elev_slider.setValue(30)
        self.azim_slider.setValue(45)

        # Stop animations
        self.animate_2d_btn.setChecked(False)
        self.animate_3d_btn.setChecked(False)
        self.animate_quantum_btn.setChecked(False)
        self.canvas_2d.stop_animation()
        self.canvas_3d.stop_animation()
        if self.quantum_canvas:
            self.quantum_canvas.stop_animation()

        # Reset visualizations
        self.canvas_2d.redraw()
        self.canvas_3d.redraw()
        if self.quantum_canvas:
            self.quantum_canvas.redraw()

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

    def export_quantum_view(self):
        """Export quantum orbital view as image."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Quantum View", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            if self.quantum_canvas:
                self.quantum_canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')

    def export_radial_view(self):
        """Export radial functions view as image."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Radial View", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            if self.radial_canvas:
                self.radial_canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')

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
            "Bohr Model Viewer v2.0\n\n"
            "An interactive tool for visualizing atomic structure using the Bohr model\n"
            "with full Schrödinger equation quantum orbital calculations.\n\n"
            "Features:\n"
            "• 2D and 3D atomic visualizations\n"
            "• Complete atomic data display\n"
            "• Electron configuration analysis\n"
            "• Quantum mechanical calculations\n"
            "• Real-time parameter adjustment\n"
            "• Quantum orbital probability densities\n"
            "• Radial wavefunction plots\n"
            "• Export capabilities\n\n"
            "Includes both classical Bohr model and modern quantum mechanics."
        )

TOOL_META = {
    "name": "Bohr Model Viewer",
    "description": "Interactive 2D/3D visualization of atomic structure using the Bohr model with complete atomic data display and quantum mechanics integration",
    "category": "Chemistry",
    "version": "2.0.0",
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
        "Beautiful modern UI design",
        "Schrödinger equation solver for hydrogen-like atoms",
        "Quantum orbital probability density visualization",
        "Radial wavefunction and probability plots",
        "Interactive quantum number controls",
        "Orbital shape analysis (s, p, d, f orbitals)",
        "Wavefunction phase visualization",
        "Expectation value calculations",
        "Advanced 3D orbital rendering",
        "Multiple visualization modes",
        "Comprehensive quantum state analysis"
    ],
    "educational_value": "Explore atomic structure, understand electron configuration, learn quantum numbers, visualize Bohr model vs quantum orbitals, master wave mechanics",
    "keywords": ["bohr model", "atomic structure", "electron configuration", "quantum numbers", "periodic table", "3d visualization", "chemistry education", "schrödinger equation", "quantum orbitals", "wave functions", "probability density", "hydrogen atom"]
}


def create_tool(parent=None):
    """Create and return a BohrModelViewer instance."""
    return BohrModelViewer(parent)