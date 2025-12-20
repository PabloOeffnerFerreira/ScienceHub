"""
Geometry Tool - Interactive 2D/3D Geometric Shape Visualization
===============================================================

A comprehensive tool for visualizing and analyzing geometric shapes
with interactive 2D and 3D views, property calculations, and educational content.

Features:
- Interactive 2D shape construction and visualization
- 3D geometric solid rendering with properties
- Coordinate geometry and transformations
- Real-time parameter adjustment and calculations
- Educational information and formulas
- Export capabilities for images and data
- Animation and interactive exploration
- Multiple visualization modes and styles

Author: ScienceHub Team
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Polygon, Rectangle, RegularPolygon, Arc
from matplotlib.collections import PatchCollection
import math

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QTextEdit,
    QSplitter, QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QSlider, QProgressBar, QTabWidget,
    QLineEdit, QMessageBox, QMenuBar, QMenu, QFileDialog, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon
from PyQt6 import sip

from sciencehub.ui.components.tool_base import ScienceHubTool


class GeometryCalculator:
    """Calculator for geometric properties and formulas."""

    @staticmethod
    def circle_properties(radius: float) -> dict:
        """Calculate circle properties."""
        return {
            'area': math.pi * radius**2,
            'circumference': 2 * math.pi * radius,
            'diameter': 2 * radius
        }

    @staticmethod
    def triangle_properties(a: float, b: float, c: float) -> dict:
        """Calculate triangle properties using side lengths."""
        # Check if valid triangle
        if a + b <= c or a + c <= b or b + c <= a:
            return {'error': 'Invalid triangle sides'}

        s = (a + b + c) / 2  # semi-perimeter
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula

        # Angles using law of cosines
        angle_a = math.degrees(math.acos((b**2 + c**2 - a**2) / (2 * b * c)))
        angle_b = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
        angle_c = math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))

        return {
            'area': area,
            'perimeter': a + b + c,
            'angles': [angle_a, angle_b, angle_c],
            'semiperimeter': s,
            'type': GeometryCalculator.triangle_type(a, b, c)
        }

    @staticmethod
    def triangle_type(a: float, b: float, c: float) -> str:
        """Determine triangle type."""
        sides = sorted([a, b, c])
        if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-10:
            return "Right-angled"
        elif sides[0]**2 + sides[1]**2 > sides[2]**2:
            return "Acute-angled"
        else:
            return "Obtuse-angled"

    @staticmethod
    def rectangle_properties(width: float, height: float) -> dict:
        """Calculate rectangle properties."""
        return {
            'area': width * height,
            'perimeter': 2 * (width + height),
            'diagonal': math.sqrt(width**2 + height**2)
        }

    @staticmethod
    def polygon_properties(sides: int, side_length: float) -> dict:
        """Calculate regular polygon properties."""
        if sides < 3:
            return {'error': 'Polygon must have at least 3 sides'}

        # Interior angle
        interior_angle = ((sides - 2) * 180) / sides

        # Apothem and area
        apothem = side_length / (2 * math.tan(math.pi / sides))
        area = (sides * side_length * apothem) / 2

        # Circumradius
        circumradius = side_length / (2 * math.sin(math.pi / sides))

        return {
            'area': area,
            'perimeter': sides * side_length,
            'interior_angle': interior_angle,
            'exterior_angle': 360 / sides,
            'apothem': apothem,
            'circumradius': circumradius
        }

    @staticmethod
    def sphere_properties(radius: float) -> dict:
        """Calculate sphere properties."""
        return {
            'volume': (4/3) * math.pi * radius**3,
            'surface_area': 4 * math.pi * radius**2
        }

    @staticmethod
    def cube_properties(side: float) -> dict:
        """Calculate cube properties."""
        return {
            'volume': side**3,
            'surface_area': 6 * side**2,
            'space_diagonal': side * math.sqrt(3)
        }

    @staticmethod
    def cylinder_properties(radius: float, height: float) -> dict:
        """Calculate cylinder properties."""
        return {
            'volume': math.pi * radius**2 * height,
            'lateral_surface_area': 2 * math.pi * radius * height,
            'total_surface_area': 2 * math.pi * radius * (radius + height)
        }

    @staticmethod
    def cone_properties(radius: float, height: float) -> dict:
        """Calculate cone properties."""
        slant_height = math.sqrt(radius**2 + height**2)
        return {
            'volume': (1/3) * math.pi * radius**2 * height,
            'lateral_surface_area': math.pi * radius * slant_height,
            'total_surface_area': math.pi * radius * (radius + slant_height),
            'slant_height': slant_height
        }

    @staticmethod
    def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance between two 2D points."""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        """Calculate distance between two 3D points."""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    @staticmethod
    def midpoint_2d(x1: float, y1: float, x2: float, y2: float) -> tuple:
        """Calculate midpoint between two 2D points."""
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def rotate_point_2d(x: float, y: float, angle_deg: float, center_x: float = 0, center_y: float = 0) -> tuple:
        """Rotate a point around a center point."""
        angle_rad = math.radians(angle_deg)
        x_translated = x - center_x
        y_translated = y - center_y

        x_rotated = x_translated * math.cos(angle_rad) - y_translated * math.sin(angle_rad)
        y_rotated = x_translated * math.sin(angle_rad) + y_translated * math.cos(angle_rad)

        return (x_rotated + center_x, y_rotated + center_y)


class Geometry2DCanvas(FigureCanvasQTAgg):
    """2D Canvas for geometric shape visualization."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data
        self.shape_data = {}
        self.show_grid = True
        self.show_labels = True
        self.show_measurements = True
        self.fill_shapes = True
        self.animation_angle = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_step)

        self.setup_plot()

    def setup_plot(self):
        """Initialize the 2D geometry plot."""
        self.axes.clear()
        self.axes.set_facecolor('#0a0a0a')

        # Set labels
        self.axes.set_xlabel('X', color='white', fontsize=12)
        self.axes.set_ylabel('Y', color='white', fontsize=12)

        # Set tick colors
        self.axes.tick_params(colors='white')

        # Set title
        self.axes.set_title('2D Geometry - Interactive Shapes', color='white', fontsize=14, fontweight='bold', pad=20)

        # Grid
        if self.show_grid:
            self.axes.grid(True, alpha=0.3, color='white')

        # Equal aspect ratio
        self.axes.set_aspect('equal')

    def update_shape(self, shape_type: str, params: dict):
        """Update the shape being displayed."""
        self.shape_data = {
            'type': shape_type,
            'params': params
        }
        self.redraw()

    def draw_circle(self, params: dict):
        """Draw a circle."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        radius = params.get('radius', 1)

        # Draw circle
        circle = Circle((center_x, center_y), radius,
                       facecolor='#4ecdc4' if self.fill_shapes else 'none',
                       edgecolor='#4ecdc4', linewidth=2, alpha=0.7)
        self.axes.add_patch(circle)

        # Draw center point
        self.axes.plot(center_x, center_y, 'ro', markersize=8, alpha=0.8)

        # Labels and measurements
        if self.show_labels:
            self.axes.text(center_x, center_y + 0.1, 'Center', ha='center', va='bottom',
                          color='white', fontsize=10)

        if self.show_measurements:
            # Radius line
            self.axes.plot([center_x, center_x + radius], [center_y, center_y],
                          'r--', linewidth=1, alpha=0.7)
            self.axes.text(center_x + radius/2, center_y + 0.1, f'r = {radius:.2f}',
                          ha='center', va='bottom', color='red', fontsize=9)

            # Properties
            props = GeometryCalculator.circle_properties(radius)
            info_text = f"Area: {props['area']:.2f}\nCircumference: {props['circumference']:.2f}"
            self.axes.text(center_x - radius, center_y - radius - 0.5, info_text,
                          color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                          facecolor='#333333', alpha=0.8))

    def draw_triangle(self, params: dict):
        """Draw a triangle."""
        vertices = params.get('vertices', [(0, 0), (1, 0), (0.5, 1)])

        # Draw triangle
        triangle = Polygon(vertices,
                          facecolor='#ff6b6b' if self.fill_shapes else 'none',
                          edgecolor='#ff6b6b', linewidth=2, alpha=0.7)
        self.axes.add_patch(triangle)

        # Draw vertices
        for i, (x, y) in enumerate(vertices):
            self.axes.plot(x, y, 'ro', markersize=6, alpha=0.8)
            if self.show_labels:
                self.axes.text(x, y + 0.1, f'P{i+1}', ha='center', va='bottom',
                              color='white', fontsize=10)

        # Calculate and display properties
        if len(vertices) == 3:
            # Calculate side lengths
            a = GeometryCalculator.distance_2d(*vertices[0], *vertices[1])
            b = GeometryCalculator.distance_2d(*vertices[1], *vertices[2])
            c = GeometryCalculator.distance_2d(*vertices[2], *vertices[0])

            if self.show_measurements:
                # Draw side lengths
                for i in range(3):
                    p1 = vertices[i]
                    p2 = vertices[(i + 1) % 3]
                    mid_x, mid_y = GeometryCalculator.midpoint_2d(*p1, *p2)
                    length = [a, b, c][i]
                    self.axes.text(mid_x, mid_y, f'{length:.2f}', ha='center', va='center',
                                  color='yellow', fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor='black', alpha=0.7))

            # Properties
            props = GeometryCalculator.triangle_properties(a, b, c)
            if 'error' not in props:
                info_text = f"Area: {props['area']:.2f}\nPerimeter: {props['perimeter']:.2f}\nType: {props['type']}"
                self.axes.text(0.02, 0.98, info_text, transform=self.axes.transAxes,
                              color='white', fontsize=8, verticalalignment='top',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='#333333', alpha=0.8))

    def draw_rectangle(self, params: dict):
        """Draw a rectangle."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        width = params.get('width', 2)
        height = params.get('height', 1)

        # Calculate corner coordinates
        x = center_x - width/2
        y = center_y - height/2

        # Draw rectangle
        rect = Rectangle((x, y), width, height,
                        facecolor='#45b7d1' if self.fill_shapes else 'none',
                        edgecolor='#45b7d1', linewidth=2, alpha=0.7)
        self.axes.add_patch(rect)

        # Draw center point
        self.axes.plot(center_x, center_y, 'ro', markersize=8, alpha=0.8)

        if self.show_measurements:
            # Draw dimensions
            self.axes.plot([x, x + width], [y, y], 'r--', linewidth=1, alpha=0.7)
            self.axes.text(center_x, y - 0.1, f'w = {width:.2f}', ha='center', va='top',
                          color='red', fontsize=9)

            self.axes.plot([x, x], [y, y + height], 'r--', linewidth=1, alpha=0.7)
            self.axes.text(x - 0.1, center_y, f'h = {height:.2f}', ha='right', va='center',
                          color='red', fontsize=9)

        # Properties
        props = GeometryCalculator.rectangle_properties(width, height)
        info_text = f"Area: {props['area']:.2f}\nPerimeter: {props['perimeter']:.2f}\nDiagonal: {props['diagonal']:.2f}"
        self.axes.text(x + width + 0.1, y + height, info_text, color='white', fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='#333333', alpha=0.8))

    def draw_polygon(self, params: dict):
        """Draw a regular polygon."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        sides = params.get('sides', 6)
        radius = params.get('radius', 1)

        # Draw polygon
        polygon = RegularPolygon((center_x, center_y), sides, radius=radius,
                               facecolor='#ffaa00' if self.fill_shapes else 'none',
                               edgecolor='#ffaa00', linewidth=2, alpha=0.7)
        self.axes.add_patch(polygon)

        # Draw center point
        self.axes.plot(center_x, center_y, 'ro', markersize=8, alpha=0.8)

        if self.show_measurements:
            # Draw radius
            self.axes.plot([center_x, center_x + radius], [center_y, center_y],
                          'r--', linewidth=1, alpha=0.7)
            self.axes.text(center_x + radius/2, center_y + 0.1, f'r = {radius:.2f}',
                          ha='center', va='bottom', color='red', fontsize=9)

        # Properties
        side_length = 2 * radius * math.sin(math.pi / sides)
        props = GeometryCalculator.polygon_properties(sides, side_length)
        if 'error' not in props:
            info_text = f"Sides: {sides}\nArea: {props['area']:.2f}\nPerimeter: {props['perimeter']:.2f}"
            self.axes.text(center_x - radius, center_y - radius - 0.5, info_text,
                          color='white', fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                          facecolor='#333333', alpha=0.8))

    def draw_coordinate_geometry(self, params: dict):
        """Draw coordinate geometry elements."""
        points = params.get('points', [])
        lines = params.get('lines', [])
        transformations = params.get('transformations', {})

        # Draw points
        for i, (x, y) in enumerate(points):
            self.axes.plot(x, y, 'ro', markersize=8, alpha=0.8)
            if self.show_labels:
                self.axes.text(x + 0.1, y + 0.1, f'P{i+1}({x:.1f}, {y:.1f})',
                              color='white', fontsize=9)

        # Draw lines
        for line in lines:
            if len(line) == 2:
                x1, y1 = line[0]
                x2, y2 = line[1]
                self.axes.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)

                if self.show_measurements:
                    distance = GeometryCalculator.distance_2d(x1, y1, x2, y2)
                    mid_x, mid_y = GeometryCalculator.midpoint_2d(x1, y1, x2, y2)
                    self.axes.text(mid_x, mid_y + 0.1, f'd = {distance:.2f}',
                                  ha='center', va='bottom', color='yellow', fontsize=8)

        # Apply transformations
        if transformations.get('rotation', 0) != 0:
            angle = transformations['rotation']
            transformed_points = []
            for x, y in points:
                new_x, new_y = GeometryCalculator.rotate_point_2d(x, y, angle)
                transformed_points.append((new_x, new_y))
                self.axes.plot(new_x, new_y, 'go', markersize=6, alpha=0.8)

    def start_animation(self):
        """Start shape animation."""
        self.animation_timer.start(50)  # 50ms intervals

    def stop_animation(self):
        """Stop shape animation."""
        self.animation_timer.stop()

    def animate_step(self):
        """Advance animation by one step."""
        self.animation_angle += 2
        self.redraw()

    def redraw(self):
        """Redraw the entire 2D geometry."""
        self.setup_plot()

        if self.shape_data:
            shape_type = self.shape_data.get('type', '')
            params = self.shape_data.get('params', {})

            if shape_type == 'circle':
                self.draw_circle(params)
            elif shape_type == 'triangle':
                self.draw_triangle(params)
            elif shape_type == 'rectangle':
                self.draw_rectangle(params)
            elif shape_type == 'polygon':
                self.draw_polygon(params)
            elif shape_type == 'coordinate':
                self.draw_coordinate_geometry(params)

        self.draw()


class Geometry3DCanvas(FigureCanvasQTAgg):
    """3D Canvas for geometric solid visualization."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#0a0a0a')
        self.axes = self.figure.add_subplot(111, projection='3d')

        super().__init__(self.figure)
        self.setParent(parent)

        # Initialize data
        self.solid_data = {}
        self.show_wireframe = False
        self.opacity = 0.7
        self.view_angle = [30, 45]  # elevation, azimuth
        self.animation_angle = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_step)

        self.setup_plot()

    def setup_plot(self):
        """Initialize the 3D geometry plot."""
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
        self.axes.set_title('3D Geometry - Solids and Surfaces', color='white', fontsize=14, fontweight='bold', pad=20)

        # Set initial view
        self.axes.view_init(elev=self.view_angle[0], azim=self.view_angle[1])

        # Set equal aspect
        self.axes.set_box_aspect([1, 1, 1])

    def update_solid(self, solid_type: str, params: dict):
        """Update the solid being displayed."""
        self.solid_data = {
            'type': solid_type,
            'params': params
        }
        self.redraw()

    def draw_sphere(self, params: dict):
        """Draw a sphere."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        center_z = params.get('center_z', 0)
        radius = params.get('radius', 1)

        # Create sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = center_x + radius * np.outer(np.cos(u), np.sin(v))
        y = center_y + radius * np.outer(np.sin(u), np.sin(v))
        z = center_z + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        if self.show_wireframe:
            self.axes.plot_wireframe(x, y, z, color='#4ecdc4', alpha=self.opacity)
        else:
            self.axes.plot_surface(x, y, z, color='#4ecdc4', alpha=self.opacity)

        # Properties
        props = GeometryCalculator.sphere_properties(radius)
        info_text = f"Volume: {props['volume']:.2f}\nSurface Area: {props['surface_area']:.2f}"
        self.axes.text2D(0.02, 0.98, info_text, transform=self.axes.transAxes,
                        color='white', fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#333333', alpha=0.8))

    def draw_cube(self, params: dict):
        """Draw a cube."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        center_z = params.get('center_z', 0)
        side = params.get('side', 1)

        # Cube vertices
        vertices = np.array([
            [-side/2, -side/2, -side/2],
            [side/2, -side/2, -side/2],
            [side/2, side/2, -side/2],
            [-side/2, side/2, -side/2],
            [-side/2, -side/2, side/2],
            [side/2, -side/2, side/2],
            [side/2, side/2, side/2],
            [-side/2, side/2, side/2]
        ])

        # Translate to center
        vertices += np.array([center_x, center_y, center_z])

        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
        ]

        # Draw faces
        for face in faces:
            face_array = np.array(face)
            if self.show_wireframe:
                self.axes.plot_wireframe(face_array[:, 0], face_array[:, 1], face_array[:, 2],
                                       color='#ff6b6b', alpha=self.opacity)
            else:
                # Create surface
                X = face_array[:, 0].reshape(2, 2)
                Y = face_array[:, 1].reshape(2, 2)
                Z = face_array[:, 2].reshape(2, 2)
                self.axes.plot_surface(X, Y, Z, color='#ff6b6b', alpha=self.opacity)

        # Properties
        props = GeometryCalculator.cube_properties(side)
        info_text = f"Volume: {props['volume']:.2f}\nSurface Area: {props['surface_area']:.2f}"
        self.axes.text2D(0.02, 0.98, info_text, transform=self.axes.transAxes,
                        color='white', fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#333333', alpha=0.8))

    def draw_cylinder(self, params: dict):
        """Draw a cylinder."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        center_z = params.get('center_z', 0)
        radius = params.get('radius', 1)
        height = params.get('height', 2)

        # Create cylinder
        z = np.linspace(center_z - height/2, center_z + height/2, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x = center_x + radius * np.cos(theta_grid)
        y = center_y + radius * np.sin(theta_grid)

        if self.show_wireframe:
            self.axes.plot_wireframe(x, y, z_grid, color='#45b7d1', alpha=self.opacity)
        else:
            self.axes.plot_surface(x, y, z_grid, color='#45b7d1', alpha=self.opacity)

        # Draw top and bottom circles
        for z_val in [center_z - height/2, center_z + height/2]:
            theta_circle = np.linspace(0, 2*np.pi, 20)
            x_circle = center_x + radius * np.cos(theta_circle)
            y_circle = center_y + radius * np.sin(theta_circle)
            z_circle = np.full_like(theta_circle, z_val)
            self.axes.plot(x_circle, y_circle, z_circle, color='#45b7d1', alpha=self.opacity)

        # Properties
        props = GeometryCalculator.cylinder_properties(radius, height)
        info_text = f"Volume: {props['volume']:.2f}\nLateral SA: {props['lateral_surface_area']:.2f}\nTotal SA: {props['total_surface_area']:.2f}"
        self.axes.text2D(0.02, 0.98, info_text, transform=self.axes.transAxes,
                        color='white', fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#333333', alpha=0.8))

    def draw_cone(self, params: dict):
        """Draw a cone."""
        center_x = params.get('center_x', 0)
        center_y = params.get('center_y', 0)
        center_z = params.get('center_z', 0)
        radius = params.get('radius', 1)
        height = params.get('height', 2)

        # Create cone
        z = np.linspace(center_z - height/2, center_z + height/2, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # Radius decreases linearly from base to apex
        r_grid = radius * (1 - (z_grid - (center_z - height/2)) / height)
        r_grid = np.maximum(r_grid, 0)  # Ensure non-negative

        x = center_x + r_grid * np.cos(theta_grid)
        y = center_y + r_grid * np.sin(theta_grid)

        if self.show_wireframe:
            self.axes.plot_wireframe(x, y, z_grid, color='#ffaa00', alpha=self.opacity)
        else:
            self.axes.plot_surface(x, y, z_grid, color='#ffaa00', alpha=self.opacity)

        # Draw base circle
        theta_circle = np.linspace(0, 2*np.pi, 20)
        x_circle = center_x + radius * np.cos(theta_circle)
        y_circle = center_y + radius * np.sin(theta_circle)
        z_circle = np.full_like(theta_circle, center_z - height/2)
        self.axes.plot(x_circle, y_circle, z_circle, color='#ffaa00', alpha=self.opacity)

        # Properties
        props = GeometryCalculator.cone_properties(radius, height)
        info_text = f"Volume: {props['volume']:.2f}\nLateral SA: {props['lateral_surface_area']:.2f}\nTotal SA: {props['total_surface_area']:.2f}"
        self.axes.text2D(0.02, 0.98, info_text, transform=self.axes.transAxes,
                        color='white', fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#333333', alpha=0.8))

    def draw_coordinate_3d(self, params: dict):
        """Draw 3D coordinate geometry elements."""
        points = params.get('points', [])
        vectors = params.get('vectors', [])

        # Draw points
        for i, (x, y, z) in enumerate(points):
            self.axes.scatter(x, y, z, color='red', s=50, alpha=0.8)
            self.axes.text(x + 0.1, y + 0.1, z + 0.1, f'P{i+1}({x:.1f}, {y:.1f}, {z:.1f})',
                          color='white', fontsize=8)

        # Draw vectors
        for vector in vectors:
            if len(vector) == 2:  # start and end points
                start, end = vector
                xs, ys, zs = start
                xe, ye, ze = end

                # Draw vector line
                self.axes.quiver(xs, ys, zs, xe-xs, ye-ys, ze-zs,
                               color='cyan', alpha=0.8, linewidth=2)

                # Calculate magnitude
                magnitude = GeometryCalculator.distance_3d(xs, ys, zs, xe, ye, ze)
                mid_x, mid_y, mid_z = (xs + xe)/2, (ys + ye)/2, (zs + ze)/2
                self.axes.text(mid_x, mid_y, mid_z, f'|{magnitude:.2f}|',
                              color='yellow', fontsize=8)

    def set_view_angle(self, elev: float, azim: float):
        """Set camera view angle."""
        self.view_angle = [elev, azim]
        self.axes.view_init(elev=elev, azim=azim)
        self.redraw()

    def start_animation(self):
        """Start 3D animation."""
        self.animation_timer.start(50)  # 50ms intervals

    def stop_animation(self):
        """Stop 3D animation."""
        self.animation_timer.stop()

    def animate_step(self):
        """Advance animation by one step."""
        self.animation_angle += 2
        self.view_angle[1] += 1  # Rotate view
        if self.view_angle[1] >= 360:
            self.view_angle[1] = 0
        self.axes.view_init(elev=self.view_angle[0], azim=self.view_angle[1])
        self.redraw()

    def redraw(self):
        """Redraw the entire 3D geometry."""
        self.setup_plot()

        if self.solid_data:
            solid_type = self.solid_data.get('type', '')
            params = self.solid_data.get('params', {})

            if solid_type == 'sphere':
                self.draw_sphere(params)
            elif solid_type == 'cube':
                self.draw_cube(params)
            elif solid_type == 'cylinder':
                self.draw_cylinder(params)
            elif solid_type == 'cone':
                self.draw_cone(params)
            elif solid_type == 'coordinate_3d':
                self.draw_coordinate_3d(params)

        self.draw()


class GeometryTool(ScienceHubTool):
    """Comprehensive Geometry Tool with 2D/3D visualization and property calculations."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize calculator
        self.calculator = GeometryCalculator()

        # Current shape/solid data
        self.current_2d_shape = 'circle'
        self.current_3d_solid = 'sphere'

        # Setup UI
        self.setup_ui()

        # Initialize with default shapes
        self.update_2d_shape()
        self.update_3d_solid()

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

        # Shape Selection
        shape_group = QGroupBox("Shape Selection")
        shape_group.setObjectName("toolCard")
        shape_layout = QVBoxLayout(shape_group)

        # 2D Shape selector
        shape_2d_layout = QHBoxLayout()
        shape_2d_layout.addWidget(QLabel("2D Shape:"))
        self.shape_2d_combo = QComboBox()
        self.shape_2d_combo.addItems(['circle', 'triangle', 'rectangle', 'polygon', 'coordinate'])
        self.shape_2d_combo.currentTextChanged.connect(self.on_2d_shape_changed)
        shape_2d_layout.addWidget(self.shape_2d_combo)
        shape_layout.addLayout(shape_2d_layout)

        # 3D Solid selector
        shape_3d_layout = QHBoxLayout()
        shape_3d_layout.addWidget(QLabel("3D Solid:"))
        self.shape_3d_combo = QComboBox()
        self.shape_3d_combo.addItems(['sphere', 'cube', 'cylinder', 'cone', 'coordinate_3d'])
        self.shape_3d_combo.currentTextChanged.connect(self.on_3d_shape_changed)
        shape_3d_layout.addWidget(self.shape_3d_combo)
        shape_layout.addLayout(shape_3d_layout)

        layout.addWidget(shape_group)

        # 2D Shape Parameters
        self.param_2d_group = QGroupBox("2D Shape Parameters")
        self.param_2d_group.setObjectName("toolCard")
        self.setup_2d_parameters()
        layout.addWidget(self.param_2d_group)

        # 3D Solid Parameters
        self.param_3d_group = QGroupBox("3D Solid Parameters")
        self.param_3d_group.setObjectName("toolCard")
        self.setup_3d_parameters()
        layout.addWidget(self.param_3d_group)

        # Visualization Controls
        vis_group = QGroupBox("Visualization Controls")
        vis_group.setObjectName("toolCard")
        vis_layout = QVBoxLayout(vis_group)

        # 2D Controls
        self.show_2d_grid = QCheckBox("Show grid (2D)")
        self.show_2d_grid.setChecked(True)
        self.show_2d_grid.stateChanged.connect(self.update_2d_visualization)
        vis_layout.addWidget(self.show_2d_grid)

        self.show_2d_labels = QCheckBox("Show labels (2D)")
        self.show_2d_labels.setChecked(True)
        self.show_2d_labels.stateChanged.connect(self.update_2d_visualization)
        vis_layout.addWidget(self.show_2d_labels)

        self.show_2d_measurements = QCheckBox("Show measurements (2D)")
        self.show_2d_measurements.setChecked(True)
        self.show_2d_measurements.stateChanged.connect(self.update_2d_visualization)
        vis_layout.addWidget(self.show_2d_measurements)

        self.fill_2d_shapes = QCheckBox("Fill shapes (2D)")
        self.fill_2d_shapes.setChecked(True)
        self.fill_2d_shapes.stateChanged.connect(self.update_2d_visualization)
        vis_layout.addWidget(self.fill_2d_shapes)

        # 3D Controls
        self.show_3d_wireframe = QCheckBox("Wireframe mode (3D)")
        self.show_3d_wireframe.stateChanged.connect(self.update_3d_visualization)
        vis_layout.addWidget(self.show_3d_wireframe)

        # Opacity control
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("3D Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("0.70")
        opacity_layout.addWidget(self.opacity_label)
        vis_layout.addLayout(opacity_layout)

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

        # Data Display Tabs
        data_tabs = QTabWidget()

        # Properties
        props_tab = self.create_properties_tab()
        data_tabs.addTab(props_tab, "Properties")

        # Formulas
        formulas_tab = self.create_formulas_tab()
        data_tabs.addTab(formulas_tab, "Formulas")

        # Calculations
        calc_tab = self.create_calculations_tab()
        data_tabs.addTab(calc_tab, "Calculations")

        layout.addWidget(data_tabs)

        return panel

    def setup_2d_parameters(self):
        """Setup 2D shape parameter controls."""
        # Clear existing layout
        layout = self.param_2d_group.layout()
        if layout:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            layout = QVBoxLayout()
            self.param_2d_group.setLayout(layout)

        # Common parameters
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center X:"))
        self.center_x_2d = QDoubleSpinBox()
        self.center_x_2d.setRange(-10, 10)
        self.center_x_2d.setValue(0)
        self.center_x_2d.valueChanged.connect(self.update_2d_shape)
        center_layout.addWidget(self.center_x_2d)

        center_layout.addWidget(QLabel("Y:"))
        self.center_y_2d = QDoubleSpinBox()
        self.center_y_2d.setRange(-10, 10)
        self.center_y_2d.setValue(0)
        self.center_y_2d.valueChanged.connect(self.update_2d_shape)
        center_layout.addWidget(self.center_y_2d)
        layout.addLayout(center_layout)

        # Shape-specific parameters
        self.setup_shape_specific_2d_params(layout)

    def setup_shape_specific_2d_params(self, layout):
        """Setup shape-specific parameter controls."""
        shape = self.current_2d_shape

        if shape == 'circle':
            radius_layout = QHBoxLayout()
            radius_layout.addWidget(QLabel("Radius:"))
            self.radius_2d = QDoubleSpinBox()
            self.radius_2d.setRange(0.1, 10)
            self.radius_2d.setValue(1)
            self.radius_2d.valueChanged.connect(self.update_2d_shape)
            radius_layout.addWidget(self.radius_2d)
            layout.addLayout(radius_layout)

        elif shape == 'triangle':
            # Triangle vertices
            vertex_layout = QVBoxLayout()
            vertex_layout.addWidget(QLabel("Triangle Vertices:"))

            # Vertex 1
            v1_layout = QHBoxLayout()
            v1_layout.addWidget(QLabel("P1 (x,y):"))
            self.v1_x = QDoubleSpinBox()
            self.v1_x.setRange(-10, 10)
            self.v1_x.setValue(0)
            self.v1_x.valueChanged.connect(self.update_2d_shape)
            v1_layout.addWidget(self.v1_x)
            self.v1_y = QDoubleSpinBox()
            self.v1_y.setRange(-10, 10)
            self.v1_y.setValue(0)
            self.v1_y.valueChanged.connect(self.update_2d_shape)
            v1_layout.addWidget(self.v1_y)
            vertex_layout.addLayout(v1_layout)

            # Vertex 2
            v2_layout = QHBoxLayout()
            v2_layout.addWidget(QLabel("P2 (x,y):"))
            self.v2_x = QDoubleSpinBox()
            self.v2_x.setRange(-10, 10)
            self.v2_x.setValue(1)
            self.v2_x.valueChanged.connect(self.update_2d_shape)
            v2_layout.addWidget(self.v2_x)
            self.v2_y = QDoubleSpinBox()
            self.v2_y.setRange(-10, 10)
            self.v2_y.setValue(0)
            self.v2_y.valueChanged.connect(self.update_2d_shape)
            v2_layout.addWidget(self.v2_y)
            vertex_layout.addLayout(v2_layout)

            # Vertex 3
            v3_layout = QHBoxLayout()
            v3_layout.addWidget(QLabel("P3 (x,y):"))
            self.v3_x = QDoubleSpinBox()
            self.v3_x.setRange(-10, 10)
            self.v3_x.setValue(0.5)
            self.v3_x.valueChanged.connect(self.update_2d_shape)
            v3_layout.addWidget(self.v3_x)
            self.v3_y = QDoubleSpinBox()
            self.v3_y.setRange(-10, 10)
            self.v3_y.setValue(1)
            self.v3_y.valueChanged.connect(self.update_2d_shape)
            v3_layout.addWidget(self.v3_y)
            vertex_layout.addLayout(v3_layout)

            layout.addLayout(vertex_layout)

        elif shape == 'rectangle':
            rect_layout = QHBoxLayout()
            rect_layout.addWidget(QLabel("Width:"))
            self.width_2d = QDoubleSpinBox()
            self.width_2d.setRange(0.1, 10)
            self.width_2d.setValue(2)
            self.width_2d.valueChanged.connect(self.update_2d_shape)
            rect_layout.addWidget(self.width_2d)

            rect_layout.addWidget(QLabel("Height:"))
            self.height_2d = QDoubleSpinBox()
            self.height_2d.setRange(0.1, 10)
            self.height_2d.setValue(1)
            self.height_2d.valueChanged.connect(self.update_2d_shape)
            rect_layout.addWidget(self.height_2d)
            layout.addLayout(rect_layout)

        elif shape == 'polygon':
            poly_layout = QHBoxLayout()
            poly_layout.addWidget(QLabel("Sides:"))
            self.sides_2d = QSpinBox()
            self.sides_2d.setRange(3, 20)
            self.sides_2d.setValue(6)
            self.sides_2d.valueChanged.connect(self.update_2d_shape)
            poly_layout.addWidget(self.sides_2d)

            poly_layout.addWidget(QLabel("Radius:"))
            self.radius_poly = QDoubleSpinBox()
            self.radius_poly.setRange(0.1, 10)
            self.radius_poly.setValue(1)
            self.radius_poly.valueChanged.connect(self.update_2d_shape)
            poly_layout.addWidget(self.radius_poly)
            layout.addLayout(poly_layout)

    def setup_3d_parameters(self):
        """Setup 3D solid parameter controls."""
        # Clear existing layout
        layout = self.param_3d_group.layout()
        if layout:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            layout = QVBoxLayout()
            self.param_3d_group.setLayout(layout)

        # Common parameters
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center X:"))
        self.center_x_3d = QDoubleSpinBox()
        self.center_x_3d.setRange(-10, 10)
        self.center_x_3d.setValue(0)
        self.center_x_3d.valueChanged.connect(self.update_3d_solid)
        center_layout.addWidget(self.center_x_3d)

        center_layout.addWidget(QLabel("Y:"))
        self.center_y_3d = QDoubleSpinBox()
        self.center_y_3d.setRange(-10, 10)
        self.center_y_3d.setValue(0)
        self.center_y_3d.valueChanged.connect(self.update_3d_solid)
        center_layout.addWidget(self.center_y_3d)

        center_layout.addWidget(QLabel("Z:"))
        self.center_z_3d = QDoubleSpinBox()
        self.center_z_3d.setRange(-10, 10)
        self.center_z_3d.setValue(0)
        self.center_z_3d.valueChanged.connect(self.update_3d_solid)
        center_layout.addWidget(self.center_z_3d)
        layout.addLayout(center_layout)

        # Shape-specific parameters
        self.setup_shape_specific_3d_params(layout)

    def setup_shape_specific_3d_params(self, layout):
        """Setup solid-specific parameter controls."""
        solid = self.current_3d_solid

        if solid in ['sphere']:
            radius_layout = QHBoxLayout()
            radius_layout.addWidget(QLabel("Radius:"))
            self.radius_3d = QDoubleSpinBox()
            self.radius_3d.setRange(0.1, 10)
            self.radius_3d.setValue(1)
            self.radius_3d.valueChanged.connect(self.update_3d_solid)
            radius_layout.addWidget(self.radius_3d)
            layout.addLayout(radius_layout)

        elif solid == 'cube':
            side_layout = QHBoxLayout()
            side_layout.addWidget(QLabel("Side Length:"))
            self.side_3d = QDoubleSpinBox()
            self.side_3d.setRange(0.1, 10)
            self.side_3d.setValue(1)
            self.side_3d.valueChanged.connect(self.update_3d_solid)
            side_layout.addWidget(self.side_3d)
            layout.addLayout(side_layout)

        elif solid == 'cylinder':
            cyl_layout = QHBoxLayout()
            cyl_layout.addWidget(QLabel("Radius:"))
            self.radius_cyl = QDoubleSpinBox()
            self.radius_cyl.setRange(0.1, 10)
            self.radius_cyl.setValue(1)
            self.radius_cyl.valueChanged.connect(self.update_3d_solid)
            cyl_layout.addWidget(self.radius_cyl)

            cyl_layout.addWidget(QLabel("Height:"))
            self.height_cyl = QDoubleSpinBox()
            self.height_cyl.setRange(0.1, 10)
            self.height_cyl.setValue(2)
            self.height_cyl.valueChanged.connect(self.update_3d_solid)
            cyl_layout.addWidget(self.height_cyl)
            layout.addLayout(cyl_layout)

        elif solid == 'cone':
            cone_layout = QHBoxLayout()
            cone_layout.addWidget(QLabel("Base Radius:"))
            self.radius_cone = QDoubleSpinBox()
            self.radius_cone.setRange(0.1, 10)
            self.radius_cone.setValue(1)
            self.radius_cone.valueChanged.connect(self.update_3d_solid)
            cone_layout.addWidget(self.radius_cone)

            cone_layout.addWidget(QLabel("Height:"))
            self.height_cone = QDoubleSpinBox()
            self.height_cone.setRange(0.1, 10)
            self.height_cone.setValue(2)
            self.height_cone.valueChanged.connect(self.update_3d_solid)
            cone_layout.addWidget(self.height_cone)
            layout.addLayout(cone_layout)

    def create_properties_tab(self) -> QWidget:
        """Create properties display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Properties display
        self.properties_text = QTextEdit()
        self.properties_text.setReadOnly(True)
        self.properties_text.setPlainText("Shape properties will be displayed here.")
        layout.addWidget(self.properties_text)

        return widget

    def create_formulas_tab(self) -> QWidget:
        """Create formulas display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Formulas display
        formulas_text = QTextEdit()
        formulas_text.setReadOnly(True)
        formulas_text.setPlainText("""
2D Shape Formulas:

Circle:
• Area: A = πr²
• Circumference: C = 2πr
• Diameter: d = 2r

Triangle:
• Area: A = √[s(s-a)(s-b)(s-c)] (Heron's formula)
• Perimeter: P = a + b + c
• Semiperimeter: s = (a+b+c)/2

Rectangle:
• Area: A = w × h
• Perimeter: P = 2(w + h)
• Diagonal: d = √(w² + h²)

Regular Polygon:
• Area: A = (n × s × a)/2
• Perimeter: P = n × s
• Interior angle: θ = ((n-2)×180°)/n

3D Solid Formulas:

Sphere:
• Volume: V = (4/3)πr³
• Surface Area: A = 4πr²

Cube:
• Volume: V = s³
• Surface Area: A = 6s²
• Space Diagonal: d = s√3

Cylinder:
• Volume: V = πr²h
• Lateral Surface Area: A = 2πrh
• Total Surface Area: A = 2πr(r + h)

Cone:
• Volume: V = (1/3)πr²h
• Lateral Surface Area: A = πrℓ
• Total Surface Area: A = πr(r + ℓ)
• Slant Height: ℓ = √(r² + h²)
        """)
        layout.addWidget(formulas_text)

        return widget

    def create_calculations_tab(self) -> QWidget:
        """Create calculations display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Calculations display
        self.calculations_text = QTextEdit()
        self.calculations_text.setReadOnly(True)
        self.calculations_text.setPlainText("Real-time calculations will be displayed here.")
        layout.addWidget(self.calculations_text)

        return widget

    def create_right_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Visualization tabs
        vis_tabs = QTabWidget()

        # 2D Geometry
        self.canvas_2d = Geometry2DCanvas(self)
        vis_tabs.addTab(self.canvas_2d, "2D Geometry")

        # 3D Geometry
        self.canvas_3d = Geometry3DCanvas(self)
        vis_tabs.addTab(self.canvas_3d, "3D Geometry")

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

    def on_2d_shape_changed(self, shape: str):
        """Handle 2D shape selection change."""
        self.current_2d_shape = shape
        self.setup_2d_parameters()
        self.update_2d_shape()

    def on_3d_shape_changed(self, solid: str):
        """Handle 3D solid selection change."""
        self.current_3d_solid = solid
        self.setup_3d_parameters()
        self.update_3d_solid()

    def update_2d_shape(self):
        """Update 2D shape visualization."""
        params = self.get_2d_shape_params()
        if self.canvas_2d:
            self.canvas_2d.update_shape(self.current_2d_shape, params)
        self.update_properties_display()

    def update_3d_solid(self):
        """Update 3D solid visualization."""
        params = self.get_3d_solid_params()
        if self.canvas_3d:
            self.canvas_3d.update_solid(self.current_3d_solid, params)
        self.update_properties_display()

    def get_2d_shape_params(self) -> dict:
        """Get current 2D shape parameters."""
        params = {
            'center_x': self.center_x_2d.value(),
            'center_y': self.center_y_2d.value()
        }

        if self.current_2d_shape == 'circle':
            params['radius'] = self.radius_2d.value()
        elif self.current_2d_shape == 'triangle':
            params['vertices'] = [
                (self.v1_x.value(), self.v1_y.value()),
                (self.v2_x.value(), self.v2_y.value()),
                (self.v3_x.value(), self.v3_y.value())
            ]
        elif self.current_2d_shape == 'rectangle':
            params['width'] = self.width_2d.value()
            params['height'] = self.height_2d.value()
        elif self.current_2d_shape == 'polygon':
            params['sides'] = self.sides_2d.value()
            params['radius'] = self.radius_poly.value()

        return params

    def get_3d_solid_params(self) -> dict:
        """Get current 3D solid parameters."""
        params = {
            'center_x': self.center_x_3d.value(),
            'center_y': self.center_y_3d.value(),
            'center_z': self.center_z_3d.value()
        }

        if self.current_3d_solid == 'sphere':
            params['radius'] = self.radius_3d.value()
        elif self.current_3d_solid == 'cube':
            params['side'] = self.side_3d.value()
        elif self.current_3d_solid == 'cylinder':
            params['radius'] = self.radius_cyl.value()
            params['height'] = self.height_cyl.value()
        elif self.current_3d_solid == 'cone':
            params['radius'] = self.radius_cone.value()
            params['height'] = self.height_cone.value()

        return params

    def update_properties_display(self):
        """Update properties display."""
        properties_text = ""

        # 2D Shape properties
        if self.current_2d_shape == 'circle':
            radius = self.radius_2d.value()
            props = GeometryCalculator.circle_properties(radius)
            properties_text += f"2D Circle (r = {radius}):\n"
            properties_text += f"• Area: {props['area']:.4f}\n"
            properties_text += f"• Circumference: {props['circumference']:.4f}\n"
            properties_text += f"• Diameter: {props['diameter']:.4f}\n\n"

        elif self.current_2d_shape == 'triangle':
            vertices = [
                (self.v1_x.value(), self.v1_y.value()),
                (self.v2_x.value(), self.v2_y.value()),
                (self.v3_x.value(), self.v3_y.value())
            ]
            a = GeometryCalculator.distance_2d(*vertices[0], *vertices[1])
            b = GeometryCalculator.distance_2d(*vertices[1], *vertices[2])
            c = GeometryCalculator.distance_2d(*vertices[2], *vertices[0])
            props = GeometryCalculator.triangle_properties(a, b, c)

            properties_text += f"2D Triangle (sides: {a:.2f}, {b:.2f}, {c:.2f}):\n"
            if 'error' not in props:
                properties_text += f"• Area: {props['area']:.4f}\n"
                properties_text += f"• Perimeter: {props['perimeter']:.4f}\n"
                properties_text += f"• Type: {props['type']}\n"
                properties_text += f"• Angles: {props['angles'][0]:.1f}°, {props['angles'][1]:.1f}°, {props['angles'][2]:.1f}°\n\n"
            else:
                properties_text += "• Invalid triangle\n\n"

        elif self.current_2d_shape == 'rectangle':
            width = self.width_2d.value()
            height = self.height_2d.value()
            props = GeometryCalculator.rectangle_properties(width, height)
            properties_text += f"2D Rectangle ({width} × {height}):\n"
            properties_text += f"• Area: {props['area']:.4f}\n"
            properties_text += f"• Perimeter: {props['perimeter']:.4f}\n"
            properties_text += f"• Diagonal: {props['diagonal']:.4f}\n\n"

        elif self.current_2d_shape == 'polygon':
            sides = self.sides_2d.value()
            radius = self.radius_poly.value()
            side_length = 2 * radius * math.sin(math.pi / sides)
            props = GeometryCalculator.polygon_properties(sides, side_length)
            properties_text += f"2D Regular {sides}-gon (r = {radius}):\n"
            properties_text += f"• Area: {props['area']:.4f}\n"
            properties_text += f"• Perimeter: {props['perimeter']:.4f}\n"
            properties_text += f"• Interior Angle: {props['interior_angle']:.1f}°\n"
            properties_text += f"• Exterior Angle: {props['exterior_angle']:.1f}°\n\n"

        # 3D Solid properties
        if self.current_3d_solid == 'sphere':
            radius = self.radius_3d.value()
            props = GeometryCalculator.sphere_properties(radius)
            properties_text += f"3D Sphere (r = {radius}):\n"
            properties_text += f"• Volume: {props['volume']:.4f}\n"
            properties_text += f"• Surface Area: {props['surface_area']:.4f}\n\n"

        elif self.current_3d_solid == 'cube':
            side = self.side_3d.value()
            props = GeometryCalculator.cube_properties(side)
            properties_text += f"3D Cube (s = {side}):\n"
            properties_text += f"• Volume: {props['volume']:.4f}\n"
            properties_text += f"• Surface Area: {props['surface_area']:.4f}\n"
            properties_text += f"• Space Diagonal: {props['space_diagonal']:.4f}\n\n"

        elif self.current_3d_solid == 'cylinder':
            radius = self.radius_cyl.value()
            height = self.height_cyl.value()
            props = GeometryCalculator.cylinder_properties(radius, height)
            properties_text += f"3D Cylinder (r = {radius}, h = {height}):\n"
            properties_text += f"• Volume: {props['volume']:.4f}\n"
            properties_text += f"• Lateral Surface Area: {props['lateral_surface_area']:.4f}\n"
            properties_text += f"• Total Surface Area: {props['total_surface_area']:.4f}\n\n"

        elif self.current_3d_solid == 'cone':
            radius = self.radius_cone.value()
            height = self.height_cone.value()
            props = GeometryCalculator.cone_properties(radius, height)
            properties_text += f"3D Cone (r = {radius}, h = {height}):\n"
            properties_text += f"• Volume: {props['volume']:.4f}\n"
            properties_text += f"• Lateral Surface Area: {props['lateral_surface_area']:.4f}\n"
            properties_text += f"• Total Surface Area: {props['total_surface_area']:.4f}\n"
            properties_text += f"• Slant Height: {props['slant_height']:.4f}\n\n"

        self.properties_text.setPlainText(properties_text)

    def update_2d_visualization(self):
        """Update 2D visualization settings."""
        if self.canvas_2d:
            self.canvas_2d.show_grid = self.show_2d_grid.isChecked()
            self.canvas_2d.show_labels = self.show_2d_labels.isChecked()
            self.canvas_2d.show_measurements = self.show_2d_measurements.isChecked()
            self.canvas_2d.fill_shapes = self.fill_2d_shapes.isChecked()
            self.canvas_2d.redraw()

    def update_3d_visualization(self):
        """Update 3D visualization settings."""
        if self.canvas_3d:
            self.canvas_3d.show_wireframe = self.show_3d_wireframe.isChecked()
            self.canvas_3d.redraw()

    def on_opacity_changed(self, value: int):
        """Handle opacity change."""
        opacity = value / 100.0
        self.opacity_label.setText(f"{opacity:.2f}")
        if self.canvas_3d:
            self.canvas_3d.opacity = opacity
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

        if self.canvas_3d:
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

    def export_properties(self):
        """Export properties as text file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Properties", "", "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if filename:
            with open(filename, 'w') as f:
                f.write("Geometry Tool - Shape Properties\n")
                f.write("=" * 40 + "\n\n")
                f.write(self.properties_text.toPlainText())

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Geometry Tool",
            "Geometry Tool v1.0\n\n"
            "An interactive tool for visualizing and analyzing geometric shapes\n"
            "with comprehensive 2D and 3D geometry calculations.\n\n"
            "Features:\n"
            "• 2D shape visualization (circle, triangle, rectangle, polygon)\n"
            "• 3D solid visualization (sphere, cube, cylinder, cone)\n"
            "• Real-time property calculations\n"
            "• Interactive parameter adjustment\n"
            "• Coordinate geometry tools\n"
            "• Animation capabilities\n"
            "• Export functionality\n\n"
            "Perfect for geometry education and analysis."
        )

TOOL_META = {
    "name": "Geometry Tool",
    "description": "Interactive 2D/3D geometric shape visualization with property calculations, coordinate geometry, and educational content",
    "category": "Mathematics",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "Interactive 2D shape construction and visualization",
        "3D geometric solid rendering with properties",
        "Coordinate geometry and transformations",
        "Real-time parameter adjustment and calculations",
        "Educational information and formulas",
        "Export capabilities for images and data",
        "Animation and interactive exploration",
        "Multiple visualization modes and styles",
        "Circle, triangle, rectangle, and polygon analysis",
        "Sphere, cube, cylinder, and cone calculations",
        "Distance, midpoint, and rotation calculations",
        "Area, volume, perimeter, and surface area formulas",
        "Wireframe and solid rendering modes",
        "Camera controls for 3D visualization",
        "Property comparison and analysis",
        "Mathematical formula reference",
        "Real-time calculation updates",
        "Educational geometry content",
        "Interactive shape manipulation",
        "Comprehensive measurement tools"
    ],
    "educational_value": "Master geometry concepts through interactive visualization, understand shape properties, learn coordinate geometry, and explore 3D solids",
    "keywords": ["geometry", "shapes", "2d visualization", "3d solids", "area calculation", "volume calculation", "coordinate geometry", "transformations", "mathematics education", "interactive learning"]
}


def create_tool(parent=None):
    """Create and return a GeometryTool instance."""
    return GeometryTool(parent)