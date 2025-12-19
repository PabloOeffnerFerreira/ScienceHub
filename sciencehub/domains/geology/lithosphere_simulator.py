"""
2D Lithosphere Stress & Deformation Simulator (ScienceHub)
=========================================================

A real-time, interactive 2D lithosphere simulator focused on:
- Plate velocities + boundary types (convergent/divergent/transform/oblique)
- Elastic + brittle damage (Mohr–Coulomb-ish) with optional temperature weakening
- Stress tensor fields + strain accumulation + emergent fault bands
- Earthquake-like rupture cascades + event catalog + magnitude proxy plots
- Stunning dark UI with pyqtgraph visuals, overlays, painting tools, presets, exports

Design goals:
- Fast enough for live interaction (vectorized numpy + light relaxation)
- Educational but visually impressive
- Single-file tool integration (no data tables here)

Dependencies:
- numpy
- PyQt6
- pyqtgraph

Author: ScienceHub Team
"""

from __future__ import annotations

import math
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSlider, QDoubleSpinBox, QCheckBox, QGroupBox, QFrame,
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog, QSplitter,
    QTabWidget, QSpinBox, QLineEdit, QScrollArea
)
import pyqtgraph.exporters
import pyqtgraph as pg

from sciencehub.ui.components.tool_base import ScienceHubTool


# -----------------------------
# Utility / math
# -----------------------------

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

def safe_log10(x: float) -> float:
    x = abs(x)
    if x <= 0:
        return -999.0
    return math.log10(x)

def principal_stresses(sig_xx: np.ndarray, sig_yy: np.ndarray, tau_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 2D principal stresses
    mean = 0.5 * (sig_xx + sig_yy)
    rad = np.sqrt((0.5 * (sig_xx - sig_yy))**2 + tau_xy**2)
    s1 = mean + rad
    s3 = mean - rad
    return s1, s3

def max_shear(sig_xx: np.ndarray, sig_yy: np.ndarray, tau_xy: np.ndarray) -> np.ndarray:
    # tau_max = (s1-s3)/2
    s1, s3 = principal_stresses(sig_xx, sig_yy, tau_xy)
    return 0.5 * (s1 - s3)

def von_mises_2d(sig_xx: np.ndarray, sig_yy: np.ndarray, tau_xy: np.ndarray) -> np.ndarray:
    # 2D plane-stress-ish von Mises proxy
    return np.sqrt(sig_xx**2 - sig_xx*sig_yy + sig_yy**2 + 3*tau_xy**2)


# -----------------------------
# Event model
# -----------------------------

@dataclass
class QuakeEvent:
    t: float
    step: int
    cells: int
    area: float
    mean_slip: float
    peak_slip: float
    stress_drop: float
    energy_proxy: float
    mag_proxy: float
    centroid: Tuple[float, float]


# -----------------------------
# Core simulation engine
# -----------------------------

class LithosphereSimCore:
    """
    Grid-based quasi-static elastic + brittle damage simulator.

    State fields (Ny x Nx):
      u, v: displacement components
      exx, eyy, exy: strain components (small strain)
      sxx, syy, sxy: stress components
      damage: [0..1] weakness / faulting
      pstrain: accumulated plastic strain invariant
      lock: "locking" proxy (failure margin) used for visuals

    Driving:
      - Plate velocity boundary conditions on left/right/top/bottom
      - Boundary type modifies coupling + shear/normal drive proportions
    """

    def __init__(self, nx: int = 180, ny: int = 120, dx: float = 1000.0, dy: float = 1000.0):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy

        self.drive_scale = 2000.0

        # time / stepping
        self.step = 0
        self.t = 0.0
        self.dt = 0.05  # pseudo-time (not physical seconds)
        self.relax_iters = 20
        self.damping = 0.35  # relaxation damping

        # material (base)
        self.E = 60e9          # Young's modulus (Pa)
        self.nu = 0.25         # Poisson ratio
        self.rho = 2800.0      # density (kg/m^3) (mostly unused here)
        self.cohesion = 25e6   # Pa
        self.friction_deg = 30.0
        self.tensile = 10e6    # Pa

        # damage / plasticity
        self.damage_rate = 0.035
        self.heal_rate = 0.002
        self.softening = 0.55     # strength reduction with damage
        self.event_threshold = 0.16  # failure margin threshold to trigger ruptures
        self.stress_drop_factor = 0.55  # fraction of shear stress dropped in rupture
        self.transfer = 0.25  # stress transfer to neighbors during rupture cascade

        # thermal / strength gradient
        self.use_temperature = False
        self.temp_surface = 273.0
        self.temp_gradient = 20.0 / 1000.0  # K per meter (20 K/km)
        self.temp_weakening = 0.004  # per K (simple exponential strength reduction)
        self.crust_thickness = 35000.0  # meters
        self.bdt_depth = 18000.0  # brittle-ductile transition proxy depth (m)
        self.ductile_strength_floor = 0.15  # minimum strength multiplier at depth

        # boundary + plates
        self.boundary_mode = "Cross-section (x vs depth)"
        self.boundaries = {
            "left":  {"type": "transform",   "vx": 0.02, "vy": 0.00, "coupling": 0.85},
            "right": {"type": "transform",   "vx": -0.02, "vy": 0.00, "coupling": 0.85},
            "top":   {"type": "free",        "vx": 0.00, "vy": 0.00, "coupling": 0.0},
            "bottom":{"type": "fixed",       "vx": 0.00, "vy": 0.00, "coupling": 1.0},
        }

        # heterogeneity paint mask (multipliers)
        self.weakness = np.ones((self.ny, self.nx), dtype=np.float32)

        # fields
        self.reset()

    def reset(self):
        ny, nx = self.ny, self.nx
        self.step = 0
        self.t = 0.0

        self.u = np.zeros((ny, nx), dtype=np.float32)
        self.v = np.zeros((ny, nx), dtype=np.float32)

        self.exx = np.zeros((ny, nx), dtype=np.float32)
        self.eyy = np.zeros((ny, nx), dtype=np.float32)
        self.exy = np.zeros((ny, nx), dtype=np.float32)

        self.sxx = np.zeros((ny, nx), dtype=np.float32)
        self.syy = np.zeros((ny, nx), dtype=np.float32)
        self.sxy = np.zeros((ny, nx), dtype=np.float32)

        self.damage = np.zeros((ny, nx), dtype=np.float32)
        self.pstrain = np.zeros((ny, nx), dtype=np.float32)
        self.lock = np.zeros((ny, nx), dtype=np.float32)  # "how close to failure"

        # temperature field (depth-based)
        self.temp = self._compute_temperature_field()

        # cached
        self._events: List[QuakeEvent] = []
        
    def _compute_temperature_field(self) -> np.ndarray:
        y = (np.arange(self.ny, dtype=np.float32) * self.dy)[:, None]  # (ny,1)
        T_col = self.temp_surface + self.temp_gradient * y            # (ny,1)
        return np.repeat(T_col, self.nx, axis=1).astype(np.float32)   # (ny,nx)

    def set_resolution(self, nx: int, ny: int):
        self.nx, self.ny = int(nx), int(ny)
        self.weakness = np.ones((self.ny, self.nx), dtype=np.float32)
        self.reset()

    def set_material(self, E: float, nu: float, cohesion: float, friction_deg: float, tensile: float):
        self.E = float(E)
        self.nu = float(nu)
        self.cohesion = float(cohesion)
        self.friction_deg = float(friction_deg)
        self.tensile = float(tensile)

    def set_thermal(self, enabled: bool, surface: float, grad_K_per_km: float, crust_km: float, weakening: float, bdt_km: float):
        self.use_temperature = bool(enabled)
        self.temp_surface = float(surface)
        self.temp_gradient = float(grad_K_per_km) / 1000.0
        self.crust_thickness = float(crust_km) * 1000.0
        self.temp_weakening = float(weakening)
        self.bdt_depth = float(bdt_km) * 1000.0
        self.temp = self._compute_temperature_field()

    def set_dynamics(self, dt: float, relax_iters: int, damping: float):
        self.dt = float(dt)
        self.relax_iters = int(relax_iters)
        self.damping = float(damping)

    def set_damage_params(self, damage_rate: float, heal_rate: float, softening: float,
                          event_threshold: float, stress_drop_factor: float, transfer: float):
        self.damage_rate = float(damage_rate)
        self.heal_rate = float(heal_rate)
        self.softening = float(softening)
        self.event_threshold = float(event_threshold)
        self.stress_drop_factor = float(stress_drop_factor)
        self.transfer = float(transfer)

    def set_boundary(self, side: str, btype: str, vx: float, vy: float, coupling: float):
        self.boundaries[side] = {"type": btype, "vx": float(vx), "vy": float(vy), "coupling": float(coupling)}

    def add_weakness_paint(self, ix: int, iy: int, radius: int, strength: float):
        """
        Paint weakness multiplier.
        strength < 1 => weaker, >1 => stronger.
        """
        ix = int(ix); iy = int(iy)
        r = max(1, int(radius))
        y0 = max(0, iy - r); y1 = min(self.ny, iy + r + 1)
        x0 = max(0, ix - r); x1 = min(self.nx, ix + r + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist2 = (yy - iy)**2 + (xx - ix)**2
        mask = dist2 <= r*r
        # soft brush: gaussian-ish
        w = np.exp(-dist2.astype(np.float32) / (0.35 * r*r + 1e-6))
        local = self.weakness[y0:y1, x0:x1]
        # blend toward strength
        target = np.ones_like(local) * float(strength)
        local[mask] = (1 - 0.35*w[mask]) * local[mask] + (0.35*w[mask]) * target[mask]
        self.weakness[y0:y1, x0:x1] = local

    # ---- mechanics ----

    def _effective_strength_multiplier(self) -> np.ndarray:
        """
        Combine damage + weakness + temperature weakening + brittle/ductile depth effect.
        """
        mult = np.ones((self.ny, self.nx), dtype=np.float32)

        # damage reduces strength
        mult *= (1.0 - self.softening * self.damage).astype(np.float32)
        mult *= self.weakness.astype(np.float32)

        if self.use_temperature:
            # simple exponential weakening with temperature and ductile transition
            T = self.temp
            # depth proxy
            y = (np.arange(self.ny, dtype=np.float32) * self.dy)[:, None]

            # temperature weakening
            tw = np.exp(-self.temp_weakening * (T - self.temp_surface)).astype(np.float32)
            tw = np.clip(tw, self.ductile_strength_floor, 1.0)

            # brittle-ductile transition smoothing: below bdt depth, strength floors
            bdt = self.bdt_depth
            smooth = 1.0 / (1.0 + np.exp((y - bdt) / (2500.0)))  # ~1 above, ~0 below
            # blend to floor below
            depth_mult = (smooth + (1 - smooth) * self.ductile_strength_floor).astype(np.float32)

            mult *= tw
            mult *= depth_mult

        return np.clip(mult, 0.05, 1.5).astype(np.float32)

    def _lame(self) -> Tuple[float, float]:
        # Lamé parameters
        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu

    def _apply_boundary_velocities(self, u_new: np.ndarray, v_new: np.ndarray):
        """
        Convert boundary velocities to displacement increments along edges.
        """
        dt = self.dt
        b = self.boundaries

        # Side behavior: type influences how much normal vs shear is coupled
        def couple(side: str) -> Tuple[float, float, float]:
            info = b[side]
            vx, vy = info["vx"], info["vy"]
            c = clamp(info["coupling"], 0.0, 1.0)
            t = info["type"]

            # free/fixed shortcuts
            if t == "free":
                return 0.0, 0.0, 0.0
            if t == "fixed":
                return 0.0, 0.0, 1.0

            # For convergent/divergent emphasize normal component (into/out of boundary)
            # For transform emphasize tangential component
            # For oblique keep both
            if side in ("left", "right"):
                # normal is x, tangential is y (in cross-section sense)
                if t == "convergent":
                    return vx * 1.0, vy * 0.25, c
                if t == "divergent":
                    return vx * 1.0, vy * 0.25, c
                if t == "transform":
                    return vx * 0.25, vy * 1.0, c
                if t == "oblique":
                    return vx * 1.0, vy * 1.0, c
            else:
                # top/bottom: normal is y
                if t == "convergent":
                    return vx * 0.25, vy * 1.0, c
                if t == "divergent":
                    return vx * 0.25, vy * 1.0, c
                if t == "transform":
                    return vx * 1.0, vy * 0.25, c
                if t == "oblique":
                    return vx * 1.0, vy * 1.0, c

            return vx, vy, c

        vxL, vyL, cL = couple("left")
        vxR, vyR, cR = couple("right")
        vxT, vyT, cT = couple("top")
        vxB, vyB, cB = couple("bottom")

        scale = self.drive_scale
        u_new[:, 0]  += cL * vxL * dt * scale
        v_new[:, 0]  += cL * vyL * dt * scale
        u_new[:, -1] += cR * vxR * dt * scale
        v_new[:, -1] += cR * vyR * dt * scale
        u_new[0, :]  += cT * vxT * dt * scale
        v_new[0, :]  += cT * vyT * dt * scale
        u_new[-1, :] += cB * vxB * dt * scale
        v_new[-1, :] += cB * vyB * dt * scale

        # If fixed boundary, clamp displacements to 0 (more rigid)
        if self.boundaries["bottom"]["type"] == "fixed":
            u_new[-1, :] = 0.0
            v_new[-1, :] = 0.0
        if self.boundaries["top"]["type"] == "fixed":
            u_new[0, :] = 0.0
            v_new[0, :] = 0.0
        if self.boundaries["left"]["type"] == "fixed":
            u_new[:, 0] = 0.0
            v_new[:, 0] = 0.0
        if self.boundaries["right"]["type"] == "fixed":
            u_new[:, -1] = 0.0
            v_new[:, -1] = 0.0

    def _compute_strain(self):
        # central differences
        u, v = self.u, self.v
        dx, dy = self.dx, self.dy

        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
        dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dy)
        du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dy)
        dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dx)

        self.exx = du_dx.astype(np.float32)
        self.eyy = dv_dy.astype(np.float32)
        self.exy = (0.5 * (du_dy + dv_dx)).astype(np.float32)

        # edges: make derivatives more stable by copying neighbor
        self.exx[:, 0] = self.exx[:, 1]
        self.exx[:, -1] = self.exx[:, -2]
        self.eyy[0, :] = self.eyy[1, :]
        self.eyy[-1, :] = self.eyy[-2, :]

        self.exy[0, :] = self.exy[1, :]
        self.exy[-1, :] = self.exy[-2, :]
        self.exy[:, 0] = self.exy[:, 1]
        self.exy[:, -1] = self.exy[:, -2]

    def _compute_stress(self):
        lam, mu = self._lame()
        exx, eyy, exy = self.exx, self.eyy, self.exy

        tr = (exx + eyy)
        self.sxx = (lam * tr + 2 * mu * exx).astype(np.float32)
        self.syy = (lam * tr + 2 * mu * eyy).astype(np.float32)
        self.sxy = (2 * mu * exy).astype(np.float32)

        # Introduce strength reduction by damage as a "softening" of shear stress too (helps fault visuals)
        soft = (1.0 - 0.35 * self.damage).astype(np.float32)
        self.sxy *= soft

    def _failure_margin(self) -> np.ndarray:
        """
        Return a failure margin in [0..1+] where >1 means failing.

        We use a Mohr–Coulomb-like proxy with principal stresses:
          tau_max > c + sigma_n * tan(phi)
        Approximate sigma_n as mean compressive stress (positive compression here).
        """
        sxx, syy, sxy = self.sxx, self.syy, self.sxy
        s1, s3 = principal_stresses(sxx, syy, sxy)

        # convention: compression positive
        sigma_n = np.maximum(0.0, 0.5 * (s1 + s3))  # mean normal
        tau = 0.5 * (s1 - s3)

        phi = math.radians(self.friction_deg)
        strength_mult = self._effective_strength_multiplier()

        # tensile cutoff: if s3 is too tensile (negative compression), encourage failure
        tensile_fail = np.maximum(0.0, (-s3 - self.tensile * strength_mult) / (self.tensile + 1e-9))

        # shear failure ratio
        tau_strength = (self.cohesion * strength_mult + sigma_n * math.tan(phi)).astype(np.float32)
        ratio = (tau / (tau_strength + 1e-9)).astype(np.float32)

        # combine
        margin = np.maximum(ratio, tensile_fail).astype(np.float32)
        return margin

    def _apply_plasticity_and_damage(self) -> Tuple[np.ndarray, float]:
        """
        If margin > 1, add plastic strain and damage.
        Returns:
          fail_mask (bool)
          mean_margin
        """
        margin = self._failure_margin()
        self.lock = np.clip(margin, 0.0, 2.0).astype(np.float32)

        fail = margin > 1.0
        if np.any(fail):
            # plastic strain increment proportional to excess margin
            excess = (margin - 1.0).astype(np.float32)
            excess[~fail] = 0.0

            # add to plastic strain (invariant proxy)
            self.pstrain += 0.55 * excess * self.dt
            self.pstrain = np.clip(self.pstrain, 0.0, 50.0).astype(np.float32)

            # damage evolves
            self.damage += self.damage_rate * excess * self.dt
            self.damage = np.clip(self.damage, 0.0, 1.0).astype(np.float32)
        else:
            # healing
            self.damage -= self.heal_rate * self.dt
            self.damage = np.clip(self.damage, 0.0, 1.0).astype(np.float32)

        return fail, float(np.mean(margin))

    def _rupture_cascade(self, fail_seed: np.ndarray) -> Optional[QuakeEvent]:
        """
        Identify contiguous failing region(s) and apply a stress-drop + transfer.
        Returns largest event (if any).
        """
        if not np.any(fail_seed):
            return None

        ny, nx = self.ny, self.nx
        visited = np.zeros((ny, nx), dtype=np.uint8)

        best_event = None
        best_cells = 0

        # 4-neighborhood flood fill
        ys, xs = np.where(fail_seed)
        for start_y, start_x in zip(ys.tolist(), xs.tolist()):
            if visited[start_y, start_x]:
                continue
            if not fail_seed[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = 1
            cells = []

            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and not visited[yy, xx] and fail_seed[yy, xx]:
                        visited[yy, xx] = 1
                        stack.append((yy, xx))

            if len(cells) < 25:
                continue  # ignore tiny pops for catalog (still affects damage)

            # compute slip proxy from margin excess
            idx_y = np.array([c[0] for c in cells], dtype=np.int32)
            idx_x = np.array([c[1] for c in cells], dtype=np.int32)

            # slip proxy proportional to local lock exceedance
            margin = self.lock[idx_y, idx_x]
            slip = np.clip(margin - 1.0, 0.0, 1.25).astype(np.float32)

            peak_slip = float(np.max(slip))
            mean_slip = float(np.mean(slip))
            area = float(len(cells) * self.dx * self.dy)

            # stress drop and transfer: drop shear stress in patch
            sxy_patch = self.sxy[idx_y, idx_x]
            stress_drop = self.stress_drop_factor * sxy_patch
            self.sxy[idx_y, idx_x] = (sxy_patch - stress_drop).astype(np.float32)

            # transfer to neighbors (very simple)
            # add a fraction of stress drop to surrounding ring
            for y, x in cells[:: max(1, len(cells)//1200)]:  # sample to reduce cost
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy = y + dy; xx = x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and not fail_seed[yy, xx]:
                        self.sxy[yy, xx] += self.transfer * float(self.sxy[y, x]) * 0.02

            # energy proxy
            energy = float(np.sum(np.abs(stress_drop)) * self.dx * self.dy)
            # magnitude proxy (not real Mw, but behaves similarly)
            mag = (2.0/3.0) * safe_log10(max(energy, 1e-6)) - 2.0

            cy = float(np.mean(idx_y))
            cx = float(np.mean(idx_x))

            event = QuakeEvent(
                t=self.t,
                step=self.step,
                cells=len(cells),
                area=area,
                mean_slip=mean_slip,
                peak_slip=peak_slip,
                stress_drop=float(np.mean(np.abs(stress_drop))) if len(stress_drop) else 0.0,
                energy_proxy=energy,
                mag_proxy=mag,
                centroid=(cx, cy),
            )

            if len(cells) > best_cells:
                best_cells = len(cells)
                best_event = event

        return best_event

    def step_once(self) -> Optional[QuakeEvent]:
        """
        Perform one simulation step:
          - boundary driving
          - relaxation iterations to spread deformation
          - compute strain/stress
          - damage + failure
          - rupture cascade event (optional)
        """
        self.step += 1
        self.t += self.dt

        # drive displacements at boundaries
        u_new = self.u.copy()
        v_new = self.v.copy()
        self._apply_boundary_velocities(u_new, v_new)

        # relaxation: diffuse displacements to approximate quasi-static elasticity
        # We use a weighted Laplacian smoothing, modulated by damage (fault zones deform more)
        for _ in range(self.relax_iters):
            # neighbor average
            u_avg = 0.25 * (np.roll(u_new, 1, 0) + np.roll(u_new, -1, 0) + np.roll(u_new, 1, 1) + np.roll(u_new, -1, 1))
            v_avg = 0.25 * (np.roll(v_new, 1, 0) + np.roll(v_new, -1, 0) + np.roll(v_new, 1, 1) + np.roll(v_new, -1, 1))

            # allow more deformation where damaged
            deform = (0.45 + 0.85 * self.damage).astype(np.float32)
            alpha = self.damping * deform

            u_new = (1 - alpha) * u_new + alpha * u_avg
            v_new = (1 - alpha) * v_new + alpha * v_avg

            # respect fixed boundaries each iter
            if self.boundaries["bottom"]["type"] == "fixed":
                u_new[-1, :] = 0.0
                v_new[-1, :] = 0.0

        self.u = u_new.astype(np.float32)
        self.v = v_new.astype(np.float32)

        self._compute_strain()
        self._compute_stress()
        fail_mask, _ = self._apply_plasticity_and_damage()

        # quake detection: fail margin above threshold
        rupture_seed = self.lock > (1.0 + self.event_threshold)
        # include current failing
        rupture_seed |= fail_mask

        evt = self._rupture_cascade(rupture_seed)
        if evt is not None:
            self._events.append(evt)

        return evt

    def events(self) -> List[QuakeEvent]:
        return list(self._events)

    # Derived fields for visualization
    def field(self, name: str) -> np.ndarray:
        if name == "von Mises":
            return von_mises_2d(self.sxx, self.syy, self.sxy)
        if name == "Max Shear":
            return max_shear(self.sxx, self.syy, self.sxy)
        if name == "σxx":
            return self.sxx
        if name == "σyy":
            return self.syy
        if name == "τxy":
            return self.sxy
        if name == "σ1 (max)":
            s1, _ = principal_stresses(self.sxx, self.syy, self.sxy)
            return s1
        if name == "σ3 (min)":
            _, s3 = principal_stresses(self.sxx, self.syy, self.sxy)
            return s3
        if name == "Pressure (mean)":
            return 0.5 * (self.sxx + self.syy)
        if name == "Strain inv":
            return np.sqrt(self.exx**2 + self.eyy**2 + 2*self.exy**2)
        if name == "Plastic strain":
            return self.pstrain
        if name == "Damage (faults)":
            return self.damage
        if name == "Locking (fail)":
            return self.lock
        if name == "Weakness paint":
            return self.weakness
        if name == "Temperature":
            return self.temp
        return self.damage


# -----------------------------
# PyQtGraph helper: clickable image view with painting
# -----------------------------

class ClickableImageItem(pg.ImageItem):
    clicked = pg.QtCore.Signal(int, int)
    dragged = pg.QtCore.Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.setZValue(0)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            p = ev.pos()  # image-local coordinates
            x = int(p.x())
            y = int(p.y())
            self.clicked.emit(x, y)
            ev.accept()
        else:
            ev.ignore()

    def mouseDragEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            p = ev.pos()
            x = int(p.x())
            y = int(p.y())
            self.dragged.emit(x, y)
            ev.accept()
        else:
            ev.ignore()


class PaintableImageView(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#0b1220")

        self.plot = self.addPlot()
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")

        self.img = ClickableImageItem()
        self.plot.addItem(self.img)

        # Make pixel coords = world coords (IMPORTANT)
        self.plot.setAspectLocked(False)

    def set_image(self, arr: np.ndarray, levels=None):
        # arr is (ny, nx)
        self.img.setImage(arr, autoLevels=(levels is None))
        if levels is not None:
            self.img.setLevels(levels)

        # Define a proper geometry so coordinates are 0..nx, 0..ny
        ny, nx = arr.shape
        self.img.setRect(pg.QtCore.QRectF(0, 0, nx, ny))
        self.plot.setRange(xRange=(0, nx), yRange=(0, ny), padding=0.0)


# -----------------------------
# Main Tool UI
# -----------------------------

class LithosphereSimulatorTool(ScienceHubTool):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sim = LithosphereSimCore()
        self.running = False

        # Visual settings
        self.field_name = "von Mises"
        self.auto_levels = True
        self.level_min = 0.0
        self.level_max = 1.0
        self.show_fault_overlay = True
        self.show_vectors = False
        self.show_boundaries = True
        self.last_event_flash_t = -999.0

        # Painting
        self.paint_mode = "Weaken"
        self.brush_radius = 6
        self.brush_strength = 0.65  # <1 weaken, >1 strengthen
        self.paint_enabled = False

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._apply_style()
        self._apply_preset("Transform (San Andreas-ish)")
        self._refresh_all()

    # ---------------- UI ----------------

    def _build_ui(self):
        root = QVBoxLayout()
        root.setSpacing(12)
        self.root_layout.addLayout(root)

        header = QFrame()
        header.setObjectName("lsHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 14, 16, 14)

        title = QLabel("Lithosphere Stress & Deformation Simulator (2D)")
        title.setObjectName("lsTitle")
        subtitle = QLabel("Drive plates. Watch stress build. See faults form. Trigger quakes.")
        subtitle.setObjectName("lsSubtitle")

        tcol = QVBoxLayout()
        tcol.addWidget(title)
        tcol.addWidget(subtitle)

        h.addLayout(tcol)
        h.addStretch()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_run)
        self.step_btn = QPushButton("Step")
        self.step_btn.clicked.connect(self.step_once)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_sim)
        self.export_btn = QPushButton("Export Snapshot")
        self.export_btn.clicked.connect(self.export_snapshot)

        h.addWidget(self.play_btn)
        h.addWidget(self.step_btn)
        h.addWidget(self.reset_btn)
        h.addWidget(self.export_btn)

        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizes([420, 980, 420])
        root.addWidget(splitter, 1)

        left = self._wrap_scroll(self._build_left_panel())
        splitter.addWidget(left)

        center = self._build_center_panel()
        splitter.addWidget(center)

        right = self._wrap_scroll(self._build_right_panel())
        splitter.addWidget(right)

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setWidget(widget)
        # optional: nicer scrolling feel
        sa.setStyleSheet("""
            QScrollArea { background: transparent; }
            QScrollBar:vertical { width: 10px; background: transparent; }
            QScrollBar::handle:vertical { background: #1e2a46; border-radius: 5px; min-height: 25px; }
            QScrollBar::handle:vertical:hover { background: #3d6bff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        return sa


    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("lsPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Presets
        g0 = QGroupBox("Presets")
        g0.setObjectName("toolCard")
        gl0 = QHBoxLayout(g0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Transform (San Andreas-ish)",
            "Convergent (Himalaya-ish)",
            "Divergent (Rift/Iceland-ish)",
            "Oblique (Subduction-ish)",
            "Locked vs Creeping (segmentation demo)",
        ])
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        gl0.addWidget(self.preset_combo)
        lay.addWidget(g0)

        # Plate driving / boundaries
        gb = QGroupBox("Boundary Driving")
        gb.setObjectName("toolCard")
        grid = QGridLayout(gb)

        self.left_type = QComboBox(); self.left_type.addItems(["transform","convergent","divergent","oblique","fixed","free"])
        self.right_type = QComboBox(); self.right_type.addItems(["transform","convergent","divergent","oblique","fixed","free"])
        self.top_type = QComboBox(); self.top_type.addItems(["free","fixed","transform","convergent","divergent","oblique"])
        self.bottom_type = QComboBox(); self.bottom_type.addItems(["fixed","free","transform","convergent","divergent","oblique"])

        self.vxL = QDoubleSpinBox(); self.vxL.setRange(-0.2,0.2); self.vxL.setDecimals(3); self.vxL.setSingleStep(0.005)
        self.vxR = QDoubleSpinBox(); self.vxR.setRange(-0.2,0.2); self.vxR.setDecimals(3); self.vxR.setSingleStep(0.005)
        self.vyL = QDoubleSpinBox(); self.vyL.setRange(-0.2,0.2); self.vyL.setDecimals(3); self.vyL.setSingleStep(0.005)
        self.vyR = QDoubleSpinBox(); self.vyR.setRange(-0.2,0.2); self.vyR.setDecimals(3); self.vyR.setSingleStep(0.005)

        self.cL = QDoubleSpinBox(); self.cL.setRange(0,1); self.cL.setDecimals(2); self.cL.setSingleStep(0.05)
        self.cR = QDoubleSpinBox(); self.cR.setRange(0,1); self.cR.setDecimals(2); self.cR.setSingleStep(0.05)

        def add_row(r, lbl, w1, w2, w3=None):
            grid.addWidget(QLabel(lbl), r, 0)
            grid.addWidget(w1, r, 1)
            grid.addWidget(w2, r, 2)
            if w3 is not None:
                grid.addWidget(w3, r, 3)

        add_row(0, "Left type", self.left_type, QLabel(""), None)
        add_row(1, "Right type", self.right_type, QLabel(""), None)

        add_row(2, "Left vx,vy", self.vxL, self.vyL, self.cL)
        add_row(3, "Right vx,vy", self.vxR, self.vyR, self.cR)

        grid.addWidget(QLabel("Coupling"), 2, 3)
        grid.addWidget(QLabel("Coupling"), 3, 3)

        grid.addWidget(QLabel("Top"), 4, 0)
        grid.addWidget(self.top_type, 4, 1, 1, 2)
        grid.addWidget(QLabel("Bottom"), 5, 0)
        grid.addWidget(self.bottom_type, 5, 1, 1, 2)

        self.apply_bc_btn = QPushButton("Apply Boundary Settings")
        self.apply_bc_btn.clicked.connect(self._apply_boundary_settings)
        grid.addWidget(self.apply_bc_btn, 6, 0, 1, 4)

        lay.addWidget(gb)

        # Materials
        gm = QGroupBox("Material & Failure")
        gm.setObjectName("toolCard")
        mgrid = QGridLayout(gm)

        self.E_spin = QDoubleSpinBox(); self.E_spin.setRange(1, 300); self.E_spin.setDecimals(1); self.E_spin.setSuffix(" GPa")
        self.nu_spin = QDoubleSpinBox(); self.nu_spin.setRange(0.05, 0.49); self.nu_spin.setDecimals(2)
        self.coh_spin = QDoubleSpinBox(); self.coh_spin.setRange(0.1, 200); self.coh_spin.setDecimals(1); self.coh_spin.setSuffix(" MPa")
        self.phi_spin = QDoubleSpinBox(); self.phi_spin.setRange(0, 50); self.phi_spin.setDecimals(0); self.phi_spin.setSuffix("°")
        self.ten_spin = QDoubleSpinBox(); self.ten_spin.setRange(0.1, 100); self.ten_spin.setDecimals(1); self.ten_spin.setSuffix(" MPa")

        mgrid.addWidget(QLabel("E"), 0, 0); mgrid.addWidget(self.E_spin, 0, 1)
        mgrid.addWidget(QLabel("ν"), 0, 2); mgrid.addWidget(self.nu_spin, 0, 3)
        mgrid.addWidget(QLabel("Cohesion"), 1, 0); mgrid.addWidget(self.coh_spin, 1, 1)
        mgrid.addWidget(QLabel("Friction φ"), 1, 2); mgrid.addWidget(self.phi_spin, 1, 3)
        mgrid.addWidget(QLabel("Tensile"), 2, 0); mgrid.addWidget(self.ten_spin, 2, 1)

        self.apply_mat_btn = QPushButton("Apply Material Settings")
        self.apply_mat_btn.clicked.connect(self._apply_material_settings)
        mgrid.addWidget(self.apply_mat_btn, 3, 0, 1, 4)

        lay.addWidget(gm)

        # Thermal
        gt = QGroupBox("Temperature / Rheology (optional)")
        gt.setObjectName("toolCard")
        tgrid = QGridLayout(gt)

        self.temp_on = QCheckBox("Enable temperature weakening")
        self.temp_on.stateChanged.connect(self._apply_thermal_settings)

        self.Ts_spin = QDoubleSpinBox(); self.Ts_spin.setRange(150, 400); self.Ts_spin.setDecimals(0); self.Ts_spin.setSuffix(" K")
        self.grad_spin = QDoubleSpinBox(); self.grad_spin.setRange(0, 60); self.grad_spin.setDecimals(0); self.grad_spin.setSuffix(" K/km")
        self.crust_spin = QDoubleSpinBox(); self.crust_spin.setRange(5, 80); self.crust_spin.setDecimals(0); self.crust_spin.setSuffix(" km")
        self.bdt_spin = QDoubleSpinBox(); self.bdt_spin.setRange(5, 60); self.bdt_spin.setDecimals(0); self.bdt_spin.setSuffix(" km")
        self.weak_spin = QDoubleSpinBox(); self.weak_spin.setRange(0.0, 0.02); self.weak_spin.setDecimals(4)

        tgrid.addWidget(self.temp_on, 0, 0, 1, 4)
        tgrid.addWidget(QLabel("Surface T"), 1, 0); tgrid.addWidget(self.Ts_spin, 1, 1)
        tgrid.addWidget(QLabel("Gradient"), 1, 2); tgrid.addWidget(self.grad_spin, 1, 3)
        tgrid.addWidget(QLabel("Crust"), 2, 0); tgrid.addWidget(self.crust_spin, 2, 1)
        tgrid.addWidget(QLabel("BDT"), 2, 2); tgrid.addWidget(self.bdt_spin, 2, 3)
        tgrid.addWidget(QLabel("Weakening"), 3, 0); tgrid.addWidget(self.weak_spin, 3, 1)

        self.apply_therm_btn = QPushButton("Apply Thermal")
        self.apply_therm_btn.clicked.connect(self._apply_thermal_settings)
        tgrid.addWidget(self.apply_therm_btn, 4, 0, 1, 4)

        lay.addWidget(gt)

        # Simulation control
        gs = QGroupBox("Simulation Quality / Speed")
        gs.setObjectName("toolCard")
        sgrid = QGridLayout(gs)

        self.dt_spin = QDoubleSpinBox(); self.dt_spin.setRange(0.005, 0.5); self.dt_spin.setDecimals(3)
        self.iters_spin = QSpinBox(); self.iters_spin.setRange(1, 120)
        self.damp_spin = QDoubleSpinBox(); self.damp_spin.setRange(0.05, 0.95); self.damp_spin.setDecimals(2)

        sgrid.addWidget(QLabel("dt"), 0, 0); sgrid.addWidget(self.dt_spin, 0, 1)
        sgrid.addWidget(QLabel("Relax iters"), 0, 2); sgrid.addWidget(self.iters_spin, 0, 3)
        sgrid.addWidget(QLabel("Damping"), 1, 0); sgrid.addWidget(self.damp_spin, 1, 1)

        self.apply_dyn_btn = QPushButton("Apply Dynamics")
        self.apply_dyn_btn.clicked.connect(self._apply_dynamics)
        sgrid.addWidget(self.apply_dyn_btn, 2, 0, 1, 4)

        lay.addWidget(gs)

        lay.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("lsPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Field selection + overlays
        topbar = QHBoxLayout()

        self.field_combo = QComboBox()
        self.field_combo.addItems([
            "von Mises", "Max Shear", "σxx", "σyy", "τxy",
            "σ1 (max)", "σ3 (min)", "Pressure (mean)",
            "Strain inv", "Plastic strain", "Damage (faults)",
            "Locking (fail)", "Weakness paint", "Temperature",
        ])
        self.field_combo.currentTextChanged.connect(self._on_field_changed)

        self.auto_levels_chk = QCheckBox("Auto scale")
        self.auto_levels_chk.setChecked(True)
        self.auto_levels_chk.stateChanged.connect(self._refresh_view)

        self.fault_chk = QCheckBox("Fault overlay")
        self.fault_chk.setChecked(True)
        self.fault_chk.stateChanged.connect(self._refresh_view)

        self.paint_chk = QCheckBox("Paint mode")
        self.paint_chk.setChecked(False)
        self.paint_chk.stateChanged.connect(self._toggle_paint)

        topbar.addWidget(QLabel("Field"))
        topbar.addWidget(self.field_combo)
        topbar.addStretch(1)
        topbar.addWidget(self.auto_levels_chk)
        topbar.addWidget(self.fault_chk)
        topbar.addWidget(self.paint_chk)

        lay.addLayout(topbar)

        # Canvas
        self.view = PaintableImageView()
        self.view.img.clicked.connect(self._on_canvas_click)
        self.view.img.dragged.connect(self._on_canvas_drag)
        lay.addWidget(self.view, 1)

        # Painter controls + mini plots
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setSizes([540, 520])
        lay.addWidget(bottom)

        painter = self._build_paint_panel()
        bottom.addWidget(painter)

        plots = self._build_plot_panel()
        bottom.addWidget(plots)

        return panel

    def _build_paint_panel(self) -> QWidget:
        g = QGroupBox("Painter / Heterogeneity")
        g.setObjectName("toolCard")
        l = QGridLayout(g)

        self.paint_mode_combo = QComboBox()
        self.paint_mode_combo.addItems(["Weaken", "Strengthen", "Heal damage", "Damage (fault seed)", "Reset paint"])
        self.paint_mode_combo.currentTextChanged.connect(self._update_paint_mode)

        self.brush_radius_spin = QSpinBox()
        self.brush_radius_spin.setRange(1, 40)
        self.brush_radius_spin.setValue(self.brush_radius)
        self.brush_radius_spin.valueChanged.connect(lambda v: setattr(self, "brush_radius", int(v)))

        self.brush_strength_spin = QDoubleSpinBox()
        self.brush_strength_spin.setRange(0.2, 2.5)
        self.brush_strength_spin.setDecimals(2)
        self.brush_strength_spin.setSingleStep(0.05)
        self.brush_strength_spin.setValue(self.brush_strength)
        self.brush_strength_spin.valueChanged.connect(lambda v: setattr(self, "brush_strength", float(v)))

        self.paint_hint = QLabel("Tip: enable Paint mode, then drag on the map.")
        self.paint_hint.setObjectName("lsHint")

        self.random_hetero_btn = QPushButton("Add Heterogeneity Noise")
        self.random_hetero_btn.clicked.connect(self._add_heterogeneity_noise)

        self.clear_damage_btn = QPushButton("Clear Damage / Faults")
        self.clear_damage_btn.clicked.connect(self._clear_damage)

        l.addWidget(QLabel("Mode"), 0, 0); l.addWidget(self.paint_mode_combo, 0, 1, 1, 3)
        l.addWidget(QLabel("Brush radius"), 1, 0); l.addWidget(self.brush_radius_spin, 1, 1)
        l.addWidget(QLabel("Strength"), 1, 2); l.addWidget(self.brush_strength_spin, 1, 3)
        l.addWidget(self.paint_hint, 2, 0, 1, 4)
        l.addWidget(self.random_hetero_btn, 3, 0, 1, 4)
        l.addWidget(self.clear_damage_btn, 4, 0, 1, 4)

        return g

    def _build_plot_panel(self) -> QWidget:
        g = QGroupBox("Live Charts")
        g.setObjectName("toolCard")
        v = QVBoxLayout(g)

        self.energy_plot = pg.PlotWidget()
        self.energy_plot.setBackground("#0a0f1d")
        self.energy_plot.showGrid(x=True, y=True, alpha=0.25)
        self.energy_plot.setLabel("left", "Energy proxy")
        self.energy_plot.setLabel("bottom", "Step")
        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen("#7cdbff", width=2))

        self.mag_plot = pg.PlotWidget()
        self.mag_plot.setBackground("#0a0f1d")
        self.mag_plot.showGrid(x=True, y=True, alpha=0.25)
        self.mag_plot.setLabel("left", "Mag (proxy)")
        self.mag_plot.setLabel("bottom", "Event #")
        self.mag_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#ff6b6b"), pen=pg.mkPen(None))
        self.mag_plot.addItem(self.mag_scatter)

        v.addWidget(self.energy_plot, 1)
        v.addWidget(self.mag_plot, 1)

        return g

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("lsPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Inspector
        gi = QGroupBox("Inspector")
        gi.setObjectName("toolCard")
        grid = QGridLayout(gi)

        self.info_step = QLabel("—")
        self.info_time = QLabel("—")
        self.info_margin = QLabel("—")
        self.info_event = QLabel("—")
        self.info_cell = QLabel("Click a cell to inspect")

        self.info_cell.setWordWrap(True)
        self.info_cell.setObjectName("lsHint")

        grid.addWidget(QLabel("Step"), 0, 0); grid.addWidget(self.info_step, 0, 1)
        grid.addWidget(QLabel("t"), 0, 2); grid.addWidget(self.info_time, 0, 3)
        grid.addWidget(QLabel("Mean fail"), 1, 0); grid.addWidget(self.info_margin, 1, 1)
        grid.addWidget(QLabel("Last event"), 1, 2); grid.addWidget(self.info_event, 1, 3)
        grid.addWidget(self.info_cell, 2, 0, 1, 4)

        lay.addWidget(gi)

        # Event feed
        ge = QGroupBox("Earthquake Catalog (proxy)")
        ge.setObjectName("toolCard")
        v = QVBoxLayout(ge)

        self.event_list = QListWidget()
        self.event_list.itemClicked.connect(self._on_event_clicked)

        btnrow = QHBoxLayout()
        self.save_catalog_btn = QPushButton("Export Catalog JSON")
        self.save_catalog_btn.clicked.connect(self.export_catalog)
        self.clear_events_btn = QPushButton("Clear Catalog")
        self.clear_events_btn.clicked.connect(self._clear_catalog)

        btnrow.addWidget(self.save_catalog_btn)
        btnrow.addWidget(self.clear_events_btn)

        v.addWidget(self.event_list, 1)
        v.addLayout(btnrow)

        lay.addWidget(ge, 1)

        return panel

    # ---------------- Style ----------------

    def _apply_style(self):
        self.setStyleSheet("""
            #lsHeader {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0b1220, stop:1 #111c33);
                border: 1px solid #1e2a46;
                border-radius: 18px;
            }
            #lsTitle {
                color: #eaf2ff;
                font-size: 20px;
                font-weight: 800;
            }
            #lsSubtitle {
                color: #9fb3d9;
                font-size: 12px;
            }
            #lsPanel {
                background: transparent;
            }
            QGroupBox#toolCard, QGroupBox {
                color: #dbe7ff;
                font-weight: 700;
                border: 1px solid #1e2a46;
                border-radius: 16px;
                margin-top: 12px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0b1220, stop:1 #0f1930);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QLabel { color: #dbe7ff; }
            QLabel#lsHint { color: #9fb3d9; font-weight: 400; }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                background: #0a0f1d;
                border: 1px solid #1e2a46;
                border-radius: 12px;
                padding: 6px 10px;
                color: #eaf2ff;
            }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1a2a54, stop:1 #0f1b38);
                border: 1px solid #2a3b63;
                color: #eaf2ff;
                padding: 8px 12px;
                border-radius: 14px;
                font-weight: 700;
            }
            QPushButton:hover { border: 1px solid #3d6bff; }
            QPushButton:pressed { background: #0b1220; }
            QCheckBox { color: #dbe7ff; }
            QListWidget {
                background: #0a0f1d;
                border: 1px solid #1e2a46;
                border-radius: 12px;
                padding: 8px;
                color: #eaf2ff;
            }
            QListWidget::item { padding: 8px; border-radius: 10px; }
            QListWidget::item:selected { background: #152447; }
        """)

    # ---------------- Presets ----------------

    def _apply_preset(self, name: str):
        # stop sim for safe parameter set
        was_running = self.running
        if was_running:
            self.toggle_run()

        # Defaults
        self.E_spin.setValue(60.0)
        self.nu_spin.setValue(0.25)
        self.coh_spin.setValue(25.0)
        self.phi_spin.setValue(30)
        self.ten_spin.setValue(10.0)

        self.dt_spin.setValue(0.05)
        self.iters_spin.setValue(20)
        self.damp_spin.setValue(0.35)

        self.temp_on.setChecked(False)
        self.Ts_spin.setValue(273)
        self.grad_spin.setValue(20)
        self.crust_spin.setValue(35)
        self.bdt_spin.setValue(18)
        self.weak_spin.setValue(0.0040)

        # boundary defaults
        if name.startswith("Transform"):
            self.left_type.setCurrentText("transform")
            self.right_type.setCurrentText("transform")
            self.vxL.setValue(0.02); self.vyL.setValue(0.00); self.cL.setValue(0.85)
            self.vxR.setValue(-0.02); self.vyR.setValue(0.00); self.cR.setValue(0.85)
            self.top_type.setCurrentText("free")
            self.bottom_type.setCurrentText("fixed")

        elif name.startswith("Convergent"):
            self.left_type.setCurrentText("convergent")
            self.right_type.setCurrentText("convergent")
            self.vxL.setValue(0.03); self.vyL.setValue(0.00); self.cL.setValue(0.95)
            self.vxR.setValue(-0.03); self.vyR.setValue(0.00); self.cR.setValue(0.95)
            self.top_type.setCurrentText("free")
            self.bottom_type.setCurrentText("fixed")
            self.coh_spin.setValue(35.0)
            self.phi_spin.setValue(35)

        elif name.startswith("Divergent"):
            self.left_type.setCurrentText("divergent")
            self.right_type.setCurrentText("divergent")
            self.vxL.setValue(-0.02); self.vyL.setValue(0.00); self.cL.setValue(0.75)
            self.vxR.setValue(0.02); self.vyR.setValue(0.00); self.cR.setValue(0.75)
            self.top_type.setCurrentText("free")
            self.bottom_type.setCurrentText("fixed")
            self.temp_on.setChecked(True)
            self.grad_spin.setValue(30)
            self.coh_spin.setValue(18.0)

        elif name.startswith("Oblique"):
            self.left_type.setCurrentText("oblique")
            self.right_type.setCurrentText("oblique")
            self.vxL.setValue(0.02); self.vyL.setValue(0.01); self.cL.setValue(0.9)
            self.vxR.setValue(-0.02); self.vyR.setValue(-0.005); self.cR.setValue(0.9)
            self.top_type.setCurrentText("free")
            self.bottom_type.setCurrentText("fixed")
            self.temp_on.setChecked(True)

        elif name.startswith("Locked vs Creeping"):
            self.left_type.setCurrentText("transform")
            self.right_type.setCurrentText("transform")
            self.vxL.setValue(0.02); self.vyL.setValue(0.00); self.cL.setValue(1.0)
            self.vxR.setValue(-0.02); self.vyR.setValue(0.00); self.cR.setValue(0.4)
            self.top_type.setCurrentText("free")
            self.bottom_type.setCurrentText("fixed")
            self.coh_spin.setValue(30.0)
            self.phi_spin.setValue(32)

        self._apply_material_settings()
        self._apply_boundary_settings()
        self._apply_dynamics()
        self._apply_thermal_settings()

        self._refresh_all()

        if was_running:
            self.toggle_run()

    # ---------------- Apply settings ----------------

    def _apply_boundary_settings(self):
        self.sim.set_boundary("left", self.left_type.currentText(), self.vxL.value(), self.vyL.value(), self.cL.value())
        self.sim.set_boundary("right", self.right_type.currentText(), self.vxR.value(), self.vyR.value(), self.cR.value())
        self.sim.set_boundary("top", self.top_type.currentText(), 0.0, 0.0, 0.0)
        self.sim.set_boundary("bottom", self.bottom_type.currentText(), 0.0, 0.0, 1.0)
        self._refresh_view()

    def _apply_material_settings(self):
        self.sim.set_material(
            E=self.E_spin.value() * 1e9,
            nu=self.nu_spin.value(),
            cohesion=self.coh_spin.value() * 1e6,
            friction_deg=self.phi_spin.value(),
            tensile=self.ten_spin.value() * 1e6,
        )
        self._refresh_view()

    def _apply_thermal_settings(self):
        self.sim.set_thermal(
            enabled=self.temp_on.isChecked(),
            surface=self.Ts_spin.value(),
            grad_K_per_km=self.grad_spin.value(),
            crust_km=self.crust_spin.value(),
            weakening=self.weak_spin.value(),
            bdt_km=self.bdt_spin.value(),
        )
        self._refresh_view()

    def _apply_dynamics(self):
        self.sim.set_dynamics(
            dt=self.dt_spin.value(),
            relax_iters=self.iters_spin.value(),
            damping=self.damp_spin.value(),
        )
        self._refresh_view()

    # ---------------- Simulation controls ----------------

    def toggle_run(self):
        self.running = not self.running
        self.play_btn.setText("Pause" if self.running else "Play")
        if self.running:
            self.timer.start(30)  # ~33 fps
        else:
            self.timer.stop()

    def step_once(self):
        evt = self.sim.step_once()
        if evt:
            self._add_event(evt)
        self._refresh_all()

    def reset_sim(self):
        self.sim.reset()
        self.event_list.clear()
        self.last_event_flash_t = -999.0
        self._refresh_all()

    def _tick(self):
        # run multiple steps per tick for more drama if dt is small
        steps = 1
        if self.sim.dt < 0.03:
            steps = 2
        if self.sim.dt < 0.015:
            steps = 3

        evt = None
        for _ in range(steps):
            evt = self.sim.step_once() or evt

        if evt:
            self._add_event(evt)

        self._refresh_all()

    # ---------------- Events ----------------

    def _add_event(self, evt: QuakeEvent):
        self.last_event_flash_t = self.sim.t
        text = f"M≈{evt.mag_proxy:.2f}  cells={evt.cells}  slip≈{evt.mean_slip:.2f}  step={evt.step}"
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, evt)
        self.event_list.insertItem(0, item)
        # keep list manageable
        if self.event_list.count() > 200:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _clear_catalog(self):
        self.event_list.clear()
        # also clear in core
        self.sim._events.clear()
        self._refresh_plots()

    def export_catalog(self):
        events = self.sim.events()
        if not events:
            QMessageBox.information(self, "Export Catalog", "No events yet.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Export Event Catalog", "", "JSON files (*.json)")
        if not filename:
            return

        payload = []
        for e in events:
            payload.append({
                "t": e.t, "step": e.step, "cells": e.cells, "area": e.area,
                "mean_slip": e.mean_slip, "peak_slip": e.peak_slip,
                "stress_drop": e.stress_drop, "energy_proxy": e.energy_proxy,
                "mag_proxy": e.mag_proxy, "centroid": {"x": e.centroid[0], "y": e.centroid[1]},
            })

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            QMessageBox.information(self, "Export Catalog", "Saved.")
        except Exception as e:
            QMessageBox.warning(self, "Export Catalog", f"Failed:\n{e}")

    def _on_event_clicked(self, item: QListWidgetItem):
        evt: QuakeEvent = item.data(Qt.ItemDataRole.UserRole)
        self.info_event.setText(f"M≈{evt.mag_proxy:.2f} • cells={evt.cells} • mean slip≈{evt.mean_slip:.2f}")
        self.info_cell.setText(f"Event centroid (grid): x≈{evt.centroid[0]:.1f}, y≈{evt.centroid[1]:.1f}")

    # ---------------- Canvas interactions ----------------

    def _toggle_paint(self, state):
        self.paint_enabled = (state == Qt.CheckState.Checked)

    def _update_paint_mode(self, mode: str):
        self.paint_mode = mode

    def _on_canvas_click(self, ix: int, iy: int):
        self._inspect_cell(ix, iy)
        if self.paint_enabled:
            self._paint_at(ix, iy)

    def _on_canvas_drag(self, ix: int, iy: int):
        if not self.paint_chk.isChecked():
            return
        self._paint_at(ix, iy)

    def _paint_at(self, ix: int, iy: int):
        # Our view indexes are in plot coordinates; map to sim grid:
        # Because image was transposed, x corresponds to ix in sim.x, and y to iy in sim.y.
        x = int(clamp(ix, 0, self.sim.nx - 1))
        y = int(clamp(iy, 0, self.sim.ny - 1))

        r = self.brush_radius

        if self.paint_mode == "Weaken":
            self.sim.add_weakness_paint(x, y, r, strength=min(self.brush_strength, 0.95))
        elif self.paint_mode == "Strengthen":
            self.sim.add_weakness_paint(x, y, r, strength=max(self.brush_strength, 1.05))
        elif self.paint_mode == "Heal damage":
            # reduce damage locally
            y0 = max(0, y - r); y1 = min(self.sim.ny, y + r + 1)
            x0 = max(0, x - r); x1 = min(self.sim.nx, x + r + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist2 = (yy - y)**2 + (xx - x)**2
            mask = dist2 <= r*r
            self.sim.damage[y0:y1, x0:x1][mask] *= 0.85
        elif self.paint_mode == "Damage (fault seed)":
            y0 = max(0, y - r); y1 = min(self.sim.ny, y + r + 1)
            x0 = max(0, x - r); x1 = min(self.sim.nx, x + r + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist2 = (yy - y)**2 + (xx - x)**2
            mask = dist2 <= r*r
            self.sim.damage[y0:y1, x0:x1][mask] = np.clip(self.sim.damage[y0:y1, x0:x1][mask] + 0.05, 0, 1)
        elif self.paint_mode == "Reset paint":
            self.sim.weakness[:] = 1.0

        self._refresh_view()

    def _inspect_cell(self, ix: int, iy: int):
        x = int(clamp(ix, 0, self.sim.nx - 1))
        y = int(clamp(iy, 0, self.sim.ny - 1))

        sxx = float(self.sim.sxx[y, x])
        syy = float(self.sim.syy[y, x])
        sxy = float(self.sim.sxy[y, x])
        exx = float(self.sim.exx[y, x])
        eyy = float(self.sim.eyy[y, x])
        exy = float(self.sim.exy[y, x])
        dmg = float(self.sim.damage[y, x])
        pst = float(self.sim.pstrain[y, x])
        lock = float(self.sim.lock[y, x])
        temp = self.sim.temp
        if temp.ndim == 2:
            xx = max(0, min(x, temp.shape[1] - 1))
            yy = max(0, min(y, temp.shape[0] - 1))
            T = float(temp[yy, xx])
        else:
            yy = max(0, min(y, temp.shape[0] - 1))
            T = float(temp[yy])


        self.info_cell.setText(
            f"Cell (x={x}, y={y})\n"
            f"Stress: σxx={sxx/1e6:.1f}MPa  σyy={syy/1e6:.1f}MPa  τxy={sxy/1e6:.1f}MPa\n"
            f"Strain: εxx={exx:.3e}  εyy={eyy:.3e}  εxy={exy:.3e}\n"
            f"Damage={dmg:.2f}  Plastic={pst:.2f}  FailMargin={lock:.2f}\n"
            f"T={T:.0f}K  Weakness×{float(self.sim.weakness[y,x]):.2f}"
        )

    # ---------------- Visualization ----------------

    def _on_field_changed(self, name: str):
        self.field_name = name
        self._refresh_view()

    def _refresh_all(self):
        self.info_step.setText(str(self.sim.step))
        self.info_time.setText(f"{self.sim.t:.2f}")
        self.info_margin.setText(f"{float(np.mean(self.sim.lock)):.2f}")
        if self.sim.events():
            self.info_event.setText(f"#{len(self.sim.events())} (last M≈{self.sim.events()[-1].mag_proxy:.2f})")
        else:
            self.info_event.setText("—")

        self._refresh_view()
        self._refresh_plots()

    def _refresh_view(self):
        arr = self.sim.field(self.field_name)

        # Fault overlay: brighten high damage
        if self.fault_chk.isChecked() and self.field_name != "Damage (faults)":
            # subtle additive overlay
            arr = arr + (self.sim.damage * (np.nanmax(arr) * 0.25 + 1e-6)).astype(arr.dtype)

        if self.auto_levels_chk.isChecked():
            # robust auto levels
            lo = float(np.nanpercentile(arr, 2))
            hi = float(np.nanpercentile(arr, 98))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-6)
            self.view.set_image(arr, levels=(lo, hi))
        else:
            self.view.set_image(arr, levels=None)

    def _refresh_plots(self):
        # Energy proxy time series from pstrain + stress
        # (This is not physical energy; it's a satisfying monotonic "stored energy" proxy.)
        stored = float(np.mean(von_mises_2d(self.sim.sxx, self.sim.syy, self.sim.sxy))) * (1 + float(np.mean(self.sim.pstrain))*0.05)
        if not hasattr(self, "_energy_hist"):
            self._energy_hist = []
        self._energy_hist.append((self.sim.step, stored))
        if len(self._energy_hist) > 500:
            self._energy_hist = self._energy_hist[-500:]
        xs = np.array([p[0] for p in self._energy_hist], dtype=float)
        ys = np.array([p[1] for p in self._energy_hist], dtype=float)
        self.energy_curve.setData(xs, ys)

        evs = self.sim.events()
        if evs:
            mags = np.array([e.mag_proxy for e in evs[-300:]], dtype=float)
            idx = np.arange(len(mags), dtype=float)
            spots = [{"pos": (idx[i], mags[i])} for i in range(len(mags))]
            self.mag_scatter.setData(spots)
        else:
            self.mag_scatter.setData([])

    # ---------------- Export snapshot ----------------

    def export_snapshot(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Snapshot PNG", "", "PNG files (*.png)")
        if not filename:
            return
        try:
            exporter = pg.exporters.ImageExporter(self.view.plot)
            exporter.export(filename)
            QMessageBox.information(self, "Export Snapshot", "Saved.")
        except Exception as e:
            QMessageBox.warning(self, "Export Snapshot", f"Failed:\n{e}")

    # ---------------- Extra buttons ----------------

    def _add_heterogeneity_noise(self):
        noise = (np.random.randn(self.sim.ny, self.sim.nx).astype(np.float32) * 0.06)
        self.sim.weakness *= np.clip(1.0 + noise, 0.75, 1.25)
        self.sim.weakness = np.clip(self.sim.weakness, 0.5, 1.7).astype(np.float32)
        self._refresh_view()

    def _clear_damage(self):
        self.sim.damage[:] = 0.0
        self.sim.pstrain[:] = 0.0
        self._refresh_view()


TOOL_META = {
    "name": "Lithosphere Stress & Deformation Simulator (2D)",
    "description": "Real-time 2D lithosphere simulator with plate driving, stress/strain tensors, fault formation, and earthquake-like rupture events",
    "category": "Earth Science",
    "version": "1.0.0",
    "author": "ScienceHub Team",
    "features": [
        "Plate velocity boundary driving (convergent/divergent/transform/oblique)",
        "Elastic stress tensor fields and derived invariants",
        "Strain accumulation + plastic strain tracking",
        "Fault formation via damage/weakening",
        "Earthquake-like rupture cascades + event catalog + magnitude proxy",
        "Temperature gradient weakening + brittle-ductile transition (optional)",
        "Painter tools for heterogeneity, weakness, and fault seeding",
        "Live plots (energy proxy, magnitude series)",
        "Presets for classic tectonic regimes",
        "Snapshot export and catalog export"
    ],
    "educational_value": "Build intuition for tectonic loading, stress localization, fault formation, and seismic release cycles in a controllable 2D sandbox",
    "keywords": ["lithosphere", "tectonics", "stress", "strain", "faults", "earthquakes", "simulation", "geophysics"]
}


def create_tool(parent=None):
    return LithosphereSimulatorTool(parent)
