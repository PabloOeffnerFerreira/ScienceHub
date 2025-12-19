# Schwarzschild Black Hole Simulator

A comprehensive educational tool for exploring Schwarzschild black holes, general relativity effects, and gravitational physics. This simulator provides detailed calculations and visualizations of black hole properties including event horizons, gravitational redshift, time dilation, and orbital mechanics.

## Features

### Core Physics Calculations
- **Schwarzschild Radius**: Event horizon calculation for any black hole mass
- **Photon Sphere**: Unstable photon orbit radius (1.5 Rs)
- **ISCO (Innermost Stable Circular Orbit)**: Minimum stable orbit radius (6 Rs)
- **Gravitational Redshift**: Frequency shift due to gravitational potential
- **Time Dilation**: Relative time flow near black holes
- **Escape Velocity**: Speed required to escape gravitational well
- **Hawking Temperature**: Quantum evaporation temperature
- **Tidal Forces**: Differential gravitational acceleration

### Advanced Features
- **Multiple Display Modes**:
  - Basic Results: Key properties and current effects
  - Detailed Physics: Comprehensive analysis with all calculations
  - Visualization: Focus on plotted data interpretation

- **Interactive Visualizations**:
  - Gravitational redshift vs. distance
  - Effective potential for orbital motion
  - Light bending/deflection angles
  - Real-time plotting with pyqtgraph

- **Preset Black Hole Scenarios**:
  - Stellar black holes (10 M☉)
  - Supermassive black holes (Sagittarius A*, M87*, TON 618)
  - Micro black holes
  - Primordial black holes

- **Flexible Input System**:
  - Mass input in multiple units (M☉, M⊕, MJ, kg)
  - Distance in various scales (km, Rs, AU, ly, pc)
  - Observer location presets (distant, near, ISCO, photon sphere, horizon)
  - Object type selection (stationary, orbiting, light rays, etc.)

- **Live Update Mode**: Automatic recalculation when inputs change
- **Configuration Save/Load**: Persist settings between sessions
- **Plot Export**: Save visualizations as PNG or SVG files

## Educational Value

This simulator helps students and researchers understand:

1. **General Relativity Concepts**:
   - Spacetime curvature around massive objects
   - Gravitational time dilation
   - Light bending and gravitational lensing

2. **Black Hole Physics**:
   - Event horizons and no-return boundaries
   - Photon spheres and light trapping
   - Ergospheres and rotational effects (future extension)

3. **Astrophysical Applications**:
   - Supermassive black holes in galaxy centers
   - Stellar mass black holes from supernovae
   - Gravitational wave sources

4. **Quantum Gravity**:
   - Hawking radiation and black hole evaporation
   - Information paradox implications

## Usage

### Basic Operation
1. **Set Black Hole Mass**: Choose mass and units (solar masses for stellar BHs, billions for SMBHs)
2. **Set Observer Distance**: Distance from black hole center in various units
3. **Select Object Type**: Stationary observer, orbiting object, light ray, etc.
4. **Choose Calculation Type**: Redshift/dilation, orbital mechanics, light bending, etc.
5. **Click Calculate**: View results and visualizations

### Advanced Features
- **Live Update**: Enable automatic recalculation for interactive exploration
- **Display Modes**: Switch between basic, detailed, and visualization modes
- **Preset Scenarios**: Quickly load famous black holes for comparison
- **Plot Options**: Toggle different visualization curves
- **Export**: Save plots and configurations

### Key Calculations

#### Schwarzschild Radius
```
Rs = 2GM/c²
```
The fundamental scale of a black hole, where G is gravitational constant, M is mass, and c is speed of light.

#### Gravitational Redshift
```
z = 1/√(1 - Rs/r) - 1
```
Frequency shift experienced by light escaping from gravitational potential.

#### Time Dilation
```
τ = t √(1 - Rs/r)
```
Proper time vs coordinate time for stationary observers.

#### Photon Sphere
```
r_photon = 1.5 Rs
```
Radius where light can orbit in unstable circular paths.

#### ISCO (Innermost Stable Circular Orbit)
```
r_isco = 6 Rs
```
Smallest radius for stable circular orbits around Schwarzschild black holes.

## Technical Implementation

### Architecture
- **SchwarzschildCalculator**: Core physics computation engine
- **PhysicsConstants**: Fundamental physical constants and unit conversions
- **SchwarzschildBlackHoleSimulator**: PyQt6-based GUI widget
- **ScienceHubTool**: Base class for embeddable tools

### Dependencies
- PyQt6: GUI framework
- pyqtgraph: Scientific plotting
- numpy: Numerical computations
- math: Standard mathematical functions

### File Structure
```
sciencehub/domains/astronomy/
├── schwarzschild_black_hole_simulator.py  # Main simulator
└── README.md                              # This documentation
```

## Examples

### Example 1: Stellar Black Hole
- Mass: 10 M☉
- Distance: 100 km
- Observer: Near black hole
- Shows extreme redshift and time dilation effects

### Example 2: Sagittarius A*
- Mass: 4.3 × 10⁶ M☉
- Distance: 10 Rs
- Observer: ISCO orbit
- Demonstrates supermassive black hole effects

### Example 3: Light Bending
- Mass: 10 M☉
- Calculation: Light Bending
- Shows deflection angles for different impact parameters

## Future Enhancements

### Kerr Black Holes
- Rotating black holes with ergospheres
- Frame-dragging effects
- Higher-order multipole moments

### Reissner-Nordström
- Charged black holes
- Cauchy horizons
- Electromagnetic effects

### Numerical Relativity
- Dynamic spacetimes
- Gravitational wave generation
- Black hole mergers

### Advanced Visualizations
- 3D spacetime diagrams
- Embedding diagrams
- Light cone visualizations

## References

1. **Schwarzschild, K.** (1916). "Über das Gravitationsfeld eines Massenpunktes nach der Einsteinschen Theorie"
2. **Hawking, S.W.** (1975). "Particle creation by black holes"
3. **Misner, C.W., Thorne, K.S., Wheeler, J.A.** (1973). *Gravitation*
4. **Carroll, S.M.** (2004). *Spacetime and Geometry: An Introduction to General Relativity*

## License

This tool is part of the ScienceHub educational platform. See main project license for details.

## Contributing

Contributions welcome! Areas for improvement:
- Additional black hole types (Kerr, Reissner-Nordström)
- More visualization options
- Performance optimizations
- Educational content expansion
- Unit test coverage