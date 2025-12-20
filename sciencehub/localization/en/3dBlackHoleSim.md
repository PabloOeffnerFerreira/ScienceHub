# 3D Schwarzschild Black Hole Simulator

An interactive 3D visualization tool for exploring Schwarzschild black holes, spacetime curvature, gravitational lensing, and general relativistic effects in three dimensions.

## Overview

This 3D simulator extends the 2D Schwarzschild black hole simulator with immersive three-dimensional visualizations that help students and researchers better understand:

- **Spacetime Geometry**: Visual representation of curved spacetime around massive objects
- **Event Horizons**: 3D rendering of the black sphere where nothing can escape
- **Photon Spheres**: Unstable orbits where light can circle the black hole
- **Gravitational Lensing**: Light ray trajectories showing bending due to spacetime curvature
- **ISCO Regions**: Innermost stable circular orbits for matter
- **Observer Perspectives**: Different viewpoints and their effects on observations

## Features

### 3D Visualizations
- **Event Horizon**: Black sphere representing the point of no return
- **Photon Sphere**: Red wireframe sphere at 1.5 Rs showing unstable photon orbits
- **ISCO Sphere**: Blue wireframe sphere at 6 Rs showing stable matter orbits
- **Light Ray Trajectories**: Cyan curves showing how light bends around the black hole
- **Coordinate Grid**: Gray grid lines representing spacetime coordinates
- **Observer Position**: Yellow sphere showing current observer location

### Interactive Controls
- **Mass Control**: Adjust black hole mass from stellar (10 M☉) to supermassive (10^9 M☉)
- **Observer Distance**: Position observer at different distances from the black hole
- **Camera Controls**: Interactive elevation and azimuth sliders
- **Preset Views**: Quick access to XY plane, XZ plane, and 3D perspective views
- **Visibility Toggles**: Show/hide different visualization elements

### Advanced Features
- **Real-time Updates**: Visualization updates instantly as parameters change
- **Multiple Units**: Mass in solar masses, Earth masses, Jupiter masses, or kg
- **Distance Units**: Measurements in Rs, km, AU, or light years
- **Animation Support**: Camera rotation and light ray animations (framework ready)
- **Educational Info**: Detailed explanations of what each element represents

## Physics Implementation

### Schwarzschild Metric
The simulator visualizes the Schwarzschild solution to Einstein's field equations:

```
ds² = -(1 - 2M/r) dt² + (1 - 2M/r)⁻¹ dr² + r² dΩ²
```

Where:
- `M` is the black hole mass (in geometric units where G = c = 1)
- `r` is the radial coordinate
- `t` is time
- `dΩ²` is the solid angle element

### Key Radii
- **Event Horizon**: `r = 2M` - The boundary beyond which nothing can escape
- **Photon Sphere**: `r = 3M` - Unstable circular orbits for light
- **ISCO**: `r = 6M` - Innermost stable circular orbit for matter

### Light Bending
Light rays follow null geodesics in curved spacetime. The simulator shows approximate trajectories using the impact parameter `b`:

```
φ(r) = ∫ dr / √[(r²/b²)(r²/b² - 1 + 2M r/b²)⁻¹]
```

## Usage

### Basic Operation
1. **Set Black Hole Mass**: Choose mass using the dropdown units (M☉ for stellar black holes)
2. **Position Observer**: Set distance from black hole center
3. **Adjust Camera**: Use elevation/azimuth sliders or preset view buttons
4. **Toggle Elements**: Show/hide different visualization components
5. **Explore**: Rotate view and change parameters to understand relativistic effects

### Advanced Exploration
- **Compare Sizes**: Switch between stellar and supermassive black holes
- **Light Deflection**: Observe how light rays bend more severely near the photon sphere
- **Horizon Crossing**: Move observer inside/outside the event horizon
- **Multiple Perspectives**: Use different preset views to understand 3D geometry

### Educational Applications
- **Gravitational Lensing**: See how light bends around massive objects
- **Black Hole Shadows**: Understand the silhouette created by photon spheres
- **Time Dilation**: Visualize how time flows differently at different radii
- **Spacetime Curvature**: See the "dent" massive objects create in spacetime

## Technical Details

### Architecture
- **BlackHole3DCanvas**: Custom matplotlib 3D canvas with PyQt6 integration
- **Schwarzschild3DSimulator**: Main widget inheriting from ScienceHubTool
- **Real-time Rendering**: matplotlib's 3D engine for smooth interactions
- **Modular Design**: Separate control panels for different parameter types

### Dependencies
- **PyQt6**: GUI framework
- **matplotlib**: 3D plotting and visualization
- **numpy**: Numerical computations
- **ScienceHub Framework**: Base tool infrastructure

### Performance
- Optimized for real-time interaction
- Efficient 3D rendering with matplotlib
- Minimal computational overhead
- Smooth parameter updates

## Examples

### Example 1: Stellar Black Hole
- Mass: 10 M☉
- Observer Distance: 50 Rs
- View: 3D perspective
- Shows: Classic black hole with clear event horizon and photon sphere

### Example 2: Supermassive Black Hole
- Mass: 4.3 × 10⁶ M☉ (Sagittarius A*)
- Observer Distance: 100 Rs
- View: XY plane
- Shows: How supermassive black holes appear to orbiting stars

### Example 3: Extreme Light Bending
- Mass: 10 M☉
- Observer Distance: 10 Rs
- View: Equatorial plane
- Shows: Severe gravitational lensing near the photon sphere

## Future Enhancements

### Kerr Black Holes
- Rotating black holes with ergospheres
- Frame-dragging effects
- Penrose process visualization

### Advanced Visualizations
- **Geodesic Motion**: Particle trajectories in curved spacetime
- **Embedding Diagrams**: 2D representations of 3D curvature
- **Light Cones**: Visualization of causality in curved spacetime
- **Hawking Radiation**: Quantum effects at the horizon

### Interactive Features
- **VR Support**: Virtual reality exploration (future)
- **Multi-observer**: Compare views from different positions
- **Time Evolution**: Dynamic simulations of orbiting objects
- **Gravitational Waves**: Ripples in spacetime geometry

## Integration

The 3D simulator is designed to work alongside the 2D Schwarzschild simulator:

```python
from sciencehub.data.functions.schwarzschild_black_hole_simulator import SchwarzschildBlackHoleSimulator
from sciencehub.data.functions.schwarzschild_3d_simulator import Schwarzschild3DSimulator

# Create both simulators
simulator_2d = SchwarzschildBlackHoleSimulator()
simulator_3d = Schwarzschild3DSimulator()

# They share the same physics engine and can be synchronized
```

## Educational Impact

This 3D simulator addresses common misconceptions about black holes:

1. **Black holes are NOT cosmic vacuum cleaners** - Shows how tidal forces vary with distance
2. **Event horizons are NOT solid surfaces** - Visualizes them as coordinate boundaries
3. **Light DOES bend around black holes** - Demonstrates gravitational lensing
4. **Spacetime curvature is REAL** - Makes abstract concepts tangible

## References

1. **Schwarzschild, K.** (1916). "Über das Gravitationsfeld eines Massenpunktes nach der Einsteinschen Theorie"
2. **Einstein, A.** (1916). "Näherungsweise Integration der Feldgleichungen der Gravitation"
3. **Misner, C.W., Thorne, K.S., Wheeler, J.A.** (1973). *Gravitation*
4. **Hartle, J.B.** (2003). *Gravity: An Introduction to Einstein's General Relativity*

## License

Part of the ScienceHub educational platform. See main project license for details.

## Contributing

Contributions welcome! Areas for improvement:
- Enhanced light ray calculations
- Kerr metric support
- Animation sequences
- VR/AR integration
- Performance optimizations
- Additional visualization modes