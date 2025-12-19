"""
ScienceHub Black Hole Simulators - Usage Examples
Demonstrates how to use the 2D, 3D, and integrated black hole simulators.
"""

# Example 1: Using the 2D Schwarzschild Simulator
print("=== 2D Schwarzschild Black Hole Simulator ===")
from sciencehub.domains.astronomy.schwarzschild_black_hole_simulator import (
    SchwarzschildBlackHoleSimulator,
    TOOL_META
)

print(f"Tool: {TOOL_META['name']}")
print(f"Features: {len(TOOL_META['features'])}")
print("Description:", TOOL_META['description'][:100] + "...")

# Example 2: Using the 3D Simulator
print("\n=== 3D Schwarzschild Black Hole Simulator ===")
from sciencehub.domains.astronomy.schwarzschild_3d_simulator import (
    Schwarzschild3DSimulator,
    TOOL_META_3D
)

print(f"Tool: {TOOL_META_3D['name']}")
print(f"Features: {len(TOOL_META_3D['features'])}")
print("Description:", TOOL_META_3D['description'][:100] + "...")

# Example 3: Using the Integrated Simulator
print("\n=== Integrated Black Hole Simulator ===")
from sciencehub.domains.astronomy.integrated_black_hole_simulator import (
    IntegratedBlackHoleSimulator,
    launch_integrated_simulator
)

print("Integrated simulator combines 2D and 3D views with synchronized parameters")
print("Features:")
print("• Real-time parameter synchronization")
print("• Side-by-side 2D/3D comparison")
print("• Comprehensive black hole physics exploration")
print("• Interactive camera controls for 3D view")

# Example usage in ScienceHub
print("\n=== Usage in ScienceHub ===")

def create_black_hole_tools():
    """Create all black hole simulation tools."""
    tools = {}

    # 2D Calculator
    tools['2d_calculator'] = {
        'name': 'Schwarzschild Calculator',
        'tool': SchwarzschildBlackHoleSimulator(),
        'description': '2D analysis with calculations and 2D plots'
    }

    # 3D Visualizer
    tools['3d_visualizer'] = {
        'name': '3D Black Hole Visualizer',
        'tool': Schwarzschild3DSimulator(),
        'description': 'Interactive 3D spacetime visualization'
    }

    # Integrated Experience
    tools['integrated'] = {
        'name': 'Integrated Simulator',
        'tool': IntegratedBlackHoleSimulator(),
        'description': 'Combined 2D/3D experience with synchronization'
    }

    return tools

# Educational scenarios
print("\n=== Educational Scenarios ===")

scenarios = {
    "Stellar Black Hole": {
        "mass": 10.0,  # M☉
        "description": "Typical stellar mass black hole from supernova"
    },
    "Sagittarius A*": {
        "mass": 4.3e6,  # M☉
        "description": "Supermassive black hole at galactic center"
    },
    "Primordial Black Hole": {
        "mass": 1e15,  # kg
        "description": "Hypothetical tiny black holes from early universe"
    }
}

for name, data in scenarios.items():
    print(f"{name}: {data['mass']} - {data['description']}")

print("\n=== Launch Commands ===")

print("# Launch 2D simulator")
print("simulator_2d = SchwarzschildBlackHoleSimulator()")
print("simulator_2d.show()")

print("\n# Launch 3D simulator")
print("simulator_3d = Schwarzschild3DSimulator()")
print("simulator_3d.show()")

print("\n# Launch integrated simulator")
print("integrated = IntegratedBlackHoleSimulator()")
print("integrated.show()")

print("\n# Or use the launcher function")
print("launch_integrated_simulator()")

print("\n=== Physics Highlights ===")

physics_features = [
    "Schwarzschild radius calculations",
    "Gravitational redshift analysis",
    "Time dilation effects",
    "Event horizon visualization",
    "Photon sphere rendering",
    "ISCO (Innermost Stable Circular Orbit)",
    "Light bending trajectories",
    "Hawking radiation temperature",
    "Tidal force calculations",
    "Effective potential plots",
    "3D spacetime curvature",
    "Interactive parameter exploration"
]

print("Comprehensive physics coverage:")
for i, feature in enumerate(physics_features, 1):
    print(f"{i:2d}. {feature}")

print("\n=== Ready for Educational Use ===")
print("All simulators are now available for exploring black hole physics!")
print("From basic calculations to immersive 3D visualizations.")