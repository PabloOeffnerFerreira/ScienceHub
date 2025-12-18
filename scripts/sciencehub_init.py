#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

# Add/remove paths freely. Trailing "/" means "directory".
TREE = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    ".gitignore",
    ".editorconfig",
    ".env.example",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",

    "scripts/",
    "scripts/bootstrap.py",
    "scripts/dev_run.py",
    "scripts/build_release.py",
    "scripts/clean_cache.py",
    "scripts/migrate_data.py",
    "scripts/generate_docs.py",

    "sciencehub/",
    "sciencehub/__init__.py",

    "sciencehub/app/",
    "sciencehub/app/__init__.py",
    "sciencehub/app/entrypoint.py",
    "sciencehub/app/app_context.py",
    "sciencehub/app/app_state.py",
    "sciencehub/app/app_events.py",
    "sciencehub/app/app_registry.py",
    "sciencehub/app/startup.py",
    "sciencehub/app/shutdown.py",
    "sciencehub/app/exception_handler.py",
    "sciencehub/app/versioning.py",

    "sciencehub/kernel/",
    "sciencehub/kernel/__init__.py",

    "sciencehub/kernel/symbols/",
    "sciencehub/kernel/symbols/__init__.py",
    "sciencehub/kernel/symbols/symbol.py",
    "sciencehub/kernel/symbols/symbol_set.py",
    "sciencehub/kernel/symbols/dimensions.py",
    "sciencehub/kernel/symbols/units.py",

    "sciencehub/kernel/expressions/",
    "sciencehub/kernel/expressions/__init__.py",
    "sciencehub/kernel/expressions/expression.py",
    "sciencehub/kernel/expressions/parser.py",
    "sciencehub/kernel/expressions/evaluator.py",
    "sciencehub/kernel/expressions/simplifier.py",
    "sciencehub/kernel/expressions/latex_renderer.py",

    "sciencehub/kernel/solvers/",
    "sciencehub/kernel/solvers/__init__.py",
    "sciencehub/kernel/solvers/algebraic_solver.py",
    "sciencehub/kernel/solvers/numeric_solver.py",
    "sciencehub/kernel/solvers/differential_solver.py",
    "sciencehub/kernel/solvers/stochastic_solver.py",
    "sciencehub/kernel/solvers/optimization_solver.py",

    "sciencehub/kernel/constants/",
    "sciencehub/kernel/constants/__init__.py",
    "sciencehub/kernel/constants/physical_constants.py",
    "sciencehub/kernel/constants/chemical_constants.py",
    "sciencehub/kernel/constants/astronomical_constants.py",
    "sciencehub/kernel/constants/biological_constants.py",

    "sciencehub/kernel/validation/",
    "sciencehub/kernel/validation/__init__.py",
    "sciencehub/kernel/validation/dimensional_analysis.py",
    "sciencehub/kernel/validation/bounds_checker.py",
    "sciencehub/kernel/validation/uncertainty_propagation.py",

    "sciencehub/domains/",
    "sciencehub/domains/__init__.py",

    "sciencehub/domains/mathematics/",
    "sciencehub/domains/mathematics/__init__.py",
    "sciencehub/domains/mathematics/core.py",
    "sciencehub/domains/mathematics/geometry.py",
    "sciencehub/domains/mathematics/calculus.py",
    "sciencehub/domains/mathematics/statistics.py",

    "sciencehub/domains/physics/",
    "sciencehub/domains/physics/__init__.py",
    "sciencehub/domains/physics/mechanics.py",
    "sciencehub/domains/physics/thermodynamics.py",
    "sciencehub/domains/physics/fluid_dynamics.py",
    "sciencehub/domains/physics/statistical_mechanics.py",
    "sciencehub/domains/physics/relativity.py",
    "sciencehub/domains/physics/quantum_basics.py",
    "sciencehub/domains/physics/nuclear_physics.py",

    "sciencehub/domains/chemistry/",
    "sciencehub/domains/chemistry/__init__.py",
    "sciencehub/domains/chemistry/stoichiometry.py",
    "sciencehub/domains/chemistry/equilibrium.py",
    "sciencehub/domains/chemistry/kinetics.py",
    "sciencehub/domains/chemistry/thermochemistry.py",
    "sciencehub/domains/chemistry/electrochemistry.py",

    "sciencehub/domains/biology/",
    "sciencehub/domains/biology/__init__.py",
    "sciencehub/domains/biology/genetics.py",
    "sciencehub/domains/biology/enzymology.py",
    "sciencehub/domains/biology/ecology.py",
    "sciencehub/domains/biology/microbiology.py",

    "sciencehub/domains/astronomy/",
    "sciencehub/domains/astronomy/__init__.py",
    "sciencehub/domains/astronomy/orbits.py",
    "sciencehub/domains/astronomy/stars.py",
    "sciencehub/domains/astronomy/cosmology.py",

    "sciencehub/domains/geology/",
    "sciencehub/domains/geology/__init__.py",
    "sciencehub/domains/geology/minerals.py",
    "sciencehub/domains/geology/tectonics.py",

    "sciencehub/domains/materials/",
    "sciencehub/domains/materials/__init__.py",
    "sciencehub/domains/materials/crystal_structures.py",
    "sciencehub/domains/materials/stress_strain.py",

    "sciencehub/domains/electricity_magnetism/",
    "sciencehub/domains/electricity_magnetism/__init__.py",
    "sciencehub/domains/electricity_magnetism/circuits.py",
    "sciencehub/domains/electricity_magnetism/fields.py",

    "sciencehub/domains/waves_optics/",
    "sciencehub/domains/waves_optics/__init__.py",
    "sciencehub/domains/waves_optics/waves.py",
    "sciencehub/domains/waves_optics/optics.py",

    "sciencehub/domains/interdisciplinary/",
    "sciencehub/domains/interdisciplinary/__init__.py",
    "sciencehub/domains/interdisciplinary/unit_conversions.py",
    "sciencehub/domains/interdisciplinary/estimation.py",

    "sciencehub/engines/",
    "sciencehub/engines/__init__.py",

    "sciencehub/engines/calculation/",
    "sciencehub/engines/calculation/__init__.py",
    "sciencehub/engines/calculation/calculator_engine.py",
    "sciencehub/engines/calculation/batch_calculator.py",
    "sciencehub/engines/calculation/symbolic_calculator.py",

    "sciencehub/engines/simulation/",
    "sciencehub/engines/simulation/__init__.py",
    "sciencehub/engines/simulation/simulation_engine.py",
    "sciencehub/engines/simulation/time_evolution.py",
    "sciencehub/engines/simulation/monte_carlo.py",
    "sciencehub/engines/simulation/agent_based.py",
    "sciencehub/engines/simulation/cellular_automata.py",

    "sciencehub/engines/animation/",
    "sciencehub/engines/animation/__init__.py",
    "sciencehub/engines/animation/animation_engine.py",
    "sciencehub/engines/animation/keyframe_generator.py",
    "sciencehub/engines/animation/interpolation.py",
    "sciencehub/engines/animation/export_video.py",

    "sciencehub/engines/optimization/",
    "sciencehub/engines/optimization/__init__.py",
    "sciencehub/engines/optimization/parameter_sweep.py",
    "sciencehub/engines/optimization/genetic_algorithm.py",
    "sciencehub/engines/optimization/gradient_methods.py",

    "sciencehub/engines/execution/",
    "sciencehub/engines/execution/__init__.py",
    "sciencehub/engines/execution/task_queue.py",
    "sciencehub/engines/execution/scheduler.py",
    "sciencehub/engines/execution/parallel_executor.py",

    "sciencehub/visualization/",
    "sciencehub/visualization/__init__.py",

    "sciencehub/visualization/plots/",
    "sciencehub/visualization/plots/__init__.py",
    "sciencehub/visualization/plots/plot_base.py",
    "sciencehub/visualization/plots/line_plot.py",
    "sciencehub/visualization/plots/scatter_plot.py",
    "sciencehub/visualization/plots/heatmap.py",
    "sciencehub/visualization/plots/phase_space.py",

    "sciencehub/visualization/diagrams/",
    "sciencehub/visualization/diagrams/__init__.py",
    "sciencehub/visualization/diagrams/free_body_diagram.py",
    "sciencehub/visualization/diagrams/circuit_diagram.py",
    "sciencehub/visualization/diagrams/molecular_diagram.py",
    "sciencehub/visualization/diagrams/optical_setup.py",

    "sciencehub/visualization/animations/",
    "sciencehub/visualization/animations/__init__.py",
    "sciencehub/visualization/animations/motion_animation.py",
    "sciencehub/visualization/animations/wave_animation.py",
    "sciencehub/visualization/animations/field_animation.py",
    "sciencehub/visualization/animations/reaction_animation.py",

    "sciencehub/visualization/exporters/",
    "sciencehub/visualization/exporters/__init__.py",
    "sciencehub/visualization/exporters/png_exporter.py",
    "sciencehub/visualization/exporters/svg_exporter.py",
    "sciencehub/visualization/exporters/gif_exporter.py",
    "sciencehub/visualization/exporters/mp4_exporter.py",

    "sciencehub/visualization/themes/",
    "sciencehub/visualization/themes/__init__.py",
    "sciencehub/visualization/themes/dark_theme.py",
    "sciencehub/visualization/themes/light_theme.py",
    "sciencehub/visualization/themes/publication_theme.py",

    "sciencehub/orchestration/",
    "sciencehub/orchestration/__init__.py",
    "sciencehub/orchestration/tool_descriptor.py",
    "sciencehub/orchestration/tool_builder.py",
    "sciencehub/orchestration/workflow.py",
    "sciencehub/orchestration/pipeline.py",
    "sciencehub/orchestration/scenario.py",
    "sciencehub/orchestration/execution_context.py",

    "sciencehub/ai/",
    "sciencehub/ai/__init__.py",

    "sciencehub/ai/assistant/",
    "sciencehub/ai/assistant/__init__.py",
    "sciencehub/ai/assistant/assistant_core.py",
    "sciencehub/ai/assistant/reasoning.py",
    "sciencehub/ai/assistant/explanation_generator.py",
    "sciencehub/ai/assistant/prompt_templates.py",

    "sciencehub/ai/retrieval/",
    "sciencehub/ai/retrieval/__init__.py",
    "sciencehub/ai/retrieval/knowledge_index.py",
    "sciencehub/ai/retrieval/embedding_store.py",
    "sciencehub/ai/retrieval/semantic_search.py",

    "sciencehub/ai/planning/",
    "sciencehub/ai/planning/__init__.py",
    "sciencehub/ai/planning/task_planner.py",
    "sciencehub/ai/planning/tool_selector.py",

    "sciencehub/ai/memory/",
    "sciencehub/ai/memory/__init__.py",
    "sciencehub/ai/memory/conversation_memory.py",
    "sciencehub/ai/memory/user_profile.py",

    "sciencehub/data/",
    "sciencehub/data/README.md",
    "sciencehub/data/constants/.keep",
    "sciencehub/data/tables/.keep",
    "sciencehub/data/datasets/.keep",
    "sciencehub/data/reference_models/.keep",
    "sciencehub/data/unit_definitions/.keep",
    "sciencehub/data/metadata/.keep",

    "sciencehub/persistence/",
    "sciencehub/persistence/__init__.py",
    "sciencehub/persistence/database.py",
    "sciencehub/persistence/schema.py",
    "sciencehub/persistence/favorites.py",
    "sciencehub/persistence/cache.py",
    "sciencehub/persistence/migrations/.keep",
    "sciencehub/persistence/history/.keep",

    "sciencehub/plugins/",
    "sciencehub/plugins/__init__.py",
    "sciencehub/plugins/plugin_base.py",
    "sciencehub/plugins/plugin_loader.py",
    "sciencehub/plugins/official/.keep",
    "sciencehub/plugins/community/.keep",
    "sciencehub/plugins/experimental/.keep",

    "sciencehub/ui/",
    "sciencehub/ui/__init__.py",
    "sciencehub/ui/state.py",
    "sciencehub/ui/events.py",
    "sciencehub/ui/layout/.keep",
    "sciencehub/ui/components/.keep",
    "sciencehub/ui/panels/.keep",
    "sciencehub/ui/themes/.keep",
    "sciencehub/ui/bindings/.keep",

    "sciencehub/api/",
    "sciencehub/api/__init__.py",
    "sciencehub/api/rest/.keep",
    "sciencehub/api/websocket/.keep",
    "sciencehub/api/cli/.keep",
    "sciencehub/api/schemas/.keep",

    "sciencehub/experiments/",
    "sciencehub/experiments/__init__.py",
    "sciencehub/experiments/sandbox.py",
    "sciencehub/experiments/performance_tests.py",
    "sciencehub/experiments/visual_demos.py",
    "sciencehub/experiments/abandoned_ideas/.keep",

    "sciencehub/diagnostics/",
    "sciencehub/diagnostics/__init__.py",
    "sciencehub/diagnostics/logger.py",
    "sciencehub/diagnostics/profiler.py",
    "sciencehub/diagnostics/telemetry.py",
    "sciencehub/diagnostics/error_reports.py",
    "sciencehub/diagnostics/health_checks.py",

    "sciencehub/localization/",
    "sciencehub/localization/README.md",
    "sciencehub/localization/en/.keep",
    "sciencehub/localization/de/.keep",
    "sciencehub/localization/pt_BR/.keep",
    "sciencehub/localization/fr/.keep",

    "sciencehub/config/",
    "sciencehub/config/__init__.py",
    "sciencehub/config/default.yaml",
    "sciencehub/config/user.yaml",
    "sciencehub/config/runtime.yaml",
    "sciencehub/config/feature_flags.yaml",

    "sciencehub/utils/",
    "sciencehub/utils/__init__.py",
    "sciencehub/utils/math_helpers.py",
    "sciencehub/utils/formatting.py",
    "sciencehub/utils/time_utils.py",
    "sciencehub/utils/file_utils.py",
    "sciencehub/utils/decorators.py",
    "sciencehub/utils/type_guards.py",

    "sciencehub/tests/",
    "sciencehub/tests/__init__.py",
    "sciencehub/tests/unit/.keep",
    "sciencehub/tests/integration/.keep",
    "sciencehub/tests/regression/.keep",
    "sciencehub/tests/performance/.keep",
    "sciencehub/tests/fixtures/.keep",
]

def create_tree(root: Path, *, force: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for item in TREE:
        is_dir = item.endswith("/")
        path = root / item.rstrip("/")

        if is_dir:
            path.mkdir(parents=True, exist_ok=True)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not force:
            continue

        # Write minimal placeholders for a few common files
        if path.name == "__init__.py":
            path.write_text('"""ScienceHub package."""\n', encoding="utf-8")
        elif path.name.endswith(".keep"):
            path.write_text("", encoding="utf-8")
        elif path.name in {"README.md"} and path.parent.name == "sciencehub":
            path.write_text("# ScienceHub\n\nGenerated scaffold.\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

def main() -> int:
    p = argparse.ArgumentParser(prog="sciencehub-init", description="Create ScienceHub scaffold.")
    p.add_argument("target", nargs="?", default=".", help="Target directory (default: current directory)")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = p.parse_args()

    create_tree(Path(args.target).resolve(), force=args.force)
    print(f"ScienceHub scaffold created at: {Path(args.target).resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
