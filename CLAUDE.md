# CLAUDE.md

## Project Conventions

### Imports

- **Always use absolute imports.** Never use relative imports (e.g., `from .foo import Bar`).
  Use the full package path instead: `from eigrpo.sampling.base_selector import BaseTrajectorySelector`.

### `__init__.py` files

- **No barrel exports.** Keep `__init__.py` files empty (or with only a module docstring).
  Do not re-export symbols from submodules. Consumers import directly from the defining module.

- **No lazy imports AND no imports wrapped in a try except.** Never use `except ImportError` and always import at the top of the file.

### Module design

- **One file, one purpose.** Each module should have a single, clear responsibility.
  Do not mix unrelated concerns (e.g., data preparation, trainer subclass, and CLI
  entry-point) in the same file. If a class or function can stand alone, give it its
  own module.

- **DRY and SOLID.** Do not duplicate logic across files. Extract shared behaviour
  into reusable modules. Prefer composition over inheritance where possible.
  Scripts in `experiments/` should be thin entry-points that import and wire up
  library code from `src/eigrpo/`.

- **Limit memory usage with Flash Attention** Always use MAX_JOBS=4 so that we don't destroy our memory