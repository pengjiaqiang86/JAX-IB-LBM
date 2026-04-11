# Python Project Import Organization

## The core problem: how does `import` work?

When you write `from src.core import D2Q9`, Python searches directories listed in `sys.path` for a directory named `src` that contains an `__init__.py`. Without installation, Python only searches the **current working directory** and the standard library — so running from the wrong folder always fails.

---

## Three layouts you will encounter

```
# ── 1. FLAT layout (simplest, no install needed for local scripts) ──────
my_project/
├── my_package/          ← importable directly: import my_package
│   ├── __init__.py
│   └── core.py
├── examples/
│   └── demo.py          ← works when run from my_project/
└── tests/

# ── 2. SRC layout (modern standard — what this project uses) ─────────────
my_project/
├── src/
│   └── my_package/      ← importable only after `pip install -e .`
│       ├── __init__.py
│       └── core.py
├── examples/
├── tests/
└── pyproject.toml

# ── 3. NAMESPACE package (no __init__.py, advanced use) ──────────────────
# Used by large organisations (google.cloud.*, apache.beam.*)
# Avoid until you need it.
```

**This project** currently uses a variant: the `src/` directory itself is the importable package (so you import `src.core`). This is unusual — the standard src layout imports the nested package (`jax_ib_lbm`).

---

## pyproject.toml controls what gets installed

```toml
[build-system]
requires      = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name    = "jax-ib-lbm"   # pip install name (hyphens OK)
version = "0.1.0"

[tool.setuptools.packages.find]
# where = ["."]   → look for packages at the project root (flat layout)
# where = ["src"] → look inside src/ (proper src layout)
where   = ["."]
include = ["src", "src.*"]   # the package named "src" and all sub-packages
```

**Critical rule**: `name` in `[project]` is the *pip name* (`pip install jax-ib-lbm`).
The *import name* is the actual directory name (`import src`). They are independent.

---

## The four ways to make an import work

| Method | Command | When to use |
|---|---|---|
| **Editable install** | `pip install -e .` | Development — changes take effect immediately |
| **Regular install** | `pip install .` | Distribution / CI |
| **PYTHONPATH** | `PYTHONPATH=/path/to/project python script.py` | Quick one-off without modifying anything |
| **sys.path hack** | `sys.path.insert(0, ...)` in script | Last resort, avoid |

**Always use editable install during development.** After running `pip install -e .` once,
every `from src.xxx import yyy` works from any directory, in any terminal, for this Python environment.

---

## `__init__.py` — the package contract

```
src/
├── __init__.py          ← makes src/ a package; controls `from src import X`
├── core/
│   ├── __init__.py      ← makes core/ a sub-package
│   ├── lattice.py
│   └── grid.py
└── fluid/
    ├── __init__.py
    └── equilibrium.py
```

**In `__init__.py` you decide what is public:**

```python
# src/core/__init__.py
from src.core.lattice import D2Q9, D3Q19, D3Q27   # re-export
from src.core.grid    import EulerianGrid

# Now users can write:
from src.core import D2Q9          # ✓  (short form via __init__)
from src.core.lattice import D2Q9  # ✓  (long form, always works)
```

**Rule of thumb**: put only the *public API* in `__init__.py`. Internal helpers stay in their module files.

---

## Relative vs absolute imports

```python
# Inside src/fluid/collision.py:

# Absolute import (always works, always clear) ← PREFER THIS
from src.core.lattice import Lattice

# Relative import (works only inside a package, fragile in scripts)
from ..core.lattice import Lattice   # ..  means "go up one level"
from .equilibrium   import compute_equilibrium #  .  means "same package"
```

**Use absolute imports everywhere.** Relative imports look clever but break when you move files.

---

## Current final state of this project

```
jax-ib-lbm/            ← project root (git repo)
├── src/               ← the installed package (import name: src)
│   ├── __init__.py
│   ├── core/
│   ├── fluid/
│   ├── boundary/
│   ├── immersed_boundary/
│   ├── forcing/
│   ├── solvers/
│   ├── postprocess/
│   └── utils/
├── examples/          ← scripts, not installed
├── tests/             ← pytest, not installed
└── pyproject.toml
```

After `pip install -e .` (already done), every example and test runs correctly
from any working directory:

```bash
python examples/2d_cylinder.py
pytest tests/
```
