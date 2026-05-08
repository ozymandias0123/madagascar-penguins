"""
strategies/__init__.py

Auto-discovers and exports all strategy classes.
Each strategy file must contain a class that inherits from BaseStrategy.

Usage in engine:
    from strategies import get_strategy
    strategy = get_strategy("ICT")        # by name
    strategy = get_strategy("SilverBullet")
"""

import importlib
import os
from strategies.base_strategy import BaseStrategy

_registry: dict[str, type] = {}


def _discover():
    """Scan strategies/ folder and register all BaseStrategy subclasses."""
    folder = os.path.dirname(__file__)
    for fname in os.listdir(folder):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        module_name = f"strategies.{fname[:-3]}"
        try:
            mod = importlib.import_module(module_name)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (isinstance(obj, type)
                        and issubclass(obj, BaseStrategy)
                        and obj is not BaseStrategy):
                    _registry[obj.name] = obj
        except Exception:
            pass


def get_strategy(name: str) -> "BaseStrategy":
    """Return an instance of the strategy with the given name."""
    if not _registry:
        _discover()
    cls = _registry.get(name)
    if cls is None:
        raise ValueError(
            f"Strategy '{name}' not found. "
            f"Available: {list(_registry.keys())}"
        )
    return cls()


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    if not _registry:
        _discover()
    return sorted(_registry.keys())


_discover()

__all__ = ["BaseStrategy", "get_strategy", "list_strategies"]
