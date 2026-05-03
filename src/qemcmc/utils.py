from typing import Any
from typing import Dict
from typing import Mapping
from typing import Iterable

__all__ = [
    "string_energies",
]


def string_energies(counts: Mapping[Any, int], model) -> Dict[str, float]:
    """Map each outcome in bitstring to its Ising energy via model.energy."""
    return {s: float(model.energy(s)) for s in counts.keys()}
