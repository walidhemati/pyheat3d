"""pyheat3d: explicit heat-equation / Jacobi stencils at the GPU memory roofline, from Python."""

from .problem import Stencil, heat_stencil, jacobi_stencil, stability_limit
from .registry import VARIANTS, available_variants, make_stepper

__version__ = "0.1.0"

__all__ = [
    "VARIANTS",
    "Stencil",
    "available_variants",
    "heat_stencil",
    "jacobi_stencil",
    "make_stepper",
    "stability_limit",
]
