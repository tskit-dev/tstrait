"""
tstrait
=======

tstrait is a quantitative trait simulator of a tree sequence data.

See https://tskit.dev/ for complete documentation.
"""
from . import _version
from .simulate_effect_size import (
    sim_trait,
    TraitSimulator,
)  # noreorder
from .simulate_phenotype import (
    GenotypeResult,
    PhenotypeResult,
    PhenotypeSimulator,
    Result,
    sim_phenotype,
)  # noreorder
from .trait_model import (
    trait_model,
    TraitModel,
    TraitModelExponential,
    TraitModelFixed,
    TraitModelGamma,
    TraitModelMultivariateNormal,
    TraitModelNormal,
    TraitModelT,
)  # noreorder

__version__ = _version.tstrait_version

__all__ = [
    "__version__",
    "sim_trait",
    "TraitSimulator",
    "GenotypeResult",
    "PhenotypeResult",
    "PhenotypeSimulator",
    "Result",
    "sim_phenotype",
    "trait_model",
    "TraitModel",
    "TraitModelExponential",
    "TraitModelFixed",
    "TraitModelGamma",
    "TraitModelMultivariateNormal",
    "TraitModelNormal",
    "TraitModelT",
]
