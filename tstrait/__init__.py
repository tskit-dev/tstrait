"""
tstrait
=======

tstrait is a quantitative trait simulator of a tree sequence data.

See https://tskit.dev/ for complete documentation.
"""
from .provenance import __version__  # NOQA
from .simulate_effect_size import (
    sim_trait,
    TraitSimulator,
)  # noreorder
from .simulate_phenotype import (
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
from .genetic_value import (
    genetic_value,
    GeneticValue,
)  # noreorder
from .simulate_environment import (
    sim_env,
    EnvSimulator,
)  # noreorder

__all__ = [
    "__version__",
    "sim_trait",
    "TraitSimulator",
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
    "genetic_value",
    "GeneticValue",
    "sim_env",
    "EnvSimulator",
]
