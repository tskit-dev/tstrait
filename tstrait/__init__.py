"""
tstrait
=======

tstrait is a quantitative trait simulator of a tree sequence data.

See https://tskit.dev/ for complete documentation.
"""
from .provenance import __version__  # NOQA
from .simulate_effect_size import (
    sim_trait,
)  # noreorder
from .simulate_phenotype import (
    PhenotypeResult,
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
    sim_genetic,
    GeneticResult,
)  # noreorder
from .simulate_environment import (
    sim_env,
)  # noreorder

__all__ = [
    "__version__",
    "sim_trait",
    "PhenotypeResult",
    "sim_phenotype",
    "trait_model",
    "TraitModel",
    "TraitModelExponential",
    "TraitModelFixed",
    "TraitModelGamma",
    "TraitModelMultivariateNormal",
    "TraitModelNormal",
    "TraitModelT",
    "sim_genetic",
    "GeneticResult",
    "sim_env",
]
