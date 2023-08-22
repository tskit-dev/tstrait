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
    "GeneticValue",
    "sim_env",
    "EnvSimulator",
]
