"""
tstrait
=======

tstrait is a quantitative trait simulator of a tree sequence data.

See https://tskit.dev/ for complete documentation.
"""
from . import _version
from .simulate_effect_size import sim_trait
from .simulate_effect_size import TraitSimulator
from .simulate_phenotype import GenotypeResult
from .simulate_phenotype import PhenotypeResult
from .simulate_phenotype import PhenotypeSimulator
from .simulate_phenotype import Result
from .simulate_phenotype import sim_phenotype
from .trait_model import trait_model
from .trait_model import TraitModel
from .trait_model import TraitModelExponential
from .trait_model import TraitModelFixed
from .trait_model import TraitModelGamma
from .trait_model import TraitModelMultivariateNormal
from .trait_model import TraitModelNormal
from .trait_model import TraitModelT

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
