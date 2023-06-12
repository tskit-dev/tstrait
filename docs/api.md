---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Python API

This lists the detailed documentation for the tstrait Python API.

## Summary

```{eval-rst}
.. currentmodule:: tstrait
```

### Phenotype Simulation

```{eval-rst}
.. autofunction:: tstrait.sim_phenotype
```

```{eval-rst}
.. autoclass:: tstrait.PhenotypeSimulator
    :members: sim_genetic_value, sim_environment
```

### Trait Model

```{eval-rst}
.. autoclass:: tstrait.TraitModel
    :members: sim_effect_size
```

```{eval-rst}
.. autoclass:: tstrait.TraitModelAdditive
    :members: sim_effect_size
```

```{eval-rst}
.. autoclass:: tstrait.TraitModelAllele
    :members: sim_effect_size
```

### Result

```{eval-rst}
.. autoclass:: tstrait.PhenotypeResult
```

```{eval-rst}
.. autoclass:: tstrait.GeneticValueResult
```