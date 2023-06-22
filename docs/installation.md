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

(sec_installation)=

# Installing tstrait

**tstrait** can be installed by using pip or [conda](https://conda.io/docs/). We recommend using `conda` for most users.

## System requirements

**tstrait** requires Python 3.8+. Python dependencies are installed automatically by `pip` or `conda`.

## Via Conda

Pre-built packages are available through `conda`, and built using [conda-forge](https://conda-forge.org/). **tstrait** can be installed by using:

```{code-block} bash

$ conda install -c conda-forge tstrait

```

## Via Pip

Installing using `pip` is more flexible than `conda` as it can support more versions of Python and dependencies can be customized. **tstrait** can be installed by using:

```{code-block} bash

$ python3 -m pip install tstrait

```