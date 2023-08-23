(install)=

# Installing tstrait

tstrait can be installed by using `pip` or [conda](https://conda.io/docs/).
We recommend using `conda` for most users.

## System requirements

tstrait requires Python 3.8+. Python dependencies are installed automatically by `pip` or `conda`.

## Via conda

Pre-built packages are available through `conda`, and built using [conda-forge](https://conda-forge.org/).
It is suggested that you create a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
conda create --name tstrait-env
```

and install tstrait into it. After activating the environment by

```
conda activate tstrait-env
```

tstrait can be installed by using

```
conda install -c conda-forge tstrait
```

## Via pip

tstrait can also be installed via `pip` from PyPI. Installing using `pip` is more flexible than `conda`,
as it can support more versions of Python and dependencies can be customized. tstrait can be installed with:

```
python3 -m pip install tstrait
```

## Dependencies

tstrait requires the following dependencies:

| Package                              | Minimum supported version |
| ------------------------------------ | ------------------------- |
| [NumPy](https://numpy.org)           | 1.20.3                    |
| [numba](https://numba.pydata.org/)   | 0.57.0                    |
| [pandas](https://pandas.pydata.org/) | 1.0                       |
| [tskit](https://tskit.dev/)          | 0.5.5                     |
