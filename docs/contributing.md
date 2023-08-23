(contributing)=

# Contributing to tstrait

All contributions, bug reports, documentation improvements and ideas are welcome. If you think
there is anything missing, please open an [issue](https://github.com/tskit-dev/tstrait/issues)
or [pull request](https://github.com/tskit-dev/tstrait/pulls) on Github.

## Quick start

### Initializing environment

- Go to tstrait [Github repo](https://github.com/tskit-dev/tstrait) and click the "fork" button
  to create your own copy of the project.
- Clone the project to your own computer:

```
git clone https://github.com/your-user-name/tstrait.git
cd tstrait
git remote add upstream https://github.com/tskit-dev/tstrait.git
git fetch upstream
```

This creates the directory `tstrait` and connects your repository to the upstream (main project)
tstrait repository.

- Install the [requirements](requirements).
- Run the tests to ensure everything has worked: `python3 -m pytest`. These should all pass.

### Develop your contribution

- Make your changes in a local branch. On each commit, a
  [pre-commit hook](https://pre-commit.com/) will check for code styles and common problems.
  You can also use `pre-commit run` to run the checks without committing.

### Submit your contribution

- You can submit your contribution by pushing your changes back to your fork on Github:

```
git push origin branch-name
```

Afterwards, go to Github and open a pull request by pressing a green Pull Request button.

- Please also refer to
  [tskit documentation](https://tskit.dev/tskit/docs/stable/development.html)
  for more details on the recommended Github workflow.

(requirements)=

## Requirements

The list of packages needed for development are listed in `requirements.development.txt`.
Install these by using either:

```
conda install --file requirements/development.txt
```

or

```
python -m pip install -r requirements/development.txt
```

## Documentation

The documentation is written in markdown, and
[Jupyter Book](https://jupyterbook.org/en/stable/intro.html) is used to build a HTML
documentation. The Jupyter Book documentation offers
[Tutorial](https://jupyterbook.org/en/stable/start/your-first-book.html), so please
refer to it if you are new to the concept.

All files that are used in the documentation are included in the `docs` directory. To build
the documentation, run the following code in the `doc` directory:

```
jupyter-book build .
```

### API Reference

The API reference is created by using the docstring written in each code. We follow the
Numpy Docstring standard, and please refer to the
[documentation](https://numpydoc.readthedocs.io/en/latest/format.html) for details.

The API reference is built aumotically from docstring by using
[sphinx.ext.autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html).

### Codes

Almost all codes in the documentation are written by using the
[IPython directive](https://matplotlib.org/sampledoc/ipython_directive.html) Sphinx
extension. The code examples in the documentation are run during the doc building, so
the code examples will always be up to date.
