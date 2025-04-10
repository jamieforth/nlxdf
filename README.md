# nlxdf-tools

This adds custom NEUROLIVE features for processing XDF files using
pyxdf-tools.

This package is installed automatically by
[nl-analysis](http://github.com/jamieforth/nl-analysis.git).

To use this package as part of other projects, or to install it within
a `venv` you are managing your yourself, install with `pip` or via
your build tool/IDE of choice (e.g. `uv`).

## `pip`

```
pip install git+https://github.com/jamieforth/nlxdf-tools.git#nlxdftools
```

To develop the package itself install in `editable` mode.

```
pip install --editable git+https://github.com/jamieforth/nlxdf-tools.git#nlxdftools
```

## `uv`

```
uv add git+https://github.com/jamieforth/nlxdf-tools.git
```

To develop the package itself clone and install in `editable` mode.

```
cd ~/some/path
git clone https://github.com/jamieforth/nlxdf-tools.git
cd ~/project/path
uv add --editable ~/some/path/nlxdf-tools
```
