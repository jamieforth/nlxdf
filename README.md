# nlxdf

This package provides custom NEUROLIVE features for processing XDF
files using [pdxdf](https://github.com/jamieforth/pdxdf).

It can be used in several ways:

1. as a standalone command-line tool
2. as a dependency to be used by other scripts and packages
3. in editable mode for development of the package itself
4. as part of [nl-analysis](http://github.com/jamieforth/nl-analysis.git)
   (private).

## Standalone command-line tool

Install [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/jamieforth/nlxdf.git
cd nlxdf
uv run snakeskin-resample --help
uv run readings-resample --help
```

### Example command-line usage

``` sh
uv run snakeskin-resample -i ./data/performances/4-snakeskin/*/*.xdf \
   -o ./data/performances/4-snakeskin/resampled/ \
   --fs 512 \
   --label 'optional text to describe batch'
```

``` sh
uv run readings-resample -i ./data/performances/5-readings/*/*.xdf \
   -o ./data/performances/5-readings/resampled/ \
   --fs 512 \
   --label 'optional text to describe batch'
```

## Install as a dependency

To use this package as part of other projects, or to install it within
a `venv` you are managing your yourself, install with `pip` or via
your build tool/IDE of choice (e.g. `uv`).

## `pip`

```sh
pip install git+https://github.com/jamieforth/nlxdf.git#nlxdf
```

### `uv`

```sh
uv add git+https://github.com/jamieforth/nlxdf.git
```

## Development

To develop the package itself clone and install in `editable` mode.

Recommended: Install within a self-contained environment managed by
`uv` to ensure a reproducible development environment.

```sh
git clone https://github.com/jamieforth/nlxdf.git
cd nlxdf
uv add . --editable --dev
```

To incorporate upstream changes:

```sh
git pull
uv sync
```


Alternatively within an externally managed environment.

```sh
pip install --editable git+https://github.com/jamieforth/nlxdf.git#nlxdf
```
