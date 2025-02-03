# nlxdf-tools

This adds custom NEUROLIVE features for processing XDF files using
pyxdf-tools.

One way to use this package is to install neurolive-analysis, which
will create a self-contained `venv` and install nlxdf-tool as a
dependency.

All dependencies can be updated with `pdm`.

```
$ pdm update
```

To use this package as part of other projects, or to install it within
a `venv` you are managing your yourself, install it with `pip` or via
your build tool/IDE of choice.


```
$ pip install -e git+https://github.com/jamieforth/nlxdf-tools.git#nlxdftools
```

NB. If you are installing manually you will also need to ensure you
keep the `pyxdftools` package up-to-date.
