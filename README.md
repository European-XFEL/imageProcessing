# ImageProcessing Device (Python)

Image processing utils

## Testing

Every Karabo device in Python is shipped as a regular python package.
In order to install the package to Karabo's own Python environment,
simply type:

``pip install -e .``

in the directory of where the ``pyproject.toml`` file is located, or use the
``karabo`` utility script:

``karabo develop imageProcessing``
