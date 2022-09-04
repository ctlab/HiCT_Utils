# HiCT_Utils -- Utilities for interaction with HiCT library and data format. 
For more info about HiCT, please visit [HiCT repo page](https://github.com/ctlab/HiCT).

## Provided Utilities
* `convert`: This tool is used to convert Cooler's in `.cool` or `.mcool` format to HiCT-supported ones. For example, use `python -m hict_utils convert -o converted.hict.hdf5 source.mcool` to convert all resolutions of `source.mcool` file into HiCT-supported format. See `python -m hict_utils --help` for other options.

A list of all tools could be obtained with `python -m hict_utils --help`. 

**NOTE**: You should have `hict` library installed before trying to use this tool. Usage of virtual environments such as `venv` is recommended.

## Building from source
You can use `python setup.py bdist_wheel` to build `hict_utils` module, then install this wheel using `pip install dist/*.whl`.
