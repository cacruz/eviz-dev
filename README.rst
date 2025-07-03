=============================================================
EVIZ: An Easy to Use Earth Modeling System Visualization Tool
=============================================================

`EViz` is a comprehensive Python-based visualization library designed specifically for 
Earth System Modelers. It processes a wide variety of model-generated output formats 
and produces high-quality diagnostic plots for data analysis and validation. EViz serves 
as an essential validation tool for earth system model data, offering both command-line and 
interactive visualization capabilities.

Features
--------
* Multi-format Support: Process NetCDF, HDF, Zarr, and other common Earth System Model data formats
* Flexible Visualization: Generate maps, time series, vertical profiles, box plots, and correlation analyses
* Customizable: Configure plot appearance through YAML files with extensive customization options
* Comparison Tools: Compare multiple datasets side-by-side for in-depth analysis
* Statistical Analysis: Calculate and display metrics like RMSE and R² values directly on plots
* Interactive Mode: Use the interactive web interface for exploratory data analysis
* Batch Processing: Generate multiple plots efficiently through command-line batch processing

Installation
------------
EViz can be installed using conda:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate viz
    pip install -e .

Documentation
-------------
For comprehensive documentation, tutorials, and examples, please visit our documentation site:
https://cacruz.github.io/eviz-dev

For questions, comments, bug reports or feature requests please use the issues section: https://github.com/cacruz/eviz-dev/issues on Github. 

Contributing
------------
We welcome contributions! Please see our `Contributing Guide <https://github.com/cacruz/eviz-dev/blob/main/CONTRIBUTING.rst>`_  for details on how to submit pull requests, report issues, or request features.

Support
-------
For questions, comments, bug reports, or feature requests, please use the issues section on GitLab.

License
-------
Eviz is distributed under the Apache license.  Please read the LICENSE document located in the root folder.

Acknowledgments
---------------
EViz is developed and maintained by the `Advanced Software Technology Group (ASTG) <https://astg.pages.smce.nasa.gov/website/>`_ at NASA.
