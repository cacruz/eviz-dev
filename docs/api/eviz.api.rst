EViz API
========

Users, developers and maintainers of earth system models often find themselves in need to visualize output produced by
those modeling systems. Output from such systems can come in a variety of formats and can also be quite voluminous.
There are existing visualization packages that can be used to visualize such output but most come with limitations
ranging from being model specific to OS dependent. Furthermore, the file format adds an additional layer of complexity.
This package is intended to provide a quick easy to use way to visualize Earth system model output using a
model-agnostic approach that is also OS independent.

Infrastructure
**************

The entire EViz infrastructure is built upon two packages: ``lib`` and ``models``.

``lib`` is a high-level OO Python package which aims to provide a framework for **EViz**.

The aim of ``lib`` is to define and provide classes that are used to construct visualizable figure objects to be either
plotted or interactively visualized. In the case of ``autoViz``, the manipulated objects are ``Matplotlib`` figure and
axes. 
One unifying aspect of the package is that all visualizable objects are ultimately transformed into ``Xarray`` objects
which provide a unified representation of data and metadata. A very important design configuration is the use of
YAML-based configuration files to specify the map output. This approach avoids having the use write any code. Instead,
the YAML files provide directives to drive the map generation. 

The ``models`` package contains the user defined modules for all the supported Earth system science models. These
modules contain code implementations to visualize a particular output definition.

The current implementation has been developed in Python 3 and tested on Mac and Linux operating systems. The required
environment can be found in the cicd/environment.yaml file.

Limitations
***********

Earth system model related data comes in a variety of formats, namely NetCDF, HDF5, Grib, plain binary as well as
text format such as CSV. The current implementation assumes NetCDF but a significant effort is being made to support
other data formats such as HDF5 and CSV. We expect that future releases will have support for most common data formats
used in Earth system modeling frameworks.

Packages
********

.. automodule:: eviz
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

   eviz.lib
   eviz.models

Main Entry Points
*****************

.. toctree::
   :maxdepth: 2

   autoviz
   metadump
