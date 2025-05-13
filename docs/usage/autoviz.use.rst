==============================
autoViz: a map-generating tool
==============================

The plotting tool, **autoviz.py** or **autoViz** for short, is a highly configurable CLI-driven tool that can be used to
generate a variety of plots used to diagnose Earth system model output. What follows is a description of several use
cases that allow users to quickly generate maps using **autoViz**.


1. `Visualize one field from a single netCDF file`_
2. `Metadump: creating the configuration files`_
3. `Visualize multiple fields field from a single netCDF file`_
4. `Visualize fields with a SPECS file`_
5. `Visualize with a predefined generic model`_
6. `Visualize with a predefined GEOS model`_
7. `Visualize NU-WRF output`_
8. `sViz: the web-based interface for EViz`_


Visualize one field from a single netCDF file
---------------------------------------------

Most Earth system model output is in NetCDF format. Therefore, it is important to try to access and visualize such data
quickly. Thus, running the following will print the file's NetCDF metadata to the screen

.. code-block::

   python autoviz.py --file /path/to/model_output.nc

If you know the name of the FIELD that you want to quickly visualize, then run

.. code-block::

   python autoviz.py --file /path/to/model_output.nc --var FIELD

To run **autoViz** for visualizing netCDF model data you need to provide a **configuration file** written in **YAML** format.
This file, hereafter referred to as the **APP** file, contains instructions that specify what data file(s) to use, what 
field(s) to plot and what type of plot(s) we want. This file can be stored anywhere and can be specified as an argument 
to **autoViz**. In addition, you have to specify the model template (or **APP**) to be used. For example:

.. code-block::
   
   python autoviz.py --configfile /path/to/my_config.yaml -s my_model

In `EViz`, my_model can be one of the following: **generic**, **geos**, **ccm**, **cf**, **wrf** ,or **lis**. These are
the 'supported models'. Supported models are data sources for which predefined configurations have been made available
within **autoViz**.

You can also specify a config directory that contains all of the YAML files stored in sub-directories named after the
source of the data. For example:

.. code-block::

   python autoviz.py --config /path/to/config -s my_model

will search for my_model.yaml in /path/to/config/my_model.

The `--config /path/to/config` option is equivalent to setting the environment variable EVIZ_CONFIG_PATH as follows:

.. code-block::

   export EVIZ_CONFIG_PATH=/path/to/config

In that case you could run **autoViz** as follows:

.. code-block::

   python autoviz.py -s my_model

Note that running **autoViz** without arguments will generate an error:


.. code-block::
   
   python autoviz.py    # <<< error


Now, let's describe the contents of the YAML file.

In order to use **autoViz** we need to tell it where to find the **input data**. The APP file must specify the filename,
the fields to plot and the corresponding plot types. Thus, the simplest YAML file must contain the following information:

.. code-block::
   
   inputs:
      - name: /path/to/data.nc4
        to_plot:
           SLP: xy  # assume data.nc4 contains a field named SLP

**Note**: In this guide we will only consider netCDF input data.

The formatting of a YAML file is important or else syntax errors will result in **autoViz** crashing unexpectedly.
In particular, correct indentation is very important in a YAML file. A YAML file is basically a text representation
of a Python dictionary with "keys" and "values" separated by a colon ":". For instance, the "inputs" `key` above has
values specified in the next line, in this case a list of elements starting with a "-" that includes the filename
(name) and the fields to plot (to_plot). Keep in mind that for a given input file the values to "to_plot" must be
specified as defined in the data source metadata, and it is case-sensitive. For example if the metadata describes a
field named SLP then you must enter SLP and not slp.

Also, if one specifies only an APP file, a “generic” model is assumed. A “generic” model is a file model abstraction
to represent a **generic netCDF data source**. In the current example, **autoViz** will attempt to generate the simplest
possible 2D map. In this simple case a “good looking” plot is not guaranteed. **Finally, note that the plot will be
displayed in a pop-up window.**

**Notes**

1. A sample APP YAML file, sample_app.yaml, with all the available options, can be found in the config/ directory.

2. On DISCOVER you can try this example by copying the file
/discover/nobackup/projects/jh_tutorials/eviz/config/my_config.yaml to a directory of your choice.

Metadump: creating the configuration files
------------------------------------------

Creating the APP and SPECS files can be time consuming. To expedite the creation of such files you can run the
`metadump` utility as follows:

.. code-block::

   python metadump.py /path/to/model_output.nc --specs model_output.yaml --app model_output_specs.yaml

The above command will create configuration files containing information about all the plottable variables as well
as the allowable plot types for each variable. Note that most settings will default to "reasonable" values and will
probably need to be tweaked if you wish to generate different output than that provided byt the defaults.

Additionally, you can create configuration files to contain details about selected variables. For example:

.. code-block::

   python metadump.py /path/to/model_output.nc --specs model_output.yaml --app model_output_specs.yaml --vars VAR1 VAR2


Visualize multiple fields field from a single netCDF file
---------------------------------------------------------

Here we repeat the above example but specifying more fields to plot. With multiple fields, the plots will be shown
one at a time, in a window.

It is more convenient to save the image plots to files. We can control our output options in the "outputs" section
of the APP file. For example:


.. code-block::
   
   inputs:
     - name: /path/to/data/my_data.nc4
        to_plot:
           SLP: xy
           H: yz


   outputs:
      print_to_file: yes

You can also specify an output directory as follows:


.. code-block::
   
   outputs:
      print_to_file: yes
      output_dir: /some/other/path/output

Note that if ``output_dir`` is not specified then the images will be stored in the top-level EViz directory under
``outputs``. You may also specify the ``output_dir`` value from the shell by setting the ``EVIZ_OUTPUT_PATH`` environment
variable. For example:

.. code-block::

   export EVIZ_OUTPUT_PATH=/some/other/path/output

which has the same effect as the entry in the APP file.

For multidimensional fields the cases above only display the "first available" field slice. So, if the field is 4D,
simple lat-lon plots will display the first vertical level and the first-time level. A simple zonal mean plot will 
display the first-time level. How do you select vertical or time slices? That's next!

Visualize fields with a SPECS file
----------------------------------

There is another YAML file that is used to *configure* the plotted fields. We call that file the **SPECS** file.
In it, we provide fine-grained specifications for the field to be plotted. For example, if it's a 4D field, we may
want to specify what vertical level to plot as well as what time level - or perhaps we want a time average.
The SPECS file is located in the same path as the APP file, and it must have the same basename with “_specs”
appended to it. For example  the “my_config.yaml” SPECS file must be named “my_config_specs.yaml”.

Note that we still run `autoViz` as before:

.. code-block::
   
   python autoviz.py --config /path/to/config -s my_model


but now, the SPECS file, if found, will be used and the configurations therein will be applied to the maps specified
in the APP file. What does the SPECS file contain? As an example consider the sea-level pressure field. A possible
SLP entry in the SPECS file could be


.. code-block::
   
   SLP:
       xyplot:
          levels:
             0: []

Here we specify the field name, the type of plot (xy or latlon), the vertical level we wish to plot,  and the contour
levels we wish to use in [].

**Notes**

#. The field name is, again, case-sensitive and must conform with the netCDF field name metadata.
#. The type of plot is "xyplot", not "xy".
#. For 2D plots the convention is to use "0" to specify that it is a 2D field and thus a single vertical level.
#. For 3D fields, the vertical level must be one of the levels specified in the netCDF metadata, for example 1000.
#. The list [] following "0" can be used to specify a list of contour values that we want to display. In our example
   the contour list is empty, and so **autoViz** will generate levels based on the range values of the SLP data. Using []
   can be particularly useful if the range of data values is not known in advance.

A field such as sea-level pressure can be defined using units of `Pa`. it can be converted to `mb` units by
defining a "unitconversion" field as shown below (also specifying a data contour range):


.. code-block::
   
   SLP:
      unitconversion: 0.01
      units: mb
      xyplot:
         levels:
            0: [700, 800, 900, 950, 975, 985, 990, 995, 1000, 1005, 1010]


For more information about the available options look at the sample SPECS YAML file, sample_specs.yaml, available in 
the config/ directory.

**Notes**

Pre-defined SPECS files are included in the config directory and are already configure to work with sample data on the
DISCOVER system. These can be used with the “supported” data sources that include **generic, geos, ccm, cf, lis and wrf**.
So, for example, we can use the predefined generic model to create plots from various data sources – as long as they
are in netCDF format (that’s what generic refers to).


Use your own APP/SPECS
^^^^^^^^^^^^^^^^^^^^^^

To use your own APP/SPECS you need to change the file paths in the APP file. These paths specify the **autoViz** input
data and output paths in your system. For example, in your APP file:

.. code-block::

    inputs:
       - name         : my_file.nc4
         location     : /discover/nobackup/$USER/data

    outputs:
        output_dir: /discover/nobackup/$USER/output

The location entry is optional as you can specify the file location with the name as follows:

.. code-block::

    inputs:
       - name         : /discover/nobackup/$USER/data/my_file.nc4

Use configurations from a config directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configurations described above are stored with the `EViz` code base, under config. In some cases you may want to
maintain files in a separate config directory and use those when running autoViz or iViz. In that case you can specify
the location of the config files using the ``EVIZ_CONFIG_PATH`` environment variable. For example:

.. code-block::

    export EVIZ_CONFIG_PATH=/home/$USER/projects/config

Also, note that on DISCOVER there are sample config files that you can copy to your work space. For example:

.. code-block::

    cp -r /discover/nobackup/projects/jh_tutorials/eviz/config $NOBACKUP


Visualize with a predefined generic model
-----------------------------------------

In this case, **autoViz** will use the predefined SPECS file corresponding to the **generic** APP file. These files can
be found in the config/generic directory. The generated plots will be as specified in the generic APP and SPECS files.
The sample generic SPECS file provided with autoViz contains settings for multiple files including: 3D data (2D+time)
from the `Climatic Research Unit <https://crudata.uea.ac.uk/cru/data/hrg/>`_,  3D data (2D+time) from the
`CESM <https://www.cesm.ucar.edu/>`_ model, and 4D data (3D+time) from 
`ERA5-REAN <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_.

For this case we run **autoViz** as follows:


.. code-block::
   
   python autoviz.py --source generic

or

.. code-block::

   python autoviz.py -s generic

The terminal output may look something like this (depending on the options):

.. code-block::

    INFO :: base (__post_init__:93) : Start init
    INFO :: config (__post_init__:119) : Start init
    INFO :: config (_init_readers:370) : Setup NetCDF reader
    INFO :: reader (__post_init__:19) : Start init
    INFO :: generic (__post_init__:32) : Start init
    INFO :: root (__post_init__:66) : Start init
    INFO :: root (plot:92) : Generate plots.
    INFO :: plotter (__post_init__:1193) : Start init
    INFO :: generic (_single_plots:186) : Plotting tas, xt plot
    INFO :: generic (_get_xt:406) : 'tas' field has 1980 time levels
    INFO :: generic (_get_xt:416) : Averaging method: point_sel
    INFO :: plotter (_time_series_plot:783) : Adding trend
    INFO :: plotter (_time_series_plot:797) :  -- polynomial degree: 5
    INFO :: root (plot:124) : Output files are in /Users/ccruz/projects/EVIZ/gitlab/eviz/demo_output/single
    INFO :: root (plot:126) : Done.
    Time taken = 0:00:01.623150

This will produce various plots in the path specified in output_dir, including

1. Lat-lon plots for all of them, xy
2. Time-series plots for CRU and CESM
3. Hovmoller plots for CRU and CESM
4. Same as (2) but with subsets
5. Zonal mean plots for ERA5-REAN
6. Zonal mean Hovmoller plots for ERA5-REAN
7. Polar plot for CRU


Visualize with a predefined GEOS model
--------------------------------------

In this case we produce plots generated by the **GEOS** model simulations including the
`Chemistry Climate Model (CCM) <(https://acd-ext.gsfc.nasa.gov/Projects/GEOSCCM/>`_, 
`Composition Forecast (CF) <https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/>`_,
simulations and `MERRA2 <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_.

For this case we run **autoViz** as follows:

.. code-block::
   
   python autoviz.py -s geos
   python autoviz.py -s ccm
   python autoviz.py -s cf


These commands will produce various plots in the path specified in output_dir including:

1. Various image plots from multiple GEOS files
2. CCM comparison zonal mean plots with tropopause overlay in one field
3. CF comparison plots

For details look in the corresponding YAML files in config/geos, config/ccm, and config/cf and feel free to change the 
settings.

Visualize NU-WRF output
-----------------------

`NU-WRF <https://nuwrf.gsfc.nasa.gov/>`_ is a modeling framework that incorporates, among others, the
`WRF <https://www.mmm.ucar.edu/weather-research-and-forecasting-model>`_ and `LIS <https://lis.gsfc.nasa.gov/>`_
modeling systems. Therefore, NU-WRF simulations produce both WRF and LIS output.

Visualize with a predefined WRF model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WRF configurations can be found in the config/wrf folder. Feel free to modify the filename (and location) entries.
To generate the sample **autoViz** plots run:


.. code-block::
   
   python autoviz.py -s wrf


This will produce simple LatLon and zonal mean plots for some fields specified in the config/wrf.yaml file.

Visualize with a predefined LIS model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LIS configurations can be found in the config/lis folder. To generate the plots run:


.. code-block::
   
   python autoviz.py -s lis


This will produce simple 2D (LatLon) plots for the fields specified in the config/lis.yaml file.


sViz: the web-based interface for EViz
--------------------------------------

Overview
^^^^^^^^
The **Streamlit driver** is the primary interface for launching and interacting with the eViz visualization tool.
It provides a streamlined, interactive dashboard for Earth System model (ESM) users to explore, visualize, and
analyze data effortlessly.

This driver leverages the eViz library's capabilities, including the configuration-based approach for visualizations
and support for various data formats, such as NetCDF, HDF, and GRIB.

Key Features
^^^^^^^^^^^^
- **Interactive Visualization**: Users can load data, adjust visualization parameters, and generate plots in real-time.
- **Seamless Configuration**: Leverages YAML configuration files for preloading visualization settings.
- **Customizable Dashboard**: Integrates dynamic widgets for fine-tuning visual outputs.

File Location
"""""""""""""
`sviz.py` is located in the sviz directory, making it easy to execute with a single command.

Example Usage
"""""""""""""
To launch the Streamlit app, run the following command in your terminal from the project's root directory:

`streamlit run sviz/sviz.py`

Extensibility
"""""""""""""
Users can modify `sViz` to:

- Add new widgets or input options.
- Customize the layout or theming of the app.
- Incorporate additional datasets or plotting capabilities.

Notes
^^^^^
- Ensure that dependencies, specified in environment.yaml, are installed in your Python environment before running the driver.
- Ensure compatibility with the expected YAML configuration schema. Use `metadump.py` for guidance on creating configurations.
- Refer to the documentation_ for details on supported data formats, fields, and advanced plotting capabilities.

.. _documentation: https://astg.pages.smce.nasa.gov/visualization/eviz
