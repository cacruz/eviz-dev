Quick Start
============

Installation from source
------------------------

1. First you need to download and set up Anaconda or Miniconda on your computer.

Note that ``eViz`` has been tested with Python version >= 3.8

2. Get the source code (use http protocol for read-only access, ssh otherwise):

.. code-block::
   
   git clone https://github.com/cacruz/eviz.git
   git clone git@github.com:cacruz/eviz.git

3. cd into the code repo: 

.. code-block::
   
   cd eviz 


4. Create the Python environment:

.. code-block::
   
   conda env create -f environment.yaml


Enter *y* when prompted. This will download all the required packages needed to run the ``eViz`` tools and install
them in a separate environment called *viz*. This may take a minute or two, so please be patient.

5. Once the installation has finished building, *activate* the installed environment by running:


.. code-block::
   
   conda activate viz


Sample data
-----------

On DISCOVER we provide some data representative of the data sources
supported by the eViz tools. The data is located here:

.. code-block::

   /discover/nobackup/projects/jh_tutorials/eviz/sample_data

Therein you will find datasets collected from various data sources that are used to produce the visualizations
described in this guide.

You can also get the sample data from our data portal:

.. code-block::

    https://portal.nccs.nasa.gov/datashare/astg/eviz/sample_data/

Web-based plots
---------------

This is the code that we use to host visualization on a web platform. We encourage developers to try it out
and experiment with setting up your own web platform.

To share your visualizations on a website, `EViz` offers a tool to generate plots accessible via a web browser.
This functionality utilizes the streamlit package, which should already be included in the `viz` environment.

To use the web interface run the following command:

.. code-block::

    streamlit run sviz/sviz.py

This command will launch a web-based interface to run autoViz and display the static plots on your local host.

For additional information please look at the streamlit documentation (https://streamlit.io/).
