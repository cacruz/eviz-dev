Data Sources
============

Earth system model (ESM) data sources can be represented and accessed in multiple ways depending on various factors
such as the specific ESM, data format, storage infrastructure, and data access protocols.

Data Representation and Access
------------------------------

Here are some common ways ESM data sources can be represented and accessed:

#. File-based Access:
    - NetCDF Files: ESM data is often stored in NetCDF files, which provide a self-describing format for storing
      multidimensional data. NetCDF files can be accessed using libraries like NetCDF4, xarray, or directly using
      low-level file I/O operations.
    - GRIB Files: Some ESMs use the GRIB (GRIdded Binary) format for storing meteorological and climate data.
      GRIB files can be read using libraries like pygrib.

#. Distributed Data Access:
    - OPeNDAP: ESM data can be accessed through OPeNDAP (Open-source Project for a Network Data Access Protocol),
      which provides a client-server framework for accessing remote data over the internet. OPeNDAP allows users to
      access ESM data directly without downloading the entire dataset.
    - THREDDS: THREDDS (Thematic Real-time Environmental Distributed Data Services) is a web-based data server that
      provides metadata catalogs and facilitates access to ESM data stored in various formats. THREDDS enables users
      to access ESM data using standard protocols like OPeNDAP and NetCDF Subset Service (NCSS).

#. Data Repositories and Portals:
    - Data Repositories: ESM data can be stored in dedicated data repositories such as the Earth System Grid Federation
      (ESGF), which provides a distributed infrastructure for archiving and accessing climate and ESM data.
    - Data Portals: Web-based portals, such as the Climate Data Gateway and NASA's Earthdata Search, provide
      user-friendly interfaces for searching, discovering, and accessing ESM data from various sources.

#. API-Based Access:
    - ESM-specific APIs: Some ESMs provide their own APIs for accessing and interacting with their data. These APIs
      often offer high-level functions and abstractions specific to the ESM, allowing users to query and retrieve
      data programmatically.
    - Standard APIs: ESM data can also be accessed through standard APIs such as the Climate Data API (CDAPI) and
      the Python Climate Index (climdex) API. These APIs provide a uniform interface for accessing climate and ESM
      data from different sources.

#. Cloud-based Access:
    - Cloud Storage: ESM data can be stored in cloud storage platforms like Amazon S3, Google Cloud Storage, or
      Microsoft Azure Blob Storage. Users can access the data directly from the cloud storage infrastructure using
      appropriate APIs or tools.
    - Cloud Computing: Cloud-based computing platforms like Amazon Web Services (AWS), Google Cloud Platform (GCP), and
      Microsoft Azure provide services for running ESM simulations and processing large-scale ESM datasets. These
      platforms offer APIs and tools for accessing and analyzing ESM data within the cloud environment.

The specific methods and tools used may vary depending on the ESM model, data infrastructure, and user requirements.

In `eViz` we currently support only (1) and the OpeNDAP option in (2) above.

Data Formats
------------

There are various types and formats of data sources used in Earth System Models (ESMs) and climate science.
For example:

#. Climate Model Output:
    - NetCDF (Network Common Data Form): NetCDF is a widely used format for storing climate and ESM data. It provides a
      self-describing structure to store multidimensional data along with metadata. NetCDF files often include variables
      such as temperature, precipitation, wind speed, and various climate model diagnostics.

#. Atmospheric Data:
    - Radiosonde Data: Radiosonde data is obtained from weather balloons equipped with instruments that measure
      atmospheric properties (temperature, humidity, pressure, wind) as they ascend through the atmosphere.
    - Weather Radar Data: Weather radar data captures information about precipitation intensity, storm movement, and
      other meteorological features in the atmosphere.
    - Satellite Data: Satellite-based observations provide a wealth of atmospheric data, including measurements of
      cloud cover, sea surface temperature, aerosol content, vegetation indices, and more.

#. Oceanographic Data:
    - Sea Surface Temperature (SST) Data: SST data represents the temperature of the ocean's surface and is an
      essential parameter for understanding climate patterns and ocean dynamics.
    - Sea Level Data: Sea level data provides measurements of changes in sea level over time, which helps monitor
      global sea level rise and ocean circulation patterns.
    - Ocean Current Data: Ocean current data captures information about the movement and circulation of ocean waters,
      which plays a crucial role in climate dynamics and marine ecosystems.

#. Land Surface Data:
    - Soil Moisture Data: Soil moisture data measures the amount of water content present in the soil, influencing
      plant growth, agriculture, and hydrological processes.
    - Vegetation Indices: Vegetation indices, such as the Normalized Difference Vegetation Index (NDVI), quantify the
      health and vigor of vegetation by analyzing the reflectance of different wavelengths of light.
    - Land Cover Data: Land cover data classifies the Earth's surface into different categories like forests,
      grasslands, croplands, urban areas, etc., providing information on land use and land cover changes.

#. Cryospheric Data:
    - Glacier and Ice Sheet Data: Data on glaciers and ice sheets help monitor changes in their extent, volume, and
      mass balance, providing insights into climate change and sea level rise.
    - Snow Cover Data: Snow cover data tracks the extent and depth of snow on land, critical for hydrological modeling,
      water resource management, and climate studies.

#. Emission and Greenhouse Gas Data:
    - CO2 and Other Greenhouse Gas Measurements: Data sources include direct measurements of atmospheric concentrations
      of carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and other greenhouse gases. These measurements are
      collected from ground-based monitoring stations, aircraft, and satellite sensors.

Furthermore data can be structured, i.e. gridded, or unstructured. As mentioned earlier each type of data has its own
specific format and may be represented in various file formats like NetCDF, GRIB, HDF, ASCII, or stored in specialized
databases.

How do we distinguish between data sources?
There are standards and conventions used to distinguish between different data sources in the context of
Earth System Models (ESMs) and climate science. These standards help categorize and identify the sources of data
based on their characteristics. Here are a few commonly used standards:

#. **Data Source Type**: Data sources can be categorized based on their type or nature, such as atmospheric data,
   oceanographic data, land surface data, cryospheric data, emissions data, etc. This categorization is based on the
   domain or component of the Earth system from which the data originates.

#. **Data Format**: Different data formats are often associated with specific data sources. For example, atmospheric
   data may commonly be stored in NetCDF (Network Common Data Form) format, satellite imagery data may be in GeoTIFF
   or JPEG format, climate model output may be in NetCDF or GRIB (GRIdded Binary) format, and so on. The data format
   can provide a clue about the source and structure of the data.

#. **Metadata**: Metadata associated with a data source can provide information about its origin and characteristics.
   Metadata typically includes details about the data source, such as the instrument used for data collection, the
   geographical coverage, temporal resolution, variable names, units, and other relevant information. By examining
   the metadata, one can often determine the source of the data.

#. **Data Repository or Archive**: Many data sources are stored in data repositories or archives that specialize in
   specific types of data. These repositories or archives often have unique identifiers or naming conventions to
   distinguish between different data sources. Examples of such repositories include the Earth System Grid Federation
   (ESGF), NASA Earth Observing System Data and Information System (EOSDIS), National Centers for Environmental
   Information (NCEI), and many others.

#. **Data Provider or Institution**: The organization or institution responsible for collecting, processing, and
   distributing the data can also be used to differentiate between data sources. Each institution may have its own
   data naming conventions or data access portals that indicate the source of the data.

While there may not be a single universal standard to distinguish between all data sources, a combination of these
approaches and conventions is typically used to identify and categorize data sources in the field of ESM and
climate science.

In `eViz` we take the approach of identifying by using the file extension. Obviously, it is impossible to cover all the
possibilities and in some cases we make "reasonable" assumptions to determine the origin (and format) of the data source.
For example, is a file has no extension, we assume it is a NetCDF4 file.