# ChangeLog

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2025-6-19

This is a significant rewrite of v0.6.3. Though eViz's general idea and all 
the previous functionality is the same, the code structure is different.
The API remains the same and the units tests are more numerous.

A significant change is the omission if iViz, which was a separate application
using the same library code. Now, the interactivity is optional and when chosen
eviz can produce interactive plots.

### Added
- Many new modules and a slightly different directory structure.
  -  Separated 'lib' code into autoviz, config, and data. This latter defines
     sources, factory (for sources), and a pipeline that defines the eviz 
     workflow.
  - The original config module (config.py) was broken down into more
    manageable pieces including a configuration adapter to bridge the 
    configuration with the pipeline.
- Many unit tests (code coverage increased from ~25% to ~45%)
- Minimum workable Grib class (and its corresponding grib data source option)
  and meta coordinates.
- Matplotlib rcParams options in specs file
- Automatic OpenDAP support (when input is a URL)
- Improved documentation

### Deprecated

### Fixed
- The above additions fixed many bugs.
- Improved metadump.py with tests
- Edited sviz/autoviz.py
- Issues with regrid() were fixed
- Coordinates standardization

### Removed
- iViz code
- The generic data source is now known as 'gridded' to emphasize its 'nature'
- Class in const.py that relies on a module-replacement approach is replaced
  by a plain module for constants and another module for env-dependent values,
  as well as paths.

### Known issues:
- Figure details are not automatically optimal and have to be tweaked
  through specs.
- The units module needs more testing and some unit conversions may not
  work as expected.
  
## [0.6.3] - 2024-12-23

### Added
- Rename eviz.py (the plotting tool) to autoviz.py
- GIF option to autoviz (specified in app YAML).
- Significant updates to streamlit interface (sviz) including
   - Add Streamlit integration and update configuration for new datasets
   - YAML display and feedback button
   - Splash screen
- Function to get project directory
- Ability to print date to figure (WRF only)
- More unit tests
- Updated documentation

### Deprecated

### Fixed
- Path issues in image generation
- Class hierarchy for NUWRF modles
- Metadump: Add ability to ignore vars in json output
- Iviz: Units label and log y of yz plot for diff plots

### Removed
- Files under lib/units 
- Iviz: Separate library from application code
   - Moved files from lib/data to models/iviz

### Known issues:
- The units module needs more testing and some unit conversions may not
  work as expected.

## [0.6.2] - 2024-09-30

### Added
- YAML file validation with more user-friendly error traceback
- CLI option to omit LOG file creation
- Ability to reference environment variables in YAML files

### Changed

### Deprecated

### Fixed
- A bug with y-ranges in zonal mean plots
- Allow multiple file input option for nc4 files
- Issues with contour level range and formatting, axes titles, and
  other minor issues

### Removed

### Known issues:
- The units module needs more testing and some unit conversions may not
  work as expected.

## [0.6.1] - 2024-09-04

### Added
- Ability to carry out approximate unit conversion that require AIRMASS field
  without user provided file

### Changed

### Deprecated

### Fixed
- Time averaging in iViz
- Bug in chem unit conversion that was repeating conversion on data after it
  was already completed

### Removed

### Known issues:
- The units module needs more testing and some unit conversions may not work as expected

## [0.6.0] - 2024-08-05

### Added
- A units module used for unit conversion of fields in comparison plots
- Added support for comparison with OMI satellite observations (SO2, O3, NO2)
  in iViz
- Add species database to access supported chemical species
- Functionality to access/process an airmass field used by the units module
- Added unit tests
- Total column computation in eViz plots
- A streamlit-based tool to allow web-based interface for static visualizations

### Changed
- Minor fixes and refactoring to metadump.py 
- Significant updates to streamlit implementation
- Update documentation

### Deprecated

### Fixed
- Tropopause height overlay which was broken since v0.3.0
- Fixed various issues with polar and scatter plots
- Fixed some issues with title positioning and axes creation
- Fixed session saving and reloading in iViz
- Fixed user plot options in iViz 
- Single python 3.10 environment working for both tools

### Removed
- overlays.py was removed and merged into processor.py

### Known issues: 
- The units module needs more testing and some unit conversions may not work as expected

## [0.5.0] - 2024-01-30

### Added
- metadump.py utility used to generate YAML configuration files

### Changed
- Rename top level directory src to eviz
- Refactor config.py (this only affects eViz)
  - Move Config() class under eviz
- Refactor plotting routines (move into separate file)
  - Move root.py to top-level models directory
- Update eViz and iViz plots
  - Update OMI, LIS and WRF readers
  - Add support for additional data sources
  - Many bug fixes
- Move geos history under separate directory
- Update streamlit implementation
- Update documentation

### Deprecated

### Fixed

### Removed

### Known issues: 
- The eViz and iViz environments are out of sync: both need slightly different environments


## [0.4.0] - 2023-06-16

### Added
- Initial support for CSV data
- AIRNOW example with basic scatter plot
- Package version attribute

### Changed
- Refactor data_source.py (this only affect eviz.py)
- Refactor lis and wrf dashboards

### Deprecated

### Fixed

### Removed

### Known bugs: 
- YZ (lat/lev) overlay tropopause line.
- Coastlines on polar plots.
- Location of image logos in some maps is not properly sized.

## [0.3.0] - 2023-05-31

### Added
- Access to Netcdf data stored on Opendap servers.
- Unit tests for lib code.
- A handful of integration tests and a script to test the eViz CLI.

### Changed
- Widgets that were in pop up panels are now in left side Tabs. 
- Renamed Frame class to Figure.
- Updated and corrected documentation.

### Deprecated

### Fixed
- Polar plots for GEOS data sources.
- YT (Lat/time) and XT (Lon/time) hov mollar plots.
- Giffable dimensions.
- Setting of plot types available for data source.
- YZ (Lat/lev) Zonal plots colorbar sharing.

### Removed

### Known bugs: 
- YZ (lat/lev) overlay tropopause line.
- Coastlines on polar plots.
- Location of image logos in some maps is not properly sized.

## [0.2.2] - 2023-02-23

### Added

### Changed

### Deprecated

### Fixed
- Add mising dependency in environment file.
- Fix relative path issue.

### Removed

### Known bugs: 

## [0.2.1] - 2023-02-21

### Added
- Enable use of environment variable for config files.
- Add use of both -s source and -i input to CLI

### Changed
- Updated user guides.
- Changed default paths in config files for using sample datasets on DISCOVER.

### Deprecated

### Fixed
- More plot issues were resolved.
- Bug fixes to LIS coastlines, differencing, colorbar sharing, LIS soil depth plots.
- Adherence to PEP8 standards.

### Removed
- config/setup_yaml.bash

### Known bugs: 

## [0.2.0] - 2023-01-25

### Added
- Comparison plots for the following models and observations
  - LIS
  - WRF
  - CF
  - AIRNOW (iViz only)
- Implementation of generic data reader allowing ingestion of
  OMI, Landsat, HDF4, and CSV data sources.
- Improved documentation using Sphinx tool with publishable project web page
- Simple config and data source tests

### Changed
- Directory structure was refactored so that lib/ and models/ directories
  are now under src/.

### Deprecated

### Fixed
- Mangled tick formats in various plots.
- Many plot issues were resolved.

### Removed
- Removed app directory

### Known Bugs
- LIS coastlines
- Session saving 
- Zonal tropopause overlay

## [0.1.0] - 2022-07-06

First eViz release. This is a developmental version not yet available for public release.
This release contains all the basic functionality to visualize Earth system model data
from selected data sources. 

### Known bugs: 
- iViz: 
   - LIS inputs in iviz.py do not work at the moment
   - Setting a projection in combination with using the regional selection tool, or 
     changing the x and y-axis limits, is currently are broken. 



