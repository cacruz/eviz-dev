# Example: Running eViz with a Remote OpenDAP Data Source

This example demonstrates how to use the eViz tool to generate plots from a remote OpenDAP data source using the configuration in `config/gridded/gridded.yaml`.

## Prerequisites

- You have installed all required dependencies (see project README).
- You have access to the `autoviz.py` script and the `config/gridded/gridded.yaml` configuration file.

## Configuration

The file `config/gridded/gridded.yaml` should look like this:

```yaml
inputs:
   - name: https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc
     description: NCEP/NCAR Reanalysis Monthly Mean Air Temperature
     to_plot:
        air: xt,xy,tx

outputs:
    print_to_file: yes
    print_format: jpg
```
This configuration tells eViz to:

* Load the air.mon.mean.nc dataset from a remote OpenDAP server.
* Generate three types of plots for the air variable: xt, xy, and tx.
* Save the output plots as .jpg files.
  
## Running the Example

From the root of your project, run the following command in your terminal:

```bash
python autoviz.py -s gridded
```

## What Happens Next?

* eViz will read the configuration from config/gridded/gridded.yaml.
* It will connect to the remote OpenDAP server and load the specified dataset.
* It will generate the requested plots and save them in the output directory (by default, ./output_plots).

## Output

After running the command, you should see .jpg files for each plot type in the output_plots directory, such as:

* air_xt_0_0.jpg
* air_xy_0_0.jpg
* air_tx_0_0.jpg

## Troubleshooting

If you see errors about missing dependencies, install them with:

```bash
pip install -r requirements.txt
```

If you encounter issues with the OpenDAP connection, ensure that the URL is correct and that you have internet access.

