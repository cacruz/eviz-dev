import logging
import warnings
from typing import Any, Union
import pandas as pd
import xarray as xr
from dataclasses import dataclass, field
import eviz.lib.autoviz.utils as pu
from eviz.lib.autoviz.figure import Figure
from eviz.models.obs_source import ObsSource

warnings.filterwarnings("ignore")


@dataclass
class Ghg(ObsSource):
    """
    Define Greenhouse Gas (GHG) inventory data and functions.
    
    This class handles time series data for greenhouse gases typically stored in CSV format.
    It supports annual or monthly data with uncertainty values and can generate various
    time series visualizations.
    
    Attributes:
        source_data: The loaded data
        _ds_attrs: Dataset attributes
        _maps_params: Mapping parameters
        frame_params: Frame parameters for plotting
    """
    source_data: Any = None
    _ds_attrs: dict = field(default_factory=dict)
    _maps_params: dict = field(default_factory=dict)
    frame_params: Any = None

    def __post_init__(self):
        self.logger.info("Start init")
        super().__post_init__()

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        return logging.getLogger(__name__)
    
    def add_data_source(self, file_path, data_source):
        """Add a data source to the model."""
        # Store the data source if needed
        if hasattr(data_source, 'dataset'):
            self.source_data = data_source.dataset
        else:
            self.source_data = data_source

    def get_data_source(self, file_path):
        """Get a data source from the model."""
        return self.config_manager.pipeline.get_data_source(file_path)

    def load_data_sources(self):
        """Load data sources for the model."""
        pass  # Handled by the pipeline

    def process_data(self, filename: str, field_name: str) -> Union[xr.Dataset, None]:
        """
        Process CSV data for a specific field from a file.
        
        Args:
            filename: The name of the file
            field_name: The name of the field (column in CSV)
            
        Returns:
            xr.Dataset: The processed data as an xarray Dataset
        """
        data_source = self.config_manager.pipeline.get_data_source(filename)
        if not data_source or not hasattr(data_source, 'dataset'):
            self.logger.error(f"No data source available for {filename}")
            return None
            
        # For CSV data, we need to convert pandas DataFrame to xarray Dataset
        if isinstance(data_source.dataset, pd.DataFrame):
            df = data_source.dataset
            
            # Check if the field exists in the DataFrame
            if field_name not in df.columns:
                self.logger.error(f"Field {field_name} not found in {filename}")
                return None
                
            # Create an xarray Dataset from the DataFrame
            # Assuming the first column is the time dimension (year or date)
            time_col = df.columns[0]
            
            # Create coordinates
            coords = {time_col: df[time_col].values}
            
            # Create data variables
            data_vars = {}
            for col in df.columns[1:]:  # Skip the time column
                data_vars[col] = xr.DataArray(
                    data=df[col].values,
                    dims=[time_col],
                    coords=coords,
                    attrs={'units': self._infer_units(col), 'long_name': col}
                )
            
            # Create the dataset
            ds = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs={'title': f'GHG data from {filename}'}
            )
            
            return ds
        else:
            # If it's already an xarray Dataset, return it
            return data_source.dataset

    @staticmethod
    def _infer_units(column_name: str) -> str:
        """
        Infer units from column name.
        
        Args:
            column_name: The name of the column
            
        Returns:
            str: The inferred units
        """
        # Add more mappings as needed
        unit_mappings = {
            'co2': 'ppm',
            'ch4': 'ppb',
            'n2o': 'ppb',
            'sf6': 'ppt',
            'inc': 'ppm/year',
            'unc': 'ppm/year',
            'growth': '%',
            'concentration': 'ppm',
        }
        
        for key, unit in unit_mappings.items():
            if key.lower() in column_name.lower():
                return unit
                
        return 'n.a.'  # Default if no match

    def process_simple_plots(self, plotter):
        """
        Generate simple plots for all fields in the dataset when no SPECS file is provided.
        
        Args:
            plotter: The plotter object to use for generating plots
        """
        map_params = self.config_manager.map_params
        field_num = 0
        self.config_manager.findex = 0
        
        for i in map_params.keys():
            field_name = map_params[i]['field']
            source_name = map_params[i]['source_name']
            self.source_name = source_name
            filename = map_params[i]['filename']
            file_index = self.config_manager.get_file_index(filename)
            
            self.source_data = self.process_data(filename, field_name)
            if self.source_data is None:
                continue
                
            self.config_manager.findex = file_index
            self.config_manager.pindex = field_num
            self.config_manager.axindex = 0
            
            for pt in map_params[i]['to_plot']:
                self.logger.info(f"Plotting {field_name}, {pt} plot")
                field_to_plot = self._get_field_for_simple_plot(field_name, pt)
                plotter.simple_plot(self.config_manager, field_to_plot)
            field_num += 1

    def _get_field_for_simple_plot(self, field_name: str, plot_type: str) -> Union[tuple, None]:
        """
        Prepare data for simple plots without SPECS file.
        
        Args:
            field_name: The name of the field
            plot_type: The type of plot to generate
            
        Returns:
            tuple: A tuple containing the data, coordinates, field name, and plot type
        """
        if self.source_data is None or field_name not in self.source_data:
            self.logger.error(f"Field {field_name} not found in source data")
            return None
            
        data = self.source_data[field_name]
        
        # For time series data (most common for GHG)
        if 'xt' in plot_type:
            # Get the time dimension (usually 'year' or 'time')
            time_dim = list(data.coords.keys())[0]
            time_values = data.coords[time_dim].values
            
            # Check if uncertainty data is available
            unc_field = None
            for potential_unc in ['unc', 'uncertainty', f'{field_name}_unc']:
                if potential_unc in self.source_data:
                    unc_field = potential_unc
                    break
            
            # Return the data for time series plot
            return data, time_values, None, field_name, plot_type
            
        # For bar plots
        elif 'bar' in plot_type:
            time_dim = list(data.coords.keys())[0]
            time_values = data.coords[time_dim].values
            return data, time_values, None, field_name, plot_type
            
        # Default to time series for GHG data
        else:
            self.logger.warning(f"Plot type {plot_type} may not be suitable for GHG data, defaulting to time series")
            time_dim = list(data.coords.keys())[0]
            time_values = data.coords[time_dim].values
            return data, time_values, None, field_name, 'xt'

    def process_single_plots(self):
        """
        Generate single plots for each source and field according to configuration.
        """
        self.logger.info("Generating single plots")

        all_data_sources = self.config_manager.pipeline.get_all_data_sources()
        if not all_data_sources:
            self.logger.error("No data sources available for single plotting.")
            return

        for idx, params in self.config_manager.map_params.items():
            field_name = params.get('field')
            if not field_name:
                continue

            filename = params.get('filename')
            
            processed_data = self.process_data(filename, field_name)
            if processed_data is None:
                continue
                
            self.config_manager.findex = idx
            self.config_manager.pindex = idx
            self.config_manager.axindex = 0

            if field_name in processed_data:
                field_data_array = processed_data[field_name]
            else:
                self.logger.error(f"Field {field_name} not found in processed data")
                continue
                
            plot_types = params.get('to_plot', ['xt'])  # Default to time series for GHG data
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
                
            for plot_type in plot_types:
                self._process_plot(field_data_array, field_name, idx, plot_type, processed_data)

        if self.config_manager.make_gif:
            pu.create_gif(self.config_manager.config)

    def _process_plot(self, data_array, field_name, file_index, plot_type, full_dataset=None):
        """
        Process a single plot type for a given field.
        
        Args:
            data_array: The data array to process
            field_name: The name of the field
            file_index: The index of the file
            plot_type: The type of plot to generate
            plotter: The plotter object to use for generating plots
            full_dataset: The full dataset (for accessing uncertainty data)
        """
        self.logger.info(f"Plotting {field_name}, {plot_type} plot")
        figure = Figure.create_eviz_figure(self.config_manager, plot_type)
        self.config_manager.ax_opts = figure.init_ax_opts(field_name)

        field_to_plot = self._prepare_field_to_plot(data_array, field_name, file_index,
                                                    plot_type, figure, full_dataset=full_dataset)
        if field_to_plot:
                plot_result = self.create_plot(field_name, field_to_plot)
                pu.print_map(self.config_manager, plot_type, self.config_manager.findex, plot_result)

    def _prepare_field_to_plot(self, data_array, field_name, file_index, plot_type, figure, time_level=None, full_dataset=None):
        """
        Prepare the data array and coordinates for plotting.
        
        Args:
            data_array: The data array to process
            field_name: The name of the field
            file_index: The index of the file
            plot_type: The type of plot to generate
            figure: The figure object
            time_level: The time level to use (not typically used for GHG data)
            full_dataset: The full dataset (for accessing uncertainty data)
            
        Returns:
            tuple: A tuple containing the data, coordinates, field name, plot type, file index, and figure
        """
        if data_array is None:
            self.logger.error(f"No data array provided for field {field_name}")
            return None

        # For time series plots (most common for GHG data)
        if 'xt' in plot_type:
            # Get the time dimension (usually 'year' or 'time')
            time_dim = list(data_array.coords.keys())[0]
            time_values = data_array.coords[time_dim].values
            
            # Check if uncertainty data is available
            unc_data = None
            if full_dataset is not None:
                for potential_unc in ['unc', 'uncertainty', f'{field_name}_unc']:
                    if potential_unc in full_dataset:
                        unc_data = full_dataset[potential_unc]
                        break
            
            # If we have uncertainty data, we can add it to the plot later
            if unc_data is not None:
                # Store uncertainty data for use in the plotter
                self.config_manager.ax_opts['uncertainty_data'] = unc_data
            
            # Return the data for time series plot
            return data_array, time_values, None, field_name, plot_type, file_index, figure
            
        # For bar plots
        elif 'bar' in plot_type:
            time_dim = list(data_array.coords.keys())[0]
            time_values = data_array.coords[time_dim].values
            return data_array, time_values, None, field_name, plot_type, file_index, figure
            
        # Default to time series for GHG data
        else:
            self.logger.warning(f"Plot type {plot_type} may not be suitable for GHG data, defaulting to time series")
            time_dim = list(data_array.coords.keys())[0]
            time_values = data_array.coords[time_dim].values
            return data_array, time_values, None, field_name, 'xt', file_index, figure

    def process_comparison_plots(self, plotter):
        """
        Generate comparison plots for paired data sources.
        
        Args:
            plotter: The plotter object to use for generating plots
        """
        self.logger.info("Generating comparison plots for GHG data")
        
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform comparison.")
            return
            
        # Get the file indices for the two files being compared
        idx1 = self.config_manager.a_list[0]
        idx2 = self.config_manager.b_list[0]
        
        # Get the field names for each file
        fields_file1 = [params['field'] for i, params in 
                        self.config_manager.map_params.items() if 
                        params['file_index'] == idx1]
        fields_file2 = [params['field'] for i, params in 
                        self.config_manager.map_params.items() if 
                        params['file_index'] == idx2]
        
        # Pair fields by order
        num_pairs = min(len(fields_file1), len(fields_file2))
        field_pairs = list(zip(fields_file1[:num_pairs], fields_file2[:num_pairs]))
        
        # Process each field pair
        for field1, field2 in field_pairs:
            # Find map_params for this field in both files
            idx1_field = next((i for i, params in self.config_manager.map_params.items()
                              if params['file_index'] == idx1 and params['field'] == field1), None)
            idx2_field = next((i for i, params in self.config_manager.map_params.items()
                              if params['file_index'] == idx2 and params['field'] == field2), None)
            
            if idx1_field is None or idx2_field is None:
                continue
                
            # Get the filenames
            filename1 = self.config_manager.map_params[idx1_field]['filename']
            filename2 = self.config_manager.map_params[idx2_field]['filename']
            
            # Process the data
            data1 = self.process_data(filename1, field1)
            data2 = self.process_data(filename2, field2)
            
            if data1 is None or data2 is None:
                continue
                
            # Get the plot types
            plot_types = self.config_manager.map_params[idx1_field].get('to_plot', ['xt'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
                
            # Process each plot type
            for plot_type in plot_types:
                self.logger.info(f"Plotting comparison of {field1} vs {field2}, {plot_type} plot")
                
                # Create a figure for the comparison
                figure = Figure.create_eviz_figure(self.config_manager, plot_type, 
                                                 nrows=2, ncols=1)
                figure.set_axes()
                
                # Plot the first dataset
                self.config_manager.findex = idx1_field
                self.config_manager.pindex = 0
                self.config_manager.axindex = 0
                self.config_manager.ax_opts = figure.init_ax_opts(field1)
                
                field_to_plot1 = self._prepare_field_to_plot(data1[field1], field1, idx1_field,
                                                             plot_type, figure, full_dataset=data1)
                if field_to_plot1:
                    plotter.comparison_plots(self.config_manager, field_to_plot1)
                
                # Plot the second dataset
                self.config_manager.findex = idx2_field
                self.config_manager.pindex = 0
                self.config_manager.axindex = 1
                self.config_manager.ax_opts = figure.init_ax_opts(field2)
                
                field_to_plot2 = self._prepare_field_to_plot(data2[field2], field2, idx2_field,
                                                             plot_type, figure, full_dataset=data2)
                if field_to_plot2:
                    plotter.comparison_plots(self.config_manager, field_to_plot2)
                
                # Print the map
                pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)

    def process_side_by_side_plots(self, plotter):
        """
        Generate side-by-side comparison plots for paired data sources.
        
        Args:
            plotter: The plotter object to use for generating plots
        """
        self.logger.info("Generating side-by-side plots for GHG data")
        
        if not self.config_manager.a_list or not self.config_manager.b_list:
            self.logger.error("a_list or b_list is empty, cannot perform side-by-side comparison.")
            return
            
        # Get the file indices for the two files being compared
        idx1 = self.config_manager.a_list[0]
        idx2 = self.config_manager.b_list[0]
        
        # Get the field names for each file
        fields_file1 = [params['field'] for i, params in 
                        self.config_manager.map_params.items() if 
                        params['file_index'] == idx1]
        fields_file2 = [params['field'] for i, params in 
                        self.config_manager.map_params.items() if 
                        params['file_index'] == idx2]
        
        # Pair fields by order
        num_pairs = min(len(fields_file1), len(fields_file2))
        field_pairs = list(zip(fields_file1[:num_pairs], fields_file2[:num_pairs]))
        
        # Process each field pair
        for field1, field2 in field_pairs:
            # Find map_params for this field in both files
            idx1_field = next((i for i, params in self.config_manager.map_params.items()
                              if params['file_index'] == idx1 and params['field'] == field1), None)
            idx2_field = next((i for i, params in self.config_manager.map_params.items()
                              if params['file_index'] == idx2 and params['field'] == field2), None)
            
            if idx1_field is None or idx2_field is None:
                continue
                
            # Get the filenames
            filename1 = self.config_manager.map_params[idx1_field]['filename']
            filename2 = self.config_manager.map_params[idx2_field]['filename']
            
            # Process the data
            data1 = self.process_data(filename1, field1)
            data2 = self.process_data(filename2, field2)
            
            if data1 is None or data2 is None:
                continue
                
            # Get the plot types
            plot_types = self.config_manager.map_params[idx1_field].get('to_plot', ['xt'])
            if isinstance(plot_types, str):
                plot_types = [pt.strip() for pt in plot_types.split(',')]
                
            # Process each plot type
            for plot_type in plot_types:
                self.logger.info(f"Plotting side-by-side {field1} vs {field2}, {plot_type} plot")
                
                # Create a figure for the side-by-side comparison
                figure = Figure.create_eviz_figure(self.config_manager, plot_type, 
                                                 nrows=1, ncols=2)
                figure.set_axes()
                
                # Plot the first dataset
                self.config_manager.findex = idx1_field
                self.config_manager.pindex = 0
                self.config_manager.axindex = 0
                self.config_manager.ax_opts = figure.init_ax_opts(field1)
                
                field_to_plot1 = self._prepare_field_to_plot(data1[field1], field1, idx1_field,
                                                             plot_type, figure, full_dataset=data1)
                if field_to_plot1:
                    plotter.comparison_plots(self.config_manager, field_to_plot1)
                
                # Plot the second dataset
                self.config_manager.findex = idx2_field
                self.config_manager.pindex = 0
                self.config_manager.axindex = 1
                self.config_manager.ax_opts = figure.init_ax_opts(field2)
                
                field_to_plot2 = self._prepare_field_to_plot(data2[field2], field2, idx2_field,
                                                             plot_type, figure, full_dataset=data2)
                if field_to_plot2:
                    plotter.comparison_plots(self.config_manager, field_to_plot2)
                
                # Print the map
                pu.print_map(self.config_manager, plot_type, self.config_manager.findex, figure)
