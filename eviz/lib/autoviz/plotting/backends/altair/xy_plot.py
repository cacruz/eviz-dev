import numpy as np
import pandas as pd
import altair as alt
from eviz.lib.autoviz.plotting.base import XYPlotter


class AltairXYPlotter(XYPlotter):
    """Altair implementation of XY plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        alt.renderers.enable('default')
        alt.data_transformers.disable_max_rows()

    def plot(self, config, data_to_plot):
        """Create an interactive XY plot using Altair.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data2d, x, y, field_name, plot_type, findex, fig)
        
        Returns:
            The created Altair chart object
        """
        data2d, x, y, field_name, plot_type, findex, _ = data_to_plot
        
        if data2d is None:
            return None
        
        ax_opts = config.ax_opts
        
        if 'fill_value' in config.spec_data[field_name]['xyplot']:
            fill_value = config.spec_data[field_name]['xyplot']['fill_value']
            data2d = data2d.where(data2d != fill_value, np.nan)
        
        # Convert xarray DataArray to pandas DataFrame for Altair
        df = self._convert_to_dataframe(data2d, x, y)
        
        self.logger.debug(f"DataFrame shape: {df.shape}")
        self.logger.debug(f"DataFrame columns: {df.columns}")
        
        if df.empty or 'value' not in df.columns:
            self.logger.error("DataFrame is empty or missing 'value' column")
            # Create a simple chart with a message
            chart = alt.Chart(pd.DataFrame({'x': [0], 'y': [0], 'text': ['No data to display']})).mark_text(
                align='center',
                baseline='middle',
                fontSize=20
            ).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=[-1, 1]), axis=None),
                y=alt.Y('y:Q', scale=alt.Scale(domain=[-1, 1]), axis=None),
                text='text:N'
            ).properties(
                width=800,
                height=500,
                title=field_name
            )
            self.plot_object = chart
            return chart
        
        self.logger.debug(f"DataFrame value range: {df['value'].min()} to {df['value'].max()}")

        # Get colormap - convert matplotlib colormap to Vega colormap
        cmap = ax_opts.get('use_cmap', 'viridis')
        # Temporary: Map common matplotlib colormaps to Vega schemes
        cmap_mapping = {
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'magma': 'magma',
            'cividis': 'cividis',
            'rainbow': 'rainbow',
            'jet': 'rainbow',  # Approximate
            'Blues': 'blues',
            'Reds': 'reds',
            'Greens': 'greens',
            'YlOrRd': 'yelloworangered',
            'RdBu': 'redblue',
            'coolwarm': 'redblue'
        }
        vega_scheme = cmap_mapping.get(cmap, 'viridis')
        self.logger.debug(f"Using colormap: {cmap} -> {vega_scheme}")
        
        # Get title
        title = field_name
        if 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "n.a."
        if 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        elif hasattr(data2d, 'attrs') and 'units' in data2d.attrs:
            units = data2d.attrs['units']
        elif hasattr(data2d, 'units'):
            units = data2d.units
        
        # Convert xarray DataArray to pandas DataFrame for Altair
        df = self._convert_to_dataframe(data2d, x, y)
        
        self.logger.debug(f"DataFrame shape: {df.shape}")
        self.logger.debug(f"DataFrame columns: {df.columns}")
        self.logger.debug(f"DataFrame value range: {df['value'].min()} to {df['value'].max()}")
        
        if df.empty:
            self.logger.error("DataFrame is empty after processing")
            return None
        
        if df['value'].isna().all():
            self.logger.error("All values are NaN")
            return None
        
        # Check for extreme outliers that might skew the color scale
        q1 = df['value'].quantile(0.01)
        q3 = df['value'].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
        if not outliers.empty:
            self.logger.debug(f"Found {len(outliers)} outliers outside range [{lower_bound}, {upper_bound}]")
        
        if ax_opts.get('clevs') is not None:
            color_domain = ax_opts['clevs']
            self.logger.debug(f"Using specified contour levels for color domain: {color_domain}")
        else:
            # Use extent
            vmin = df['value'].min()
            vmax = df['value'].max()
            
            # If there are extreme outliers, consider using robust bounds
            if len(outliers) > 0.05 * len(df):  # If more than 5% are outliers
                vmin = max(lower_bound, vmin)
                vmax = min(upper_bound, vmax)
                self.logger.debug(f"Using robust color domain: [{vmin}, {vmax}]")
            
            color_domain = [vmin, vmax]
        
        try:
            x_name = x.name if hasattr(x, 'name') else 'X'
            y_name = y.name if hasattr(y, 'name') else 'Y'
                        
            # TODO: needs testing
            y_sort = 'ascending'
            
            chart = alt.Chart(df).mark_square(size=100).encode(
                x=alt.X('x:Q', 
                       title=x_name,
                       scale=alt.Scale(domain=[df['x'].min(), df['x'].max()], nice=False)),
                y=alt.Y('y:Q', 
                       title=y_name,
                       sort=y_sort,
                       scale=alt.Scale(domain=[df['y'].min(), df['y'].max()], nice=False)),
                color=alt.Color('value:Q', 
                               scale=alt.Scale(scheme=vega_scheme, domain=color_domain),
                               title=f"{field_name} ({units})")
            ).properties(
                width=800,
                height=500,
                title=title
            ).interactive()
            
            chart = chart.encode(
                tooltip=[
                    alt.Tooltip('x:Q', title=x_name),
                    alt.Tooltip('y:Q', title=y_name),
                    alt.Tooltip('value:Q', title=field_name, format='.3f')
                ]
            )
            
            if ax_opts.get('line_contours', False) and ax_opts.get('clevs') is not None:
                levels = ax_opts['clevs']
                
                contour_chart = alt.Chart(df).mark_square(size=100).encode(
                    x=alt.X('x:Q', title=x_name),
                    y=alt.Y('y:Q', title=y_name, sort=y_sort),
                    color=alt.Color('value:Q', 
                                   scale=alt.Scale(domain=list(levels), scheme=vega_scheme),
                                   title=f"{field_name} ({units})")
                ).properties(
                    width=800,
                    height=500
                ).interactive()
                
                # Layer the contour chart on top of the heatmap
                chart = alt.layer(chart, contour_chart)
        
        except Exception as e:
            self.logger.error(f"Error creating Altair chart: {e}")
            chart = alt.Chart(df).mark_point().encode(
                x='x:Q',
                y='y:Q',
                color='value:Q'
            ).properties(
                width=800,
                height=500,
                title=f"{title} (fallback visualization)"
            )
        
        self.plot_object = chart
        
        return chart

    def _convert_to_dataframe(self, data2d, x, y):
        """Convert xarray DataArray to pandas DataFrame for Altair.
        
        Args:
            data2d: xarray DataArray with 2D data
            x: x-coordinate values
            y: y-coordinate values
            
        Returns:
            pandas DataFrame with columns 'x', 'y', 'value'
        """
        try:
            x_vals = x.values if hasattr(x, 'values') else x
            y_vals = y.values if hasattr(y, 'values') else y
            
            if isinstance(data2d, pd.DataFrame):
                if 'x' in data2d.columns and 'y' in data2d.columns:
                    if 'value' not in data2d.columns:
                        # Try to find a value column
                        value_cols = [col for col in data2d.columns 
                                    if col not in ['x', 'y']]
                        if value_cols:
                            data2d = data2d.rename(columns={value_cols[0]: 'value'})
                        else:
                            # Create a dummy value column
                            data2d['value'] = 0
                    return data2d
            
            data_values = data2d.values if hasattr(data2d, 'values') else data2d
                       
            if np.isnan(data_values).all():
                self.logger.warning("All values are NaN, creating empty DataFrame")
                return pd.DataFrame(columns=['x', 'y', 'value'])
            
            self.logger.debug(f"Data values range: {np.nanmin(data_values)} to {np.nanmax(data_values)}")
            self.logger.debug(f"Data values NaN count: {np.isnan(data_values).sum()}")
            
            rows = []
            for j, y_val in enumerate(y_vals):
                for i, x_val in enumerate(x_vals):
                    if j < data_values.shape[0] and i < data_values.shape[1]:
                        value = data_values[j, i]
                        if not np.isnan(value):
                            rows.append({'x': x_val, 'y': y_val, 'value': value})
            
            df = pd.DataFrame(rows)
            
            # If DataFrame is empty, create a minimal one with a single point
            if df.empty:
                self.logger.warning("Created empty DataFrame, adding a single point")
                df = pd.DataFrame([{'x': x_vals[0], 'y': y_vals[0], 'value': 0}])
            
            # Log the result
            self.logger.debug(f"Created DataFrame with {len(df)} rows")
            
            return df

        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            # Create a minimal DataFrame as fallback
            return pd.DataFrame({
                'x': [0, 1],
                'y': [0, 1],
                'value': [0, 1]
            })
    
    def show(self):
        """Display the plot."""
        pass

    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass
