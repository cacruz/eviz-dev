import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from eviz.lib.autoviz.plotting.base import BoxPlotter


class MatplotlibBoxPlotter(BoxPlotter):
    """Matplotlib implementation of Box plotting."""
    
    def __init__(self):
        super().__init__()
        self.plot_object = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def plot(self, config, data_to_plot):
        """Create a box plot using Matplotlib.
        
        Args:
            config: Configuration manager
            data_to_plot: Tuple containing (data, categories, values, field_name,
                          plot_type, findex, fig)
        
        Returns:
            The created Matplotlib figure and axes
        """
        data, _, _, field_name, plot_type, findex, fig_input = data_to_plot
         
        if data is None:
            self.logger.warning("No data to plot")
            return None
        
        if isinstance(data, pd.DataFrame):
            df = data
            self.logger.debug(f"Using provided DataFrame with {len(df)} rows")
        else:
            self.logger.warning("Expected DataFrame for box plot, got something else")
            return None
        
        title = field_name
        if hasattr(config, 'spec_data') and field_name in config.spec_data and 'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        units = "n.a."
        if hasattr(config, 'spec_data') and field_name in config.spec_data and 'units' in config.spec_data[field_name]:
            units = config.spec_data[field_name]['units']
        
        try:
            # Determine the category column
            category_col = None
            if 'time' in df.columns:
                category_col = 'time'
            elif 'category' in df.columns:
                category_col = 'category'
            else:
                # Use the first column that's not 'value'
                for col in df.columns:
                    if col != 'value':
                        category_col = col
                        break
            
            if category_col is None:
                self.logger.warning("No category column found for box plot")
                return None
            
            if fig_input is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = fig_input
                if hasattr(fig, 'gca'):
                    ax = fig.gca()
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
            
            color_cycle = config.ax_opts.get('color_cycle', plt.rcParams['axes.prop_cycle'].by_key()['color'])
            
            # Group data by category
            grouped_data = df.groupby(category_col)['value'].apply(list).to_dict()
            categories = list(grouped_data.keys())
            values = [grouped_data[cat] for cat in categories]
            
            box_plot = ax.boxplot(values, patch_artist=True, labels=categories)
            
            for i, box in enumerate(box_plot['boxes']):
                box_color = color_cycle[i % len(color_cycle)]
                box.set(facecolor=box_color, alpha=0.7)
                
            ax.set_title(title)
            ax.set_xlabel(category_col)
            ax.set_ylabel(f"{title} ({units})")
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels if they're long or there are many categories
            if len(categories) > 5 or any(len(str(cat)) > 10 for cat in categories):
                plt.xticks(rotation=45, ha='right')
            
            fig.tight_layout()
            
            self.plot_object = (fig, ax)
            
            return (fig, ax)
            
        except Exception as e:
            self.logger.error(f"Error creating matplotlib box plot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        if self.plot_object is not None:
            try:
                fig, ax = self.plot_object
                
                # Get DPI from kwargs or use default
                dpi = kwargs.get('dpi', 300)
                
                # Save figure
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                self.logger.info(f"Saved box plot to {filename}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
        else:
            self.logger.warning("No plot to save")

    def show(self):
        """Display the plot."""
        if self.plot_object is not None:
            try:
                fig, ax = self.plot_object
                plt.show()
            except Exception as e:
                self.logger.error(f"Error showing plot: {e}")
        else:
            self.logger.warning("No plot to show")
