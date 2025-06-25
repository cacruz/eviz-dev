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

        box_colors = None
        if config.compare or config.compare_diff:
            box_colors = config.box_colors

        else:
            if field_name in config.spec_data and 'box_color' in config.spec_data[field_name]['boxplot']:
                box_colors = config.spec_data[field_name]['boxplot'].get('box_color', 'blue')
        
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

            if fig_input is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = fig_input
                if hasattr(fig, 'gca'):
                    ax = fig.gca()
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))

                # Standardize box_colors to always be a list
                if box_colors is None:
                    # Use default color cycle
                    box_colors = config.ax_opts.get('color_cycle', plt.rcParams['axes.prop_cycle'].by_key()['color'])
                elif isinstance(box_colors, str):
                    box_colors = [box_colors]

           
            # Check if we have multiple experiments for side-by-side box plots
            has_multiple_experiments = 'experiment' in df.columns and len(df['experiment'].unique()) > 1

            if has_multiple_experiments and hasattr(config, 'overlay') and config.overlay:
                experiments = df['experiment'].unique()
                num_experiments = len(experiments)
                
                all_categories = df[category_col].unique()
                num_categories = len(all_categories)
                self.logger.debug(f"Found {num_experiments} experiments and {num_categories} categories")   

                if category_col == 'time':
                    # Try to convert time strings to datetime for proper sorting
                    try:
                        # Convert time strings to datetime objects
                        time_categories = pd.to_datetime(all_categories)
                        # Sort and get the sorted indices
                        sorted_indices = np.argsort(time_categories)
                        # Reorder categories
                        all_categories = [all_categories[i] for i in sorted_indices]
                    except:
                        # If conversion fails, try numeric sorting
                        try:
                            # Try to convert to numeric values
                            numeric_categories = [float(cat) for cat in all_categories]
                            sorted_indices = np.argsort(numeric_categories)
                            all_categories = [all_categories[i] for i in sorted_indices]
                        except:
                            # If all else fails, use lexicographic sorting
                            all_categories = sorted(all_categories)
                
                # Create a figure with enough width
                if fig_input is None:
                    fig, ax = plt.subplots(figsize=(max(10, num_categories * 2), 6))
                
                positions = []
                box_data = []
                box_colors = []
                box_labels = []  # For debugging
                
                group_width = 0.8
                box_width = group_width / num_experiments
                
                for i, category in enumerate(all_categories):
                    category_center = i + 1  # Center position for this category group
                    
                    for j, experiment in enumerate(experiments):
                        # Calculate offset from center for this experiment's box
                        offset = (j - (num_experiments - 1) / 2) * box_width
                        position = category_center + offset
                        
                        mask = (df[category_col] == category) & (df['experiment'] == experiment)
                        values = df.loc[mask, 'value'].values
                        
                        if len(values) > 0:
                            positions.append(position)
                            box_data.append(values)
                            box_colors.append(box_colors[j % len(box_colors)])
                            box_labels.append(f"{category}_{experiment}")  # For debugging
                
                self.logger.info(f"Box positions: {positions}")
                self.logger.info(f"Box labels: {box_labels}")
                
                box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=box_width * 0.9)
                
                for i, box in enumerate(box_plot['boxes']):
                    box.set(facecolor=box_colors[i % len(box_colors)], alpha=0.7)
                
                category_centers = [i + 1 for i in range(num_categories)]
                ax.set_xticks(category_centers)
                ax.set_xticklabels(all_categories)
                
                ax.set_xlim(0.5, num_categories + 0.5)
                
                legend_handles = [plt.Rectangle((0, 0), 1, 1, color=box_colors[i % len(box_colors)], alpha=0.7) 
                                for i in range(num_experiments)]
                ax.legend(legend_handles, experiments, loc='best')

            else:
                # Standard box plot (single experiment or not overlaid)
                # Group data by category
                grouped_data = df.groupby(category_col)['value'].apply(list).to_dict()
                categories = list(grouped_data.keys())
                
                if category_col == 'time':
                    try:
                        # Convert time strings to datetime objects
                        time_categories = pd.to_datetime(categories)
                        # Sort and get the sorted indices
                        sorted_indices = np.argsort(time_categories)
                        # Reorder categories
                        sorted_categories = [categories[i] for i in sorted_indices]
                        # Reorder values to match sorted categories
                        values = [grouped_data[categories[i]] for i in sorted_indices]
                        categories = sorted_categories
                    except:
                        # If conversion fails, try numeric sorting
                        try:
                            # Try to convert to numeric values
                            numeric_categories = [float(cat) for cat in categories]
                            sorted_indices = np.argsort(numeric_categories)
                            sorted_categories = [categories[i] for i in sorted_indices]
                            values = [grouped_data[categories[i]] for i in sorted_indices]
                            categories = sorted_categories
                        except:
                            # If all else fails, use lexicographic sorting
                            sorted_categories = sorted(categories)
                            values = [grouped_data[cat] for cat in sorted_categories]
                            categories = sorted_categories
                else:
                    values = [grouped_data[cat] for cat in categories]
                
                box_plot = ax.boxplot(values, patch_artist=True, labels=categories)
                
                for i, box in enumerate(box_plot['boxes']):
                    box_color = box_colors[i % len(box_colors)]
                    box.set(facecolor=box_color, alpha=0.7)

            ax.set_title(title)
            ax.set_xlabel(category_col)
            ax.set_ylabel(f"{title} ({units})")

            ax.grid(True, linestyle='--', alpha=0.7)

            # Rotate x-axis labels if they're long or there are many categories
            if len(df[category_col].unique()) > 5 or any(len(str(cat)) > 10 for cat in df[category_col].unique()):
                plt.xticks(rotation=45, ha='right')

            fig.tight_layout()
                    
            return fig

        except Exception as e:
            self.logger.error(f"Error creating matplotlib box plot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save(self, filename, **kwargs):
        """Save the plot to a file."""
        pass

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
