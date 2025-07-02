import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from .base import MatplotlibBasePlotter


class MatplotlibBoxPlotter(MatplotlibBasePlotter):
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
        data, _, _, field_name, plot_type, findex, fig = data_to_plot

        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return fig

        required_cols = {'value', 'experiment'}
        if not required_cols.issubset(data.columns):
            self.logger.warning(f"DataFrame missing required columns: "
                                f"{required_cols - set(data.columns)}")
            return fig

        self.source_name = config.source_names[config.ds_index]
        self.units = self.get_units(config, 
                                    field_name, 
                                    data, 
                                    findex)
        self.fig = fig
        ax_opts = config.ax_opts
        df = data

        if not config.compare and not config.compare_diff and not config.overlay:
            fig.set_axes()
        
        ax_temp = fig.get_axes()
        axes_shape = fig.subplots
        
        if axes_shape == (3, 1):
            if ax_opts['is_diff_field']:
                self.ax = ax_temp[2]
            else:
                self.ax = ax_temp[config.axindex]
        elif axes_shape == (2, 2):
            if ax_opts['is_diff_field']:
                self.ax = ax_temp[2]
                if config.ax_opts['add_extra_field_type']:
                    self.ax = ax_temp[3]
            else:
                self.ax = ax_temp[config.axindex]
        elif axes_shape == (1, 2) or axes_shape == (1, 3):
            if isinstance(ax_temp, list):
                self.ax = ax_temp[config.axindex]
            else:
                self.ax = ax_temp
        else:
            self.ax = ax_temp[0]

        ax_opts = fig.update_ax_opts(field_name, self.ax, 'box', level=0)
        fig.plot_text(field_name, self.ax, 'box', data=df)
        
        self._plot_box_data(config, self.ax, ax_opts, fig, df, field_name, findex)
        
        # Handle title for comparison plots
        if config.compare_diff:
            title_text = field_name
            fig.suptitle_eviz(title_text, 
                            fontweight='bold', fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))
        elif config.compare:
            title_text = config.map_params[findex].get('field', 'No name')
            if hasattr(config, 'spec_data') and field_name in config.spec_data and \
                    'name' in config.spec_data[field_name]:
                title_text = config.spec_data[field_name]['name']
            fig.suptitle_eviz(title_text, 
                            fontweight='bold', fontstyle='italic',
                            fontsize=self._image_font_size(fig.subplots))
            
            if config.add_logo:
                self._add_logo_ax(fig, desired_width_ratio=0.05)
                
        return fig
    
    def _plot_box_data(self, config, ax, ax_opts, fig, df, field_name, findex):
        """Helper method that plots the time series (xt) data."""        

        title = field_name
        if hasattr(config, 'spec_data') and \
                field_name in config.spec_data and \
                'name' in config.spec_data[field_name]:
            title = config.spec_data[field_name]['name']
        
        color_settings = None
        if config.compare or config.compare_diff or config.overlay:
            color_settings = config.box_colors
        else:
            if field_name in config.spec_data and \
                    'boxplot' in config.spec_data[field_name] and \
                    'box_color' in config.spec_data[field_name]['boxplot']:
                color_settings = config.spec_data[field_name]['boxplot'].get('box_color', 'blue')

        try:
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

            if not color_settings:
                color_settings = config.ax_opts.get('color_cycle', plt.rcParams['axes.prop_cycle'].by_key()['color'])
            elif isinstance(color_settings, str):
                color_settings = [color_settings]
            self.logger.debug(f"Using color settings: {color_settings}")

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
                
                positions = []
                box_data = []
                box_colors = []
                
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
                            
                            if len(color_settings) == 1:
                                box_colors.append(color_settings[0])
                            else:
                                box_colors.append(color_settings[j % len(color_settings)])

                exp_ids = df['experiment'].unique().tolist()
                box_data = [df[df['experiment'] == eid]['value'] for eid in exp_ids]

                box_plot = ax.boxplot(box_data, 
                            positions=positions, 
                            patch_artist=True, 
                            labels=None,  # Don't set labels here
                            widths=box_width * 0.9)
                
                for i, box in enumerate(box_plot['boxes']):
                    box.set(facecolor=box_colors[i % len(box_colors)], alpha=0.7)

                if not config.add_legend:
                    if len(all_categories) == 1:
                        # Only one category: label by experiment
                        ax.set_xticks(positions)
                        ax.set_xticklabels(experiments)
                    else:
                        # Multiple categories: label by category
                        category_centers = [i + 1 for i in range(num_categories)]
                        ax.set_xticks(category_centers)
                        ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=8)
                else:
                    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=box_colors[i % len(box_colors)], alpha=0.7) 
                                    for i in range(num_experiments)]
                    ax.legend(legend_handles, experiments, loc='best')
                    
                if config.add_legend:
                    ax.set_xlabel(category_col)
                ax.set_xlim(0.5, num_categories + 0.5)
                ax.set_ylabel(f"{title} ({self.units})")
                ax.grid(ax_opts['add_grid'], linestyle='--', alpha=0.7)

            else:
                # Standard box plot (single experiment or not overlaid)
                # Group data by category
                grouped_data = df.groupby(category_col)['value'].apply(list).to_dict()
                categories = list(grouped_data.keys())
                values = [grouped_data[cat] for cat in categories]

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

                # Format labels
                formatted_labels = []
                for cat in categories:
                    try:
                        dt = pd.to_datetime(cat)
                        formatted_labels.append(dt.strftime('%H:%M'))  # Just show hour:minute
                    except:
                        formatted_labels.append(cat)

                exp_id = df['experiment'].iloc[0] if 'experiment' in df.columns else ''
                # If only one experiment, use exp_id as label
                if 'experiment' in df.columns and df['experiment'].nunique() == 1:
                    exp_id = df['experiment'].iloc[0]
                    box_plot = ax.boxplot(values, 
                                patch_artist=True, 
                                labels=[exp_id]*len(values))
                    ax.set_xticklabels([exp_id]*len(values), fontsize=10)
                else:
                    box_plot = ax.boxplot(values, 
                                patch_artist=True, 
                                labels=categories)
                    ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=10)

                ax.set_xlabel(category_col)
                ax.set_ylabel(f"{title} ({self.units})")
                ax.grid(ax_opts['add_grid'], linestyle='--', alpha=0.7)

            if config.compare:
                fig.suptitle_eviz(text=field_name, 
                                fontweight='bold',
                                fontstyle='italic',
                                fontsize=self._image_font_size(fig.subplots))

                if config.add_logo:
                    self._add_logo_ax(fig, desired_width_ratio=0.05)

        except Exception as e:
            self.logger.error(f"Error creating matplotlib box plot: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    @staticmethod
    def format_time_label(label):
        try:
            # Try parsing as datetime
            dt = pd.to_datetime(label)
            # Option 1: '2018-05-23 06'
            return dt.strftime('%Y-%m-%d %H')
            # Option 2: '05-23 06'
            # return dt.strftime('%m-%d %H')
        except Exception:
            return label[:13]  # fallback
