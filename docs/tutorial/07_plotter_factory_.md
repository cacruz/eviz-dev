# Chapter 7: Plotter Factory

Welcome back! In [Chapter 6: Plotter Abstraction](06_plotter_abstraction_.md), we learned about the **Plotter Abstraction**, which gave us a standard way (a universal instruction manual) to interact with any plotting tool, regardless of whether it draws an XY map, an XT time series, or uses Matplotlib or HvPlot. We saw the `BasePlotter` abstract class and how different plot types and backend-specific classes inherit from it.

But even with that standard interface, we still have a question: how does eViz automatically get the *correct, specific* Plotter object it needs? For example, when the configuration says to plot temperature with `plot_type: xy` and `plotting_backend: matplotlib`, how does eViz know to create an instance of `MatplotlibXYPlotter` specifically?

This is where the **Plotter Factory** comes in.

## What is the Plotter Factory?

Let's extend our drawing tool analogy. The Plotter Abstraction is like agreeing on standard actions for *any* drawing tool: `plot()`, `save()`, `show()`. The problem is, when you go to the tool shed, you don't just grab a generic "Plotter" – you need a *specific* tool: a **Matplotlib XY Plotter** for drawing maps with Matplotlib, or an **HvPlot XT Plotter** for interactive time series.

The **Plotter Factory** is like the **tool request counter** in the shed. You go to the counter and say, "I need a tool to draw an **XY map** using the **Matplotlib brand**." The person at the counter (the Factory) doesn't ask you how to *use* the tool (that's the job of the Plotter Abstraction and the tool itself), they just look up which tool matches your request, find it, and hand you the correct one, ready to go.

The Plotter Factory's job is simple but crucial:
1.  Receive a request specifying the **type of plot** you need (e.g., `xy`, `xt`, `sc`).
2.  Receive a request specifying the **plotting backend** you want to use (e.g., `matplotlib`, `hvplot`, `altair`).
3.  Look up which specific Plotter class corresponds to that exact combination.
4.  Create an instance of that specific class.
5.  Return the created Plotter object to the part of eViz that needs it.

This keeps the code that needs a plotter very clean. It doesn't have to import and check for dozens of different specific plotter classes. It just interacts with the Factory.

## Our Use Case: Getting the Right Plotter for Temperature

Remember our config snippet:

```yaml
# --- Snippet from a config file ---
inputs:
  # ... data source definition ...
  to_plot:
    temperature: xy # Need an 'xy' plot for 'temperature'
# ...
outputs:
  plotting_backend: matplotlib # Use 'matplotlib' backend
  # ... other settings ...
```

When eViz is processing this instruction to plot 'temperature', it knows it needs:
*   Data for 'temperature' (already loaded and processed by the [Data Processing Pipeline](05_data_processing_pipeline_.md)).
*   A plot of type `xy`.
*   Using the `matplotlib` backend.

At the point where the system needs to create the actual plotting object, it will ask the Plotter Factory: "Please give me a plotter for type 'xy' and backend 'matplotlib'."

The Plotter Factory will know that this combination requires the `MatplotlibXYPlotter` class, create an instance of it, and return it.

## How the Plotter Factory Works (High-Level)

Let's trace this request for the 'temperature' XY map using Matplotlib:

```{mermaid}
sequenceDiagram
    participant PlottingLogic as Plotting Logic (e.g., within a Model)
    participant PlotterFactory as Plotter Factory
    participant MatplotlibXYPlotter as MatplotlibXYPlotter

    PlottingLogic->>PlotterFactory: "Need Plotter for type 'xy', backend 'matplotlib'"
    PlotterFactory->>PlotterFactory: Look up ('xy', 'matplotlib')
    PlotterFactory->>MatplotlibXYPlotter: Create new MatplotlibXYPlotter()
    MatplotlibXYPlotter-->>PlotterFactory: Return MatplotlibXYPlotter object
    PlotterFactory-->>PlottingLogic: Return MatplotlibXYPlotter object

    PlottingLogic->>MatplotlibXYPlotter: Call plot(config, data)
    MatplotlibXYPlotter->>MatplotlibXYPlotter: (Uses Matplotlib internally)
    MatplotlibXYPlotter-->>PlottingLogic: Plotting step complete

    PlottingLogic->>MatplotlibXYPlotter: Call save(filename)
    MatplotlibXYPlotter->>MatplotlibXYPlotter: (Uses Matplotlib internally)
    MatplotlibXYPlotter-->>PlottingLogic: Saving step complete
```

The `Plotting Logic` (which lives elsewhere, perhaps in the [Source/Model Specific Logic](08_source_model_specific_logic_.md)) doesn't directly create `MatplotlibXYPlotter`. It calls the `PlotterFactory`, provides the required `plot_type` and `backend`, and receives a Plotter object. It then uses this object via the standard methods (`plot`, `save`, etc.) learned in [Chapter 6: Plotter Abstraction](06_plotter_abstraction_.md).

## Inside the Code: The `PlotterFactory` Class

The `PlotterFactory` class is found in `eviz/lib/autoviz/plotting/factory.py`. It's quite simple because its main job is just the lookup and creation.

Let's look at a simplified version of the `PlotterFactory` and its `create_plotter` method:

```python
# --- File: eviz/lib/autoviz/plotting/factory.py (simplified) ---

# Import all the specific plotter classes
from .backends.matplotlib.xy_plot import MatplotlibXYPlotter
from .backends.matplotlib.xt_plot import MatplotlibXTPlotter
# ... import other matplotlib plotters ...
from .backends.hvplot.xy_plot import HvplotXYPlotter
from .backends.hvplot.xt_plot import HvplotXTPlotter
# ... import other hvplot plotters ...
# ... import altair plotters ...

class PlotterFactory:
    """Factory for creating appropriate plotters."""

    @staticmethod # This means you can call it like PlotterFactory.create_plotter(...)
    def create_plotter(plot_type, backend="matplotlib"):
        """Create a plotter for the given plot type and backend."""

        # The heart of the factory: a dictionary mapping (type, backend) to the Class
        plotters_registry = {
            ("xy", "matplotlib"): MatplotlibXYPlotter, # XY plots with Matplotlib
            ("xt", "matplotlib"): MatplotlibXTPlotter, # XT plots with Matplotlib
            # ... other Matplotlib mappings ...

            ("xy", "hvplot"): HvplotXYPlotter,       # XY plots with HvPlot
            ("xt", "hvplot"): HvplotXTPlotter,       # XT plots with HvPlot
            # ... other HvPlot mappings ...

            # ... Altair mappings ...
        }

        # Create a key based on the requested type and backend
        key = (plot_type, backend)

        # Look up the class in our dictionary
        if key in plotters_registry:
            # If found, get the Class...
            plotter_class = plotters_registry[key]
            # ...create an instance of that Class...
            plotter_instance = plotter_class() # Calls the __init__ of the specific plotter
            # ...and return the instance!
            return plotter_instance
        else:
            # If the combination isn't in our dictionary, we don't support it
            raise ValueError(f"No plotter available for plot_type={plot_type}, backend={backend}")

```

Let's break this down:

1.  **Imports:** The factory needs to know about all the specific plotter classes it might be asked to create, so they are imported at the top.
2.  **`@staticmethod`:** The `create_plotter` method is marked `@staticmethod`. This means you don't need to create a `PlotterFactory` object first; you can call the method directly on the class like `PlotterFactory.create_plotter(...)`.
3.  **`plotters_registry`:** This dictionary is the core lookup mechanism. The *keys* are tuples containing the requested `plot_type` and `backend` strings (e.g., `("xy", "matplotlib")`). The *values* are the actual Python **class objects** themselves (e.g., `MatplotlibXYPlotter`).
4.  **Lookup:** The code creates the lookup `key` from the input arguments `plot_type` and `backend`. It then checks if this `key` exists in the `plotters_registry` dictionary.
5.  **Creation:** If the key is found, it retrieves the corresponding `plotter_class` from the dictionary. It then creates a new *instance* of that class by calling `plotter_class()` (which in turn calls the `__init__` method of that specific plotter).
6.  **Return:** The newly created `plotter_instance` is returned.
7.  **Error Handling:** If the requested `plot_type` and `backend` combination isn't found in the dictionary, it means eViz doesn't have a plotter for that combination, and a `ValueError` is raised.

This simple structure effectively separates the "decision of which plotter to use" from the "code that uses the plotter".

## Where is the Plotter Factory Used?

The `PlotterFactory.create_plotter` method is called by the part of eViz that is responsible for generating plots based on the configuration. This is often within the **Source/Model Specific Logic** ([Chapter 8](08_source_model_specific_logic_.md)) classes, like `GriddedModel`.

These classes iterate through the `map_params` (the list of plot tasks derived from your `to_plot` configuration, as seen in [Chapter 2: Configuration Management](02_configuration_management_.md)) and, for each required plot, they use the `PlotterFactory` to get the right tool.

```python
# --- Snippet from a Source/Model Specific Logic class (simplified) ---

# This code would be part of a class like GriddedModel,
# after data has been loaded and processed.

class GriddedModel:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        # ... other setup ...

    def run(self):
        """Main plotting logic for this model."""
        # ... get processed data ...

        # Get the list of plot tasks from the config manager
        plot_tasks = self.config_manager.map_params.values()

        # Get the requested plotting backend from config
        backend = self.config_manager.output_config.plotting_backend

        # Loop through each plot task defined in the config
        for task_index, task_details in enumerate(plot_tasks):
            field_name = task_details['field_name']
            plot_type = task_details['plot_type']

            # Get the processed data for this specific variable (field_name)
            # This data comes from the Data Processing Pipeline (Chapter 5)
            data_for_this_plot = self._get_data_for_plot(field_name, task_details)

            if data_for_this_plot is None:
                 self.logger.warning(f"Skipping plot for {field_name}: data not available.")
                 continue

            self.logger.info(f"Creating {plot_type} plot for {field_name} using {backend} backend...")

            try:
                # *** THIS IS WHERE THE PLOTTER FACTORY IS USED ***
                plotter = PlotterFactory.create_plotter(plot_type=plot_type, backend=backend)

                # Now use the plotter via the standard BasePlotter interface!
                # Prepare plot-specific data structure expected by the plotter's plot method
                data_tuple_for_plotter = (
                    data_for_this_plot, # The xarray DataArray
                    ..., # Include necessary x, y, or time coordinates
                    field_name,
                    plot_type,
                    task_details['findex'], # File index
                    None # Matplotlib figure object (might be created/passed differently)
                )

                # Tell the plotter to create the plot internally
                # This calls the specific Matplotlib/HvPlot plot method
                plotter.plot(self.config_manager, data_tuple_for_plotter)

                # Get the output filename from the config manager
                output_filename = self._determine_output_filename(field_name, plot_type, task_details)

                # Tell the plotter to save the plot
                # This calls the specific Matplotlib/HvPlot save method
                plotter.save(output_filename)

                # Tell the plotter to show the plot (if requested)
                # if self.config_manager.output_config.print_to_screen:
                #     plotter.show()

            except ValueError as e:
                self.logger.error(f"Could not create plotter: {e}")
            except Exception as e:
                self.logger.error(f"Error during plotting for {field_name}: {e}")

        # ... cleanup ...
```

This snippet shows how a class like `GriddedModel` iterates through the plot requests from the configuration (`plot_tasks`). For each task, it determines the required `plot_type` and uses the configured `backend`. It then passes these directly to `PlotterFactory.create_plotter()`. The factory handles the complexity of deciding *which specific class* (like `MatplotlibXYPlotter`) to instantiate and returns the object. The `GriddedModel` code then simply calls the standard methods (`plot`, `save`, `show`) on the returned `plotter` object, trusting that it knows how to draw and save based on its internal type and backend.

This is the core benefit of the Factory pattern: the client code (`GriddedModel` in this case) is decoupled from the specific implementations of the plotters.

## Summary

In this chapter, we focused on the **Plotter Factory**, the component responsible for creating the correct, specific Plotter object instance needed for a visualization task.

*   The Plotter Factory acts as a tool request counter, taking the desired **plot type** (`xy`, `xt`, etc.) and **plotting backend** (`matplotlib`, `hvplot`, etc.) as input.
*   It uses an internal lookup (a dictionary) to find the specific Plotter class (like `MatplotlibXYPlotter`) that matches the requested combination.
*   It creates an instance of that specific class and returns it.
*   This pattern hides the complexity of selecting the right class from the code that uses the plotter.
*   The Factory's `create_plotter` method is typically called by the **Source/Model Specific Logic** ([Chapter 8](08_source_model_specific_logic_.md)) during the plotting phase, allowing that logic to work with Plotter objects via the standard interface provided by [Chapter 6: Plotter Abstraction](06_plotter_abstraction_.md).

We've now covered how eViz reads configuration, loads and processes data, defines a standard interface for plotting tools, and automatically selects the right plotting tool using the Factory. The next step is to see how all these pieces come together in the code that is specific to handling a particular *type* of data or model output – the **Source/Model Specific Logic**.

[Source/Model Specific Logic](08_source_model_specific_logic_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)