from dataclasses import dataclass, field
import os
import logging
from typing import Dict, Any


@dataclass
class OutputConfig:
    app_data: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "./output_plots"
    print_to_file: bool = False
    print_format: str = "png"
    add_logo: bool = False
    make_pdf: bool = False
    print_basic_stats: bool = False
    mpl_style: str = "classic"
    make_gif: bool = False
    gif_fps: int = 10

    def initialize(self):
        """Initialize output configuration."""
        outputs = self.app_data.outputs

        self.add_logo = outputs.get("add_logo", False)
        self.print_to_file = outputs.get("print_to_file", False)
        self.print_format = outputs.get("print_format", "png")
        self.make_pdf = outputs.get("make_pdf", False)
        self.print_basic_stats = outputs.get("print_basic_stats", False)
        self.mpl_style = outputs.get("mpl_style", "classic")
        self.fig_style = outputs.get("fig_style", "default")
        self.make_gif = outputs.get("make_gif", False)
        self.gif_fps = outputs.get("gif_fps", 10)
        self.dpi = outputs.get("dpi", 300)
        self.backend = "matplotlib"
        self._init_visualization(outputs)

        self._set_output_dir()

    def _init_visualization(self, outputs: Dict[str, Any]) -> None:
        """Initialize parameters in the `visualization` subsection ."""
        # Set default values if not already set
        if not hasattr(self, 'backend'):
            self.backend = "matplotlib"
        if not hasattr(self, 'colormap'):
            self.colormap = "coolwarm"
        if not hasattr(self, 'fig_style'):
            self.fig_style = "default"
        if not hasattr(self, 'dpi'):
            self.dpi = 300
        if not hasattr(self, 'gif_fps'):
            self.gif_fps = 10
        if not hasattr(self, 'mpl_style'):
            self.mpl_style = "classic"
            
        # Override with visualization-specific settings if present
        if "visualization" in outputs:
            outputs_config = outputs["visualization"]
            self.backend = outputs_config.get("backend", self.backend)
            self.colormap = outputs_config.get("colormap", self.colormap)
            self.fig_style = outputs_config.get("fig_style", self.fig_style)
            self.dpi = outputs_config.get("dpi", self.dpi)
            self.gif_fps = outputs_config.get("gif_fps", self.gif_fps)
            self.mpl_style = outputs_config.get("mpl_style", self.mpl_style)

    def _set_output_dir(self):
        """Set the output directory."""
        if self.print_to_file:
            if not os.path.exists(self.output_dir):
                logging.info(f"Creating output directory: {self.output_dir}")
                os.makedirs(self.output_dir)

    @property
    def logger(self):
        """Return the logger for this class."""
        return logging.getLogger(__name__)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the output configuration."""
        return {
            "backend": self.backend,
            "colormap": self.colormap,
            "fig_style": self.fig_style,
            "dpi": self.dpi,
            "gif_fps": self.gif_fps,
            "mpl_style": self.mpl_style,
            "output_dir": self.output_dir,
            "print_to_file": self.print_to_file,
            "print_format": self.print_format,
            "add_logo": self.add_logo,
            "make_pdf": self.make_pdf,
            "print_basic_stats": self.print_basic_stats,
            "make_gif": self.make_gif,
        }
