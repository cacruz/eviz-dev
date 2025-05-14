from dataclasses import dataclass, field
import os
import logging
from typing import Dict, Any
from eviz.lib.utils import log_method


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

    @log_method
    def initialize(self):
        """Initialize output configuration."""
        outputs = self.app_data.outputs

        self.add_logo = outputs.get('add_logo', False)
        self.print_to_file = outputs.get('print_to_file', False)
        self.print_format = outputs.get('print_format', 'png')
        self.make_pdf = outputs.get('make_pdf', False)
        self.print_basic_stats = outputs.get('print_basic_stats', False)
        self.mpl_style = outputs.get('mpl_style', 'classic')
        self.make_gif = outputs.get('make_gif', False)
        self.gif_fps = outputs.get('gif_fps', 10)

        self._set_output_dir()

        self.logger.debug(f"OutputConfig initialized with: "
                          f"add_logo={self.add_logo}, "
                          f"print_to_file={self.print_to_file}, "
                          f"print_format={self.print_format}, "
                          f"make_pdf={self.make_pdf}, "
                          f"print_basic_stats={self.print_basic_stats}, "
                          f"mpl_style={self.mpl_style}, "
                          f"make_gif={self.make_gif}, "
                          f"gif_fps={self.gif_fps}, "
                          f"output_dir={self.output_dir}")

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
            "output_dir": self.output_dir,
            "print_to_file": self.print_to_file,
            "print_format": self.print_format,
            "add_logo": self.add_logo,
            "make_pdf": self.make_pdf,
            "print_basic_stats": self.print_basic_stats,
            "mpl_style": self.mpl_style,
            "make_gif": self.make_gif,
            "gif_fps": self.gif_fps,
        }
    