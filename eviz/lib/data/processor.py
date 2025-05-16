"""
Deprecated processor module. Use eviz.lib.data.pipeline.processor instead.
"""
import warnings
from dataclasses import dataclass
from typing import Any, List
import logging
import xarray as xr

from eviz.lib.autoviz.config_manager import ConfigManager
from eviz.lib.data.pipeline.processor import DataProcessor as PipelineProcessor


def _deprecation_warning(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future version. "
        f"Use {new_name} from eviz.lib.data.pipeline.processor instead.",
        DeprecationWarning,
        stacklevel=2
    )

@dataclass
class Overlays:
    """Deprecated: Use DataProcessor from pipeline.processor instead."""
    
    config: ConfigManager
    plot_type: str

    def __post_init__(self):
        _deprecation_warning("Overlays", "DataProcessor")
        self._processor = PipelineProcessor(self.config)
        self.tropp = None
        self.tropp_conversion = None
        self.trop_ok = False

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def process_data(self):
        """Deprecated: Use DataProcessor.process_data_source instead."""
        _deprecation_warning("Overlays.process_data", "DataProcessor.process_data_source")
        return self._processor._apply_tropopause_height(self.config.data_source)

    def sphum_field(self, ds_meta, findex=0):
        """Deprecated: Use DataProcessor.process_data_source instead."""
        _deprecation_warning("Overlays.sphum_field", "DataProcessor.process_data_source")
        return self._processor._apply_sphum_conversion(self.config.data_source)

@dataclass
class Interp:
    """Deprecated: Use DataProcessor from pipeline.processor instead."""
    
    config_manager: ConfigManager
    data: List[Any]

    def __post_init__(self):
        _deprecation_warning("Interp", "DataProcessor")
        self._processor = PipelineProcessor(self.config_manager)
        self.logger.info("Start init")

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def regrid(self, pid):
        """Deprecated: Use DataProcessor.regrid instead."""
        _deprecation_warning("Interp.regrid", "DataProcessor.regrid")
        if len(self.data) != 2:
            return None, None, None
        
        data1, data2 = self.data
        dim1, dim2 = self.config_manager.get_dim_names(pid)
        
        try:
            regridded1, regridded2 = self._processor.regrid(data1, data2, dim1, dim2)
            return regridded1, regridded1[dim1].values, regridded1[dim2].values
        except Exception as e:
            self.logger.error(f"Error during regridding: {e}")
            return None, None, None

class DataProcessor:
    """Deprecated: Use DataProcessor from pipeline.processor instead."""

    def __init__(self, file_list):
        _deprecation_warning("DataProcessor", "DataProcessor from pipeline.processor")
        self.file_list = file_list
        self.datasets = []
        self._processor = PipelineProcessor()

    def process_files(self):
        """Deprecated: Use DataProcessor.process_data_source instead."""
        _deprecation_warning("DataProcessor.process_files", "DataProcessor.process_data_source")
        for file_path in self.file_list:
            try:
                with xr.open_dataset(file_path) as ds:
                    processed_ds = self._processor._process_dataset(ds)
                    self.datasets.append(processed_ds)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
