from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AppData:
    """Data class for application-level configuration."""
    inputs: Dict[str, Any] = field(default_factory=dict)
    for_inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    system_opts: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    plot_params: Dict[str, Any] = field(default_factory=dict)
