from dataclasses import dataclass, field
from eviz.lib.config.app_data import AppData


@dataclass
class HistoryConfig:
    app_data: AppData = field(default_factory=AppData)  
    use_history: bool = False
    history_dir: str = None

    def initialize(self):
        """Initialize history configuration."""
        history = self.app_data.history.get('history', {})
        self.use_history = history.get('use_history', False)
        self.history_dir = history.get('history_dir', None)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the HistoryConfig."""
        return {
            "history": self.history,
            "use_history": self.use_history,
            "history_dir": self.history_dir,
        }   