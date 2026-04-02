from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    model: Any = None
    stores: Any = None
    oil: Any = None
    holidays: Any = None
    encoders: Any = None
    history: Any = None


state = AppState()
