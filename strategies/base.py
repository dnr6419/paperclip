from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """
    buy=1, sell=-1, hold=0.
    stop_loss and take_profit are fractional ratios (e.g. 0.04 = 4%).
    position_size is fraction of total capital (e.g. 0.02 = 2%).
    """
    direction: int        # 1=buy, -1=sell, 0=hold
    stop_loss: float      # fraction below entry (e.g. 0.04)
    take_profit: float    # fraction above entry (e.g. 0.10)
    position_size: float  # fraction of capital to deploy
    take_profit2: Optional[float] = None  # second target if partial exit


class BaseStrategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of 1/0/-1 aligned with df.index."""
        raise NotImplementedError

    def get_signal_params(self) -> Signal:
        """Return default risk params for this strategy."""
        raise NotImplementedError
