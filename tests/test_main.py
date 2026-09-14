import runpy

import pandas as pd
import pytest

from nifty_quant.domain.strategies.base import Strategy


_compute_signal_scores = runpy.run_path("main.py")["_compute_signal_scores"]


def test_compute_signal_scores_surfaces_strategy_failure():
    class FailingStrategy(Strategy):
        @property
        def min_history_days(self):
            return 1

        def compute_signals(self, **kwargs):
            raise ValueError("invalid signal input")

        def select_and_weight(self, **kwargs):
            raise NotImplementedError

    with pytest.raises(RuntimeError, match="Failed to compute recommendation signals.") as error:
        _compute_signal_scores(
            strategy=FailingStrategy(),
            prices=pd.DataFrame(),
            daily_returns=pd.DataFrame(),
            as_of=pd.Timestamp("2024-01-01"),
        )

    assert isinstance(error.value.__cause__, ValueError)
