from abc import ABC, abstractmethod


class ExecutionModel(ABC):
    """
    Abstract execution and cost model.
    """

    @abstractmethod
    def apply_costs(
        self,
        notional: float,
        turnover: float,
    ) -> float:
        """
        Returns total execution cost given traded notional.
        """
        raise NotImplementedError
