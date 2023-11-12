from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABC, abstractmethod


class HistGradientBoostingBase(BaseEstimator, ABC):
    """Absract base class for hist gradient boosting estimator"""
    @abstractmethod
    def __init__(self):
        ...

    def fit(self):
        # Define fit logic here
        ...

    @abstractmethod
    def predict(self):
        ...


class HistGradientBoostingRegressor(HistGradientBoostingBase, RegressorMixin):
    """Concrete implementation for hist gradient boosting regressor"""
    def __init__(self):
        super().__init__()
