from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union


# Base abstract class for all time series prediction models
class BaseModel(ABC):
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (fitted: {self.is_fitted})" 