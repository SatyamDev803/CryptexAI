from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union


class BaseModel(ABC):
    """Base abstract class for all time series prediction models."""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        Args:
            X_train: Training features of shape (samples, timesteps, features)
            y_train: Target values of shape (samples,)
            **kwargs: Additional parameters for specific model implementations
            
        Returns:
            Dict containing training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given data.
        
        Args:
            X: Input features of shape (samples, timesteps, features)
            
        Returns:
            Predictions of shape (samples,)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (fitted: {self.is_fitted})" 