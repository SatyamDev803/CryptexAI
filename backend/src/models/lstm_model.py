import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, List, Union

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsolutePercentageError

from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""
    
    def __init__(
        self,
        name: str = "LSTM",
        lstm_units: List[int] = [50, 50, 50],
        dropout_rates: List[float] = [0.2, 0.2, 0.2],
        optimizer: str = "adam",
        loss: Union[str, Huber] = Huber(),
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the LSTM model.
        
        Args:
            name: Model name
            lstm_units: Number of units in each LSTM layer
            dropout_rates: Dropout rate after each LSTM layer
            optimizer: Optimizer for training
            loss: Loss function for training
            input_shape: Input shape of (timesteps, features)
        """
        super().__init__(name=name)
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        self.optimizer = optimizer
        self.loss = loss
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Input shape of (timesteps, features)
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout_rates[0]))
        
        # Middle LSTM layers
        for i in range(1, len(self.lstm_units) - 1):
            model.add(LSTM(units=self.lstm_units[i], return_sequences=True))
            model.add(Dropout(self.dropout_rates[i]))
        
        # Last LSTM layer (if more than one)
        if len(self.lstm_units) > 1:
            model.add(LSTM(units=self.lstm_units[-1], return_sequences=False))
            model.add(Dropout(self.dropout_rates[-1]))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[MeanAbsolutePercentageError()]
        )
        
        self.model = model
        self.input_shape = input_shape
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping: bool = True,
        patience: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features of shape (samples, timesteps, features)
            y_train: Target values of shape (samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training history and metrics
        """
        if self.model is None:
            if self.input_shape is None:
                self.input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(self.input_shape)
        
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ))
        # Add learning rate scheduler
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ))
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            **kwargs
        )
        
        self.is_fitted = True
        
        # Return training metrics
        return {
            "history": history.history,
            "epochs_completed": len(history.history['loss']),
            "final_loss": history.history['loss'][-1],
            "final_val_loss": history.history['val_loss'][-1] if validation_split > 0 else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given data.
        
        Args:
            X: Input features of shape (samples, timesteps, features)
            
        Returns:
            Predictions of shape (samples,)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        return self.model.predict(X).flatten()
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the Keras model
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        self.model = load_model(path)
        self.is_fitted = True
        
        # Update input shape
        self.input_shape = (self.model.input_shape[1], self.model.input_shape[2]) 