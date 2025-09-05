import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, List

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .base_model import BaseModel


class GRUModel(BaseModel):
    def __init__(
        self,
        name: str = "GRU",
        gru_units: List[int] = [50, 50, 50],
        dropout_rates: List[float] = [0.2, 0.2, 0.2],
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(name=name)
        self.gru_units = gru_units
        self.dropout_rates = dropout_rates
        self.optimizer = optimizer
        self.loss = loss
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(
            units=self.gru_units[0],
            return_sequences=True if len(self.gru_units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout_rates[0]))
        
        # Middle GRU layers
        for i in range(1, len(self.gru_units) - 1):
            model.add(GRU(units=self.gru_units[i], return_sequences=True))
            model.add(Dropout(self.dropout_rates[i]))
        
        # Last GRU layer (if more than one)
        if len(self.gru_units) > 1:
            model.add(GRU(units=self.gru_units[-1], return_sequences=False))
            model.add(Dropout(self.dropout_rates[-1]))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
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
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        return self.model.predict(X).flatten()
    
    def save(self, path: str) -> None:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the Keras model
        self.model.save(path)
        
    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        self.model = load_model(path)
        self.is_fitted = True
        
        # Update input shape
        self.input_shape = (self.model.input_shape[1], self.model.input_shape[2]) 