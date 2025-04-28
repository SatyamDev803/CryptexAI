import numpy as np
import os
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional, List

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping

from .base_model import BaseModel


class TransformerModel(BaseModel):
    """Transformer model for time series prediction"""
    
    def __init__(
        self,
        name: str = "Transformer",
        num_layers: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        dff: int = 128,
        dropout_rate: float = 0.1,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the Transformer model.
        
        Args:
            name: Model name
            num_layers: Number of transformer layers
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dff: Dimension of the feed forward network
            dropout_rate: Dropout rate
            optimizer: Optimizer for training
            loss: Loss function for training
            input_shape: Input shape of (timesteps, features)
        """
        super().__init__(name=name)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the Transformer model architecture.
        
        Args:
            input_shape: Input shape of (timesteps, features)
        """
        # Input shape should be (batch_size, seq_len, features)
        inputs = Input(shape=input_shape)
        x = inputs

        # Add static positional encoding via Lambda
        seq_len, feature_dim = input_shape
        pos_encoding = self._get_positional_encoding(seq_len, feature_dim)
        x = tf.keras.layers.Lambda(lambda t: t + tf.constant(pos_encoding, dtype=tf.float32))(x)
        # Add dropout
        x = Dropout(self.dropout_rate)(x)
        # Project features to model dimension
        x = Dense(self.d_model)(x)
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = self._transformer_block(x, self.d_model, self.num_heads, self.dff, self.dropout_rate)
            
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Final dense layer
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
        self.model = model
        self.input_shape = input_shape
    
    def _get_positional_encoding(self, seq_len: int, feature_dim: int) -> np.ndarray:
        """Compute static positional encoding matrix."""
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(feature_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(feature_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return angle_rads[np.newaxis, ...]
    
    def _transformer_block(self, inputs, d_model, num_heads, dff, dropout_rate):
        """Creates a transformer block."""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        
        # First residual connection and normalization
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])(out1)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        
        # Second residual connection and normalization
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
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
        Train the Transformer model.
        
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