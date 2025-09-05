import numpy as np
import optuna
import json
import os
from typing import Dict, Any, List, Callable, Union, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    def __init__(
        self,
        model_builder: Callable,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        validation_split: float = 0.1,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: str = "bitcoin_prediction",
        storage: Optional[str] = None,
        save_path: Optional[str] = "results/hyperparameter_tuning",
    ):

        self.model_builder = model_builder
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.validation_split = validation_split
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        self.save_path = save_path
        
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize study
        self.study = None
    
    def objective(self, trial: optuna.Trial) -> float:
        raise NotImplementedError("Subclasses must implement objective method")
    
    def optimize(self) -> Dict[str, Any]:
        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="minimize"
        )
        
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Get best parameters and trial
        best_params = self.study.best_params
        best_value = self.study.best_value
        best_trial = self.study.best_trial
        
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_value}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        if self.save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.save_path, f"results_{timestamp}.json")
            
            results = {
                "best_params": best_params,
                "best_value": best_value,
                "best_trial": best_trial.number,
                "study_name": self.study_name,
                "n_trials": self.n_trials,
                "timestamp": timestamp,
                "trials": [
                    {
                        "number": trial.number,
                        "params": trial.params,
                        "value": trial.value,
                        "state": str(trial.state)
                    }
                    for trial in self.study.trials
                ]
            }
            
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Results saved to {result_path}")
            
            # Plot optimization history
            self._plot_optimization_history(os.path.join(self.save_path, f"history_{timestamp}.png"))
            
            # Plot parameter importances if enough trials
            if len(self.study.trials) >= 10:
                self._plot_param_importances(os.path.join(self.save_path, f"importance_{timestamp}.png"))
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": best_trial,
            "study": self.study
        }
    
    def _plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Optimization history plot saved to {save_path}")
    
    def _plot_param_importances(self, save_path: Optional[str] = None) -> None:
        try:
            plt.figure(figsize=(10, 8))
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Parameter importance plot saved to {save_path}")
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Could not plot parameter importances: {e}")


class LSTMTuner(HyperparameterTuner):
    def objective(self, trial: optuna.Trial) -> float:
        # Define hyperparameters to tune
        lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=16)
        lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=16)
        dropout_rate_1 = trial.suggest_float('dropout_rate_1', 0.1, 0.5, step=0.1)
        dropout_rate_2 = trial.suggest_float('dropout_rate_2', 0.1, 0.5, step=0.1)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Build model with these hyperparameters
        model = self.model_builder(
            lstm_units=[lstm_units_1, lstm_units_2],
            dropout_rates=[dropout_rate_1, dropout_rate_2],
            optimizer=optimizer,
            learning_rate=learning_rate
        )
        
        # Train model
        if self.x_val is not None and self.y_val is not None:
            history = model.train(
                self.x_train, self.y_train,
                epochs=50, batch_size=batch_size,
                validation_data=(self.x_val, self.y_val),
                early_stopping=True,
                patience=5,
                verbose=0
            )
            val_loss = history.get('final_val_loss')
        else:
            history = model.train(
                self.x_train, self.y_train,
                epochs=50, batch_size=batch_size,
                validation_split=self.validation_split,
                early_stopping=True,
                patience=5,
                verbose=0
            )
            val_loss = history.get('final_val_loss')
        
        return val_loss


class GRUTuner(HyperparameterTuner):
    def objective(self, trial: optuna.Trial) -> float:
        # Define hyperparameters to tune
        gru_units_1 = trial.suggest_int('gru_units_1', 32, 128, step=16)
        gru_units_2 = trial.suggest_int('gru_units_2', 32, 128, step=16)
        dropout_rate_1 = trial.suggest_float('dropout_rate_1', 0.1, 0.5, step=0.1)
        dropout_rate_2 = trial.suggest_float('dropout_rate_2', 0.1, 0.5, step=0.1)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Build model with these hyperparameters
        model = self.model_builder(
            gru_units=[gru_units_1, gru_units_2],
            dropout_rates=[dropout_rate_1, dropout_rate_2],
            optimizer=optimizer,
            learning_rate=learning_rate
        )
        
        # Train model
        if self.x_val is not None and self.y_val is not None:
            history = model.train(
                self.x_train, self.y_train,
                epochs=50, batch_size=batch_size,
                validation_data=(self.x_val, self.y_val),
                early_stopping=True,
                patience=5,
                verbose=0
            )
            val_loss = history.get('final_val_loss')
        else:
            history = model.train(
                self.x_train, self.y_train,
                epochs=50, batch_size=batch_size,
                validation_split=self.validation_split,
                early_stopping=True,
                patience=5,
                verbose=0
            )
            val_loss = history.get('final_val_loss')
        
        return val_loss 