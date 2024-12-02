import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

class StackingEnsemble:
    def __init__(self, base_models: List[BaseEstimator],
                 meta_model: BaseEstimator,
                 n_splits: int = 5):
        """Ensemble usando stacking com validação temporal.
        
        Args:
            base_models: Lista de modelos base
            meta_model: Modelo para combinar predições
            n_splits: Número de splits para validação
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.base_predictions = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina ensemble usando stacking temporal.
        
        Args:
            X: Features de treino
            y: Target de treino
        """
        # Configura splits temporais
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Array para meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # Para cada modelo base
        for i, model in enumerate(self.base_models):
            # Para cada split temporal
            for train_idx, val_idx in tscv.split(X):
                # Separa dados de treino e validação
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Treina modelo e faz predições
                model.fit(X_train, y_train)
                meta_features[val_idx, i] = model.predict(X_val)
        
        # Treina metamodelo
        self.meta_model.fit(meta_features, y)
        
        # Treina modelos base com todos os dados
        for model in self.base_models:
            model.fit(X, y)
            
        # Salva predições dos modelos base
        self.base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições usando o ensemble.
        
        Args:
            X: Features para predição
            
        Returns:
            Array com predições
        """
        # Coleta predições dos modelos base
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        
        # Usa metamodelo para predição final
        return self.meta_model.predict(meta_features)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Retorna importância relativa dos modelos base.
        
        Returns:
            Dict com pesos dos modelos
        """
        if hasattr(self.meta_model, 'coef_'):
            weights = self.meta_model.coef_
        elif hasattr(self.meta_model, 'feature_importances_'):
            weights = self.meta_model.feature_importances_
        else:
            return {}
        
        return {
            f'model_{i}': weight
            for i, weight in enumerate(weights)
        }
    
    def evaluate_models(self, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, float]:
        """Avalia performance individual dos modelos.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dict com métricas de performance
        """
        results = {}
        
        # Avalia modelos base
        for i, model in enumerate(self.base_models):
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            results[f'model_{i}_mse'] = mse
        
        # Avalia ensemble
        ensemble_pred = self.predict(X_test)
        results['ensemble_mse'] = mean_squared_error(y_test, ensemble_pred)
        
        return results

class TemporalStackingEnsemble(StackingEnsemble):
    """Versão do Stacking adaptada para dados temporais."""
    
    def prepare_temporal_features(self, X: np.ndarray,
                                window_size: int = 5) -> np.ndarray:
        """Prepara features temporais para o metamodelo.
        
        Args:
            X: Features originais
            window_size: Tamanho da janela temporal
            
        Returns:
            Array com features temporais
        """
        # Adiciona features de tendência
        trend_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            trend = np.zeros_like(pred)
            
            for t in range(window_size, len(pred)):
                trend[t] = np.mean(pred[t-window_size:t])
            
            trend_features[:, i] = trend
        
        return np.hstack([X, trend_features])
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina ensemble com features temporais."""
        # Prepara features temporais
        X_temporal = self.prepare_temporal_features(X)
        
        # Chama método fit da classe pai
        super().fit(X_temporal, y)