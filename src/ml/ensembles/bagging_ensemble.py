import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample

class BaggingEnsemble:
    def __init__(self, base_estimator: BaseEstimator,
                 n_estimators: int = 10,
                 max_samples: float = 0.8,
                 max_features: float = 0.8,
                 feature_importance: bool = True):
        """Ensemble usando bagging com amostragem temporal.
        
        Args:
            base_estimator: Modelo base a ser replicado
            n_estimators: Número de estimadores
            max_samples: Fração de amostras para cada modelo
            max_features: Fração de features para cada modelo
            feature_importance: Se deve calcular importância
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.feature_importance = feature_importance
        
        self.estimators = []
        self.feature_indices = []
        self.feature_importances_ = None
        
    def _get_sample_indices(self, X: np.ndarray) -> np.ndarray:
        """Gera índices para amostragem preservando ordem temporal."""
        n_samples = int(self.max_samples * X.shape[0])
        
        # Garante que amostras são contíguas
        start_idx = np.random.randint(0, X.shape[0] - n_samples)
        return np.arange(start_idx, start_idx + n_samples)
    
    def _get_feature_indices(self, X: np.ndarray) -> np.ndarray:
        """Seleciona features aleatoriamente."""
        n_features = int(self.max_features * X.shape[1])
        return np.random.choice(
            np.arange(X.shape[1]),
            size=n_features,
            replace=False
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina ensemble usando bagging.
        
        Args:
            X: Features de treino
            y: Target de treino
        """
        self.estimators = []
        self.feature_indices = []
        
        for _ in range(self.n_estimators):
            # Clona estimador base
            estimator = clone(self.base_estimator)
            
            # Seleciona amostras e features
            sample_indices = self._get_sample_indices(X)
            feature_indices = self._get_feature_indices(X)
            
            # Treina modelo
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            
            estimator.fit(X_subset, y_subset)
            
            self.estimators.append(estimator)
            self.feature_indices.append(feature_indices)
        
        if self.feature_importance:
            self._compute_feature_importance(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições usando o ensemble.
        
        Args:
            X: Features para predição
            
        Returns:
            Array com predições
        """
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (estimator, features) in enumerate(zip(
            self.estimators, self.feature_indices
        )):
            X_subset = X[:, features]
            predictions[:, i] = estimator.predict(X_subset)
        
        return np.mean(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Faz predições probabilísticas.
        
        Args:
            X: Features para predição
            
        Returns:
            Array com probabilidades
        """
        if not hasattr(self.base_estimator, 'predict_proba'):
            raise AttributeError('Base estimator has no predict_proba method')
            
        probas = np.zeros((X.shape[0], self.n_estimators, 2))  # Para classificação binária
        
        for i, (estimator, features) in enumerate(zip(
            self.estimators, self.feature_indices
        )):
            X_subset = X[:, features]
            probas[:, i, :] = estimator.predict_proba(X_subset)
        
        return np.mean(probas, axis=1)
    
    def _compute_feature_importance(self, X: np.ndarray):
        """Calcula importância das features."""
        if not hasattr(self.base_estimator, 'feature_importances_'):
            return
            
        feature_importances = np.zeros(X.shape[1])
        counts = np.zeros(X.shape[1])
        
        for estimator, features in zip(self.estimators, self.feature_indices):
            feature_importances[features] += estimator.feature_importances_
            counts[features] += 1
            
        # Evita divisão por zero
        counts[counts == 0] = 1
        self.feature_importances_ = feature_importances / counts
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features."""
        if self.feature_importances_ is None:
            return {}
            
        return {
            f'feature_{i}': imp
            for i, imp in enumerate(self.feature_importances_)
        }
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula score do modelo.
        
        Args:
            X: Features de teste
            y: Target de teste
            
        Returns:
            Score (R² para regressão, acurácia para classificação)
        """
        pred = self.predict(X)
        
        if self.base_estimator._estimator_type == 'classifier':
            return (pred == y).mean()
        else:
            return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()