import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreparation:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.features = []
        
    def prepare_sequences(self, data: pd.DataFrame,
                         feature_columns: List[str],
                         target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara sequências para modelos temporais.
        
        Args:
            data: DataFrame com features e target
            feature_columns: Lista de colunas de features
            target_column: Nome da coluna alvo
            
        Returns:
            Tupla (X, y) com sequências e targets
        """
        # Salva nomes das features
        self.features = feature_columns
        
        # Separa features e target
        features = data[feature_columns].values
        target = data[target_column].values.reshape(-1, 1)
        
        # Normaliza dados
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)
        
        # Cria sequências
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:(i + self.sequence_length)])
            y.append(target_scaled[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_features(self, data: pd.DataFrame,
                        feature_columns: List[str]) -> np.ndarray:
        """Prepara features para modelos não sequenciais.
        
        Args:
            data: DataFrame com features
            feature_columns: Lista de colunas de features
            
        Returns:
            Array com features normalizadas
        """
        # Salva nomes das features
        self.features = feature_columns
        
        # Seleciona e normaliza features
        features = data[feature_columns].values
        return self.feature_scaler.fit_transform(features)
    
    def prepare_target(self, data: pd.DataFrame,
                      target_column: str) -> np.ndarray:
        """Prepara target para treino.
        
        Args:
            data: DataFrame com target
            target_column: Nome da coluna alvo
            
        Returns:
            Array com target normalizado
        """
        target = data[target_column].values.reshape(-1, 1)
        return self.target_scaler.fit_transform(target)
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Reverte normalização do target.
        
        Args:
            y_scaled: Array com valores normalizados
            
        Returns:
            Array com valores originais
        """
        return self.target_scaler.inverse_transform(y_scaled)
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray,
                               test_size: float = 0.2,
                               shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray]:
        """Cria divisão treino/teste preservando ordem temporal.
        
        Args:
            X: Array de features
            y: Array de targets
            test_size: Fração para teste
            shuffle: Se deve embaralhar dados
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if shuffle:
            # Embaralha mantendo correspondência entre X e y
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        # Divide mantendo ordem temporal
        train_size = int(len(X) * (1 - test_size))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def save_scalers(self, path: str):
        """Salva scalers para uso futuro.
        
        Args:
            path: Caminho para salvar
        """
        import joblib
        
        joblib.dump(self.feature_scaler, f"{path}/feature_scaler.pkl")
        joblib.dump(self.target_scaler, f"{path}/target_scaler.pkl")
        
    def load_scalers(self, path: str):
        """Carrega scalers salvos.
        
        Args:
            path: Caminho onde estão os scalers
        """
        import joblib
        
        self.feature_scaler = joblib.load(f"{path}/feature_scaler.pkl")
        self.target_scaler = joblib.load(f"{path}/target_scaler.pkl")