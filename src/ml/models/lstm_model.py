import numpy as np
import pandas as pd
from typing import Tuple, List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self, lookback: int = 60, features: List[str] = None):
        self.lookback = lookback
        self.features = features or ['close', 'volume', 'rsi', 'macd']
        self.model = None
        self.scaler = MinMaxScaler()
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para LSTM."""
        # Seleciona features
        X = data[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Cria sequências
        X_seq, y = [], []
        for i in range(len(X_scaled) - self.lookback):
            X_seq.append(X_scaled[i:(i + self.lookback)])
            y.append(X_scaled[i + self.lookback, 0])  # Predição do preço
            
        return np.array(X_seq), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Constrói modelo LSTM."""
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
              validation_split: float = 0.2):
        """Treina o modelo."""
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        return self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        if self.model is None:
            raise ValueError("Modelo não treinado")
            
        return self.model.predict(X)
    
    def generate_signals(self, predictions: np.ndarray,
                        threshold: float = 0.001) -> np.ndarray:
        """Gera sinais baseados nas predições."""
        signals = np.zeros_like(predictions)
        signals[predictions > threshold] = 1
        signals[predictions < -threshold] = -1
        
        return signals