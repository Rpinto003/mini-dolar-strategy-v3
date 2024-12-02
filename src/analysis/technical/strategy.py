import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
import talib

class TechnicalStrategy:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos."""
        df = data.copy()
        
        # Tendência
        df['ema9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema21'] = talib.EMA(df['close'], timeperiod=21)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        
        # Momentum
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Volatilidade
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(
            df['close'], timeperiod=20
        )
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Gera sinais de trading."""
        df = data.copy()
        signals = pd.Series(0, index=df.index)
        
        # Regras de entrada
        long_condition = (
            (df['ema9'] > df['ema21']) &
            (df['rsi'] > 40) & (df['rsi'] < 60) &
            (df['close'] > df['bbands_middle'])
        )
        
        short_condition = (
            (df['ema9'] < df['ema21']) &
            (df['rsi'] > 40) & (df['rsi'] < 60) &
            (df['close'] < df['bbands_middle'])
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def calculate_risk_params(self, data: pd.DataFrame,
                           signal: int) -> Dict[str, float]:
        """Calcula parâmetros de risco."""
        current_price = data['close'].iloc[-1]
        atr = data['atr'].iloc[-1]
        
        if signal == 1:  # Long
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        elif signal == -1:  # Short
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        else:
            return {}
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': abs(take_profit - current_price) / 
                          abs(stop_loss - current_price)
        }