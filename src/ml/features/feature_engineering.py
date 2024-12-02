import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria features técnicas.
        
        Args:
            data: DataFrame com dados OHLCV
            
        Returns:
            DataFrame com features técnicas
        """
        df = data.copy()
        
        # Retornos
        df['returns'] = df['close'].pct_change()
        df['returns_volatility'] = df['returns'].rolling(20).std()
        
        # Médias Móveis
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff()
        
        # Volatilidade
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['daily_range'] = (df['high'] - df['low'])
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum
        df['momentum'] = df['close'].diff(5)
        df['acceleration'] = df['momentum'].diff()
        
        return df
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais.
        
        Args:
            data: DataFrame com índice temporal
            
        Returns:
            DataFrame com features temporais
        """
        df = data.copy()
        
        # Componentes temporais
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['dayofweek'] = df.index.dayofweek
        
        # Sessões de trading
        df['is_morning'] = df['hour'].between(9, 12)
        df['is_afternoon'] = df['hour'].between(13, 16)
        df['is_market_open'] = df['hour'].between(9, 16)
        
        # Períodos específicos
        df['is_open_hour'] = df['hour'] == 9
        df['is_close_hour'] = df['hour'] == 16
        df['is_lunch_time'] = df['hour'].between(12, 13)
        
        # Converte para variáveis numéricas
        for col in ['is_morning', 'is_afternoon', 'is_market_open',
                    'is_open_hour', 'is_close_hour', 'is_lunch_time']:
            df[col] = df[col].astype(int)
        
        return df
    
    def create_fundamental_features(self, data: pd.DataFrame,
                                  fundamental_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Cria features fundamentais.
        
        Args:
            data: DataFrame com dados de preço
            fundamental_data: Dicionário com dados fundamentais
            
        Returns:
            DataFrame com features fundamentais
        """
        df = data.copy()
        
        # Adiciona indicadores econômicos
        for name, fund_df in fundamental_data.items():
            # Faz merge preservando índice temporal
            df = df.join(fund_df, how='left')
            # Forward fill para dados ausentes
            df[fund_df.columns] = df[fund_df.columns].fillna(method='ffill')
        
        return df
    
    def select_features(self, data: pd.DataFrame,
                       target: str,
                       n_features: int = 20) -> List[str]:
        """Seleciona features mais importantes.
        
        Args:
            data: DataFrame com todas as features
            target: Nome da coluna alvo
            n_features: Número de features para selecionar
            
        Returns:
            Lista com nomes das features mais importantes
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Remove colunas com dados ausentes
        df = data.dropna(axis=1)
        
        # Separa features e target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Treina Random Forest para importância
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Calcula importância
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        # Armazena importância
        self.feature_importance = dict(zip(
            importance['feature'],
            importance['importance']
        ))
        
        return importance['feature'].head(n_features).tolist()
    
    def prepare_features(self, data: pd.DataFrame,
                        fundamental_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Prepara todas as features.
        
        Args:
            data: DataFrame com dados OHLCV
            fundamental_data: Dados fundamentais (opcional)
            
        Returns:
            DataFrame com todas as features preparadas
        """
        # Features técnicas
        df = self.create_technical_features(data)
        
        # Features temporais
        df = self.create_temporal_features(df)
        
        # Features fundamentais
        if fundamental_data:
            df = self.create_fundamental_features(df, fundamental_data)
        
        # Remove linhas com dados ausentes
        df = df.dropna()
        
        return df