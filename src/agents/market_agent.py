import numpy as np
import pandas as pd
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class MarketAgent(ABC):
    """Classe base para agentes de mercado."""
    
    def __init__(self):
        self.current_state = None
        self.history = []
    
    @abstractmethod
    def observe(self, state: Dict[str, Any]) -> None:
        """Observa estado atual do mercado.
        
        Args:
            state: Dicionário com informações do mercado
        """
        self.current_state = state
        self.history.append(state)
    
    @abstractmethod
    def act(self) -> Dict[str, Any]:
        """Define ação baseada no estado atual.
        
        Returns:
            Dict com ação a ser tomada
        """
        pass
    
    @abstractmethod
    def learn(self) -> None:
        """Aprende com as experiências passadas."""
        pass

class RiskManagementAgent(MarketAgent):
    """Agente responsável por gestão de risco."""
    
    def __init__(self, max_drawdown: float = -0.05,
                 max_position_size: int = 3):
        super().__init__()
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        
    def observe(self, state: Dict[str, Any]) -> None:
        """Observa métricas de risco atuais."""
        super().observe(state)
        
    def act(self) -> Dict[str, Any]:
        """Avalia e ajusta parâmetros de risco."""
        if not self.current_state:
            return {}
        
        # Calcula drawdown atual
        equity = self.current_state.get('equity', [])
        if len(equity) > 0:
            peak = max(equity)
            current = equity[-1]
            drawdown = (current - peak) / peak
            
            # Reduz posições se drawdown exceder limite
            if drawdown < self.max_drawdown:
                return {
                    'action': 'reduce_risk',
                    'max_positions': 1,
                    'reason': 'drawdown_exceeded'
                }
        
        return {
            'action': 'maintain',
            'max_positions': self.max_position_size
        }
    
    def learn(self) -> None:
        """Atualiza limites baseado em experiência."""
        if len(self.history) < 2:
            return
            
        # Analisa padrões de drawdown
        drawdowns = []
        for state in self.history:
            equity = state.get('equity', [])
            if len(equity) > 0:
                peak = max(equity)
                current = equity[-1]
                drawdowns.append((current - peak) / peak)
        
        if drawdowns:
            # Ajusta max_drawdown baseado em percentil
            self.max_drawdown = np.percentile(drawdowns, 5)

class MarketRegimeAgent(MarketAgent):
    """Agente para identificação de regimes de mercado."""
    
    def __init__(self, window_size: int = 20):
        super().__init__()
        self.window_size = window_size
        self.regimes = []
    
    def observe(self, state: Dict[str, Any]) -> None:
        """Observa preços e volatilidade."""
        super().observe(state)
        
    def act(self) -> Dict[str, Any]:
        """Identifica regime atual do mercado."""
        if not self.current_state:
            return {}
            
        # Calcula indicadores de regime
        returns = self.current_state.get('returns', [])
        volume = self.current_state.get('volume', [])
        
        if len(returns) >= self.window_size:
            # Volatilidade recente
            recent_vol = np.std(returns[-self.window_size:])
            
            # Volume anômalo
            vol_z_score = (volume[-1] - np.mean(volume)) / np.std(volume)
            
            # Classifica regime
            regime = self._classify_regime(recent_vol, vol_z_score)
            self.regimes.append(regime)
            
            return {
                'action': 'regime_identified',
                'regime': regime,
                'volatility': recent_vol,
                'volume_zscore': vol_z_score
            }
        
        return {'action': 'insufficient_data'}
    
    def _classify_regime(self, volatility: float,
                        volume_zscore: float) -> str:
        """Classifica regime de mercado."""
        if volatility > np.percentile(self.history_volatility(), 75):
            if volume_zscore > 1.5:
                return 'high_volatility_high_volume'
            return 'high_volatility_normal_volume'
            
        if volume_zscore > 1.5:
            return 'normal_volatility_high_volume'
            
        return 'normal'
    
    def history_volatility(self) -> List[float]:
        """Calcula histórico de volatilidade."""
        volatility = []
        for state in self.history:
            returns = state.get('returns', [])
            if len(returns) >= self.window_size:
                volatility.append(
                    np.std(returns[-self.window_size:])
                )
        return volatility
    
    def learn(self) -> None:
        """Atualiza classificação de regimes."""
        if len(self.regimes) < self.window_size:
            return
            
        # Analisa transições entre regimes
        transitions = pd.Series(self.regimes).value_counts()
        
        # Pode ajustar parâmetros baseado em transições
        most_common = transitions.index[0]
        if most_common == 'high_volatility_high_volume':
            self.window_size = min(30, self.window_size + 5)
        else:
            self.window_size = max(10, self.window_size - 5)

class FeatureSelectionAgent(MarketAgent):
    """Agente para seleção dinâmica de features."""
    
    def __init__(self, initial_features: List[str],
                 correlation_threshold: float = 0.7):
        super().__init__()
        self.features = initial_features
        self.correlation_threshold = correlation_threshold
        self.feature_importance = {}
    
    def observe(self, state: Dict[str, Any]) -> None:
        """Observa performance das features."""
        super().observe(state)
        
    def act(self) -> Dict[str, Any]:
        """Seleciona features mais relevantes."""
        if not self.current_state:
            return {}
            
        # Analisa correlações entre features
        features_data = self.current_state.get('features_data', {})
        if features_data:
            df = pd.DataFrame(features_data)
            corr = df.corr()
            
            # Remove features altamente correlacionadas
            to_drop = set()
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > self.correlation_threshold:
                        # Mantém feature com maior importância
                        feat1, feat2 = corr.columns[i], corr.columns[j]
                        if self.feature_importance.get(feat1, 0) < \
                           self.feature_importance.get(feat2, 0):
                            to_drop.add(feat1)
                        else:
                            to_drop.add(feat2)
            
            selected_features = [f for f in self.features if f not in to_drop]
            
            return {
                'action': 'feature_selection',
                'selected_features': selected_features,
                'dropped_features': list(to_drop)
            }
        
        return {'action': 'no_change'}
    
    def learn(self) -> None:
        """Atualiza importância das features."""
        if not self.history:
            return
            
        # Acumula importância das features ao longo do tempo
        for state in self.history:
            importance = state.get('feature_importance', {})
            for feature, value in importance.items():
                if feature in self.feature_importance:
                    self.feature_importance[feature] = \
                        0.9 * self.feature_importance[feature] + 0.1 * value
                else:
                    self.feature_importance[feature] = value