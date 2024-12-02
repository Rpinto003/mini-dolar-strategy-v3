from typing import List, Dict, Any
from src.agents.market_agent import MarketAgent

class AgentCoordinator:
    """Coordena múltiplos agentes de mercado."""
    
    def __init__(self, agents: List[MarketAgent]):
        """Inicializa coordenador.
        
        Args:
            agents: Lista de agentes para coordenar
        """
        self.agents = agents
        self.state_history = []
        self.action_history = []
    
    def update_state(self, state: Dict[str, Any]) -> None:
        """Atualiza estado para todos os agentes.
        
        Args:
            state: Estado atual do mercado
        """
        self.state_history.append(state)
        
        # Notifica todos os agentes
        for agent in self.agents:
            agent.observe(state)
    
    def get_actions(self) -> Dict[str, Any]:
        """Coleta e combina ações dos agentes.
        
        Returns:
            Dict com ações combinadas
        """
        actions = {}
        
        # Coleta ações de cada agente
        for agent in self.agents:
            agent_action = agent.act()
            
            # Registra ações
            actions[agent.__class__.__name__] = agent_action
            
            # Aprende com ações tomadas
            agent.learn()
        
        self.action_history.append(actions)
        return self._combine_actions(actions)
    
    def _combine_actions(self, actions: Dict[str, Dict]) -> Dict[str, Any]:
        """Combina ações dos diferentes agentes.
        
        Args:
            actions: Dicionário com ações de cada agente
            
        Returns:
            Dict com ações combinadas
        """
        combined = {}
        
        # Prioriza ações de risco
        risk_action = actions.get('RiskManagementAgent', {})
        if risk_action.get('action') == 'reduce_risk':
            combined['risk_level'] = 'high'
            combined['max_positions'] = risk_action['max_positions']
        else:
            combined['risk_level'] = 'normal'
        
        # Considera regime de mercado
        regime_action = actions.get('MarketRegimeAgent', {})
        if regime_action.get('action') == 'regime_identified':
            combined['market_regime'] = regime_action['regime']
            combined['volatility'] = regime_action['volatility']
        
        # Adiciona features selecionadas
        feature_action = actions.get('FeatureSelectionAgent', {})
        if feature_action.get('action') == 'feature_selection':
            combined['selected_features'] = feature_action['selected_features']
        
        return combined
    
    def get_agent_insights(self) -> Dict[str, Any]:
        """Retorna insights de todos os agentes.
        
        Returns:
            Dict com insights de cada agente
        """
        insights = {}
        
        for agent in self.agents:
            agent_name = agent.__class__.__name__
            
            # Coleta métricas específicas de cada agente
            if agent_name == 'RiskManagementAgent':
                insights[agent_name] = {
                    'max_drawdown': agent.max_drawdown,
                    'max_position_size': agent.max_position_size,
                    'current_risk_level': 'high' if agent.act().get('action') == 'reduce_risk' else 'normal'
                }
            elif agent_name == 'MarketRegimeAgent':
                insights[agent_name] = {
                    'current_regime': agent.regimes[-1] if agent.regimes else None,
                    'regime_transitions': len(set(agent.regimes)),
                    'volatility_trend': self._calculate_trend(agent.history_volatility())
                }
            elif agent_name == 'FeatureSelectionAgent':
                insights[agent_name] = {
                    'n_selected_features': len(agent.features),
                    'top_features': sorted(
                        agent.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5],
                    'feature_stability': self._calculate_feature_stability(agent)
                }
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência de uma série de valores."""
        if len(values) < 2:
            return 'undefined'
            
        # Calcula média móvel
        window = min(len(values), 5)
        ma = sum(values[-window:]) / window
        prev_ma = sum(values[-(window+1):-1]) / window
        
        # Determina direção
        if ma > prev_ma * 1.05:
            return 'increasing'
        elif ma < prev_ma * 0.95:
            return 'decreasing'
        return 'stable'
    
    def _calculate_feature_stability(self, agent) -> float:
        """Calcula estabilidade da seleção de features."""
        if len(self.action_history) < 2:
            return 1.0
            
        # Compara features selecionadas entre ações consecutivas
        stability_scores = []
        for i in range(1, len(self.action_history)):
            prev_features = set(self.action_history[i-1].get(
                'FeatureSelectionAgent', {}).get('selected_features', []
            ))
            curr_features = set(self.action_history[i].get(
                'FeatureSelectionAgent', {}).get('selected_features', []
            ))
            
            if prev_features and curr_features:
                intersection = len(prev_features & curr_features)
                union = len(prev_features | curr_features)
                stability_scores.append(intersection / union)
        
        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0
    
    def reset(self) -> None:
        """Reseta estado de todos os agentes."""
        self.state_history.clear()
        self.action_history.clear()
        
        for agent in self.agents:
            agent.__init__()  # Reinicializa cada agente