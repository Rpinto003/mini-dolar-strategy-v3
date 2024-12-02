import pytest
import numpy as np
from src.agents.market_agent import RiskManagementAgent, MarketRegimeAgent, FeatureSelectionAgent
from src.agents.agent_coordinator import AgentCoordinator

@pytest.fixture
def sample_agents():
    return [
        RiskManagementAgent(max_drawdown=-0.05),
        MarketRegimeAgent(window_size=20),
        FeatureSelectionAgent(['sma_20', 'rsi', 'macd'])
    ]

@pytest.fixture
def sample_state():
    return {
        'equity': [100000, 98000, 97000],
        'returns': np.random.randn(50).tolist(),
        'volume': np.random.randint(1000, 2000, 50).tolist(),
        'features_data': {
            'sma_20': np.random.randn(50),
            'rsi': np.random.randn(50),
            'macd': np.random.randn(50)
        }
    }

def test_coordinator_initialization(sample_agents):
    coordinator = AgentCoordinator(sample_agents)
    
    assert len(coordinator.agents) == len(sample_agents)
    assert len(coordinator.state_history) == 0
    assert len(coordinator.action_history) == 0

def test_update_state(sample_agents, sample_state):
    coordinator = AgentCoordinator(sample_agents)
    
    coordinator.update_state(sample_state)
    
    assert len(coordinator.state_history) == 1
    assert coordinator.state_history[0] == sample_state
    
    # Verifica se todos os agentes receberam o estado
    for agent in coordinator.agents:
        assert agent.current_state == sample_state

def test_get_actions(sample_agents, sample_state):
    coordinator = AgentCoordinator(sample_agents)
    coordinator.update_state(sample_state)
    
    actions = coordinator.get_actions()
    
    # Verifica se retornou ações combinadas
    assert isinstance(actions, dict)
    assert 'risk_level' in actions
    assert 'market_regime' in actions or 'selected_features' in actions
    
    # Verifica histórico de ações
    assert len(coordinator.action_history) == 1

def test_combine_actions(sample_agents):
    coordinator = AgentCoordinator(sample_agents)
    
    test_actions = {
        'RiskManagementAgent': {
            'action': 'reduce_risk',
            'max_positions': 1
        },
        'MarketRegimeAgent': {
            'action': 'regime_identified',
            'regime': 'high_volatility',
            'volatility': 0.02
        },
        'FeatureSelectionAgent': {
            'action': 'feature_selection',
            'selected_features': ['sma_20', 'rsi']
        }
    }
    
    combined = coordinator._combine_actions(test_actions)
    
    assert combined['risk_level'] == 'high'
    assert combined['max_positions'] == 1
    assert combined['market_regime'] == 'high_volatility'
    assert 'selected_features' in combined

def test_get_agent_insights(sample_agents, sample_state):
    coordinator = AgentCoordinator(sample_agents)
    coordinator.update_state(sample_state)
    coordinator.get_actions()  # Gera algumas ações primeiro
    
    insights = coordinator.get_agent_insights()
    
    # Verifica se há insights para cada agente
    assert 'RiskManagementAgent' in insights
    assert 'MarketRegimeAgent' in insights
    assert 'FeatureSelectionAgent' in insights
    
    # Verifica conteúdo dos insights
    risk_insights = insights['RiskManagementAgent']
    assert 'max_drawdown' in risk_insights
    assert 'current_risk_level' in risk_insights
    
    regime_insights = insights['MarketRegimeAgent']
    assert 'current_regime' in regime_insights
    assert 'volatility_trend' in regime_insights

def test_coordinator_reset(sample_agents, sample_state):
    coordinator = AgentCoordinator(sample_agents)
    
    # Adiciona alguns estados e ações
    coordinator.update_state(sample_state)
    coordinator.get_actions()
    
    # Reseta
    coordinator.reset()
    
    # Verifica se tudo foi limpo
    assert len(coordinator.state_history) == 0
    assert len(coordinator.action_history) == 0
    
    # Verifica se os agentes foram resetados
    for agent in coordinator.agents:
        assert agent.current_state is None
        assert len(agent.history) == 0

def test_coordinator_multiple_updates(sample_agents):
    coordinator = AgentCoordinator(sample_agents)
    
    # Gera vários estados
    states = [
        {
            'equity': [100000 - i*1000 for i in range(3)],
            'returns': np.random.randn(50).tolist(),
            'volume': np.random.randint(1000, 2000, 50).tolist(),
            'features_data': {
                'sma_20': np.random.randn(50),
                'rsi': np.random.randn(50),
                'macd': np.random.randn(50)
            }
        }
        for _ in range(5)
    ]
    
    # Atualiza múltiplas vezes
    for state in states:
        coordinator.update_state(state)
        actions = coordinator.get_actions()
        
        assert isinstance(actions, dict)
        assert 'risk_level' in actions
    
    # Verifica histórico
    assert len(coordinator.state_history) == len(states)
    assert len(coordinator.action_history) == len(states)