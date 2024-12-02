import pytest
import numpy as np
from src.agents.market_agent import RiskManagementAgent, MarketRegimeAgent, FeatureSelectionAgent

@pytest.fixture
def sample_equity_data():
    return {
        'equity': [100000, 102000, 101000, 99000, 98000]
    }

@pytest.fixture
def sample_market_data():
    returns = np.random.normal(0, 0.01, 100)
    volume = np.random.normal(1000, 100, 100)
    return {
        'returns': returns.tolist(),
        'volume': volume.tolist()
    }

@pytest.fixture
def sample_features_data():
    np.random.seed(42)
    n_samples = 100
    data = {
        'sma_20': np.random.randn(n_samples),
        'rsi': np.random.randn(n_samples),
        'macd': np.random.randn(n_samples),
        'volume_ma': np.random.randn(n_samples)
    }
    return {'features_data': data}

def test_risk_management_agent(sample_equity_data):
    agent = RiskManagementAgent(max_drawdown=-0.02)
    
    # Testa observação inicial
    agent.observe(sample_equity_data)
    assert agent.current_state == sample_equity_data
    
    # Testa ação com drawdown
    action = agent.act()
    assert action['action'] == 'reduce_risk'
    assert action['max_positions'] == 1
    
    # Testa aprendizado
    agent.learn()
    assert agent.max_drawdown < 0  # Deve ter ajustado o limite

def test_market_regime_agent(sample_market_data):
    agent = MarketRegimeAgent(window_size=20)
    
    # Testa observação inicial
    agent.observe(sample_market_data)
    assert agent.current_state == sample_market_data
    
    # Testa identificação de regime
    action = agent.act()
    assert action['action'] == 'regime_identified'
    assert 'regime' in action
    assert 'volatility' in action
    
    # Testa aprendizado
    agent.learn()
    assert agent.window_size >= 10  # Deve manter window size razoável

def test_feature_selection_agent(sample_features_data):
    initial_features = ['sma_20', 'rsi', 'macd', 'volume_ma']
    agent = FeatureSelectionAgent(initial_features, correlation_threshold=0.7)
    
    # Testa observação inicial
    agent.observe(sample_features_data)
    assert agent.current_state == sample_features_data
    
    # Testa seleção de features
    action = agent.act()
    assert action['action'] == 'feature_selection'
    assert 'selected_features' in action
    assert len(action['selected_features']) <= len(initial_features)
    
    # Testa aprendizado
    agent.learn()
    assert len(agent.feature_importance) > 0

def test_agent_interaction():
    """Testa interação entre diferentes agentes."""
    risk_agent = RiskManagementAgent()
    regime_agent = MarketRegimeAgent()
    feature_agent = FeatureSelectionAgent(['sma_20', 'rsi'])
    
    # Cria estado combinado
    state = {
        'equity': [100000, 99000],
        'returns': np.random.randn(50).tolist(),
        'volume': np.random.randn(50).tolist(),
        'features_data': {
            'sma_20': np.random.randn(50),
            'rsi': np.random.randn(50)
        }
    }
    
    # Testa resposta de cada agente
    for agent in [risk_agent, regime_agent, feature_agent]:
        agent.observe(state)
        action = agent.act()
        assert isinstance(action, dict)
        agent.learn()

def test_agent_reset():
    """Testa reset dos agentes."""
    agents = [
        RiskManagementAgent(),
        MarketRegimeAgent(),
        FeatureSelectionAgent(['sma_20'])
    ]
    
    state = {'equity': [100000]}
    
    for agent in agents:
        # Adiciona alguns dados
        agent.observe(state)
        agent.act()
        
        # Reseta
        agent.__init__()
        
        # Verifica se foi resetado
        assert agent.current_state is None
        assert len(agent.history) == 0