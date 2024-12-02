import pytest
from datetime import datetime
from src.integrador.agent_base import (
    BaseAgent, MarketData, TechnicalIndicators, AgentDecision
)

@pytest.fixture
def sample_market_data(sample_market_data):
    """Converte dados de mercado do fixture existente para novo formato"""
    last_data = sample_market_data.iloc[-1]
    return MarketData(
        timestamp=last_data.name,
        open=last_data['open'],
        high=last_data['high'],
        low=last_data['low'],
        close=last_data['close'],
        volume=last_data['volume']
    )

@pytest.fixture
def sample_technical_indicators():
    """Cria indicadores técnicos de exemplo"""
    return TechnicalIndicators(
        rsi=45.5,
        macd={'line': 0.5, 'signal': 0.3, 'histogram': 0.2},
        bollinger={'upper': 5010, 'middle': 5000, 'lower': 4990},
        volume_profile={'poc': 5000, 'vah': 5010, 'val': 4990},
        additional={'momentum': 0.5}
    )

class TestAgent(BaseAgent):
    """Agente de teste que implementa a interface base"""
    def analyze(self) -> AgentDecision:
        return AgentDecision(
            timestamp=datetime.now(),
            action='HOLD',
            confidence=0.5,
            target_price=None,
            stop_loss=None,
            indicators_used=['RSI', 'MACD'],
            reasoning="Teste",
            metadata={}
        )

def test_agent_initialization():
    """Testa inicialização básica do agente"""
    agent = TestAgent("TestAgent")
    assert agent.name == "TestAgent"
    assert agent.market_data is None
    assert agent.technical_indicators is None

def test_market_data_validation(sample_market_data):
    """Testa validação de dados de mercado"""
    agent = TestAgent("TestAgent")
    
    # Teste com dados válidos
    assert agent.validate_market_data(sample_market_data)
    
    # Teste com dados inválidos
    invalid_data = MarketData(
        timestamp=datetime.now(),
        open=-1,  # Preço negativo inválido
        high=100,
        low=90,
        close=95,
        volume=1000
    )
    assert not agent.validate_market_data(invalid_data)

def test_technical_indicators_validation(sample_technical_indicators):
    """Testa validação de indicadores técnicos"""
    agent = TestAgent("TestAgent")
    
    # Teste com indicadores válidos
    assert agent.validate_technical_indicators(sample_technical_indicators)
    
    # Teste com RSI inválido
    invalid_indicators = TechnicalIndicators(
        rsi=150,  # RSI deve estar entre 0 e 100
        macd=sample_technical_indicators.macd,
        bollinger=sample_technical_indicators.bollinger,
        volume_profile=sample_technical_indicators.volume_profile,
        additional={}
    )
    assert not agent.validate_technical_indicators(invalid_indicators)

def test_agent_update_and_analyze(sample_market_data, sample_technical_indicators):
    """Testa fluxo completo de atualização e análise"""
    agent = TestAgent("TestAgent")
    
    # Atualiza o agente
    success = agent.update(sample_market_data, sample_technical_indicators)
    assert success
    
    # Verifica se os dados foram armazenados
    assert agent.market_data == sample_market_data
    assert agent.technical_indicators == sample_technical_indicators
    
    # Testa análise
    decision = agent.analyze()
    assert isinstance(decision, AgentDecision)
    assert decision.action in ['BUY', 'SELL', 'HOLD', 'CALL', 'PUT']
    assert 0 <= decision.confidence <= 1

def test_agent_state_persistence(tmp_path, sample_market_data, sample_technical_indicators):
    """Testa persistência do estado do agente"""
    agent = TestAgent("TestAgent")
    agent.update(sample_market_data, sample_technical_indicators)
    
    # Salva estado
    state_file = tmp_path / "agent_state.json"
    agent.save_state(str(state_file))
    
    # Verifica se arquivo foi criado
    assert state_file.exists()
    
    # Verifica conteúdo
    state = agent.to_dict()
    assert state['name'] == "TestAgent"
    assert 'market_data' in state
    assert 'technical_indicators' in state