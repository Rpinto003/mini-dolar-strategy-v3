import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_market_data():
    """Gera dados de mercado sintéticos para testes."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 5000,
        'high': np.random.randn(100).cumsum() + 5010,
        'low': np.random.randn(100).cumsum() + 4990,
        'close': np.random.randn(100).cumsum() + 5000,
        'volume': np.random.randint(100, 1000, 100),
        'tick_volume': np.random.randint(50, 500, 100),
        'spread': np.ones(100)
    }, index=dates)
    return data

@pytest.fixture
def sample_economic_data():
    """Gera dados econômicos sintéticos para testes."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'selic': pd.Series(np.ones(100) * 13.75, index=dates),
        'ipca': pd.Series(np.random.randn(100) * 0.1 + 0.5, index=dates),
        'fed_rate': pd.Series(np.ones(100) * 5.25, index=dates)
    }
    return data

@pytest.fixture
def sample_news_data():
    """Gera dados de notícias sintéticos para testes."""
    return [
        {
            'title': 'Fed mantém taxa de juros',
            'content': 'Federal Reserve mantém taxa de juros em reunião',
            'date': '2023-01-01',
            'source': 'test'
        },
        {
            'title': 'COPOM aumenta SELIC',
            'content': 'Banco Central aumenta taxa SELIC em 0.5%',
            'date': '2023-01-02',
            'source': 'test'
        }
    ]