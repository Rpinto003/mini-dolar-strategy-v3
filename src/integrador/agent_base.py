from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import pandas as pd

@dataclass
class MarketData:
    """Market data standardized structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_size: float = 0.5  # Tick size for mini dollar
    contract: str = "WDO"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_size": self.tick_size,
            "contract": self.contract
        }

@dataclass
class TechnicalIndicators:
    """Technical indicators standardized structure"""
    rsi: float
    macd: Dict[str, float]  # {'line': float, 'signal': float, 'histogram': float}
    bollinger_bands: Dict[str, float]  # {'upper': float, 'middle': float, 'lower': float}
    volume: float
    additional: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "rsi": self.rsi,
            "macd": self.macd,
            "bollinger_bands": self.bollinger_bands,
            "volume": self.volume,
            "additional": self.additional
        }

@dataclass
class FundamentalIndicators:
    """Fundamental indicators standardized structure"""
    selic_rate: float
    inflation: float
    spot_dollar: float
    additional: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "selic_rate": self.selic_rate,
            "inflation": self.inflation,
            "spot_dollar": self.spot_dollar,
            "additional": self.additional
        }

@dataclass
class AgentDecision:
    """Standardized agent decision"""
    timestamp: datetime
    action: str  # 'CALL', 'PUT', 'HOLD'
    confidence: float
    indicators_used: List[str]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "confidence": self.confidence,
            "indicators_used": self.indicators_used,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "metadata": self.metadata or {}
        }

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or self._setup_logger()
        self.market_data: Optional[MarketData] = None
        self.technical_indicators: Optional[TechnicalIndicators] = None
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(f"logs/agent_{self.name}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def update(self, market_data: MarketData, 
               technical_indicators: TechnicalIndicators) -> bool:
        """Updates agent state with new data"""
        try:
            self.market_data = market_data
            self.technical_indicators = technical_indicators
            self.logger.info(f"Agent {self.name} updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            return False
    
    @abstractmethod
    def analyze(self) -> AgentDecision:
        """
        Abstract method that must be implemented by each specific agent
        to perform its analysis and return a decision
        """
        pass