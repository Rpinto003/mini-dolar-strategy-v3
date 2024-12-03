from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import pandas as pd
from dataclasses import dataclass

# Importação necessária da classe BaseAgent
from .agent_base import BaseAgent, MarketData, TechnicalIndicators, AgentDecision

@dataclass
class AnalysisResult:
    """Resultado consolidado das análises"""
    timestamp: datetime
    technical_score: float  # -1 a 1 (bearish a bullish)
    fundamental_score: float  # -1 a 1
    ml_score: float  # -1 a 1
    final_decision: str  # 'CALL', 'PUT', 'HOLD'
    confidence: float  # 0 a 1
    indicators: Dict[str, Any]
    metadata: Dict[str, Any]

class IntegrationManager:
    """Gerencia a integração entre agentes, análises e relatórios"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.agents: Dict[str, BaseAgent] = {}
        self.last_analysis: Optional[AnalysisResult] = None
        self.analysis_history: List[AnalysisResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("integration_manager")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para arquivo
        fh = logging.FileHandler("logs/integration.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Registra um novo agente"""
        self.agents[agent.name] = agent
        self.logger.info(f"Agente registrado: {agent.name}")
    
    def update_market_data(self, market_data: MarketData,
                          technical_indicators: TechnicalIndicators) -> bool:
        """Atualiza todos os agentes com novos dados de mercado"""
        try:
            success = True
            for agent in self.agents.values():
                if not agent.update(market_data, technical_indicators):
                    self.logger.error(f"Falha ao atualizar agente: {agent.name}")
                    success = False
            return success
        except Exception as e:
            self.logger.error(f"Erro na atualização de dados: {str(e)}")
            return False

    def collect_agent_decisions(self) -> Dict[str, AgentDecision]:
        """Coleta decisões de todos os agentes"""
        decisions = {}
        for name, agent in self.agents.items():
            try:
                decisions[name] = agent.analyze()
            except Exception as e:
                self.logger.error(f"Erro ao coletar decisão do agente {name}: {str(e)}")
        return decisions
    
    def consolidate_analysis(self, 
                           agent_decisions: Dict[str, AgentDecision],
                           fundamental_data: Optional[Dict] = None,
                           ml_predictions: Optional[Dict] = None) -> AnalysisResult:
        """Consolida análises de diferentes fontes"""
        try:
            # Processamento das decisões dos agentes técnicos
            tech_scores = []
            for decision in agent_decisions.values():
                score = self._convert_decision_to_score(decision)
                tech_scores.append(score * decision.confidence)
            
            technical_score = sum(tech_scores) / len(tech_scores) if tech_scores else 0
            
            # Processamento da análise fundamental
            fundamental_score = self._process_fundamental_data(fundamental_data) if fundamental_data else 0
            
            # Processamento das previsões ML
            ml_score = self._process_ml_predictions(ml_predictions) if ml_predictions else 0
            
            # Pesos para cada tipo de análise
            weights = {
                'technical': 0.4,
                'fundamental': 0.3,
                'ml': 0.3
            }
            
            # Cálculo do score final ponderado
            final_score = (
                technical_score * weights['technical'] +
                fundamental_score * weights['fundamental'] +
                ml_score * weights['ml']
            )
            
            # Determinação da decisão final
            final_decision, confidence = self._determine_final_decision(final_score)
            
            result = AnalysisResult(
                timestamp=datetime.now(),
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                ml_score=ml_score,
                final_decision=final_decision,
                confidence=confidence,
                indicators={
                    'technical': {name: d.indicators_used for name, d in agent_decisions.items()},
                    'fundamental': fundamental_data.get('indicators', {}) if fundamental_data else {},
                    'ml': ml_predictions.get('features', {}) if ml_predictions else {}
                },
                metadata={
                    'agent_decisions': {name: d.to_dict() for name, d in agent_decisions.items()},
                    'fundamental_data': fundamental_data,
                    'ml_predictions': ml_predictions
                }
            )
            
            self.last_analysis = result
            self.analysis_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na consolidação das análises: {str(e)}")
            raise
    
    def _convert_decision_to_score(self, decision: AgentDecision) -> float:
        """Converte decisão do agente para score numérico"""
        action_scores = {
            'BUY': 1.0,
            'CALL': 1.0,
            'SELL': -1.0,
            'PUT': -1.0,
            'HOLD': 0.0
        }
        return action_scores.get(decision.action, 0.0)
    
    def _process_fundamental_data(self, data: Dict) -> float:
        """Processa dados fundamentalistas em score"""
        # Implementação específica para seus indicadores fundamentalistas
        return data.get('score', 0.0)
    
    def _process_ml_predictions(self, predictions: Dict) -> float:
        """Processa previsões ML em score"""
        # Implementação específica para seus modelos ML
        return predictions.get('probability', 0.0) * 2 - 1  # Converte prob [0,1] para score [-1,1]
    
def _determine_final_decision(self, score: float) -> tuple[str, float]:
    """Determina decisão final baseada no score consolidado"""
    confidence = abs(score)
    
    # Reduzindo thresholds para gerar mais sinais
    if score > 0.1:  # Reduzido de 0.15
        return 'CALL', confidence
    elif score < -0.1:  # Reduzido de -0.15
        return 'PUT', confidence
    else:
        return 'HOLD', confidence