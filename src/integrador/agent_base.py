from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import logging
from datetime import datetime

@dataclass
class MarketData:
    """Dados de mercado padronizados"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_size: float = 0.5  # Tamanho do tick para mini dólar
    contract: str = "WDO"

@dataclass
class IndicadoresTecnicos:
    rsi: float
    macd: Dict[str, float]
    bandas_bollinger: Dict[str, float]
    volume: float
    outros: Dict[str, float]

@dataclass
class IndicadoresFundamentais:
    taxa_selic: float
    inflacao: float
    dolar_spot: float
    outros: Dict[str, float]

class IntegradorAgenteModelo:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def validar_dados(self, dados: pd.DataFrame) -> bool:
        """Valida se todos os campos necessários estão presentes e com valores válidos"""
        campos_obrigatorios = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        try:
            for campo in campos_obrigatorios:
                if campo not in dados.columns:
                    self.logger.error(f"Campo obrigatório ausente: {campo}")
                    return False
                
            if dados.isnull().any().any():
                self.logger.error("Dados contêm valores nulos")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Erro na validação dos dados: {str(e)}")
            return False
    
    def preparar_dados_modelo(self, 
                            dados_mercado: pd.DataFrame,
                            ind_tecnicos: IndicadoresTecnicos,
                            ind_fundamentais: Optional[IndicadoresFundamentais] = None) -> Dict:
        """Prepara e unifica os dados para envio ao modelo"""
        try:
            if not self.validar_dados(dados_mercado):
                raise ValueError("Dados de mercado inválidos")
            
            dados_unificados = {
                'timestamp': datetime.now().isoformat(),
                'mercado': {
                    'ultimo': dados_mercado['close'].iloc[-1],
                    'volume': dados_mercado['volume'].iloc[-1],
                    'variacao': dados_mercado['close'].pct_change().iloc[-1]
                },
                'tecnicos': {
                    'rsi': ind_tecnicos.rsi,
                    'macd': ind_tecnicos.macd,
                    'bb': ind_tecnicos.bandas_bollinger,
                    'outros': ind_tecnicos.outros
                },
                'fundamentais': None if not ind_fundamentais else {
                    'selic': ind_fundamentais.taxa_selic,
                    'inflacao': ind_fundamentais.inflacao,
                    'dolar_spot': ind_fundamentais.dolar_spot,
                    'outros': ind_fundamentais.outros
                }
            }
            
            self.logger.info("Dados preparados com sucesso")
            return dados_unificados
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar dados: {str(e)}")
            raise
    
    def registrar_decisao(self, 
                         dados_entrada: Dict,
                         predicao: str,
                         confianca: float,
                         metadados: Optional[Dict] = None) -> Dict:
        """Registra a decisão do modelo com todos os dados relevantes"""
        registro = {
            'timestamp': datetime.now().isoformat(),
            'dados_entrada': dados_entrada,
            'predicao': predicao,
            'confianca': confianca,
            'metadados': metadados or {}
        }
        
        self.logger.info(f"Decisão registrada: {predicao} (confiança: {confianca})")
        return registro
