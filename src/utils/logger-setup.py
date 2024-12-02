import logging
from pathlib import Path
from typing import Optional
import sys
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configura um logger com output para arquivo e console
    
    Args:
        name: Nome do logger
        log_dir: Diretório para arquivos de log
        level: Nível de logging
    
    Returns:
        Logger configurado
    """
    # Cria diretório de logs se não existir
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Cria logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove handlers existentes
    logger.handlers = []
    
    # Formato do log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para arquivo
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        log_path / f"{name}_{today}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Função para criar logger específico de trade
def setup_trade_logger(
    strategy_name: str,
    log_dir: str = "logs/trades"
) -> logging.Logger:
    """
    Configura logger específico para trades
    
    Args:
        strategy_name: Nome da estratégia
        log_dir: Diretório para logs de trade
    
    Returns:
        Logger configurado para trades
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"trade_{strategy_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Formato específico para trades
    trade_formatter = logging.Formatter(
        '%(asctime)s;%(levelname)s;%(message)s'
    )
    
    # Handler para arquivo de trades
    today = datetime.now().strftime('%Y-%m-%d')
    trade_handler = logging.FileHandler(
        log_path / f"trades_{strategy_name}_{today}.log",
        encoding='utf-8'
    )
    trade_handler.setFormatter(trade_formatter)
    logger.addHandler(trade_handler)
    
    return logger
