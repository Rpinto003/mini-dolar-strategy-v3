from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import json
import logging
from pathlib import Path

class DetailedReportingSystem:
    """Sistema de relatórios detalhados com análises e métricas"""
    
    def __init__(self, output_dir: str = "reports", logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("detailed_reporting")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        fh = logging.FileHandler("logs/reporting.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def generate_detailed_trade_report(self, 
                                     order: OrderDetails,
                                     analysis_result: AnalysisResult) -> Dict:
        """Gera relatório detalhado para uma operação específica"""
        report = {
            "informações_gerais": {
                "order_id": order.order_id,
                "timestamp": order.timestamp.isoformat(),
                "tipo": order.order_type.value,
                "status": order.status.value
            },
            "preços": {
                "entrada": order.entry_price,
                "saída": order.exit_price,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit
            },
            "resultado": {
                "valor": order.result,
                "duração": str(order.exit_timestamp - order.timestamp) if order.exit_timestamp else None
            },
            "análise_técnica": {
                "indicadores": order.technical_indicators,
                "score": analysis_result.technical_score
            },
            "análise_fundamental": {
                "indicadores": order.fundamental_data,
                "score": analysis_result.fundamental_score
            },
            "análise_ml": {
                "previsões": order.ml_predictions,
                "score": analysis_result.ml_score
            },
            "metadados": order.metadata
        }
        
        # Salva relatório em arquivo
        filename = f"{order.order_id}_detailed_report.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Relatório detalhado gerado: {filepath}")
        
        return report
    
    def generate_daily_summary(self, 
                             orders: List[OrderDetails],
                             date: Optional[datetime] = None) -> Dict:
        """Gera resumo diário das operações"""
        date = date or datetime.now()
        date_str = date.strftime('%Y-%m-%d')
        
        # Filtra ordens do dia
        daily_orders = [
            o for o in orders
            if o.timestamp.date() == date.date()
        ]
        
        if not daily_orders:
            return {"message": f"Nenhuma operação no dia {date_str}"}
        
        # Análise de performance
        total_result = sum(o.result for o in daily_orders if o.result)
        winning_trades = len([o for o in daily_orders if o.result and o.result > 0])
        total_trades = len(daily_orders)
        
        summary = {
            "data": date_str,
            "resultado": {
                "total": total_result,
                "win_rate": f"{(winning_trades/total_trades)*100:.2f}%",
                "número_operações": total_trades
            },
            "operações": {
                "CALL": len([o for o in daily_orders if o.order_type == OrderType.CALL]),
                "PUT": len([o for o in daily_orders if o.order_type == OrderType.PUT])
            },
            "detalhamento": [
                {
                    "id": order.order_id,
                    "tipo": order.order_type.value,
                    "entrada": order.entry_price,
                    "saída": order.exit_price,
                    "resultado": order.result,
                    "status": order.status.value
                }
                for order in daily_orders
            ]
        }
        
        # Salva resumo em arquivo
        filename = f"daily_summary_{date_str}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            