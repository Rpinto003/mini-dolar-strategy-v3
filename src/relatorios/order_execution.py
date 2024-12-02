from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import pandas as pd

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    STOPPED = "STOPPED"  # Stop loss ou take profit atingido

class OrderType(Enum):
    CALL = "CALL"
    PUT = "PUT"

@dataclass
class OrderDetails:
    """Detalhes completos de uma ordem"""
    order_id: str
    timestamp: datetime
    order_type: OrderType
    entry_price: float
    quantity: int
    status: OrderStatus
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    result: Optional[float] = None
    technical_indicators: Dict[str, Any] = None
    fundamental_data: Dict[str, Any] = None
    ml_predictions: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

class OrderExecutionSystem:
    """Sistema de execução e gestão de ordens"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.active_orders: Dict[str, OrderDetails] = {}
        self.completed_orders: List[OrderDetails] = []
        self.order_counter = 0
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("order_execution")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para arquivo
        fh = logging.FileHandler("logs/orders.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Handler para console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def _generate_order_id(self) -> str:
        """Gera ID único para ordem"""
        self.order_counter += 1
        return f"ORD{datetime.now().strftime('%Y%m%d')}-{self.order_counter:04d}"
    
    def create_order(self, 
                    analysis_result: AnalysisResult,
                    current_price: float,
                    quantity: int = 1) -> Optional[OrderDetails]:
        """Cria uma nova ordem baseada na análise"""
        try:
            if analysis_result.final_decision == 'HOLD':
                self.logger.info("Análise sugere HOLD - nenhuma ordem criada")
                return None
            
            order_type = OrderType.CALL if analysis_result.final_decision == 'CALL' else OrderType.PUT
            
            # Cálculo de stop loss e take profit baseado na volatilidade ou outros fatores
            atr = analysis_result.indicators.get('technical', {}).get('ATR', 0)
            stop_loss = current_price - (atr * 1.5) if order_type == OrderType.CALL else current_price + (atr * 1.5)
            take_profit = current_price + (atr * 2) if order_type == OrderType.CALL else current_price - (atr * 2)
            
            order = OrderDetails(
                order_id=self._generate_order_id(),
                timestamp=datetime.now(),
                order_type=order_type,
                entry_price=current_price,
                quantity=quantity,
                status=OrderStatus.PENDING,
                stop_loss=stop_loss,
                take_profit=take_profit,
                technical_indicators=analysis_result.indicators.get('technical'),
                fundamental_data=analysis_result.indicators.get('fundamental'),
                ml_predictions=analysis_result.indicators.get('ml'),
                metadata={
                    'confidence': analysis_result.confidence,
                    'technical_score': analysis_result.technical_score,
                    'fundamental_score': analysis_result.fundamental_score,
                    'ml_score': analysis_result.ml_score
                }
            )
            
            self.active_orders[order.order_id] = order
            self.logger.info(f"Ordem criada: {order.order_id} - {order_type.value} @ {current_price}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Erro ao criar ordem: {str(e)}")
            return None
    
    def update_order(self, order_id: str, current_price: float,
                    current_time: datetime) -> None:
        """Atualiza status de uma ordem com base no preço atual"""
        if order_id not in self.active_orders:
            self.logger.error(f"Ordem não encontrada: {order_id}")
            return
            
        order = self.active_orders[order_id]
        
        # Verifica stop loss e take profit
        if order.order_type == OrderType.CALL:
            if current_price <= order.stop_loss:
                self._close_order(order_id, current_price, current_time, OrderStatus.STOPPED)
            elif current_price >= order.take_profit:
                self._close_order(order_id, current_price, current_time, OrderStatus.FILLED)
        else:  # PUT
            if current_price >= order.stop_loss:
                self._close_order(order_id, current_price, current_time, OrderStatus.STOPPED)
            elif current_price <= order.take_profit:
                self._close_order(order_id, current_price, current_time, OrderStatus.FILLED)
    
    def _close_order(self, order_id: str, exit_price: float,
                    exit_time: datetime, status: OrderStatus) -> None:
        """Finaliza uma ordem ativa"""
        order = self.active_orders[order_id]
        order.exit_price = exit_price
        order.exit_timestamp = exit_time
        order.status = status
        
        # Calcula resultado
        multiplier = 1 if order.order_type == OrderType.CALL else -1
        order.result = (exit_price - order.entry_price) * multiplier * order.quantity
        
        # Move para ordens completadas
        self.completed_orders.append(order)
        del self.active_orders[order_id]
        
        self.logger.info(
            f"Ordem {order_id} fechada: {status.value} - "
            f"Resultado: {order.result:.2f}"
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem ativa"""
        if order_id not in self.active_orders:
            self.logger.error(f"Ordem não encontrada: {order_id}")
            return False
            
        order = self.active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.exit_timestamp = datetime.now()
        
        self.completed_orders.append(order)
        del self.active_orders[order_id]
        
        self.logger.info(f"Ordem cancelada: {order_id}")
        return True
    
    def get_active_orders(self) -> List[OrderDetails]:
        """Retorna lista de ordens ativas"""
        return list(self.active_orders.values())
    
    def get_completed_orders(self, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[OrderDetails]:
        """Retorna ordens completadas em um período"""
        if not start_time and not end_time:
            return self.completed_orders
            
        return [
            order for order in self.completed_orders
            if (not start_time or order.timestamp >= start_time) and
               (not end_time or order.timestamp <= end_time)
        ]
    
    def generate_performance_report(self, 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Dict:
        """Gera relatório de performance"""
        orders = self.get_completed_orders(start_time, end_time)
        
        if not orders:
            return {"message": "Nenhuma ordem no período especificado"}
        
        # Cálculo de métricas
        total_orders = len(orders)
        winning_orders = len([o for o in orders if o.result and o.result > 0])
        losing_orders = len([o for o in orders if o.result and o.result < 0])
        
        total_profit = sum(o.result for o in orders if o.result and o.result > 0)
        total_loss = sum(o.result for o in orders if o.result and o.result < 0)
        
        win_rate = (winning_orders / total_orders) * 100 if total_orders > 0 else 0
        
        return {
            "período": {
                "início": start_time.isoformat() if start_time else "início",
                "fim": end_time.isoformat() if end_time else "atual"
            },
            "métricas_gerais": {
                "total_ordens": total_orders,
                "ordens_ganhadoras": winning_orders,
                "ordens_perdedoras": losing_orders,
                "win_rate": f"{win_rate:.2f}%",
                "resultado_total": total_profit + total_loss,
                "lucro_total": total_profit,
                "prejuízo_total": total_loss
            },
            "médias": {
                "lucro_médio": total_profit / winning_orders if winning_orders > 0 else 0,
                "prejuízo_médio": total_loss / losing_orders if losing_orders > 0 else 0,
                "resultado_por_ordem": (total_profit + total_loss) / total_orders
            },
            "detalhamento": {
                "CALL": len([o for o in orders if o.order_type == OrderType.CALL]),
                "PUT": len([o for o in orders if o.order_type == OrderType.PUT])
            }
        }
