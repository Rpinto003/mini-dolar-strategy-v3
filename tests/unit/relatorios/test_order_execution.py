import pytest
from datetime import datetime, timedelta
from src.relatorios.order_execution import (
    OrderExecutionSystem, OrderDetails, OrderStatus, OrderType
)
from src.integrador.agent_base import AnalysisResult

@pytest.fixture
def sample_analysis_result():
    return AnalysisResult(
        timestamp=datetime.now(),
        technical_score=0.7,
        fundamental_score=0.5,
        ml_score=0.6,
        final_decision='CALL',
        confidence=0.8,
        indicators={
            'technical': {'RSI': 65, 'MACD': {'line': 0.5}},
            'fundamental': {'selic': 13.75},
            'ml': {'probability': 0.8}
        },
        metadata={'volatility': 0.02}
    )

@pytest.fixture
def execution_system():
    return OrderExecutionSystem()

def test_create_order(execution_system, sample_analysis_result):
    """Testa criação de ordem"""
    current_price = 5000.0
    
    order = execution_system.create_order(
        sample_analysis_result,
        current_price,
        quantity=1
    )
    
    assert order is not None
    assert order.order_type == OrderType.CALL
    assert order.entry_price == current_price
    assert order.status == OrderStatus.PENDING
    assert order.stop_loss is not None
    assert order.take_profit is not None

def test_order_lifecycle(execution_system, sample_analysis_result):
    """Testa ciclo de vida completo de uma ordem"""
    # Cria ordem
    current_price = 5000.0
    order = execution_system.create_order(sample_analysis_result, current_price)
    order_id = order.order_id
    
    # Verifica ordem ativa
    active_orders = execution_system.get_active_orders()
    assert len(active_orders) == 1
    assert active_orders[0].order_id == order_id
    
    # Atualiza preço (stop loss atingido)
    stop_price = order.stop_loss
    execution_system.update_order(order_id, stop_price, datetime.now())
    
    # Verifica se ordem foi fechada
    assert order_id not in {o.order_id for o in execution_system.get_active_orders()}
    completed = execution_system.get_completed_orders()
    assert len(completed) == 1
    assert completed[0].status == OrderStatus.STOPPED

def test_performance_report(execution_system, sample_analysis_result):
    """Testa geração de relatório de performance"""
    # Cria algumas ordens
    prices = [5000.0, 5010.0, 4990.0]
    for price in prices:
        order = execution_system.create_order(sample_analysis_result, price)
        # Simula fechamento com lucro/prejuízo
        execution_system.update_order(
            order.order_id,
            price * (1.01 if order.order_type == OrderType.CALL else 0.99),
            datetime.now() + timedelta(minutes=5)
        )
    
    # Gera relatório
    report = execution_system.generate_performance_report()
    
    assert 'métricas_gerais' in report
    assert 'total_ordens' in report['métricas_gerais']
    assert report['métricas_gerais']['total_ordens'] == 3
    assert 'win_rate' in report['métricas_gerais']

def test_order_cancellation(execution_system, sample_analysis_result):
    """Testa cancelamento de ordem"""
    order = execution_system.create_order(
        sample_analysis_result,
        current_price=5000.0
    )
    
    # Cancela ordem
    success = execution_system.cancel_order(order.order_id)
    assert success
    
    # Verifica status
    completed = execution_system.get_completed_orders()
    assert completed[-1].status == OrderStatus.CANCELLED

def test_multiple_orders_tracking(execution_system, sample_analysis_result):
    """Testa rastreamento de múltiplas ordens"""
    # Cria várias ordens
    orders = []
    for price in [5000.0, 5010.0, 4990.0]:
        order = execution_system.create_order(sample_analysis_result, price)
        orders.append(order)
    
    # Verifica rastreamento
    active = execution_system.get_active_orders()
    assert len(active) == 3
    
    # Fecha algumas ordens
    execution_system.update_order(
        orders[0].order_id,
        5020.0,
        datetime.now()
    )
    
    # Verifica atualização
    active = execution_system.get_active_orders()
    assert len(active) == 2
    completed = execution_system.get_completed_orders()
    assert len(completed) == 1