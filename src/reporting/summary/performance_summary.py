import pandas as pd
import numpy as np
from typing import Dict, List

class PerformanceSummary:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calcula métricas principais de performance.
        
        Args:
            trades: Lista de trades realizados
            
        Returns:
            Dict com métricas de performance
        """
        if not trades:
            return {}
            
        # Converte para DataFrame
        df = pd.DataFrame(trades)
        
        # Retornos
        total_pnl = sum(trade['pnl'] for trade in trades)
        wins = len([t for t in trades if t['pnl'] > 0])
        losses = len([t for t in trades if t['pnl'] < 0])
        
        # Cálculo do equity
        equity = [self.initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade['pnl'])
        equity = np.array(equity)
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Métricas
        metrics = {
            'total_trades': len(trades),
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': wins / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'return': total_pnl / self.initial_capital,
            'max_drawdown': max_drawdown,
            'profit_factor': abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) /
                               sum(t['pnl'] for t in trades if t['pnl'] < 0))
                               if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else np.inf,
            'avg_win': np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins else 0,
            'avg_loss': abs(np.mean([t['pnl'] for t in trades if t['pnl'] < 0])) if losses else 0
        }
        
        # Adiciona Sharpe Ratio se houver retornos diários
        if 'exit_date' in df.columns:
            daily_returns = df.set_index('exit_date')['pnl'].resample('D').sum()
            metrics['sharpe_ratio'] = self._calculate_sharpe(daily_returns)
            
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calcula Sharpe Ratio anualizado."""
        excess_returns = returns - (risk_free_rate / 252)  # Taxa livre de risco diária
        if len(excess_returns) > 1:
            return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        return 0.0
    
    def generate_summary(self, trades: List[Dict], format: str = 'text') -> str:
        """Gera sumário formatado da performance.
        
        Args:
            trades: Lista de trades realizados
            format: Formato de saída ('text' ou 'html')
            
        Returns:
            String formatada com sumário
        """
        metrics = self.calculate_metrics(trades)
        
        if format == 'text':
            summary = "=== Sumário de Performance ===\n\n"
            summary += f"Capital Inicial: R$ {self.initial_capital:,.2f}\n"
            summary += f"Resultado Total: R$ {metrics['total_pnl']:,.2f}\n"
            summary += f"Retorno: {metrics['return']*100:.1f}%\n"
            summary += f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%\n\n"
            
            summary += "--- Estatísticas de Trading ---\n"
            summary += f"Total de Trades: {metrics['total_trades']}\n"
            summary += f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
            summary += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            summary += f"Ganho Médio: R$ {metrics['avg_win']:,.2f}\n"
            summary += f"Perda Média: R$ {metrics['avg_loss']:,.2f}\n"
            
            if 'sharpe_ratio' in metrics:
                summary += f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}"
                
        elif format == 'html':
            # Implementar formato HTML se necessário
            pass
            
        return summary