import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class SimulationResult:
    returns: np.ndarray
    equity_curves: np.ndarray
    metrics: Dict[str, float]
    var: float
    cvar: float

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 1000,
                 confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def simulate_returns(self, historical_returns: pd.Series,
                        initial_capital: float = 100000) -> SimulationResult:
        """Simula retornos usando Monte Carlo.
        
        Args:
            historical_returns: Série de retornos históricos
            initial_capital: Capital inicial
        """
        # Parâmetros da distribuição
        mu = historical_returns.mean()
        sigma = historical_returns.std()
        
        # Gera simulações
        n_days = len(historical_returns)
        returns = np.random.normal(
            mu, sigma,
            size=(self.n_simulations, n_days)
        )
        
        # Calcula curvas de equity
        equity_curves = initial_capital * (1 + returns).cumprod(axis=1)
        
        # Calcula métricas
        final_values = equity_curves[:, -1]
        total_returns = (final_values - initial_capital) / initial_capital
        
        # Calcula VaR e CVaR
        var_threshold = np.percentile(
            total_returns,
            (1 - self.confidence_level) * 100
        )
        cvar = np.mean(total_returns[total_returns <= var_threshold])
        
        metrics = {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curves),
            'win_rate': np.mean(total_returns > 0)
        }
        
        return SimulationResult(
            returns=returns,
            equity_curves=equity_curves,
            metrics=metrics,
            var=var_threshold,
            cvar=cvar
        )
    
    def simulate_trades(self, trades: List[Dict],
                       n_sequences: int = 1000) -> SimulationResult:
        """Simula sequências de trades.
        
        Args:
            trades: Lista de trades históricos
            n_sequences: Número de sequências a simular
        """
        # Extrai retornos dos trades
        returns = [trade['pnl'] for trade in trades]
        
        # Simula sequências de trades
        sequences = []
        for _ in range(n_sequences):
            sequence = np.random.choice(returns, size=len(returns))
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        
        # Calcula equity curves
        equity_curves = np.zeros((n_sequences, len(returns)))
        equity_curves[:, 0] = 100000 + sequences[:, 0]
        
        for i in range(1, len(returns)):
            equity_curves[:, i] = equity_curves[:, i-1] + sequences[:, i]
        
        # Calcula métricas
        final_values = equity_curves[:, -1]
        initial_capital = 100000
        total_returns = (final_values - initial_capital) / initial_capital
        
        # VaR e CVaR
        var_threshold = np.percentile(
            total_returns,
            (1 - self.confidence_level) * 100
        )
        cvar = np.mean(total_returns[total_returns <= var_threshold])
        
        metrics = {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curves),
            'win_rate': np.mean(total_returns > 0)
        }
        
        return SimulationResult(
            returns=sequences,
            equity_curves=equity_curves,
            metrics=metrics,
            var=var_threshold,
            cvar=cvar
        )
    
    def _calculate_max_drawdown(self, equity_curves: np.ndarray) -> float:
        """Calcula máximo drawdown médio."""
        rolling_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (equity_curves - rolling_max) / rolling_max
        return np.mean(np.min(drawdowns, axis=1))
    
    def plot_simulations(self, result: SimulationResult):
        """Plota resultados das simulações."""
        import matplotlib.pyplot as plt
        
        # Equity curves
        plt.figure(figsize=(12, 6))
        plt.plot(result.equity_curves.T, alpha=0.1, color='blue')
        plt.plot(result.equity_curves.mean(axis=0), 'r--',
                 label='Média')
        plt.title('Simulações Monte Carlo')
        plt.xlabel('Dias')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        
        # Distribuição de retornos
        plt.figure(figsize=(12, 6))
        plt.hist(result.returns[:, -1], bins=50,
                 alpha=0.7, color='blue')
        plt.axvline(result.var, color='r', linestyle='--',
                    label=f'VaR {self.confidence_level:.0%}')
        plt.title('Distribuição dos Retornos')
        plt.xlabel('Retorno')
        plt.ylabel('Frequência')
        plt.legend()
        plt.grid(True)
        
        plt.show()