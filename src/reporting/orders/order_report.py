import pandas as pd
from typing import List, Dict
from datetime import datetime

class OrderReport:
    def __init__(self):
        pass
        
    def generate_orders_report(self, trades: List[Dict], format: str = 'text') -> str:
        """Gera relatório detalhado das ordens.
        
        Args:
            trades: Lista de trades realizados
            format: Formato de saída ('text' ou 'csv')
            
        Returns:
            String formatada com relatório de ordens
        """
        if not trades:
            return "Nenhuma ordem executada."
            
        # Converte para DataFrame
        df = pd.DataFrame(trades)
        
        if format == 'text':
            report = "=== Relatório de Ordens ===\n\n"
            
            for i, trade in enumerate(trades, 1):
                report += f"Ordem #{i}:\n"
                report += f"Data Entrada: {trade['entry_date']}\n"
                report += f"Preço Entrada: R$ {trade['entry_price']:.2f}\n"
                report += f"Direção: {'Compra' if trade['direction'] == 'long' else 'Venda'}\n"
                
                if trade['status'] == 'closed':
                    report += f"Data Saída: {trade['exit_date']}\n"
                    report += f"Preço Saída: R$ {trade['exit_price']:.2f}\n"
                    report += f"Resultado: R$ {trade['pnl']:.2f}\n"
                    
                report += f"Stop Loss: R$ {trade['stop_loss']:.2f}\n"
                report += f"Take Profit: R$ {trade['take_profit']:.2f}\n"
                report += "---\n"
                
        elif format == 'csv':
            return df.to_csv(index=False)
            
        return report
    
    def analyze_trade_distribution(self, trades: List[Dict]) -> Dict:
        """Analisa distribuição dos trades.
        
        Args:
            trades: Lista de trades realizados
            
        Returns:
            Dict com análise da distribuição
        """
        if not trades:
            return {}
            
        df = pd.DataFrame(trades)
        
        # Distribuição por horário
        df['hour'] = pd.to_datetime(df['entry_date']).dt.hour
        hourly_dist = df.groupby('hour')['pnl'].agg(['count', 'mean', 'sum'])
        
        # Distribuição por direção
        direction_dist = df.groupby('direction')['pnl'].agg(['count', 'mean', 'sum'])
        
        # Duração média dos trades
        df['duration'] = pd.to_datetime(df['exit_date']) - pd.to_datetime(df['entry_date'])
        avg_duration = df['duration'].mean()
        
        return {
            'hourly_distribution': hourly_dist.to_dict(),
            'direction_distribution': direction_dist.to_dict(),
            'average_duration': avg_duration,
            'best_trade': df.loc[df['pnl'].idxmax()].to_dict(),
            'worst_trade': df.loc[df['pnl'].idxmin()].to_dict()
        }
    
    def generate_detailed_report(self, trades: List[Dict]) -> str:
        """Gera relatório detalhado combinando ordens e análise.
        
        Args:
            trades: Lista de trades realizados
            
        Returns:
            String formatada com relatório completo
        """
        distribution = self.analyze_trade_distribution(trades)
        
        report = self.generate_orders_report(trades)
        report += "\n=== Análise da Distribuição ===\n\n"
        
        report += "Distribuição por Horário:\n"
        for hour, stats in distribution['hourly_distribution'].items():
            report += f"{hour}h: {stats['count']} trades, Média: R$ {stats['mean']:.2f}\n"
        
        report += "\nDistribuição por Direção:\n"
        for direction, stats in distribution['direction_distribution'].items():
            report += f"{direction}: {stats['count']} trades, Média: R$ {stats['mean']:.2f}\n"
        
        report += f"\nDuração Média: {distribution['average_duration']}\n"
        
        report += "\nMelhores/Piores Trades:\n"
        best = distribution['best_trade']
        worst = distribution['worst_trade']
        report += f"Melhor: R$ {best['pnl']:.2f} ({best['entry_date']})\n"
        report += f"Pior: R$ {worst['pnl']:.2f} ({worst['entry_date']})\n"
        
        return report