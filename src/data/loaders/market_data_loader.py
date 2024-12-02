import pandas as pd
import sqlite3
from datetime import datetime
import logging

class MarketDataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Carrega dados do banco SQLite."""
        try:
            query = "SELECT * FROM candles"
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append(f"time >= '{start_date}'")
                if end_date:
                    conditions.append(f"time <= '{end_date}'")
                query += " WHERE " + " AND ".join(conditions)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, parse_dates=['time'])
                
            return df.set_index('time')
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            return pd.DataFrame()