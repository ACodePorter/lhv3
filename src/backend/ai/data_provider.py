from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..data.fetcher import DataFetcher
from ..api.backtest_service import BacktestService


class KlineDataProvider:
    def __init__(self, db: Optional[Session] = None):
        self.fetcher = DataFetcher()
        self.db = db

    def get_kline(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime],
        data_source: str = "database",
        frequency: str = "1d",
    ) -> pd.DataFrame:
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d") if end_time else None

        if data_source == "database":
            if self.db is None:
                return pd.DataFrame()
            service = BacktestService(self.db)
            data = service.get_backtest_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_source="database",
                features=None,
            )
        else:
            data = self.fetcher.fetch_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source,
                frequency=frequency,
            )

        if data is None or data.empty:
            return pd.DataFrame()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date")
            data = data[data["date"] >= start_time]
            if end_time is not None:
                data = data[data["date"] <= end_time]
        return data.reset_index(drop=True)
