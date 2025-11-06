"""
高股息MA偏离小T策略

理念：
- 面向波动较小、适合长期持有的高股息股票。
- 始终保留一部分长期基线仓位，当价格相对长期均线偏离到达阈值时，进行小额T买入或卖出。

参数：
- long_ma (int): 长期均线周期，例如 120/200。
- dev_threshold_pct (float): 相对长期均线的偏离阈值百分比，超过则触发。
- baseline_long_position (float): 长期基线仓位占比（0-1）。
- t_chunk_count (int): 小T分为多少份（分批次数）。
- total_max_position (float): 仓位的上限（含基线+小T）。

建议：
- 小T每次操作的仓位 = (total_max_position - baseline_long_position) / t_chunk_count。
- 当价格低于均线超过阈值时买入一个小T份额；高于均线超过阈值时卖出一个小T份额。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from .templates.strategy_template import StrategyTemplate

logger = logging.getLogger(__name__)


class DividendMATStrategy(StrategyTemplate):
    """高股息MA偏离小T策略实现"""

    def __init__(self, name: str = "高股息MA偏离小T策略", data: pd.DataFrame = None, parameters: Dict[str, Any] = None):
        default_params = {
            "long_ma": 120,
            "dev_threshold_pct": 0.08,  # 8%
            "baseline_long_position": 0.6,  # 60%
            "t_chunk_count": 5,
            "total_max_position": 1.0,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(name=name, data=data, parameters=default_params)

    def set_parameters(self, parameters: Dict[str, Any]):
        if parameters:
            self.parameters.update(parameters)
        logger.info(f"更新策略参数: {self.parameters}")
        return self.parameters

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None or len(self.data) == 0:
            logger.error("数据为空，无法生成信号")
            return pd.DataFrame()

        df = self.data.copy()

        # 参数提取
        long_ma = int(self.parameters.get("long_ma", 120))
        threshold = float(self.parameters.get("dev_threshold_pct", 0.08))
        baseline = float(self.parameters.get("baseline_long_position", 0.6))
        chunk_count = int(self.parameters.get("t_chunk_count", 5))
        max_pos = float(self.parameters.get("total_max_position", 1.0))

        # 临时调试日志：参数快照
        try:
            self.log(
                f"参数快照 long_ma={long_ma}, threshold={threshold}, baseline={baseline}, chunk_count={chunk_count}, max_pos={max_pos}",
                "INFO",
            )
        except Exception:
            pass

        # 计算长期均线与偏离
        df["long_ma"] = df["close"].rolling(window=long_ma, min_periods=long_ma).mean()
        df["dev_pct"] = (df["close"] - df["long_ma"]) / df["long_ma"]

        # 初始化信号列
        df["signal"] = 0
        df["trigger_reason"] = ""
        df["position_size"] = np.nan
        df["t_position"] = 0.0
        df["cumulative_position"] = baseline  # 基线仓位 + t_position

        # 小T单次仓位
        t_capacity = max(max_pos - baseline, 0.0)
        t_chunk = t_capacity / chunk_count if chunk_count > 0 else 0.0

        try:
            self.log(
                f"T容量与分批 t_capacity={t_capacity:.4f}, t_chunk={t_chunk:.4f}",
                "DEBUG",
            )
            if t_capacity <= 0:
                self.log("t_capacity<=0，只有基线仓位，将不会执行小T交易", "WARNING")
            if chunk_count <= 0:
                self.log("chunk_count<=0，分批次数为零，无法进行小T分批", "WARNING")
        except Exception:
            pass

        # 顺序遍历，跟踪小T仓位
        current_t = 0.0
        for i in range(len(df)):
            ma_val = df.at[df.index[i], "long_ma"]
            dev = df.at[df.index[i], "dev_pct"]

            # MA尚未就绪
            if pd.isna(ma_val):
                df.at[df.index[i], "signal"] = 0
                df.at[df.index[i], "trigger_reason"] = "ma_not_ready"
                df.at[df.index[i], "t_position"] = current_t
                df.at[df.index[i], "cumulative_position"] = baseline + current_t
                df.at[df.index[i], "position_size"] = baseline + current_t
                try:
                    self.log(
                        f"{df.index[i]} MA未就绪: long_ma={long_ma}",
                        "DEBUG",
                    )
                except Exception:
                    pass
                continue

            # 触发条件：偏离超过阈值
            if dev <= -threshold and (current_t + t_chunk) <= t_capacity:
                # 低于均线阈值 => 买入一个小T份
                current_t += t_chunk
                df.at[df.index[i], "signal"] = 1
                df.at[df.index[i], "trigger_reason"] = f"buy_deviation_{dev:.4f}"
                try:
                    self.log(
                        f"{df.index[i]} 触发买入: dev={dev:.4f} <= -{threshold}, t_chunk={t_chunk:.4f}, t_pos={current_t:.4f}",
                        "INFO",
                    )
                except Exception:
                    pass
            elif dev >= threshold and (current_t - t_chunk) >= 0.0:
                # 高于均线阈值 => 卖出一个小T份
                current_t -= t_chunk
                df.at[df.index[i], "signal"] = -1
                df.at[df.index[i], "trigger_reason"] = f"sell_deviation_{dev:.4f}"
                try:
                    self.log(
                        f"{df.index[i]} 触发卖出: dev={dev:.4f} >= {threshold}, t_chunk={t_chunk:.4f}, t_pos={current_t:.4f}",
                        "INFO",
                    )
                except Exception:
                    pass
            else:
                df.at[df.index[i], "signal"] = 0
                df.at[df.index[i], "trigger_reason"] = "hold"
                try:
                    self.log(
                        f"{df.index[i]} 持有: dev={dev:.4f}, threshold=±{threshold}, t_pos={current_t:.4f}",
                        "DEBUG",
                    )
                except Exception:
                    pass

            df.at[df.index[i], "t_position"] = current_t
            df.at[df.index[i], "cumulative_position"] = baseline + current_t
            df.at[df.index[i], "position_size"] = baseline + current_t

        # 附加符号列（若可用）
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"]

        # 临时调试日志：统计摘要
        try:
            buy_cnt = int((df["signal"] == 1).sum())
            sell_cnt = int((df["signal"] == -1).sum())
            self.log(f"统计摘要: 买入={buy_cnt}, 卖出={sell_cnt}", "INFO")
        except Exception:
            pass

        return df

    def suggest_position_size(self, signal: float, row=None) -> float:
        # 使用计算好的 position_size 作为建议
        if row is not None and "position_size" in row:
            val = float(row["position_size"]) if not pd.isna(row["position_size"]) else None
            if val is not None:
                return max(0.0, min(1.0, val))
        return None  # 交给默认仓位控制

    def get_strategy_info(self) -> dict:
        return {
            "name": self.name,
            "description": "保留长期仓位，围绕长期均线偏离进行小额T的高股息策略",
            "parameters": self.parameters,
            "key_features": [
                "长期均线基线仓位",
                "偏离阈值触发买卖",
                "分批小T交易",
                "建议仓位输出"
            ],
        }