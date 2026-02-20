from typing import Any, Dict, List, Tuple, Optional

import logging
import numpy as np
import pandas as pd

from .model_provider import AIModel


logger = logging.getLogger(__name__)


class AiInvestmentEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        models: Dict[str, AIModel],
        initial_capital: float = 100000.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.data = data.copy()
        self.models = models
        self.initial_capital = initial_capital
        self.config = config or {}

    def run(self) -> Dict[str, Any]:
        if self.data is None or self.data.empty:
            logger.warning("AI投资引擎收到空数据，直接返回空结果")
            return {
                "records": [],
                "metrics": {},
                "equity_curves": {},
            }

        if "date" in self.data.columns:
            self.data["date"] = pd.to_datetime(self.data["date"])
            self.data = self.data.sort_values("date").reset_index(drop=True)

        symbol = ""
        if "symbol" in self.data.columns and not self.data["symbol"].empty:
            symbol = str(self.data["symbol"].iloc[0])

        buy_threshold = float(self.config.get("buy_threshold", 0.002))
        sell_threshold = float(self.config.get("sell_threshold", -0.002))
        stop_loss_pct = float(self.config.get("stop_loss_pct", 0.05))
        take_profit_pct = float(self.config.get("take_profit_pct", 0.1))
        commission_rate = float(self.config.get("commission_rate", 0.0))
        slippage_rate = float(self.config.get("slippage_rate", 0.0))
        window = int(self.config.get("window", 20))
        if window <= 1:
            window = 2

        logger.info(
            "启动AI投资引擎: symbol=%s, 数据条数=%d, 窗口=%d, buy=%.4f, sell=%.4f, "
            "stop_loss=%.4f, take_profit=%.4f, 模型列表=%s",
            symbol,
            len(self.data),
            window,
            buy_threshold,
            sell_threshold,
            stop_loss_pct,
            take_profit_pct,
            list(self.models.keys()),
        )

        state: Dict[str, Dict[str, Any]] = {}
        for name in self.models.keys():
            state[name] = {
                "cash": float(self.initial_capital),
                "position": 0.0,
                "entry_price": 0.0,
                "cumulative_pnl": 0.0,
                "equity_history": [],
                "trades": [],
                "pending_commission": 0.0,
            }
            logger.debug(
                "初始化模型账户状态: model=%s, 初始资金=%.2f",
                name,
                self.initial_capital,
            )

        records: List[Dict[str, Any]] = []

        for i in range(window, len(self.data) - 1):
            history = self.data.iloc[: i + 1]
            current_row = self.data.iloc[i]
            next_row = self.data.iloc[i + 1]
            current_price = float(current_row["close"])
            next_price = float(next_row["close"])
            timestamp = next_row["date"] if "date" in next_row else None

            logger.debug(
                "时间步: idx=%d, 时间=%s, 当前价=%.6f, 下一根价=%.6f",
                i,
                timestamp,
                current_price,
                next_price,
            )

            for name, model in self.models.items():
                model_state = state[name]
                logger.debug(
                    "调用模型预测: model=%s, 账户状态: cash=%.2f, position=%.6f, entry_price=%.6f, cum_pnl=%.2f",
                    name,
                    model_state["cash"],
                    model_state["position"],
                    model_state["entry_price"],
                    model_state["cumulative_pnl"],
                )
                context = {
                    "symbol": symbol,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "window": window,
                    "cash": model_state["cash"],
                    "position": model_state["position"],
                    "entry_price": model_state["entry_price"],
                    "cumulative_pnl": model_state["cumulative_pnl"],
                    "recent_trades": model_state.get("trades", []),
                    "timestamp": timestamp,
                }
                predicted_price = model.predict_next_price(history, context)
                if current_price <= 0:
                    change = 0.0
                else:
                    change = (predicted_price - current_price) / current_price

                action = "HOLD"
                pnl = 0.0
                trigger_reason = "未触发买卖、止损或止盈条件"

                equity_before = model_state["cash"] + model_state["position"] * current_price

                if model_state["position"] > 0 and model_state["entry_price"] > 0:
                    drawdown = (current_price - model_state["entry_price"]) / model_state["entry_price"]
                    if drawdown <= -stop_loss_pct:
                        action = "SELL"
                        trigger_reason = (
                            f"持仓回撤 {drawdown:.4f} 小于等于止损阈值 {-stop_loss_pct:.4f}"
                        )
                        logger.info(
                            "模型%s触发止损: 时间=%s, 当前价=%.6f, 入场价=%.6f, 回撤=%.4f",
                            name,
                            timestamp,
                            current_price,
                            model_state["entry_price"],
                            drawdown,
                        )
                    elif drawdown >= take_profit_pct:
                        action = "SELL"
                        trigger_reason = (
                            f"持仓收益 {drawdown:.4f} 大于等于止盈阈值 {take_profit_pct:.4f}"
                        )
                        logger.info(
                            "模型%s触发止盈: 时间=%s, 当前价=%.6f, 入场价=%.6f, 收益=%.4f",
                            name,
                            timestamp,
                            current_price,
                            model_state["entry_price"],
                            drawdown,
                        )

                if action == "HOLD":
                    if change >= buy_threshold and model_state["position"] == 0:
                        action = "BUY"
                        trigger_reason = (
                            f"预测涨幅 {change:.4f} 大于等于买入阈值 {buy_threshold:.4f} 且当前无持仓"
                        )
                        logger.info(
                            "模型%s生成买入信号: 时间=%s, 当前价=%.6f, 预测价=%.6f, 涨幅=%.4f",
                            name,
                            timestamp,
                            current_price,
                            predicted_price,
                            change,
                        )
                    elif change <= sell_threshold and model_state["position"] > 0:
                        action = "SELL"
                        trigger_reason = (
                            f"预测跌幅 {change:.4f} 小于等于卖出阈值 {sell_threshold:.4f} 且当前有持仓"
                        )
                        logger.info(
                            "模型%s生成卖出信号: 时间=%s, 当前价=%.6f, 预测价=%.6f, 跌幅=%.4f",
                            name,
                            timestamp,
                            current_price,
                            predicted_price,
                            change,
                        )

                if action == "BUY" and model_state["position"] == 0 and current_price > 0:
                    execution_price_buy = current_price * (1.0 + slippage_rate)
                    if execution_price_buy <= 0:
                        execution_price_buy = current_price
                    if commission_rate > 0:
                        quantity = model_state["cash"] / (execution_price_buy * (1.0 + commission_rate))
                    else:
                        quantity = model_state["cash"] / execution_price_buy
                    trade_value = execution_price_buy * quantity
                    buy_commission = trade_value * commission_rate if commission_rate > 0 else 0.0
                    model_state["cash"] -= trade_value + buy_commission
                    model_state["position"] = quantity
                    model_state["entry_price"] = execution_price_buy
                    model_state["pending_commission"] = buy_commission
                    model_state.setdefault("trades", []).append(
                        {
                            "timestamp": timestamp,
                            "action": "BUY",
                            "price": float(execution_price_buy),
                            "quantity": float(quantity),
                            "pnl": 0.0,
                        }
                    )
                    logger.info(
                        "执行买入: model=%s, 时间=%s, 买入价=%.6f, 数量=%.6f, 账户现金=%.2f",
                        name,
                        timestamp,
                        execution_price_buy,
                        quantity,
                        model_state["cash"],
                    )
                elif action == "SELL" and model_state["position"] > 0 and current_price > 0:
                    quantity = model_state["position"]
                    execution_price_sell = current_price * (1.0 - slippage_rate)
                    if execution_price_sell <= 0:
                        execution_price_sell = current_price
                    revenue = quantity * execution_price_sell
                    sell_commission = revenue * commission_rate if commission_rate > 0 else 0.0
                    total_commission = sell_commission + float(model_state.get("pending_commission", 0.0))
                    cost = quantity * model_state["entry_price"]
                    trade_pnl = revenue - total_commission - cost
                    model_state["cash"] += revenue - sell_commission
                    model_state["position"] = 0.0
                    model_state["entry_price"] = 0.0
                    model_state["pending_commission"] = 0.0
                    pnl = trade_pnl
                    model_state["cumulative_pnl"] += trade_pnl
                    model_state.setdefault("trades", []).append(
                        {
                            "timestamp": timestamp,
                            "action": "SELL",
                            "price": float(execution_price_sell),
                            "quantity": float(quantity),
                            "pnl": float(trade_pnl),
                        }
                    )
                    logger.info(
                        "执行卖出: model=%s, 时间=%s, 卖出价=%.6f, 数量=%.6f, 本次盈亏=%.2f, 累计盈亏=%.2f, 账户现金=%.2f",
                        name,
                        timestamp,
                        execution_price_sell,
                        quantity,
                        trade_pnl,
                        model_state["cumulative_pnl"],
                        model_state["cash"],
                    )

                equity = model_state["cash"] + model_state["position"] * next_price
                model_state["equity_history"].append((timestamp, equity))

                logger.debug(
                    "时间步结束: model=%s, 时间=%s, 预测价=%.6f, 实际价=%.6f, 操作=%s, 持仓=%.6f, "
                    "单笔盈亏=%.2f, 累计盈亏=%.2f, 账户权益=%.2f",
                    name,
                    timestamp,
                    predicted_price,
                    next_price,
                    action,
                    model_state["position"],
                    pnl,
                    model_state["cumulative_pnl"],
                    equity,
                )

                record = {
                    "model_type": name,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "predicted_price": float(predicted_price),
                    "actual_price": float(next_price),
                    "action": action,
                    "position": float(model_state["position"]),
                    "pnl": float(pnl),
                    "cumulative_pnl": float(model_state["cumulative_pnl"]),
                    "equity": float(equity),
                    "trigger_reason": trigger_reason,
                }
                records.append(record)

        metrics: Dict[str, Dict[str, float]] = {}
        equity_curves: Dict[str, List[Dict[str, Any]]] = {}

        for name, model_state in state.items():
            history_list: List[Tuple[Any, float]] = model_state["equity_history"]
            if not history_list:
                metrics[name] = {
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                }
                equity_curves[name] = []
                continue

            dates = [item[0] for item in history_list]
            equities = [item[1] for item in history_list]
            equity_series = pd.Series(equities, index=pd.to_datetime(dates))

            total_return = 0.0
            if self.initial_capital > 0:
                total_return = float(equity_series.iloc[-1] / self.initial_capital - 1.0)

            max_drawdown = self._calculate_max_drawdown(equity_series.values)
            sharpe_ratio = self._calculate_sharpe_ratio(equity_series.values)

            metrics[name] = {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
            }

            equity_curves[name] = [
                {"date": dates[i], "equity": float(equities[i])} for i in range(len(dates))
            ]

            logger.info(
                "模型%s回放完成: 总收益=%.4f, 最大回撤=%.4f, 夏普比率=%.4f",
                name,
                total_return,
                max_drawdown,
                sharpe_ratio,
            )

        logger.info("AI投资引擎运行完成: symbol=%s, 总记录数=%d", symbol, len(records))

        return {
            "records": records,
            "metrics": metrics,
            "equity_curves": equity_curves,
        }

    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        arr = np.array(equity, dtype=float)
        if arr.size == 0:
            return 0.0
        cummax = np.maximum.accumulate(arr)
        drawdowns = (arr - cummax) / cummax
        return float(drawdowns.min()) if drawdowns.size > 0 else 0.0

    def _calculate_sharpe_ratio(self, equity: List[float]) -> float:
        arr = np.array(equity, dtype=float)
        if arr.size < 2:
            return 0.0
        returns = np.diff(arr) / arr[:-1]
        if returns.size == 0:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0:
            return 0.0
        scale = float(np.sqrt(len(returns)))
        return float(mean / std * scale)
