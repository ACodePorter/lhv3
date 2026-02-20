from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Any as AnyType

from datetime import datetime
import json
import logging
import re
from urllib import request, error

import pandas as pd

from ..config import API_KEYS
from ..models.base import SessionLocal
from ..models.strategy import AiPromptSetting


logger = logging.getLogger(__name__)


class AIModel(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def predict_next_price(self, history: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> float:
        raise NotImplementedError


class SimplePriceModel(AIModel):
    def predict_next_price(self, history: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> float:
        if history is None or history.empty:
            logger.warning("模型%s收到空历史数据，返回0.0", self.name)
            return 0.0
        if "close" not in history.columns:
            value = float(history.iloc[-1].iloc[0])
            logger.debug("模型%s使用非close列数据进行预测，值=%f", self.name, value)
            return value
        window = int(self.config.get("window", 5))
        if window <= 0:
            window = 1
        close = history["close"].tail(window)
        if close.empty:
            last_price = float(history["close"].iloc[-1])
            logger.debug("模型%s窗口内无数据，使用最后收盘价=%f", self.name, last_price)
            return last_price
        mean_price = float(close.mean())
        logger.debug("模型%s根据最近%d根K线预测价格: %.6f", self.name, window, mean_price)
        return mean_price


class DeepSeekPriceModel(AIModel):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.fallback = SimplePriceModel(name, config)
        self.call_logs: List[Dict[str, Any]] = []

    def _get_system_prompt(self) -> str:
        value = self.config.get("system_prompt")
        if isinstance(value, str) and value.strip():
            return value
        db = None
        try:
            db = SessionLocal()
            query = db.query(AiPromptSetting).filter(AiPromptSetting.model_type == self.name)
            setting = query.order_by(AiPromptSetting.updated_at.desc()).first()
            if setting and isinstance(setting.system_prompt, str) and setting.system_prompt.strip():
                return setting.system_prompt
        except Exception as e:
            logger.error("获取模型%s系统提示词失败: %s", self.name, str(e))
        finally:
            if db is not None:
                db.close()
        return "你是一个专业的金融量化模型，只输出数字。"

    def _build_prompt(self, history: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> str:
        window = int(self.config.get("window", 5))
        if window <= 0:
            window = 1
        tail = history.tail(window)
        kline_lines: List[str] = []
        for _, row in tail.iterrows():
            date_value = str(row.get("date", ""))
            close_value = float(row.get("close", 0.0))
            open_value = float(row.get("open", close_value))
            high_value = float(row.get("high", close_value))
            low_value = float(row.get("low", close_value))
            volume_value = float(row.get("volume", 0.0))
            line = f"{date_value},{open_value},{high_value},{low_value},{close_value},{volume_value}"
            kline_lines.append(line)

        symbol = ""
        if "symbol" in history.columns and not history["symbol"].empty:
            try:
                symbol = str(history["symbol"].iloc[-1])
            except Exception:
                symbol = ""

        frequency = str(self.config.get("frequency", "unknown"))

        buy_threshold = float(self.config.get("buy_threshold", 0.002))
        sell_threshold = float(self.config.get("sell_threshold", -0.002))
        stop_loss_pct = float(self.config.get("stop_loss_pct", 0.05))
        take_profit_pct = float(self.config.get("take_profit_pct", 0.1))

        cash = 0.0
        position = 0.0
        entry_price = 0.0
        cumulative_pnl = 0.0
        recent_trades: List[Dict[str, AnyType]] = []

        if context:
            cash = float(context.get("cash", cash))
            position = float(context.get("position", position))
            entry_price = float(context.get("entry_price", entry_price))
            cumulative_pnl = float(context.get("cumulative_pnl", cumulative_pnl))
            trades_value = context.get("recent_trades") or []
            if isinstance(trades_value, list):
                recent_trades = trades_value

        account_lines: List[str] = []
        account_lines.append(f"当前现金: {cash:.4f}")
        account_lines.append(f"当前持仓数量: {position:.6f}")
        account_lines.append(f"当前持仓成本: {entry_price:.6f}")
        account_lines.append(f"累计盈亏: {cumulative_pnl:.4f}")

        trade_lines: List[str] = []
        max_trades = int(self.config.get("max_trades", 10))
        if max_trades <= 0:
            max_trades = 1
        if recent_trades:
            selected_trades = recent_trades[-max_trades:]
            for t in selected_trades:
                ts = str(t.get("timestamp", ""))
                action = str(t.get("action", ""))
                price = float(t.get("price", 0.0))
                qty = float(t.get("quantity", 0.0))
                pnl_value = float(t.get("pnl", 0.0))
                trade_line = f"{ts},{action},{price},{qty},{pnl_value}"
                trade_lines.append(trade_line)

        header_parts: List[str] = []
        header_parts.append("你是一个严格遵守指令的金融量化模型。")
        header_parts.append(
            f"标的: {symbol if symbol else 'unknown'}；周期: {frequency}。"
        )
        header_parts.append(
            f"参考参数: 买入阈值={buy_threshold}, 卖出阈值={sell_threshold}, 止损比例={stop_loss_pct}, 止盈比例={take_profit_pct}。"
        )
        header_parts.append(
            "下面是该标的最近若干根K线数据，按时间从早到晚排列，"
            "格式为: 日期,开盘价,最高价,最低价,收盘价,成交量。"
        )
        header_parts.append(
            "接着是当前账户状态和最近的部分成交记录，用于帮助你理解交易上下文。"
        )
        header_parts.append(
            "你的任务是根据这些信息预测“下一根K线的收盘价”，单位与输入数据一致。"
        )
        header_parts.append(
            "严格要求：只输出一个数字，不要输出任何其他内容（不要文字、不要单位、不要解释）。"
        )

        sections: List[str] = []
        sections.append("\n".join(header_parts))
        sections.append("K线数据（date,open,high,low,close,volume）:")
        sections.append("\n".join(kline_lines))
        sections.append("账户状态:")
        sections.append("\n".join(account_lines))
        sections.append("最近成交记录（timestamp,action,price,qty,pnl），如无则为空:")
        if trade_lines:
            sections.append("\n".join(trade_lines))
        else:
            sections.append("无")

        return "\n".join(sections)

    def _call_api(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> float:
        api_key = self.config.get("api_key") or API_KEYS.get("deepseek", "")
        if not api_key:
            logger.error("模型%s未配置DeepSeek API密钥，无法调用DeepSeek接口", self.name)
        url = "https://api.deepseek.com/v1/chat/completions"
        system_prompt = self._get_system_prompt()
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        call_timestamp = None
        if context and isinstance(context, dict) and context.get("timestamp"):
            call_timestamp = context.get("timestamp")
            if isinstance(call_timestamp, datetime):
                call_timestamp = call_timestamp.isoformat()
            else:
                call_timestamp = str(call_timestamp)
        else:
            call_timestamp = datetime.utcnow().isoformat()
        log_entry: Dict[str, Any] = {
            "timestamp": call_timestamp,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "request": {
                "model": "deepseek-chat",
                "temperature": 0.1,
            },
            "context": context or {},
            "success": False,
        }
        if not api_key:
            log_entry["error_type"] = "config"
            log_entry["error_message"] = "DeepSeek API key is missing"
            self.call_logs.append(log_entry)
            raise RuntimeError("DeepSeek API key is missing")
        url = "https://api.deepseek.com/v1/chat/completions"
        req = request.Request(url, data=data_bytes, headers=headers, method="POST")
        body = ""
        try:
            with request.urlopen(req, timeout=15) as resp:
                status_code = None
                try:
                    status_code = resp.getcode()
                except Exception:
                    status_code = getattr(resp, "status", None)
                body = resp.read().decode("utf-8")
                log_entry["http_status"] = status_code
                log_entry["raw_response"] = body
        except error.HTTPError as e:
            logger.error("模型%s调用DeepSeek HTTP错误: %s %s", self.name, e.code, e.reason)
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            log_entry["error_type"] = "http_error"
            log_entry["error_message"] = f"{e.code} {e.reason}"
            log_entry["http_status"] = e.code
            if error_body:
                log_entry["raw_response"] = error_body
            self.call_logs.append(log_entry)
            raise RuntimeError(f"DeepSeek HTTP error: {e.code} {e.reason}")
        except error.URLError as e:
            logger.error("模型%s调用DeepSeek网络错误: %s", self.name, str(e))
            log_entry["error_type"] = "network_error"
            log_entry["error_message"] = str(e)
            self.call_logs.append(log_entry)
            raise RuntimeError(f"DeepSeek network error: {e}")
        except Exception as e:
            logger.error("模型%s调用DeepSeek异常: %s", self.name, str(e))
            log_entry["error_type"] = "request_exception"
            log_entry["error_message"] = str(e)
            self.call_logs.append(log_entry)
            raise RuntimeError(f"DeepSeek request exception: {e}")
        try:
            obj = json.loads(body)
        except Exception as e:
            logger.error("模型%s解析DeepSeek响应失败: %s", self.name, str(e))
            log_entry["error_type"] = "parse_json_error"
            log_entry["error_message"] = str(e)
            log_entry["raw_response"] = body
            self.call_logs.append(log_entry)
            raise RuntimeError(f"DeepSeek response parse error: {e}")
        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("模型%s获取DeepSeek返回内容失败: %s", self.name, str(e))
            log_entry["error_type"] = "response_format_error"
            log_entry["error_message"] = str(e)
            log_entry["raw_response"] = body
            self.call_logs.append(log_entry)
            raise RuntimeError(f"DeepSeek response format error: {e}")
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", content)
        if not match:
            logger.error("模型%s从DeepSeek返回中未找到数字: %s", self.name, content)
            log_entry["error_type"] = "no_number_found"
            log_entry["error_message"] = content
            log_entry["response"] = content
            self.call_logs.append(log_entry)
            raise RuntimeError("DeepSeek response does not contain a valid number")
        try:
            value = float(match.group(0))
            logger.info("模型%s从DeepSeek获得预测价格: %.6f", self.name, value)
            log_entry["success"] = True
            log_entry["response"] = content
            log_entry["parsed_value"] = value
            self.call_logs.append(log_entry)
            return value
        except Exception as e:
            logger.error("模型%s解析DeepSeek数字失败: %s", self.name, str(e))
            log_entry["error_type"] = "parse_number_error"
            log_entry["error_message"] = str(e)
            log_entry["response"] = content
            self.call_logs.append(log_entry)
            raise RuntimeError(f"DeepSeek number parse error: {e}")

    def predict_next_price(self, history: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> float:
        if history is None or history.empty:
            logger.warning("模型%s收到空历史数据，使用回退模型", self.name)
            return self.fallback.predict_next_price(history, context)
        prompt = self._build_prompt(history, context)
        value = self._call_api(prompt, context)
        if callable(value):
            return value(history, context)
        return value


def create_model(model_type: str, config: Dict[str, Any]) -> AIModel:
    name = model_type
    logger.info("创建AI模型实例: type=%s, name=%s, config=%s", model_type, name, config)
    t = model_type.lower()
    if t == "deepseek":
        if not config.get("api_key"):
            key = API_KEYS.get("deepseek", "")
            if key:
                cfg = dict(config)
                cfg["api_key"] = key
                config = cfg
        return DeepSeekPriceModel(name, config)
    return SimplePriceModel(name, config)
