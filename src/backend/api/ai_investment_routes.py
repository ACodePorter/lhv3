from datetime import datetime
from typing import Any, Dict, List, Optional

import base64
import json
import logging
import threading
import zlib

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..ai.data_provider import KlineDataProvider
from ..ai.engine import AiInvestmentEngine
from ..ai.model_provider import create_model
from ..config import API_KEYS
from ..models import (
    AiInvestmentRun,
    AiInvestmentRecord,
    AiInvestmentLog,
    AiPromptSetting,
    get_db,
)
from ..models.base import SessionLocal


router = APIRouter(tags=["ai_investment"])
logger = logging.getLogger(__name__)


def _parse_iso_datetime(value: str) -> datetime:
    v = value.strip()
    if v.endswith("Z"):
        v = v.replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _compress_json(data: Any) -> Optional[str]:
    if data is None:
        return None
    try:
        raw = json.dumps(data, ensure_ascii=False)
        compressed = zlib.compress(raw.encode("utf-8"))
        return base64.b64encode(compressed).decode("ascii")
    except Exception:
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return None


def _decompress_json(data: Optional[str]) -> Any:
    if not data:
        return None
    try:
        compressed = base64.b64decode(data.encode("ascii"))
        raw = zlib.decompress(compressed).decode("utf-8")
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(data)
        except Exception:
            return None

class AiRunRequest(BaseModel):
    name: Optional[str] = None
    symbol: str
    start_time: str
    end_time: Optional[str] = None
    data_source: str = "database"
    frequency: str = "1d"
    models: List[str]
    initial_capital: float = 100000.0
    buy_threshold: float = 0.002
    sell_threshold: float = -0.002
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    window: int = 20
    commission_rate: float = 0.0015
    slippage_rate: float = 0.001


class AiRunResponse(BaseModel):
    status: str
    message: str
    run_id: int
    metrics: Dict[str, Dict[str, float]]
    equity_curves: Dict[str, List[Dict[str, Any]]]
    price_series: List[Dict[str, Any]]
    name: Optional[str] = None
    symbol: Optional[str] = None
    data_source: Optional[str] = None
    frequency: Optional[str] = None
    models: List[str] = []
    initial_capital: Optional[float] = None
    config: Optional[Dict[str, Any]] = None


class AiRunResumeRequest(BaseModel):
    name: Optional[str] = None
    end_time: Optional[str] = None


class AiRunLogItem(BaseModel):
    id: int
    timestamp: datetime
    level: str
    category: str
    message: str
    ai_input: Optional[Dict[str, Any]]
    ai_output: Optional[Dict[str, Any]]
    extra: Optional[Dict[str, Any]]


class AiRunLogResponse(BaseModel):
    total: int
    items: List[AiRunLogItem]


class AiCallEstimateResponse(BaseModel):
    data_length: int
    window: int
    engine_window: int
    per_model_calls: int
    model_count: int
    total_calls: int
    formula: str
    message: str


def _build_run_logs(
    run_id: int,
    request: AiRunRequest,
    result: Dict[str, Any],
    model_instances: Dict[str, Any],
    parent_run_id: Optional[int],
    engine_window: int,
    data_length: int,
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []

    now_iso = datetime.utcnow().isoformat()
    total_call_count = 0
    error_call_count = 0

    for model_name, model in model_instances.items():
        call_logs = getattr(model, "call_logs", None)
        if not call_logs:
            continue
        for entry in call_logs:
            ts = entry.get("timestamp") or now_iso
            ai_output_value: Dict[str, Any] = {}
            if entry.get("response") is not None:
                ai_output_value["content"] = entry.get("response")
            if entry.get("parsed_value") is not None:
                ai_output_value["value"] = entry.get("parsed_value")
            if entry.get("raw_response") is not None:
                ai_output_value["raw_response"] = entry.get("raw_response")
            if entry.get("error_type") is not None:
                ai_output_value["error_type"] = entry.get("error_type")
            if entry.get("error_message") is not None:
                ai_output_value["error_message"] = entry.get("error_message")
            if entry.get("http_status") is not None:
                ai_output_value["http_status"] = entry.get("http_status")
            success_flag = entry.get("success")
            total_call_count += 1
            if not success_flag:
                error_call_count += 1
            logs.append(
                {
                    "timestamp": ts,
                    "level": "INFO" if success_flag else "ERROR",
                    "category": "ai_call",
                    "message": "AI模型调用",
                    "ai_input": {
                        "system_prompt": entry.get("system_prompt"),
                        "prompt": entry.get("prompt"),
                        "request": entry.get("request"),
                        "context": entry.get("context"),
                    },
                    "ai_output": ai_output_value,
                    "extra": {
                        "model_type": model_name,
                        "success": success_flag,
                        "error_type": entry.get("error_type"),
                        "error_message": entry.get("error_message"),
                        "http_status": entry.get("http_status"),
                    },
                    "model_type": model_name,
                }
            )

    error_message = result.get("error")
    status_value = "failed" if error_message else "completed"
    level_value = "ERROR" if error_message else "INFO"
    message_value = "AI投资运行失败" if error_message else "AI投资运行完成"
    logs.insert(
        0,
        {
            "timestamp": now_iso,
            "level": level_value,
            "category": "run",
            "message": message_value,
            "ai_input": {
                "symbol": request.symbol,
                "models": request.models,
                "start_time": request.start_time,
                "end_time": request.end_time,
                "data_source": request.data_source,
                "frequency": request.frequency,
            },
            "ai_output": {
                "metrics": result.get("metrics", {}),
                "error": error_message,
                "ai_call_count": total_call_count,
                "ai_call_error_count": error_call_count,
            },
            "extra": {
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "initial_capital": request.initial_capital,
                "status": status_value,
                "ai_call_count": total_call_count,
                "ai_call_error_count": error_call_count,
            },
            "model_type": None,
        },
    )

    if total_call_count == 0:
        models_list = request.models or []
        models_lower = [str(m).lower() for m in models_list]
        has_deepseek = "deepseek" in models_lower
        formula_str = "max(0, data_length - window - 1)"
        if has_deepseek:
            reason = (
                "已选择大模型 'deepseek'，但有效K线数量 data_length="
                f"{data_length}，窗口大小 window={engine_window}。"
                "根据调用次数计算公式: max(0, data_length - window - 1)，"
                f"代入得到: max(0, {data_length} - {engine_window} - 1) = 0，"
                "因此本次未触发大模型HTTP调用。"
            )
        else:
            reason = (
                "本次运行的模型列表不包含大模型 'deepseek'，"
                "因此不会触发大模型HTTP调用。"
            )
        logs.append(
            {
                "timestamp": now_iso,
                "level": "INFO",
                "category": "ai_call",
                "message": "本次运行未触发AI模型HTTP调用",
                "ai_input": {
                    "symbol": request.symbol,
                    "models": request.models,
                    "start_time": request.start_time,
                    "end_time": request.end_time,
                    "data_source": request.data_source,
                    "frequency": request.frequency,
                },
                "ai_output": {
                    "call_count": 0,
                    "reason": reason,
                    "formula": formula_str,
                    "data_length": data_length,
                    "window": engine_window,
                    "has_deepseek": has_deepseek,
                },
                "extra": {
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "status": status_value,
                },
                "model_type": None,
            }
        )

    records_data: List[Dict[str, Any]] = result.get("records", [])
    for rec in records_data:
        action = rec.get("action")
        if action not in ("BUY", "SELL"):
            continue
        ts_value = rec.get("timestamp")
        if isinstance(ts_value, datetime):
            ts = ts_value.isoformat()
        elif isinstance(ts_value, str) and ts_value:
            ts = ts_value
        else:
            ts = now_iso
        model_type = rec.get("model_type")
        message = (
            f"模型{model_type}执行买入"
            if action == "BUY"
            else f"模型{model_type}执行卖出"
        )
        logs.append(
            {
                "timestamp": ts,
                "level": "INFO",
                "category": "trade",
                "message": message,
                "ai_input": {
                    "symbol": rec.get("symbol"),
                    "model_type": model_type,
                    "predicted_price": rec.get("predicted_price"),
                    "actual_price": rec.get("actual_price"),
                    "action": action,
                },
                "ai_output": {
                    "position": rec.get("position"),
                    "pnl": rec.get("pnl"),
                    "cumulative_pnl": rec.get("cumulative_pnl"),
                    "equity": rec.get("equity"),
                },
                "extra": {
                    "action": action,
                },
                "model_type": model_type,
            }
        )

    return logs


def _async_persist_run_logs(run_id: int, logs: List[Dict[str, Any]]) -> None:
    if not logs:
        return

    def worker():
        db = SessionLocal()
        try:
            record_map: Dict[tuple, int] = {}
            try:
                records = (
                    db.query(AiInvestmentRecord)
                    .filter(AiInvestmentRecord.run_id == run_id)
                    .all()
                )
                for rec in records:
                    key = (
                        (rec.model_type or "").lower(),
                        rec.timestamp.isoformat() if rec.timestamp else "",
                    )
                    record_map[key] = rec.id
            except Exception:
                record_map = {}

            for item in logs:
                ts_iso = item.get("timestamp")
                try:
                    ts = (
                        datetime.fromisoformat(ts_iso)
                        if isinstance(ts_iso, str)
                        else datetime.utcnow()
                    )
                except Exception:
                    ts = datetime.utcnow()

                model_type = item.get("model_type")
                record_id = None
                if model_type:
                    key = (str(model_type).lower(), ts.isoformat())
                    record_id = record_map.get(key)

                log_row = AiInvestmentLog(
                    run_id=run_id,
                    record_id=record_id,
                    timestamp=ts,
                    level=str(item.get("level") or "INFO").upper(),
                    category=str(item.get("category") or "system"),
                    message=str(item.get("message") or ""),
                    ai_input_compressed=_compress_json(item.get("ai_input")),
                    ai_output_compressed=_compress_json(item.get("ai_output")),
                    extra=item.get("extra") or None,
                )
                db.add(log_row)

            db.commit()
        except Exception as e:
            logger.error("保存AI投资运行日志失败: %s", str(e))
            db.rollback()
        finally:
            db.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def _perform_ai_run(
    request: AiRunRequest,
    db: Session,
    parent_run_id: Optional[int] = None,
) -> AiRunResponse:
    if not request.models:
        raise HTTPException(status_code=400, detail="至少需要选择一个模型")

    try:
        start_time = _parse_iso_datetime(request.start_time)
    except Exception:
        raise HTTPException(status_code=400, detail="start_time 格式无效")

    end_time: Optional[datetime] = None
    if request.end_time:
        try:
            end_time = _parse_iso_datetime(request.end_time)
        except Exception:
            raise HTTPException(status_code=400, detail="end_time 格式无效")

    provider = KlineDataProvider(db)

    data_source = request.data_source
    frequency = request.frequency
    if data_source == "database":
        frequency = "1d"
    data = provider.get_kline(
        symbol=request.symbol,
        start_time=start_time,
        end_time=end_time,
        data_source=data_source,
        frequency=frequency,
    )

    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="无法获取指定条件的K线数据")

    run_name = request.name or f"AI投资-{request.symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    run = AiInvestmentRun(
        name=run_name,
        symbol=request.symbol,
        models=request.models,
        data_source=request.data_source,
        frequency=request.frequency,
        initial_capital=request.initial_capital,
        status="running",
        parent_run_id=parent_run_id,
        config={
            "start_time": request.start_time,
            "end_time": request.end_time,
            "buy_threshold": request.buy_threshold,
            "sell_threshold": request.sell_threshold,
            "stop_loss_pct": request.stop_loss_pct,
            "take_profit_pct": request.take_profit_pct,
            "window": request.window,
            "commission_rate": request.commission_rate,
            "slippage_rate": request.slippage_rate,
        },
        performance_metrics={},
        created_at=datetime.now(),
        updated_at=datetime.now(),
        completed_at=None,
    )

    db.add(run)
    db.flush()

    model_instances: Dict[str, Any] = {}
    for model_name in request.models:
        config: Dict[str, Any] = {
            "api_key": API_KEYS.get(model_name, ""),
            "window": request.window,
            "frequency": frequency,
            "buy_threshold": request.buy_threshold,
            "sell_threshold": request.sell_threshold,
            "stop_loss_pct": request.stop_loss_pct,
            "take_profit_pct": request.take_profit_pct,
            "commission_rate": request.commission_rate,
            "slippage_rate": request.slippage_rate,
        }
        config["run_id"] = run.id
        model_instances[model_name] = create_model(model_name, config)

    engine_window = int(request.window)
    if engine_window <= 1:
        engine_window = 2
    data_length = len(data)
    if data_length <= engine_window + 1:
        candidate = data_length - 2
        if candidate < 2:
            candidate = 2
        engine_window = candidate

    engine_config: Dict[str, Any] = {
        "buy_threshold": request.buy_threshold,
        "sell_threshold": request.sell_threshold,
        "stop_loss_pct": request.stop_loss_pct,
        "take_profit_pct": request.take_profit_pct,
        "window": engine_window,
    }

    engine = AiInvestmentEngine(
        data=data,
        models=model_instances,
        initial_capital=request.initial_capital,
        config=engine_config,
    )

    result: Dict[str, Any] = {}
    run_status = "completed"
    error_message: Optional[str] = None
    try:
        result = engine.run()
    except Exception as e:
        logger.error("AI投资运行失败: %s", str(e))
        run_status = "failed"
        error_message = str(e)
        result = {
            "metrics": {},
            "records": [],
            "equity_curves": {},
            "error": error_message,
        }

    run.status = run_status
    run.performance_metrics = result.get("metrics", {})
    run.updated_at = datetime.now()
    run.completed_at = datetime.now() if run_status == "completed" else None

    records_data: List[Dict[str, Any]] = result.get("records", [])
    for rec in records_data:
        record = AiInvestmentRecord(
            run_id=run.id,
            model_type=rec.get("model_type"),
            symbol=request.symbol,
            timestamp=rec.get("timestamp"),
            predicted_price=rec.get("predicted_price"),
            actual_price=rec.get("actual_price"),
            action=rec.get("action"),
            position=rec.get("position"),
            pnl=rec.get("pnl"),
            cumulative_pnl=rec.get("cumulative_pnl"),
            equity=rec.get("equity"),
            extra={"trigger_reason": rec.get("trigger_reason")} if rec.get("trigger_reason") else None,
        )
        db.add(record)

    db.commit()

    try:
        logs = _build_run_logs(
            run.id,
            request,
            result,
            model_instances,
            parent_run_id,
            engine_window=engine_window,
            data_length=len(data),
        )
        _async_persist_run_logs(run.id, logs)
    except Exception as e:
        logger.error("构建或保存AI运行日志失败: %s", str(e))

    price_series: List[Dict[str, Any]] = []
    if "date" in data.columns and "close" in data.columns:
        for _, row in data.iterrows():
            price_series.append(
                {
                    "date": row["date"],
                    "close": float(row["close"]),
                }
            )

    response_common = dict(
        run_id=run.id,
        metrics=result.get("metrics", {}),
        equity_curves=result.get("equity_curves", {}),
        price_series=price_series,
        name=run.name,
        symbol=run.symbol,
        data_source=run.data_source,
        frequency=run.frequency,
        models=request.models,
        initial_capital=request.initial_capital,
        config=run.config,
    )

    if run_status == "completed":
        return AiRunResponse(
            status="success",
            message="AI投资回放完成",
            **response_common,
        )
    return AiRunResponse(
        status="failed",
        message=f"AI投资回放失败: {error_message or '未知错误'}",
        **response_common,
    )


@router.post("/ai-investment/run", response_model=AiRunResponse)
def run_ai_investment(request: AiRunRequest, db: Session = Depends(get_db)):
    return _perform_ai_run(request, db, parent_run_id=None)


@router.post("/ai-investment/estimate-calls", response_model=AiCallEstimateResponse)
def estimate_ai_calls(request: AiRunRequest, db: Session = Depends(get_db)):
    try:
        start_time = _parse_iso_datetime(request.start_time)
    except Exception:
        raise HTTPException(status_code=400, detail="start_time 格式无效")

    end_time: Optional[datetime] = None
    if request.end_time:
        try:
            end_time = _parse_iso_datetime(request.end_time)
        except Exception:
            raise HTTPException(status_code=400, detail="end_time 格式无效")

    provider = KlineDataProvider(db)

    data_source = request.data_source
    frequency = request.frequency
    if data_source == "database":
        frequency = "1d"

    data = provider.get_kline(
        symbol=request.symbol,
        start_time=start_time,
        end_time=end_time,
        data_source=data_source,
        frequency=frequency,
    )

    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="无法获取指定条件的K线数据")

    data_length = len(data)
    engine_window = int(request.window)
    if engine_window <= 1:
        engine_window = 2

    per_model_calls = max(0, data_length - engine_window - 1)
    model_count = len(request.models or [])
    total_calls = per_model_calls * model_count
    formula = f"max(0, data_length - window - 1) = max(0, {data_length} - {engine_window} - 1)"
    message = (
        f"在当前时间范围内共有 {data_length} 根K线，窗口={engine_window}，"
        f"单个模型理论预测步数约为 {per_model_calls} 次，"
        f"{model_count} 个模型合计约 {total_calls} 次。"
    )

    return AiCallEstimateResponse(
        data_length=data_length,
        window=request.window,
        engine_window=engine_window,
        per_model_calls=per_model_calls,
        model_count=model_count,
        total_calls=total_calls,
        formula=formula,
        message=message,
    )


class AiRunItem(BaseModel):
    id: int
    name: str
    symbol: str
    models: List[str]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]


@router.get("/ai-investment/runs", response_model=List[AiRunItem])
def list_ai_runs(db: Session = Depends(get_db)):
    runs = (
        db.query(AiInvestmentRun)
        .order_by(AiInvestmentRun.created_at.desc())
        .all()
    )
    items: List[AiRunItem] = []
    for run in runs:
        models: List[str] = []
        if isinstance(run.models, list):
            models = run.models
        items.append(
            AiRunItem(
                id=run.id,
                name=run.name,
                symbol=run.symbol,
                models=models,
                status=run.status,
                created_at=run.created_at,
                completed_at=run.completed_at,
            )
        )
    return items


class AiRecordItem(BaseModel):
    id: int
    run_id: int
    model_type: str
    timestamp: datetime
    predicted_price: float
    actual_price: float
    action: str
    position: float
    pnl: float
    cumulative_pnl: float
    equity: float
    trigger_reason: Optional[str] = None


@router.get("/ai-investment/run/{run_id}/records", response_model=List[AiRecordItem])
def get_ai_run_records(run_id: int, db: Session = Depends(get_db)):
    records = (
        db.query(AiInvestmentRecord)
        .filter(AiInvestmentRecord.run_id == run_id)
        .order_by(AiInvestmentRecord.timestamp.asc())
        .all()
    )
    items: List[AiRecordItem] = []
    for rec in records:
        items.append(
            AiRecordItem(
                id=rec.id,
                run_id=rec.run_id,
                model_type=rec.model_type,
                timestamp=rec.timestamp,
                predicted_price=rec.predicted_price or 0.0,
                actual_price=rec.actual_price or 0.0,
                action=rec.action or "",
                position=rec.position or 0.0,
                pnl=rec.pnl or 0.0,
                cumulative_pnl=rec.cumulative_pnl or 0.0,
                equity=rec.equity or 0.0,
                trigger_reason=(rec.extra or {}).get("trigger_reason") if rec.extra else None,
            )
        )
    return items


@router.get("/ai-investment/run/{run_id}", response_model=AiRunResponse)
def get_ai_run_detail(run_id: int, db: Session = Depends(get_db)):
    run = db.query(AiInvestmentRun).filter(AiInvestmentRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="运行记录不存在")

    metrics: Dict[str, Dict[str, float]] = {}
    if isinstance(run.performance_metrics, dict):
        metrics = run.performance_metrics

    models: List[str] = []
    if isinstance(run.models, list):
        models = run.models

    records = (
        db.query(AiInvestmentRecord)
        .filter(AiInvestmentRecord.run_id == run.id)
        .order_by(AiInvestmentRecord.timestamp.asc())
        .all()
    )

    equity_curves: Dict[str, List[Dict[str, Any]]] = {}
    price_series: List[Dict[str, Any]] = []
    seen_dates = set()

    for rec in records:
        model_type = rec.model_type or ""
        if rec.timestamp and rec.equity is not None:
            equity_curves.setdefault(model_type, []).append(
                {"date": rec.timestamp, "equity": float(rec.equity)}
            )
        if rec.timestamp and rec.actual_price is not None:
            key = rec.timestamp.isoformat()
            if key not in seen_dates:
                seen_dates.add(key)
                price_series.append(
                    {"date": rec.timestamp, "close": float(rec.actual_price)}
                )

    return AiRunResponse(
        status="success",
        message="获取AI投资运行详情成功",
        run_id=run.id,
        metrics=metrics,
        equity_curves=equity_curves,
        price_series=price_series,
        name=run.name,
        symbol=run.symbol,
        data_source=run.data_source,
        frequency=run.frequency,
        models=models,
        initial_capital=run.initial_capital,
        config=run.config or {},
    )


@router.get("/ai-investment/run/{run_id}/logs", response_model=AiRunLogResponse)
def get_ai_run_logs(
    run_id: int,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(50, ge=1, le=200, description="每页数量"),
    level: Optional[str] = Query(None, description="日志级别过滤"),
    category: Optional[str] = Query(None, description="日志类别过滤"),
    keyword: Optional[str] = Query(None, description="关键字搜索"),
    record_id: Optional[int] = Query(None, description="关联交易记录ID"),
):
    query = db.query(AiInvestmentLog).filter(AiInvestmentLog.run_id == run_id)

    if level:
        query = query.filter(AiInvestmentLog.level == level.upper())
    if category:
        query = query.filter(AiInvestmentLog.category == category)
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(AiInvestmentLog.message.ilike(like))
    if record_id is not None:
        query = query.filter(AiInvestmentLog.record_id == record_id)

    total = query.count()

    logs = (
        query.order_by(AiInvestmentLog.timestamp.asc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )

    items: List[AiRunLogItem] = []
    for log in logs:
        items.append(
            AiRunLogItem(
                id=log.id,
                timestamp=log.timestamp,
                level=log.level,
                category=log.category,
                message=log.message,
                ai_input=_decompress_json(log.ai_input_compressed),
                ai_output=_decompress_json(log.ai_output_compressed),
                extra=log.extra,
            )
        )

    return AiRunLogResponse(total=total, items=items)


@router.post("/ai-investment/run/{run_id}/resume", response_model=AiRunResponse)
def resume_ai_run(
    run_id: int,
    request: AiRunResumeRequest,
    db: Session = Depends(get_db),
):
    base_run = db.query(AiInvestmentRun).filter(AiInvestmentRun.id == run_id).first()
    if base_run is None:
        raise HTTPException(status_code=404, detail="运行记录不存在")

    config = base_run.config or {}

    last_record = (
        db.query(AiInvestmentRecord)
        .filter(AiInvestmentRecord.run_id == base_run.id)
        .order_by(AiInvestmentRecord.timestamp.desc())
        .first()
    )

    start_time: Optional[str] = None
    if last_record and last_record.timestamp:
        start_time = last_record.timestamp.isoformat()
    elif base_run.completed_at:
        start_time = base_run.completed_at.isoformat()
    else:
        start_time = config.get("end_time") or config.get("start_time")

    if not start_time:
        raise HTTPException(status_code=400, detail="原始运行缺少起始时间，无法续跑")

    end_time = request.end_time

    models: List[str] = []
    if isinstance(base_run.models, list):
        models = base_run.models

    initial_capital = base_run.initial_capital
    if last_record and last_record.equity is not None:
        initial_capital = float(last_record.equity)

    resume_request = AiRunRequest(
        name=request.name or f"{base_run.name}-续跑",
        symbol=base_run.symbol,
        start_time=start_time,
        end_time=end_time,
        data_source=base_run.data_source,
        frequency=base_run.frequency,
        models=models,
        initial_capital=initial_capital,
        buy_threshold=float(config.get("buy_threshold", 0.002)),
        sell_threshold=float(config.get("sell_threshold", -0.002)),
        stop_loss_pct=float(config.get("stop_loss_pct", 0.05)),
        take_profit_pct=float(config.get("take_profit_pct", 0.1)),
        window=int(config.get("window", 20)),
    )

    return _perform_ai_run(resume_request, db, parent_run_id=base_run.id)


class AiPromptSettingItem(BaseModel):
    id: int
    model_type: str
    scene: Optional[str]
    system_prompt: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class AiPromptSettingCreate(BaseModel):
    model_type: str
    scene: Optional[str] = "ai_investment"
    system_prompt: str
    description: Optional[str] = None


@router.get("/ai-investment/prompt-settings", response_model=List[AiPromptSettingItem])
def list_ai_prompt_settings(
    model_type: Optional[str] = None,
    scene: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(AiPromptSetting)
    if model_type:
        query = query.filter(AiPromptSetting.model_type == model_type)
    if scene:
        scoped_query = query.filter(AiPromptSetting.scene == scene)
        records = scoped_query.order_by(AiPromptSetting.created_at.desc()).all()
        if not records:
            records = query.order_by(AiPromptSetting.created_at.desc()).all()
    else:
        records = query.order_by(AiPromptSetting.created_at.desc()).all()
    items: List[AiPromptSettingItem] = []
    for rec in records:
        items.append(
            AiPromptSettingItem(
                id=rec.id,
                model_type=rec.model_type,
                scene=rec.scene,
                system_prompt=rec.system_prompt,
                description=rec.description,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
            )
        )
    return items


@router.post("/ai-investment/prompt-settings", response_model=AiPromptSettingItem)
def save_ai_prompt_setting(
    request: AiPromptSettingCreate,
    db: Session = Depends(get_db),
):
    now = datetime.now()
    setting = (
        db.query(AiPromptSetting)
        .filter(AiPromptSetting.model_type == request.model_type)
        .filter(AiPromptSetting.scene == request.scene)
        .first()
    )
    if setting is None:
        setting = AiPromptSetting(
            model_type=request.model_type,
            scene=request.scene,
            system_prompt=request.system_prompt,
            description=request.description,
            created_at=now,
            updated_at=now,
        )
        db.add(setting)
    else:
        setting.system_prompt = request.system_prompt
        setting.description = request.description
        setting.updated_at = now
    db.commit()
    db.refresh(setting)
    return AiPromptSettingItem(
        id=setting.id,
        model_type=setting.model_type,
        scene=setting.scene,
        system_prompt=setting.system_prompt,
        description=setting.description,
        created_at=setting.created_at,
        updated_at=setting.updated_at,
    )
