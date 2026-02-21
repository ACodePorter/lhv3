"""
回测状态管理API路由
提供基于新架构的回测状态和历史记录管理功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, and_
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel

from ..models import get_db
from ..models.strategy import BacktestStatus, BacktestHistory, StrategySnapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest-status", tags=["backtest-status"])

# 请求模型
class UpdateBacktestRequest(BaseModel):
    """更新回测请求模型"""
    new_name: Optional[str] = None
    new_description: Optional[str] = None  # 添加描述字段
    update_to_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: Optional[float] = None
    instruments: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    position_config: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

# 响应模型
class BacktestStatusResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class BacktestHistoryResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_backtest_status(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
    name: Optional[str] = Query(None, description="回测名称过滤"),
    status: Optional[str] = Query(None, description="状态过滤")
):
    """获取回测状态列表（最新状态）"""
    try:
        offset = (page - 1) * size
        
        # 构建查询
        query = db.query(BacktestStatus).options(
            joinedload(BacktestStatus.strategy_snapshot)
        )
        
        # 应用过滤条件
        if name:
            query = query.filter(BacktestStatus.name.like(f"%{name}%"))
        if status:
            query = query.filter(BacktestStatus.status == status)
        
        # 执行查询
        backtest_statuses = query.order_by(desc(BacktestStatus.updated_at)).offset(offset).limit(size).all()
        
        result = []
        for status in backtest_statuses:
            strategy_name = None
            if status.strategy_snapshot:
                strategy_name = status.strategy_snapshot.name
            
            # 提取性能指标
            performance_metrics = status.performance_metrics or {}
            total_return = performance_metrics.get('total_return', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            
            result.append({
                "id": status.id,
                "name": status.name,
                "description": status.description,
                "strategy_name": strategy_name,
                "start_date": status.start_date.isoformat() if status.start_date else None,
                "end_date": status.end_date.isoformat() if status.end_date else None,
                "initial_capital": status.initial_capital,
                "instruments": status.instruments,
                "status": status.status,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "parameters": status.parameters,  # 添加参数配置
                "created_at": status.created_at.isoformat(),
                "updated_at": status.updated_at.isoformat(),
                "completed_at": status.completed_at.isoformat() if status.completed_at else None,
                "performance_metrics": performance_metrics,
                "trade_records": status.trade_records  # 添加交易记录数据
            })
        
        return result
        
    except Exception as e:
        logger.error(f"获取回测状态列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取回测状态列表失败: {str(e)}")

@router.get("/{status_id}", response_model=Dict[str, Any])
async def get_backtest_status(
    status_id: int,
    db: Session = Depends(get_db)
):
    """获取回测状态详情"""
    try:
        status = db.query(BacktestStatus).options(
            joinedload(BacktestStatus.strategy_snapshot)
        ).filter(BacktestStatus.id == status_id).first()
        
        if not status:
            raise HTTPException(status_code=404, detail="回测状态不存在")
        
        strategy_name = None
        if status.strategy_snapshot:
            strategy_name = status.strategy_snapshot.name
        
        return {
            "status": "success",
            "data": {
                "id": status.id,
                "name": status.name,
                "description": status.description,
                "strategy_name": strategy_name,
                "strategy_id": status.strategy_id,
                "strategy_snapshot_id": status.strategy_snapshot_id,
                "start_date": status.start_date.isoformat() if status.start_date else None,
                "end_date": status.end_date.isoformat() if status.end_date else None,
                "initial_capital": status.initial_capital,
                "instruments": status.instruments,
                "parameters": status.parameters,
                "position_config": status.position_config,
                "results": status.results,
                "equity_curve": status.equity_curve,
                "trade_records": status.trade_records,
                "performance_metrics": status.performance_metrics,
                "logs": status.logs,
                "status": status.status,
                "created_at": status.created_at.isoformat(),
                "updated_at": status.updated_at.isoformat(),
                "completed_at": status.completed_at.isoformat() if status.completed_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测状态详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取回测状态详情失败: {str(e)}")

@router.get("/{status_id}/history", response_model=List[Dict[str, Any]])
async def get_backtest_history(
    status_id: int,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """获取回测历史记录"""
    try:
        # 验证状态记录存在
        status = db.query(BacktestStatus).filter(BacktestStatus.id == status_id).first()
        if not status:
            raise HTTPException(status_code=404, detail="回测状态不存在")
        
        offset = (page - 1) * size
        
        # 获取历史记录
        history_records = db.query(BacktestHistory).filter(
            BacktestHistory.status_id == status_id
        ).order_by(desc(BacktestHistory.created_at)).offset(offset).limit(size).all()
        
        result = []
        for record in history_records:
            # 提取性能指标
            performance_metrics = record.performance_metrics or {}
            total_return = performance_metrics.get('total_return', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            
            result.append({
                "id": record.id,
                "status_id": record.status_id,
                "start_date": record.start_date.isoformat() if record.start_date else None,
                "end_date": record.end_date.isoformat() if record.end_date else None,
                "initial_capital": record.initial_capital,
                "instruments": record.instruments,
                "status": record.status,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "created_at": record.created_at.isoformat(),
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                "operation_type": record.operation_type,
                "performance_metrics": performance_metrics,
                "logs": record.logs
            })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测历史记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取回测历史记录失败: {str(e)}")

@router.delete("/{status_id}/history/{history_id}", response_model=Dict[str, Any])
async def delete_backtest_history(
    status_id: int,
    history_id: int,
    db: Session = Depends(get_db)
):
    """删除单条回测历史记录"""
    try:
        status = db.query(BacktestStatus).filter(BacktestStatus.id == status_id).first()
        if not status:
            raise HTTPException(status_code=404, detail="回测状态不存在")
        
        history_record = db.query(BacktestHistory).filter(
            BacktestHistory.id == history_id,
            BacktestHistory.status_id == status_id
        ).first()
        if not history_record:
            raise HTTPException(status_code=404, detail="历史记录不存在")
        
        db.delete(history_record)
        db.commit()
        
        logger.info(f"已删除回测历史记录: 状态ID={status_id}, 历史ID={history_id}")
        
        return {
            "status": "success",
            "message": "历史记录已删除"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除回测历史记录失败: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除回测历史记录失败: {str(e)}")

@router.post("/{status_id}/update", response_model=Dict[str, Any])
async def update_backtest_status(
    status_id: int,
    update_request: UpdateBacktestRequest,
    db: Session = Depends(get_db)
):
    """更新回测状态（重新运行回测）"""
    try:
        # 获取现有状态记录
        status = db.query(BacktestStatus).filter(BacktestStatus.id == status_id).first()
        if not status:
            raise HTTPException(status_code=404, detail="回测状态不存在")
        
        logger.info(f"开始更新回测状态 {status_id}: {update_request.dict()}")
        
        # 导入回测服务
        from .backtest_service import BacktestService
        
        # 创建回测服务实例
        backtest_service = BacktestService(db)
        
        # 准备回测参数 - 使用请求中的新参数或保持原有参数
        # 处理日期参数：如果提供了update_to_date，则使用原start_date到update_to_date
        logger.info(f"处理日期参数: update_request.update_to_date={update_request.update_to_date}")
        logger.info(f"状态记录中的日期: start_date={status.start_date}, end_date={status.end_date}")
        
        if update_request.update_to_date:
            # 确保start_date是字符串格式
            if isinstance(status.start_date, str):
                start_date = status.start_date
            else:
                start_date = status.start_date.strftime('%Y-%m-%d')
            end_date = update_request.update_to_date
            logger.info(f"使用update_to_date: start_date={start_date}, end_date={end_date}")
        else:
            # 确保日期是字符串格式
            if isinstance(status.start_date, str):
                start_date = update_request.start_date or status.start_date
            else:
                start_date = update_request.start_date or status.start_date.strftime('%Y-%m-%d')
            
            # 如果没有指定end_date，则获取最新的可用数据
            if update_request.end_date:
                end_date = update_request.end_date
            else:
                # 获取最新的可用数据日期
                from datetime import timedelta
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"未指定end_date，使用最新可用日期: {end_date}")
            logger.info(f"使用常规日期参数: start_date={start_date}, end_date={end_date}")
        
        initial_capital = update_request.initial_capital or status.initial_capital
        instruments = update_request.instruments or status.instruments
        parameters = update_request.parameters or status.parameters or {}
        position_config = update_request.position_config or status.position_config
        
        # 处理新名称和描述
        new_name = update_request.new_name or status.name
        new_description = update_request.new_description if update_request.new_description is not None else status.description
        logger.info(f"处理新名称: update_request.new_name={update_request.new_name}, status.name={status.name}, new_name={new_name}")
        logger.info(f"处理新描述: update_request.new_description={update_request.new_description}, status.description={status.description}, new_description={new_description}")
        
        # 获取股票代码
        symbol = instruments[0] if instruments else 'TSLA'
        
        # 获取策略ID
        strategy_id = status.strategy_id
        if not strategy_id:
            raise HTTPException(status_code=400, detail="无法获取策略ID")
        
        # 合并参数
        if position_config:
            parameters['positionConfig'] = position_config
        
        # 重新运行回测
        logger.info(f"重新运行回测: 策略ID={strategy_id}, 股票={symbol}, 期间={start_date}至{end_date}, 初始资金={initial_capital}")
        
        result = backtest_service.run_backtest(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            parameters=parameters,
            data_source='database',
            features=[]
        )
        
        if result.get('status') != 'success':
            raise HTTPException(status_code=500, detail=f"回测运行失败: {result.get('message', '未知错误')}")
        
        # 获取实际的回测结果数据
        backtest_data = result.get('data', {})
        if not backtest_data:
            raise HTTPException(status_code=500, detail="回测结果为空")
        
        # 调试：打印回测结果
        logger.info(f"回测结果状态: {result.get('status')}")
        logger.info(f"回测结果消息: {result.get('message')}")
        logger.info(f"回测数据键: {backtest_data.keys() if isinstance(backtest_data, dict) else 'Not a dict'}")
        
        # 更新状态记录 - 包括新的参数
        status.name = new_name
        status.description = new_description
        status.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        status.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        status.initial_capital = initial_capital
        status.instruments = instruments
        status.parameters = parameters
        status.position_config = position_config
        status.results = backtest_data
        status.equity_curve = backtest_data.get('equity_curve')
        status.trade_records = backtest_data.get('trades')  # 回测引擎返回的字段名是'trades'
        
        # 调试：打印字段更新信息
        logger.info(f"更新字段: trade_records={len(status.trade_records) if status.trade_records else 0}条记录")
        logger.info(f"更新字段: equity_curve={len(status.equity_curve) if status.equity_curve else 0}条记录")
        status.performance_metrics = {
            'total_return': backtest_data.get('total_return'),
            'annual_return': backtest_data.get('annual_return'),  # 添加年化收益率
            'max_drawdown': backtest_data.get('max_drawdown'),
            'sharpe_ratio': backtest_data.get('sharpe_ratio'),
            'volatility': backtest_data.get('volatility'),
            'win_rate': backtest_data.get('win_rate'),
            'profit_factor': backtest_data.get('profit_factor')
        }
        status.updated_at = datetime.now()
        status.completed_at = datetime.now()
        
        # 创建新的历史记录 - 使用更新后的参数
        history_record = BacktestHistory(
            status_id=status.id,
            start_date=status.start_date,
            end_date=status.end_date,
            initial_capital=status.initial_capital,
            instruments=status.instruments,
            parameters=status.parameters,
            position_config=status.position_config,
            # 保存回测结果数据到历史记录
            results=backtest_data,
            equity_curve=backtest_data.get('equity_curve'),
            trade_records=backtest_data.get('trades'),  # 回测引擎返回的字段名是'trades'
            performance_metrics=status.performance_metrics,
            status=status.status,
            completed_at=status.completed_at,
            operation_type='update'
        )
        db.add(history_record)
        
        db.commit()
        db.refresh(status)
        db.refresh(history_record)
        
        logger.info(f"回测状态更新成功: 状态ID={status.id}, 历史ID={history_record.id}")
        
        return {
            "status": "success",
            "message": "回测更新成功",
            "data": {
                "status_id": status.id,
                "history_id": history_record.id,
                "new_backtest_name": new_name,
                "update_range": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "updated_at": status.updated_at.isoformat(),
                "performance_metrics": status.performance_metrics,
                "updated_parameters": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_capital": initial_capital,
                    "instruments": instruments,
                    "parameters": parameters,
                    "position_config": position_config
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新回测状态失败: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新回测状态失败: {str(e)}")

@router.delete("/{status_id}", response_model=Dict[str, Any])
async def delete_backtest_status(
    status_id: int,
    db: Session = Depends(get_db)
):
    """删除回测状态及其历史记录"""
    try:
        status = db.query(BacktestStatus).filter(BacktestStatus.id == status_id).first()
        if not status:
            raise HTTPException(status_code=404, detail="回测状态不存在")
        
        # 删除状态记录（级联删除历史记录）
        db.delete(status)
        db.commit()
        
        return {
            "status": "success",
            "message": "回测状态已删除"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除回测状态失败: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除回测状态失败: {str(e)}")

@router.get("/stats/summary", response_model=Dict[str, Any])
async def get_backtest_stats_summary(db: Session = Depends(get_db)):
    """获取回测统计摘要"""
    try:
        # 统计总数
        total_count = db.query(BacktestStatus).count()
        completed_count = db.query(BacktestStatus).filter(BacktestStatus.status == 'completed').count()
        running_count = db.query(BacktestStatus).filter(BacktestStatus.status == 'running').count()
        failed_count = db.query(BacktestStatus).filter(BacktestStatus.status == 'failed').count()
        
        # 统计历史记录总数
        total_history = db.query(BacktestHistory).count()
        
        return {
            "status": "success",
            "data": {
                "total_backtests": total_count,
                "completed_backtests": completed_count,
                "running_backtests": running_count,
                "failed_backtests": failed_count,
                "total_history_records": total_history,
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"获取回测统计摘要失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取回测统计摘要失败: {str(e)}")
