from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import Base

class Strategy(Base):
    """策略模型"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    code = Column(String)  # 策略代码
    parameters = Column(String, nullable=True)  # 策略参数，JSON字符串格式
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_template = Column(Boolean, default=False)  # 是否为模板策略
    template = Column(String, nullable=True)  # 策略模板类型
    
    # 一对多关系：一个策略可以有多个回测
    backtests = relationship("Backtest", back_populates="strategy")
    
    # 参数优化相关关系
    parameter_spaces = relationship("StrategyParameterSpace", back_populates="strategy", cascade="all, delete-orphan")
    parameter_sets = relationship("ParameterSet", back_populates="strategy", cascade="all, delete-orphan")
    optimization_jobs = relationship("OptimizationJob", back_populates="strategy", cascade="all, delete-orphan")

class StrategySnapshot(Base):
    """策略快照模型 - 保存策略的镜像"""
    __tablename__ = "strategy_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    code = Column(Text)  # 策略代码镜像
    parameters = Column(Text, nullable=True)  # 策略参数镜像，JSON字符串格式
    template = Column(String, nullable=True)  # 策略模板类型镜像
    created_at = Column(DateTime, default=datetime.now)
    
    # 多对一关系：多个快照属于一个策略
    strategy = relationship("Strategy")
    # 一对多关系：一个快照可以有多个回测
    backtests = relationship("Backtest", back_populates="strategy_snapshot")

class Backtest(Base):
    """回测模型"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    
    # 策略相关
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)  # 当前策略ID（可能为空）
    strategy_snapshot_id = Column(Integer, ForeignKey("strategy_snapshots.id"), nullable=False)  # 策略快照ID
    
    # 回测参数
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    instruments = Column(JSON)  # 回测标的，JSON格式
    parameters = Column(JSON, nullable=True)  # 回测参数，JSON格式
    
    # 仓位控制参数
    position_config = Column(JSON, nullable=True)  # 仓位控制配置
    
    # 回测结果
    results = Column(JSON, nullable=True)  # 回测结果，JSON格式
    equity_curve = Column(JSON, nullable=True)  # 权益曲线数据
    trade_records = Column(JSON, nullable=True)  # 交易记录
    performance_metrics = Column(JSON, nullable=True)  # 性能指标
    
    # 状态和时间
    status = Column(String, default='running')  # running, completed, failed
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    # 关系
    strategy = relationship("Strategy", back_populates="backtests")
    strategy_snapshot = relationship("StrategySnapshot", back_populates="backtests")
    trades = relationship("Trade", back_populates="backtest")

class BacktestStatus(Base):
    """回测状态表 - 存储最新状态"""
    __tablename__ = "backtest_status"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)  # 回测名称，唯一
    description = Column(String, nullable=True)
    
    # 策略相关
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    strategy_snapshot_id = Column(Integer, ForeignKey("strategy_snapshots.id"), nullable=False)
    
    # 回测参数
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    instruments = Column(JSON)
    parameters = Column(JSON, nullable=True)
    position_config = Column(JSON, nullable=True)
    
    # 最新回测结果
    results = Column(JSON, nullable=True)
    equity_curve = Column(JSON, nullable=True)
    trade_records = Column(JSON, nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)  # 策略执行日志
    
    # 状态和时间
    status = Column(String, default='running')
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    # 关系
    strategy = relationship("Strategy")
    strategy_snapshot = relationship("StrategySnapshot")
    history_records = relationship("BacktestHistory", back_populates="status_record", cascade="all, delete-orphan")

class BacktestHistory(Base):
    """回测历史表 - 记录每次更新"""
    __tablename__ = "backtest_history"
    
    id = Column(Integer, primary_key=True, index=True)
    status_id = Column(Integer, ForeignKey("backtest_status.id"), nullable=False)
    
    # 回测参数（保存当时的参数）
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    instruments = Column(JSON)
    parameters = Column(JSON, nullable=True)
    position_config = Column(JSON, nullable=True)
    
    # 回测结果
    results = Column(JSON, nullable=True)
    equity_curve = Column(JSON, nullable=True)
    trade_records = Column(JSON, nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)  # 策略执行日志
    
    # 状态和时间
    status = Column(String, default='running')
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    # 操作类型
    operation_type = Column(String, default='create')  # create, update, rerun
    
    # 关系
    status_record = relationship("BacktestStatus", back_populates="history_records")

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"))
    datetime = Column(DateTime)
    symbol = Column(String, index=True)
    direction = Column(String)  # BUY or SELL
    price = Column(Float)
    quantity = Column(Float)
    commission = Column(Float)
    backtest = relationship("Backtest", back_populates="trades")


class AiInvestmentRun(Base):
    __tablename__ = "ai_investment_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    symbol = Column(String, index=True)
    models = Column(JSON)
    data_source = Column(String, default="yahoo")
    frequency = Column(String, default="1m")
    initial_capital = Column(Float, default=100000.0)
    status = Column(String, default="running")
    parent_run_id = Column(Integer, ForeignKey("ai_investment_runs.id"), nullable=True, index=True)
    config = Column(JSON, nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    completed_at = Column(DateTime, nullable=True)

    records = relationship("AiInvestmentRecord", back_populates="run", cascade="all, delete-orphan")
    logs = relationship("AiInvestmentLog", back_populates="run", cascade="all, delete-orphan")


class AiInvestmentRecord(Base):
    __tablename__ = "ai_investment_records"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("ai_investment_runs.id"), index=True)
    model_type = Column(String, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    predicted_price = Column(Float)
    actual_price = Column(Float)
    action = Column(String)
    position = Column(Float)
    pnl = Column(Float)
    cumulative_pnl = Column(Float)
    equity = Column(Float)
    extra = Column(JSON, nullable=True)

    run = relationship("AiInvestmentRun", back_populates="records")


class AiInvestmentLog(Base):
    __tablename__ = "ai_investment_logs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("ai_investment_runs.id"), index=True)
    record_id = Column(Integer, ForeignKey("ai_investment_records.id"), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    level = Column(String, index=True)
    category = Column(String, index=True)
    message = Column(Text)
    ai_input_compressed = Column(Text, nullable=True)
    ai_output_compressed = Column(Text, nullable=True)
    extra = Column(JSON, nullable=True)

    run = relationship("AiInvestmentRun", back_populates="logs")
    record = relationship("AiInvestmentRecord")


class AiPromptSetting(Base):
    __tablename__ = "ai_prompt_settings"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String, index=True)
    scene = Column(String, nullable=True)
    system_prompt = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
