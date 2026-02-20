import os
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    ForeignKey,
    Text,
    create_engine,
    UniqueConstraint,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# 创建Base类
Base = declarative_base()

# 数据源模型
class DataSource(Base):
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    api_key = Column(String(200), nullable=True)
    api_secret = Column(String(200), nullable=True)
    base_url = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系：一个数据源可以有多个股票
    stocks = relationship("Stock", back_populates="source")
    
    def __repr__(self):
        return f"<DataSource(name='{self.name}')>"

# 股票模型
class Stock(Base):
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    type = Column(String(20), nullable=False, comment="股票类型: A股/港股/美股/期货/加密货币等")
    exchange = Column(String(20), nullable=True)
    industry = Column(String(50), nullable=True)
    sector = Column(String(50), nullable=True)
    source_id = Column(Integer, ForeignKey('data_sources.id'), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=True)
    total_records = Column(Integer, default=0, comment="总记录数")
    first_date = Column(Date, nullable=True, comment="数据开始日期")
    last_date = Column(Date, nullable=True, comment="数据结束日期")
    
    # 关系：一个股票属于一个数据源
    source = relationship("DataSource", back_populates="stocks")
    
    # 关系：一个股票可以有多个数据记录
    data = relationship("StockData", back_populates="stock", cascade="all, delete-orphan")
    
    # 创建唯一约束：每个股票代码在每个数据源中只能有一个
    __table_args__ = (UniqueConstraint('symbol', 'source_id', name='uix_stock_symbol_source'),)
    
    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"

# 股票数据模型
class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=True)
    
    # 其他可选字段
    turnover = Column(Float, nullable=True, comment="换手率")
    pe_ratio = Column(Float, nullable=True, comment="市盈率")
    pb_ratio = Column(Float, nullable=True, comment="市净率")
    dividend_yield = Column(Float, nullable=True, comment="股息率")
    
    # 关系：每条数据属于一个股票
    stock = relationship("Stock", back_populates="data")
    
    # 创建联合唯一约束：每个股票在每一天只能有一条数据
    __table_args__ = (UniqueConstraint('stock_id', 'date', name='uix_stock_date'),)
    
    def __repr__(self):
        return f"<StockData(stock_id={self.stock_id}, date='{self.date}')>"

# 技术指标模型
class TechnicalIndicator(Base):
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False)
    indicator_name = Column(String(50), nullable=False, comment="指标名称: MA, RSI, MACD等")
    indicator_value = Column(Float, nullable=False)
    parameter = Column(String(50), nullable=True, comment="指标参数, 如MA的周期")
    
    # 关系: 每个指标属于一个股票
    stock = relationship("Stock")
    
    # 创建联合唯一约束:每个股票每天每种指标(包括参数)只能有一个值
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', 'indicator_name', 'parameter', name='uix_stock_indicator'),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicator(stock_id={self.stock_id}, indicator='{self.indicator_name}({self.parameter})')>"

# 每日价格模型 (兼容旧系统)
class DailyPrice(Base):
    __tablename__ = 'daily_prices'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=True)
    
    __table_args__ = (UniqueConstraint('stock_id', 'date', name='uix_stock_date_price'),)

# 数据库连接和会话创建函数
def get_engine(db_url=None):
    """创建数据库引擎"""
    if db_url is None:
        from ..config import DATABASE_URL
        db_url = DATABASE_URL
    return create_engine(db_url, echo=False)

def get_session(engine=None):
    """创建数据库会话"""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def get_db():
    """FastAPI依赖项，提供数据库会话"""
    db = get_session()
    try:
        yield db
    finally:
        db.close()

def init_db(db_url=None):
    """初始化数据库，创建所有表"""
    # 导入所有模型以确保它们被注册到对应的 Base.metadata 中
    from . import strategy  # 导入策略相关模型（使用 .base.Base）
    from . import optimization  # 导入优化相关模型（使用 .base.Base）
    from .base import Base as CoreBase  # 回测/策略等核心模型
    
    engine = get_engine(db_url)
    
    # 1. 创建当前文件中定义的数据相关表（DataSource / Stock / StockData / TechnicalIndicator / DailyPrice）
    Base.metadata.create_all(engine)
    # 2. 创建核心回测与AI投资相关表（Strategy / Backtest / AiInvestmentRun / AiInvestmentRecord 等）
    CoreBase.metadata.create_all(engine)

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if "ai_investment_runs" in tables:
            columns = [col["name"] for col in inspector.get_columns("ai_investment_runs")]
            if "parent_run_id" not in columns:
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            "ALTER TABLE ai_investment_runs ADD COLUMN parent_run_id INTEGER"
                        )
                    )
                    conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS ix_ai_investment_runs_parent_run_id ON ai_investment_runs (parent_run_id)"
                        )
                    )
    except Exception:
        pass

    return engine 
