from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import os
import sys
import json
from sqlalchemy import text
import traceback

# 添加系统路径，确保可以导入后端模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..models import get_db, init_db
from ..data import DataFetcher, DataProcessor
from ..strategy.templates import MovingAverageCrossover, BollingerBandsStrategy, MACDStrategy, RSIStrategy
from ..backtest import BacktestEngine, PerformanceAnalyzer
from ..config import FRONTEND_URL
from . import data_routes

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="量化交易回测系统API",
    description="提供数据管理、策略回测和分析的API接口",
    version="0.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，方便调试
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"收到请求: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"请求处理完成: {request.method} {request.url.path} - 状态码: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"请求处理错误: {request.method} {request.url.path} - 错误: {str(e)}")
        raise

# 注册数据管理路由
app.include_router(data_routes.router, prefix="/api/data", tags=["数据管理"])

# 注册策略管理路由
from .strategy_routes import router as strategy_router
app.include_router(strategy_router, prefix="/api/strategies", tags=["策略管理"])

# 注册回测路由
from .backtest_routes import router as backtest_router
from .backtest_status_routes import router as backtest_status_router
app.include_router(backtest_router, prefix="/api/backtest", tags=["回测"])
app.include_router(backtest_status_router, tags=["回测状态"])

# 注册参数优化路由
from .optimization_routes import router as optimization_router
app.include_router(optimization_router, tags=["参数优化"])

# 注册异步任务路由
from .async_routes import router as async_router
app.include_router(async_router, tags=["异步任务"])

# 注册AI实时投资跑数路由
from .ai_investment_routes import router as ai_investment_router
app.include_router(ai_investment_router, prefix="/api", tags=["AI投资"])


# 初始化数据库
@app.on_event("startup")
async def startup():
    try:
        # 确保每次启动时初始化数据库表结构
        init_db()
        logger.info("数据库初始化成功，所有表已创建")
        
        # 初始化任务队列系统
        from ..utils.task_queue import get_task_queue
        task_queue = get_task_queue()
        logger.info("异步任务队列系统已初始化")
        
        # 初始化第一个示例策略
        initialize_default_strategy()
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """应用关闭时的清理工作"""
    try:
        # 关闭任务队列系统
        from ..utils.task_queue import get_task_queue
        task_queue = get_task_queue()
        await task_queue.shutdown()
        logger.info("异步任务队列系统已关闭")
    except Exception as e:
        logger.error(f"关闭任务队列系统失败: {e}")


def initialize_default_strategy():
    """初始化第一个示例策略到数据库"""
    try:
        from ..models.strategy import Strategy as StrategyModel
        from sqlalchemy import text
        
        # 获取数据库会话
        db = next(get_db())
        
        # 检查是否已存在默认策略
        table_check = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")).fetchone()
        if not table_check:
            logger.warning("strategies表不存在，无法初始化默认策略")
            return
            
        # 检查是否已存在移动平均交叉策略
        existing = db.query(StrategyModel).filter(StrategyModel.name == "MA交叉策略").first()
        if existing:
            logger.info("默认MA交叉策略已存在，跳过初始化")
            return
            
        # MA交叉策略代码
        ma_strategy_code = '''from .strategy_template import StrategyTemplate
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MACrossoverStrategy(StrategyTemplate):
    """
    移动平均线交叉策略
    当MA5上穿MA20时买入，当MA5下穿MA20时卖出
    """
    
    def __init__(self, parameters=None):
        """初始化策略"""
        default_params = {
            "short_window": 5,   # 短期移动平均窗口
            "long_window": 20,   # 长期移动平均窗口
        }
        
        # 合并用户参数与默认参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MA交叉策略", default_params)
        
    def generate_signals(self) -> pd.DataFrame:
        """
        生成交易信号
        
        Returns:
            包含信号的DataFrame，包括:
            - signal: 交易信号 (1: 买入, -1: 卖出, 0: 不操作)
            - trigger_reason: 信号触发原因
        """
        if self.data is None or self.data.empty:
            logger.warning("未设置数据或数据为空，无法生成信号")
            return pd.DataFrame()
        
        # 获取参数
        short_window = self.parameters["short_window"]
        long_window = self.parameters["long_window"]
        
        logger.info(f"生成MA交叉信号: 短期窗口={short_window}, 长期窗口={long_window}")
        
        # 计算指标
        df = self.data.copy()
        
        # 计算移动平均线
        df[f"ma_{short_window}"] = df["close"].rolling(window=short_window).mean()
        df[f"ma_{long_window}"] = df["close"].rolling(window=long_window).mean()
        
        # 计算当前日期和前一日期的移动平均线差值
        df["ma_diff"] = df[f"ma_{short_window}"] - df[f"ma_{long_window}"]
        df["prev_ma_diff"] = df["ma_diff"].shift(1)
        
        # 初始化信号列
        df["signal"] = 0
        df["trigger_reason"] = ""
        
        # 生成买入信号：短期均线从下方上穿长期均线
        buy_signal = (df["ma_diff"] > 0) & (df["prev_ma_diff"] <= 0)
        df.loc[buy_signal, "signal"] = 1
        df.loc[buy_signal, "trigger_reason"] = f"MA{short_window}从下方上穿MA{long_window}"
        
        # 生成卖出信号：短期均线从上方下穿长期均线
        sell_signal = (df["ma_diff"] < 0) & (df["prev_ma_diff"] >= 0)
        df.loc[sell_signal, "signal"] = -1
        df.loc[sell_signal, "trigger_reason"] = f"MA{short_window}从上方下穿MA{long_window}"
        
        # 统计信号数量
        buy_count = (df["signal"] == 1).sum()
        sell_count = (df["signal"] == -1).sum()
        logger.info(f"信号统计: 买入信号={buy_count}个, 卖出信号={sell_count}个")
        
        return df
'''
        
        # 默认参数
        default_parameters = {
            "short_window": 5,
            "long_window": 20
        }
        
        # 创建策略记录
        new_strategy = StrategyModel(
            name="MA交叉策略",
            description="当短期移动平均线(MA5)上穿长期移动平均线(MA20)时买入，下穿时卖出",
            code=ma_strategy_code,
            parameters=json.dumps(default_parameters),
            is_template=True,
            template="ma_crossover",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 添加到数据库
        db.add(new_strategy)
        db.commit()
        
        logger.info(f"成功创建默认MA交叉策略 (ID: {new_strategy.id})")
    except Exception as e:
        logger.error(f"初始化默认策略失败: {e}")
        # 回滚数据库
        try:
            db.rollback()
        except:
            pass

# 健康检查端点
@app.get("/")
async def root():
    return {"status": "success", "message": "量化交易回测系统API服务已启动"}

@app.get("/health")
async def health_check():
    return {"status": "success", "service": "API Server", "version": "0.1.0"}

# 数据API
@app.get("/api/data/symbols")
async def get_symbols(db: Session = Depends(get_db)):
    """获取支持的股票列表"""
    try:
        # 这里可以从数据库中获取股票列表
        # 示例数据
        symbols = [
            {"symbol": "AAPL", "name": "苹果公司", "exchange": "纳斯达克"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "纳斯达克"},
            {"symbol": "MSFT", "name": "微软公司", "exchange": "纳斯达克"},
            {"symbol": "AMZN", "name": "亚马逊公司", "exchange": "纳斯达克"},
            {"symbol": "600000.SS", "name": "浦发银行", "exchange": "上证"},
            {"symbol": "000001.SZ", "name": "平安银行", "exchange": "深证"},
        ]
        return {"status": "success", "data": symbols}
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 注意：此路由已移动到 data_routes.py 中，使用 POST /api/data/fetch 进行自动抓取

# 策略API
@app.get("/api/strategies")
async def get_strategies(name: Optional[str] = None, include_templates: bool = True, db: Session = Depends(get_db)):
    """获取所有策略列表或按名称搜索"""
    try:
        logger.info(f"获取策略列表请求: 名称过滤={name}")
        
        # 导入策略模型
        from ..models.strategy import Strategy as StrategyModel
        
        # 构建查询
        query = db.query(StrategyModel)
        
        # 如果提供了名称参数，进行过滤
        if name:
            query = query.filter(StrategyModel.name.like(f"%{name}%"))
            logger.info(f"应用名称过滤条件: %{name}%")
        
        # 执行查询
        strategies = query.all()
        logger.info(f"找到 {len(strategies)} 个策略")
        
        # 处理结果并返回
        result_data = []
        for strategy in strategies:
            # 解析参数字符串为字典
            params_dict = {}
            if strategy.parameters:
                try:
                    params_dict = json.loads(strategy.parameters)
                except Exception as e:
                    logger.error(f"解析策略 {strategy.id} 的参数失败: {e}")
            
            # 构建单个策略数据
            strategy_data = {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "parameters": params_dict,
                "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
                "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
                "is_template": strategy.is_template,
                "template": strategy.template
            }
            result_data.append(strategy_data)
        
        logger.info(f"最终返回 {len(result_data)} 个策略数据")
        
        return {
            "status": "success",
            "data": result_data
        }
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")

@app.post("/api/strategies")
async def create_strategy_endpoint(request: Request):
    try:
        data = await request.json()
        strategy = create_strategy(data)
        return strategy
    except Exception as e:
        logger.exception(f"创建策略端点错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建策略失败: {str(e)}")

def create_strategy(data):
    """创建新策略"""
    try:
        logger.info(f"处理策略创建请求: {data}")
        
        from ..models.strategy import Strategy as StrategyModel
        from sqlalchemy import text
        
        # 获取SQLAlchemy数据库会话
        db = next(get_db())
        
        name = data.get("name")
        
        if not name:
            logger.error("请求中没有策略名称")
            raise HTTPException(status_code=400, detail="缺少策略名称")
        
        # 检查是否存在具有相同名称的策略
        check_sql = f"SELECT * FROM strategies WHERE name = '{name}'"
        logger.info(f"执行SQL查询: {check_sql}")
        
        existing = db.query(StrategyModel).filter(StrategyModel.name == name).first()
        
        # 处理参数字段
        parameters = data.get("parameters")
        template_type = data.get("template")  # 获取模板类型
        
        if parameters is not None:
            if isinstance(parameters, dict):
                parameters = json.dumps(parameters)
                logger.info(f"已将参数字典序列化为JSON字符串: {parameters}")
            elif isinstance(parameters, str):
                # 验证是否为有效的JSON字符串
                try:
                    json.loads(parameters)
                    logger.info("参数已经是有效的JSON字符串")
                except Exception as e:
                    logger.error(f"提供的参数字符串不是有效的JSON: {e}")
                    raise HTTPException(status_code=400, detail="提供的参数不是有效的JSON格式")
            else:
                logger.error(f"不支持的参数类型: {type(parameters)}")
                raise HTTPException(status_code=400, detail="不支持的参数类型")
        
        if existing:
            logger.info(f"存在同名策略，更新现有策略: {existing.name} (ID: {existing.id})")
            # 更新现有策略字段
            existing.description = data.get("description", existing.description)
            existing.code = data.get("code", existing.code)
            existing.parameters = parameters
            existing.template = template_type  # 更新模板类型
            existing.is_template = data.get("is_template", existing.is_template)
            existing.updated_at = datetime.now()
            
            # 使用现有策略
            strategy = existing
        else:
            # 实例化新策略模型
            strategy = StrategyModel(
                name=data.get("name"),
                description=data.get("description"),
                code=data.get("code"),
                parameters=parameters,  # 使用处理后的参数
                template=template_type,  # 使用模板类型
                is_template=data.get("is_template", False),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            # 添加到会话
            db.add(strategy)
        
        logger.info(f"创建的策略对象参数: parameters={strategy.parameters}, 类型={type(strategy.parameters)}")
        
        # 执行SQL语句检查，使用text()函数
        check_table_sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'"
        logger.info(f"执行SQL查询: {check_table_sql}")
        table_check = db.execute(text(check_table_sql)).fetchone()
        logger.info(f"数据库表检查结果: {table_check}")
        
        if not table_check:
            logger.error("strategies表不存在!")
            raise HTTPException(status_code=500, detail="数据库表不存在")
            
        # 获取ID
        db.flush()  # 获取ID但不提交
        generated_id = strategy.id
        logger.info(f"生成新策略ID: {generated_id}")
        
        # 记录当前策略对象状态
        logger.info(f"策略对象状态: {strategy.__dict__}")
        
        # 提交到数据库
        db.commit()
        
        # 验证创建是否成功
        verification_sql = f"SELECT * FROM strategies WHERE id = {strategy.id}"
        logger.info(f"执行验证SQL: {verification_sql}")
        verification_result = db.execute(text(verification_sql)).fetchone()
        logger.info(f"验证结果: {verification_result}")
        
        # 使用原生SQL查询再次检查
        raw_check_sql = f"SELECT COUNT(*) FROM strategies WHERE id = {strategy.id}"
        logger.info(f"执行计数SQL: {raw_check_sql}")
        raw_check = db.execute(text(raw_check_sql)).scalar()
        logger.info(f"原生SQL验证结果: 找到{raw_check}条记录")
        
        # 使用ORM查询验证
        new_strategy = db.query(StrategyModel).get(strategy.id)
        if not new_strategy:
            logger.error(f"策略创建验证失败: 未找到ID为{strategy.id}的策略")
            raise HTTPException(status_code=500, detail="策略创建成功但无法验证")
        
        logger.info(f"ORM查询到的策略信息: {new_strategy.__dict__}")
        
        # 解析参数
        params_dict = {}
        if new_strategy.parameters:
            try:
                params_dict = json.loads(new_strategy.parameters)
                logger.info(f"成功解析参数: {params_dict}")
            except Exception as e:
                logger.error(f"解析参数失败: {e}")
                params_dict = {}
        
        # 返回创建的策略数据
        result_data = {
            "id": new_strategy.id,
            "name": new_strategy.name,
            "description": new_strategy.description,
            "parameters": params_dict,
            "created_at": new_strategy.created_at.isoformat() if new_strategy.created_at else None,
            "updated_at": new_strategy.updated_at.isoformat() if new_strategy.updated_at else None,
            "is_template": new_strategy.is_template,
            "template": new_strategy.template
        }
        
        logger.info(f"返回给客户端的数据: {result_data}")
        
        # 记录操作结果
        if existing:
            logger.info(f"成功更新策略: {strategy.name} (ID: {strategy.id})")
            operation_message = "策略更新成功"
        else:
            logger.info(f"成功创建策略: {strategy.name} (ID: {strategy.id})")
            operation_message = "策略创建成功"
        
        # 验证操作是否成功
        return {
            "status": "success",
            "message": operation_message,
            "data": result_data
        }
    except HTTPException as he:
        # 直接重新抛出HTTP异常
        raise he
    except Exception as e:
        logger.error(f"处理策略创建请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建策略时发生错误: {str(e)}")

@app.get("/api/strategies/{strategy_id}")
async def get_strategy(strategy_id: int, db: Session = Depends(get_db)):
    try:
        logger.info(f"获取策略请求: ID={strategy_id}")
        
        # 导入策略模型（如果尚未导入）
        from ..models.strategy import Strategy as StrategyModel
        
        # 构建SQL查询
        sql_query = f"SELECT * FROM strategies WHERE id = {strategy_id}"
        logger.info(f"执行SQL查询: {sql_query}")
        
        strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
        if not strategy:
            logger.warning(f"未找到策略: ID={strategy_id}")
            raise HTTPException(status_code=404, detail=f"未找到ID为{strategy_id}的策略")
        
        logger.info(f"找到策略: {strategy.name} (ID: {strategy.id})")
        
        # 解析参数字符串为字典
        params_dict = {}
        if strategy.parameters:
            try:
                params_dict = json.loads(strategy.parameters)
                logger.info(f"成功解析参数: {params_dict}")
            except Exception as e:
                logger.error(f"解析参数失败: {e}")
                params_dict = {}
        
        # 构建返回数据
        result_data = {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "code": strategy.code,
            "parameters": params_dict,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
            "is_template": strategy.is_template,
            "template": strategy.template  # 添加模板字段
        }
        
        logger.info(f"返回策略数据: ID={strategy.id}, 名称={strategy.name}")
        
        return {
            "status": "success",
            "data": result_data
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"获取策略时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取策略时发生错误: {str(e)}")

@app.put("/api/strategies/{strategy_id}")
async def update_strategy(strategy_id: int, request: Request, db: Session = Depends(get_db)):
    """更新现有策略"""
    try:
        data = await request.json()
        logger.info(f"更新策略请求: ID={strategy_id}, 数据={data}")
        
        # 从数据库获取策略
        from ..models.strategy import Strategy as StrategyModel
        
        # 构建SQL查询
        sql_query = f"SELECT * FROM strategies WHERE id = {strategy_id}"
        logger.info(f"执行SQL查询: {sql_query}")
        
        db_strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
        
        if not db_strategy:
            logger.warning(f"未找到要更新的策略: ID={strategy_id}")
            raise HTTPException(status_code=404, detail=f"找不到ID为{strategy_id}的策略")
        
        logger.info(f"找到要更新的策略: {db_strategy.name} (ID: {db_strategy.id})")
        
        # 处理参数字段
        if "parameters" in data:
            parameters = data["parameters"]
            if parameters is not None:
                if isinstance(parameters, dict):
                    parameters = json.dumps(parameters)
                    logger.info(f"已将参数字典序列化为JSON字符串: {parameters}")
                elif isinstance(parameters, str):
                    # 验证是否为有效的JSON字符串
                    try:
                        json.loads(parameters)
                        logger.info("参数已经是有效的JSON字符串")
                    except Exception as e:
                        logger.error(f"提供的参数字符串不是有效的JSON: {e}")
                        raise HTTPException(status_code=400, detail="提供的参数不是有效的JSON格式")
                else:
                    logger.error(f"不支持的参数类型: {type(parameters)}")
                    raise HTTPException(status_code=400, detail="不支持的参数类型")
                
                data["parameters"] = parameters
        
        # 更新策略字段
        if "name" in data:
            db_strategy.name = data["name"]
        if "description" in data:
            db_strategy.description = data["description"]
        if "code" in data:
            db_strategy.code = data["code"]
        if "parameters" in data:
            db_strategy.parameters = data["parameters"]
        if "is_template" in data:
            db_strategy.is_template = data["is_template"]
        if "template" in data:
            db_strategy.template = data["template"]
        
        db_strategy.updated_at = datetime.now()
        
        # 记录更新状态
        logger.info(f"更新后的策略对象: {db_strategy.__dict__}")
        
        # 保存到数据库
        update_query = f"UPDATE strategies SET name='{db_strategy.name}', description='{db_strategy.description or ''}', template='{db_strategy.template or ''}', parameters='{db_strategy.parameters or ''}' WHERE id={strategy_id}"
        logger.info(f"执行更新SQL: {update_query}")
        
        db.commit()
        logger.info(f"策略更新成功: ID={db_strategy.id}, 名称={db_strategy.name}")
        
        # 解析参数字符串为字典
        params_dict = {}
        if db_strategy.parameters:
            try:
                params_dict = json.loads(db_strategy.parameters)
                logger.info(f"成功解析参数: {params_dict}")
            except Exception as e:
                logger.error(f"解析参数失败: {e}")
                params_dict = {}
        
        # 返回更新后的策略
        result_data = {
            "id": db_strategy.id,
            "name": db_strategy.name,
            "description": db_strategy.description,
            "code": db_strategy.code,
            "parameters": params_dict,
            "created_at": db_strategy.created_at.isoformat() if db_strategy.created_at else None,
            "updated_at": db_strategy.updated_at.isoformat() if db_strategy.updated_at else None,
            "is_template": db_strategy.is_template,
            "template": db_strategy.template
        }
        
        return {
            "status": "success",
            "message": "策略更新成功",
            "data": result_data
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"更新策略时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新策略时发生错误: {str(e)}")

@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(strategy_id: int, db: Session = Depends(get_db)):
    """删除策略"""
    try:
        # 从数据库获取策略
        from ..models.strategy import Strategy as StrategyModel
        db_strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
        
        if not db_strategy:
            raise HTTPException(status_code=404, detail=f"找不到ID为{strategy_id}的策略")
        
        # 删除策略
        db.delete(db_strategy)
        db.commit()
        
        logger.info(f"删除策略成功: {db_strategy.name}(ID: {strategy_id})")
        
        return {"status": "success", "message": "策略删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除策略失败: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/test")
async def test_strategy(
    data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """测试交易策略"""
    try:
        # 获取请求参数
        strategy_id = data.get("strategy_id")
        symbol = data.get("symbol")
        start_date = data.get("start_date")
        end_date = data.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        initial_capital = float(data.get("initial_capital", 100000))
        parameters = data.get("parameters", {})
        data_source = data.get("data_source", "database")  # 默认从数据库获取
        features = data.get("features", [])
        
        logger.info("=" * 80)
        logger.info(f"开始策略回测 - 股票: {symbol}, 策略: {strategy_id}")
        logger.info(f"回测参数: 日期范围={start_date}至{end_date}, 初始资金={initial_capital}, 数据源={data_source}")
        logger.info(f"策略参数: {parameters}")
        logger.info(f"特征列表: {features}")
        logger.info("-" * 80)
        
        # 参数检查
        if not strategy_id:
            raise ValueError("未提供策略ID")
        if not symbol:
            raise ValueError("未提供股票代码")
        if not start_date:
            raise ValueError("未提供开始日期")
        
        # 初始化回测服务
        from .backtest_service import BacktestService
        backtest_service = BacktestService(db)
        
        # 运行回测
        result = backtest_service.run_backtest(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            parameters=parameters,
            data_source=data_source,
            features=features
        )
        
        return result

    except ValueError as ve:
        logger.error(f"参数错误: {str(ve)}")
        return {
            "status": "error",
            "message": str(ve),
            "data": None
        }
    except Exception as e:
        logger.error(f"测试策略失败: {str(e)}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"测试策略失败: {str(e)}",
            "data": None
        }


# 兼容旧前端：某些前端版本调用 /api/strategies/backtest
# 将请求转发到现有的 test_strategy 实现，保持行为一致
@app.post("/api/strategies/backtest")
async def backtest_strategy_alias(
    data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """兼容接口：转发到 /api/strategies/test 的实现"""
    # 直接复用已有实现
    return await test_strategy(data, db)

# 导出报告API
@app.post("/api/backtest/report")
async def generate_report(
    data: Dict[str, Any]
):
    """生成回测报告"""
    try:
        # 获取请求参数
        backtest_results = data.get("backtest_results")
        
        if not backtest_results:
            raise ValueError("未提供回测结果")
            
        # 分析回测结果
        analyzer = PerformanceAnalyzer(backtest_results)
        report_path = analyzer.generate_report()
        
        return {"status": "success", "data": {"report_path": report_path}}
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/templates")
async def get_templates(db: Session = Depends(get_db)):
    """获取所有策略模板"""
    try:
        # 预定义策略模板
        predefined_strategies = [
            {
                "id": "ma_crossover",
                "name": "移动平均交叉策略",
                "description": "利用短期和长期移动平均线交叉产生买卖信号",
                "parameters": {
                    "short_window": {"type": "int", "default": 5, "min": 1, "max": 30, "description": "短期移动平均窗口"},
                    "long_window": {"type": "int", "default": 20, "min": 5, "max": 120, "description": "长期移动平均窗口"},
                },
                "code": """# 策略示例：移动平均线交叉策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param short_window: int = 20
    # @param long_window: int = 60
    context.params = {
        'symbol': '000300.SH',
        'short_window': 20,
        'long_window': 60
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算移动平均线
    df['short_ma'] = talib.SMA(df['close'], timeperiod=params['short_window'])
    df['long_ma'] = talib.SMA(df['close'], timeperiod=params['long_window'])
    
    # 生成交易信号
    df['signal'] = 0
    # 短均线上穿长均线，买入信号
    df.loc[(df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1)), 'signal'] = 1
    # 短均线下穿长均线，卖出信号
    df.loc[(df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1)), 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[df['signal'] == 1]:
        df.loc[idx, 'trigger_reason'] = f"短期均线({params['short_window']}日)上穿长期均线({params['long_window']}日)"
    
    for idx in df.index[df['signal'] == -1]:
        df.loc[idx, 'trigger_reason'] = f"短期均线({params['short_window']}日)下穿长期均线({params['long_window']}日)"
    
    return df['signal']
"""
            },
            {
                "id": "bollinger_bands",
                "name": "布林带策略",
                "description": "利用价格突破布林带上下轨产生买卖信号",
                "parameters": {
                    "window": {"type": "int", "default": 20, "min": 5, "max": 100, "description": "布林带窗口"},
                    "num_std": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "description": "标准差倍数"},
                },
                "code": """# 策略示例：布林带策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param window: int = 20
    # @param num_std: float = 2.0
    context.params = {
        'symbol': '000300.SH',
        'window': 20,
        'num_std': 2.0
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算布林带指标
    df['middle'] = df['close'].rolling(window=params['window'], min_periods=1).mean()
    df['std'] = df['close'].rolling(window=params['window'], min_periods=1).std()
    df['upper'] = df['middle'] + (df['std'] * params['num_std'])
    df['lower'] = df['middle'] - (df['std'] * params['num_std'])
    
    # 初始化信号列
    df['signal'] = 0
    
    # 价格由下方突破下轨，买入信号
    buy_signal = (df['close'] >= df['lower']) & (df['close'].shift(1) < df['lower'].shift(1))
    df.loc[buy_signal, 'signal'] = 1
    
    # 价格由上方突破上轨，卖出信号
    sell_signal = (df['close'] <= df['upper']) & (df['close'].shift(1) > df['upper'].shift(1))
    df.loc[sell_signal, 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[buy_signal]:
        df.loc[idx, 'trigger_reason'] = f"价格从下方突破布林带下轨（{params['window']}日，{params['num_std']}倍标准差）"
    
    for idx in df.index[sell_signal]:
        df.loc[idx, 'trigger_reason'] = f"价格从上方突破布林带上轨（{params['window']}日，{params['num_std']}倍标准差）"
    
    return df['signal']
"""
            },
            {
                "id": "macd",
                "name": "MACD策略",
                "description": "利用MACD指标的金叉和死叉产生买卖信号",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 5, "max": 50, "description": "快速EMA周期"},
                    "slow_period": {"type": "int", "default": 26, "min": 10, "max": 100, "description": "慢速EMA周期"},
                    "signal_period": {"type": "int", "default": 9, "min": 3, "max": 30, "description": "信号线周期"},
                },
                "code": """# 策略示例：MACD策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param fast_period: int = 12
    # @param slow_period: int = 26
    # @param signal_period: int = 9
    context.params = {
        'symbol': '000300.SH',
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算MACD指标
    macd, signal, hist = talib.MACD(
        df['close'], 
        fastperiod=params['fast_period'], 
        slowperiod=params['slow_period'], 
        signalperiod=params['signal_period']
    )
    
    df['macd'] = macd
    df['signal_line'] = signal
    df['hist'] = hist
    
    # 初始化信号列
    df['signal'] = 0
    
    # MACD金叉，买入信号
    buy_signal = (df['hist'] > 0) & (df['hist'].shift(1) <= 0)
    df.loc[buy_signal, 'signal'] = 1
    
    # MACD死叉，卖出信号
    sell_signal = (df['hist'] < 0) & (df['hist'].shift(1) >= 0)
    df.loc[sell_signal, 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[buy_signal]:
        df.loc[idx, 'trigger_reason'] = f"MACD金叉（快线{params['fast_period']}日，慢线{params['slow_period']}日，信号线{params['signal_period']}日）"
    
    for idx in df.index[sell_signal]:
        df.loc[idx, 'trigger_reason'] = f"MACD死叉（快线{params['fast_period']}日，慢线{params['slow_period']}日，信号线{params['signal_period']}日）"
    
    return df['signal']
"""
            },
            {
                "id": "rsi",
                "name": "RSI策略",
                "description": "利用相对强弱指数的超买超卖产生买卖信号",
                "parameters": {
                    "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "RSI周期"},
                    "overbought": {"type": "int", "default": 70, "min": 60, "max": 90, "description": "超买阈值"},
                    "oversold": {"type": "int", "default": 30, "min": 10, "max": 40, "description": "超卖阈值"},
                },
                "code": """# 策略示例：RSI策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param rsi_period: int = 14
    # @param overbought: int = 70
    # @param oversold: int = 30
    context.params = {
        'symbol': '000300.SH',
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算RSI指标
    df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
    
    # 初始化信号列
    df['signal'] = 0
    
    # RSI超卖后回升，买入信号
    buy_signal = (df['rsi'] > params['oversold']) & (df['rsi'].shift(1) <= params['oversold'])
    df.loc[buy_signal, 'signal'] = 1
    
    # RSI超买后回落，卖出信号
    sell_signal = (df['rsi'] < params['overbought']) & (df['rsi'].shift(1) >= params['overbought'])
    df.loc[sell_signal, 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[buy_signal]:
        df.loc[idx, 'trigger_reason'] = f"RSI({params['rsi_period']}日)从超卖区域({params['oversold']})回升，当前值: {df.loc[idx, 'rsi']:.2f}"
    
    for idx in df.index[sell_signal]:
        df.loc[idx, 'trigger_reason'] = f"RSI({params['rsi_period']}日)从超买区域({params['overbought']})回落，当前值: {df.loc[idx, 'rsi']:.2f}"
    
    return df['signal']
"""
            }
        ]
        
        # 如果有名称过滤条件，筛选模板
        return {
            "status": "success",
            "data": predefined_strategies
        }
    except Exception as e:
        logger.error(f"获取策略模板列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/templates/{template_id}")
async def get_template(template_id: str, db: Session = Depends(get_db)):
    """获取单个策略模板"""
    try:
        # 预定义策略模板
        predefined_strategies = {
            "ma_crossover": {
                "id": "ma_crossover",
                "name": "移动平均交叉策略",
                "description": "利用短期和长期移动平均线交叉产生买卖信号",
                "parameters": {
                    "short_window": {"type": "int", "default": 5, "min": 1, "max": 30, "description": "短期移动平均窗口"},
                    "long_window": {"type": "int", "default": 20, "min": 5, "max": 120, "description": "长期移动平均窗口"},
                },
                "code": """# 策略示例：移动平均线交叉策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param short_window: int = 20
    # @param long_window: int = 60
    context.params = {
        'symbol': '000300.SH',
        'short_window': 20,
        'long_window': 60
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算移动平均线
    df['short_ma'] = talib.SMA(df['close'], timeperiod=params['short_window'])
    df['long_ma'] = talib.SMA(df['close'], timeperiod=params['long_window'])
    
    # 生成交易信号
    df['signal'] = 0
    # 短均线上穿长均线，买入信号
    df.loc[(df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1)), 'signal'] = 1
    # 短均线下穿长均线，卖出信号
    df.loc[(df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1)), 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[df['signal'] == 1]:
        df.loc[idx, 'trigger_reason'] = f"短期均线({params['short_window']}日)上穿长期均线({params['long_window']}日)"
    
    for idx in df.index[df['signal'] == -1]:
        df.loc[idx, 'trigger_reason'] = f"短期均线({params['short_window']}日)下穿长期均线({params['long_window']}日)"
    
    return df['signal']
"""
            },
            "bollinger_bands": {
                "id": "bollinger_bands",
                "name": "布林带策略",
                "description": "利用价格突破布林带上下轨产生买卖信号",
                "parameters": {
                    "window": {"type": "int", "default": 20, "min": 5, "max": 100, "description": "布林带窗口"},
                    "num_std": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "description": "标准差倍数"},
                },
                "code": """# 策略示例：布林带策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param window: int = 20
    # @param num_std: float = 2.0
    context.params = {
        'symbol': '000300.SH',
        'window': 20,
        'num_std': 2.0
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算布林带指标
    df['middle'] = df['close'].rolling(window=params['window'], min_periods=1).mean()
    df['std'] = df['close'].rolling(window=params['window'], min_periods=1).std()
    df['upper'] = df['middle'] + (df['std'] * params['num_std'])
    df['lower'] = df['middle'] - (df['std'] * params['num_std'])
    
    # 初始化信号列
    df['signal'] = 0
    
    # 价格由下方突破下轨，买入信号
    buy_signal = (df['close'] >= df['lower']) & (df['close'].shift(1) < df['lower'].shift(1))
    df.loc[buy_signal, 'signal'] = 1
    
    # 价格由上方突破上轨，卖出信号
    sell_signal = (df['close'] <= df['upper']) & (df['close'].shift(1) > df['upper'].shift(1))
    df.loc[sell_signal, 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[buy_signal]:
        df.loc[idx, 'trigger_reason'] = f"价格从下方突破布林带下轨（{params['window']}日，{params['num_std']}倍标准差）"
    
    for idx in df.index[sell_signal]:
        df.loc[idx, 'trigger_reason'] = f"价格从上方突破布林带上轨（{params['window']}日，{params['num_std']}倍标准差）"
    
    return df['signal']
"""
            },
            "macd": {
                "id": "macd",
                "name": "MACD策略",
                "description": "利用MACD指标的金叉和死叉产生买卖信号",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 5, "max": 50, "description": "快速EMA周期"},
                    "slow_period": {"type": "int", "default": 26, "min": 10, "max": 100, "description": "慢速EMA周期"},
                    "signal_period": {"type": "int", "default": 9, "min": 3, "max": 30, "description": "信号线周期"},
                },
                "code": """# 策略示例：MACD策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param fast_period: int = 12
    # @param slow_period: int = 26
    # @param signal_period: int = 9
    context.params = {
        'symbol': '000300.SH',
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算MACD指标
    macd, signal, hist = talib.MACD(
        df['close'], 
        fastperiod=params['fast_period'], 
        slowperiod=params['slow_period'], 
        signalperiod=params['signal_period']
    )
    
    df['macd'] = macd
    df['signal_line'] = signal
    df['hist'] = hist
    
    # 初始化信号列
    df['signal'] = 0
    
    # MACD金叉，买入信号
    buy_signal = (df['hist'] > 0) & (df['hist'].shift(1) <= 0)
    df.loc[buy_signal, 'signal'] = 1
    
    # MACD死叉，卖出信号
    sell_signal = (df['hist'] < 0) & (df['hist'].shift(1) >= 0)
    df.loc[sell_signal, 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[buy_signal]:
        df.loc[idx, 'trigger_reason'] = f"MACD金叉（快线{params['fast_period']}日，慢线{params['slow_period']}日，信号线{params['signal_period']}日）"
    
    for idx in df.index[sell_signal]:
        df.loc[idx, 'trigger_reason'] = f"MACD死叉（快线{params['fast_period']}日，慢线{params['slow_period']}日，信号线{params['signal_period']}日）"
    
    return df['signal']
"""
            },
            "rsi": {
                "id": "rsi",
                "name": "RSI策略",
                "description": "利用相对强弱指数的超买超卖产生买卖信号",
                "parameters": {
                    "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "RSI周期"},
                    "overbought": {"type": "int", "default": 70, "min": 60, "max": 90, "description": "超买阈值"},
                    "oversold": {"type": "int", "default": 30, "min": 10, "max": 40, "description": "超卖阈值"},
                },
                "code": """# 策略示例：RSI策略
import pandas as pd
import numpy as np
import talib

def initialize(context):
    '''初始化策略参数'''
    # @param symbol: str = '000300.SH'
    # @param rsi_period: int = 14
    # @param overbought: int = 70
    # @param oversold: int = 30
    context.params = {
        'symbol': '000300.SH',
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30
    }

def handle_data(context, data):
    '''处理每个交易日的数据'''
    params = context.params
    df = data[params['symbol']]
    
    # 计算RSI指标
    df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
    
    # 初始化信号列
    df['signal'] = 0
    
    # RSI超卖后回升，买入信号
    buy_signal = (df['rsi'] > params['oversold']) & (df['rsi'].shift(1) <= params['oversold'])
    df.loc[buy_signal, 'signal'] = 1
    
    # RSI超买后回落，卖出信号
    sell_signal = (df['rsi'] < params['overbought']) & (df['rsi'].shift(1) >= params['overbought'])
    df.loc[sell_signal, 'signal'] = -1
    
    # 记录信号触发原因
    for idx in df.index[buy_signal]:
        df.loc[idx, 'trigger_reason'] = f"RSI({params['rsi_period']}日)从超卖区域({params['oversold']})回升，当前值: {df.loc[idx, 'rsi']:.2f}"
    
    for idx in df.index[sell_signal]:
        df.loc[idx, 'trigger_reason'] = f"RSI({params['rsi_period']}日)从超买区域({params['overbought']})回落，当前值: {df.loc[idx, 'rsi']:.2f}"
    
    return df['signal']
"""
            }
        }
        
        # 获取模板详情
        template = predefined_strategies.get(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"模板ID {template_id} 不存在")
            
        return {
            "status": "success",
            "data": template
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略模板 {template_id} 失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略模板失败: {str(e)}")
