from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import sys
import os
import importlib.util
import traceback
import inspect

# 导入数据库模型
from ..models import get_db
from ..models.strategy import Strategy as StrategyModel
from ..utils.strategy_validator import StrategyValidator

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_strategies(
    name: Optional[str] = None, 
    include_templates: bool = True, 
    db: Session = Depends(get_db)
):
    """获取所有策略列表或按名称搜索"""
    try:
        logger.info(f"获取策略列表请求: 名称过滤={name}")
        
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
        
        return {"status": "success", "data": result_data}
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}")
async def get_strategy(strategy_id: int, db: Session = Depends(get_db)):
    """获取策略详情"""
    try:
        logger.info(f"获取策略详情请求: ID={strategy_id}")
        
        # 查询策略
        strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
        if not strategy:
            logger.warning(f"未找到策略: ID={strategy_id}")
            raise HTTPException(status_code=404, detail=f"未找到ID为{strategy_id}的策略")
        
        # 解析参数
        params_dict = {}
        if strategy.parameters:
            try:
                params_dict = json.loads(strategy.parameters)
            except Exception as e:
                logger.error(f"解析参数失败: {e}")
        
        # 构建响应数据
        result_data = {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "code": strategy.code,  # 返回策略代码
            "parameters": params_dict,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
            "is_template": strategy.is_template,
            "template": strategy.template
        }
        
        return {"status": "success", "data": result_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def create_strategy(request: Request, db: Session = Depends(get_db)):
    """创建新策略"""
    try:
        data = await request.json()
        logger.info(f"创建策略请求: {data}")
        
        # 校验必要字段
        name = data.get("name")
        code = data.get("code")
        
        if not name:
            raise HTTPException(status_code=400, detail="缺少必要字段: name")
        if not code:
            raise HTTPException(status_code=400, detail="缺少必要字段: code")
        
        # 验证策略代码
        is_valid, errors = StrategyValidator.validate_strategy_code(code)
        if not is_valid:
            error_message = "策略代码验证失败:\n" + "\n".join(errors)
            logger.warning(f"策略代码验证失败: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # 处理参数字段
        parameters = data.get("parameters")
        parameters_json = None
        
        if parameters is not None:
            if isinstance(parameters, dict):
                parameters_json = json.dumps(parameters)
            elif isinstance(parameters, str):
                try:
                    # 验证是有效的JSON字符串
                    json.loads(parameters)
                    parameters_json = parameters
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"参数不是有效的JSON格式: {e}")
            else:
                raise HTTPException(status_code=400, detail=f"不支持的参数类型: {type(parameters)}")
        
        # 检查是否存在同名策略
        existing = db.query(StrategyModel).filter(StrategyModel.name == name).first()
        
        if existing:
            # 更新已有策略
            existing.description = data.get("description", existing.description)
            existing.code = code
            existing.parameters = parameters_json
            existing.template = data.get("template", existing.template)
            existing.is_template = data.get("is_template", existing.is_template)
            existing.updated_at = datetime.now()
            
            strategy = existing
            db.flush()
            db.commit()
            
            logger.info(f"更新策略成功: {strategy.name} (ID: {strategy.id})")
            message = "策略更新成功"
        else:
            # 创建新策略
            strategy = StrategyModel(
                name=name,
                description=data.get("description"),
                code=code,
                parameters=parameters_json,
                template=data.get("template"),
                is_template=data.get("is_template", False),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            db.add(strategy)
            db.flush()
            db.commit()
            
            logger.info(f"创建策略成功: {strategy.name} (ID: {strategy.id})")
            message = "策略创建成功"
        
        # 解析参数为字典
        params_dict = {}
        if strategy.parameters:
            try:
                params_dict = json.loads(strategy.parameters)
            except Exception:
                params_dict = {}
        
        # 返回结果
        result_data = {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "parameters": params_dict,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
            "is_template": strategy.is_template,
            "template": strategy.template
        }
        
        return {
            "status": "success",
            "message": message,
            "data": result_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建策略失败: {e}")
        # 回滚数据库
        db.rollback()
        raise HTTPException(status_code=500, detail=f"创建策略失败: {str(e)}")

@router.put("/{strategy_id}")
async def update_strategy(strategy_id: int, request: Request, db: Session = Depends(get_db)):
    """更新策略"""
    try:
        data = await request.json()
        logger.info(f"更新策略请求: ID={strategy_id}, 数据={data}")
        
        # 查询策略
        strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
        if not strategy:
            raise HTTPException(status_code=404, detail=f"未找到ID为{strategy_id}的策略")
        
        # 如果提供了代码，验证代码有效性
        code = data.get("code")
        if code is not None:
            is_valid, errors = StrategyValidator.validate_strategy_code(code)
            if not is_valid:
                error_message = "策略代码验证失败:\n" + "\n".join(errors)
                logger.warning(f"策略代码验证失败: {error_message}")
                raise HTTPException(status_code=400, detail=error_message)
            
            # 更新代码
            strategy.code = code
        
        # 处理参数字段
        parameters = data.get("parameters")
        if parameters is not None:
            if isinstance(parameters, dict):
                strategy.parameters = json.dumps(parameters)
            elif isinstance(parameters, str):
                try:
                    # 验证是有效的JSON字符串
                    json.loads(parameters)
                    strategy.parameters = parameters
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"参数不是有效的JSON格式: {e}")
            else:
                raise HTTPException(status_code=400, detail=f"不支持的参数类型: {type(parameters)}")
        
        # 更新其他字段
        if "name" in data:
            strategy.name = data["name"]
        if "description" in data:
            strategy.description = data["description"]
        if "template" in data:
            strategy.template = data["template"]
        if "is_template" in data:
            strategy.is_template = data["is_template"]
        
        # 更新时间
        strategy.updated_at = datetime.now()
        
        # 提交更新
        db.commit()
        
        # 解析参数
        params_dict = {}
        if strategy.parameters:
            try:
                params_dict = json.loads(strategy.parameters)
            except Exception:
                params_dict = {}
        
        # 返回更新后的策略
        result_data = {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "parameters": params_dict,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
            "is_template": strategy.is_template,
            "template": strategy.template
        }
        
        logger.info(f"更新策略成功: {strategy.name} (ID: {strategy.id})")
        
        return {
            "status": "success",
            "message": "策略更新成功",
            "data": result_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新策略失败: {e}")
        # 回滚数据库
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新策略失败: {str(e)}")

@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: int, db: Session = Depends(get_db)):
    """删除策略"""
    try:
        logger.info(f"删除策略请求: ID={strategy_id}")
        
        # 查询策略
        strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
        if not strategy:
            raise HTTPException(status_code=404, detail=f"未找到ID为{strategy_id}的策略")
        
        # 保存策略名称用于日志
        strategy_name = strategy.name
        
        # 删除策略
        db.delete(strategy)
        db.commit()
        
        logger.info(f"删除策略成功: {strategy_name} (ID: {strategy_id})")
        
        return {
            "status": "success",
            "message": f"策略'{strategy_name}'已成功删除"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除策略失败: {e}")
        # 回滚数据库
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除策略失败: {str(e)}")

@router.post("/test")
async def test_strategy(request: Request, db: Session = Depends(get_db)):
    """测试策略代码"""
    try:
        data = await request.json()
        code = data.get("code")
        test_data = data.get("data")
        parameters = data.get("parameters", {})
        strategy_id = data.get("strategy_id")
        
        logger.info(f"收到策略测试请求: strategy_id={strategy_id}, 参数={parameters}")
        
        # 如果提供了strategy_id，则从数据库获取策略代码
        if strategy_id and not code:
            logger.info(f"从数据库获取策略代码，策略ID: {strategy_id}")
            strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
            if not strategy:
                error_msg = f"未找到ID为{strategy_id}的策略"
                logger.error(error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            code = strategy.code
            logger.info(f"成功获取策略代码，策略名称: {strategy.name}")
            
            # 如果没有提供参数，使用策略默认参数
            if not parameters and strategy.parameters:
                try:
                    parameters = json.loads(strategy.parameters)
                    logger.info(f"使用策略默认参数: {parameters}")
                except Exception as e:
                    logger.error(f"解析策略参数失败: {e}")
        
        if not code:
            error_msg = "缺少必要字段: code或strategy_id"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 验证策略代码
        logger.info("开始验证策略代码...")
        is_valid, errors = StrategyValidator.validate_strategy_code(code)
        if not is_valid:
            error_message = "策略代码验证失败:\n" + "\n".join(errors)
            logger.error(f"策略代码验证失败: {error_message}")
            return {
                "status": "error",
                "message": error_message,
                "data": {
                    "is_valid": False,
                    "errors": errors
                }
            }
        
        logger.info("策略代码验证通过")
        
        # 如果提供了测试数据，进行简单回测
        if test_data:
            try:
                logger.info("开始加载策略实例...")
                # 导入与实例化策略
                strategy_instance = load_strategy_from_code(code, test_data, parameters)
                logger.info(f"策略实例加载成功: {type(strategy_instance).__name__}")
                
                # 准备数据
                if isinstance(test_data, list):
                    df = pd.DataFrame(test_data)
                else:
                    df = pd.DataFrame()
                
                if not df.empty:
                    logger.info(f"测试数据加载成功，数据量: {len(df)}行")
                    # 设置数据
                    strategy_instance.set_data(df)
                    
                    # 生成信号
                    logger.info("开始生成交易信号...")
                    signals = strategy_instance.generate_signals()
                    
                    # 统计信号
                    buy_count = (signals['signal'] == 1).sum()
                    sell_count = (signals['signal'] == -1).sum()
                    logger.info(f"信号生成完成: 买入信号 {buy_count}个, 卖出信号 {sell_count}个")
                    
                    result = {
                        "signals": signals.to_dict(orient='records'),
                        "statistics": {
                            "total_records": len(signals),
                            "buy_signals": int(buy_count),
                            "sell_signals": int(sell_count)
                        }
                    }
                else:
                    logger.error("测试数据为空")
                    result = {"error": "测试数据为空"}
            except Exception as e:
                logger.error(f"策略测试失败: {e}")
                traceback.print_exc()
                result = {"error": f"策略测试失败: {str(e)}"}
        else:
            # 仅验证代码是否有效
            result = {"is_valid": True}
        
        logger.info("策略测试完成")
        return {
            "status": "success",
            "message": "策略代码验证通过",
            "data": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"测试策略失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"测试策略失败: {str(e)}")

def load_strategy_from_code(code: str, parameters: Dict[str, Any] = None, data: pd.DataFrame = None, globals_dict: Dict[str, Any] = None):
    """
    从代码字符串加载策略类并实例化
    
    Args:
        code: 策略代码字符串
        parameters: 策略参数
        data: 策略数据
        globals_dict: 用于执行代码的全局命名空间字典
        
    Returns:
        策略实例
    """
    temp_module_name = f"temp_strategy_module_{hash(code) % 10000}"
    
    # 记录参数信息，用于调试
    if parameters:
        logger.info(f"加载策略时传入的参数: {parameters}")
    
    try:
        # 预处理代码，修复可能存在的导入问题
        code = preprocess_strategy_code(code)
        
        # 创建临时模块
        logger.debug(f"创建临时模块: {temp_module_name}")
        spec = importlib.util.spec_from_loader(temp_module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[temp_module_name] = module
        
        # 准备执行环境
        if globals_dict is None:
            globals_dict = module.__dict__
        else:
            # 合并提供的globals_dict和模块的__dict__
            for key, value in globals_dict.items():
                module.__dict__[key] = value
        
        # 添加必要的导入
        module.__dict__['pd'] = __import__('pandas')
        module.__dict__['np'] = __import__('numpy')
        
        # 确保可以访问到StrategyTemplate
        try:
            import src.backend.strategy.templates.strategy_template
            module.__dict__['StrategyTemplate'] = src.backend.strategy.templates.strategy_template.StrategyTemplate
        except ImportError:
            logger.warning("无法导入StrategyTemplate，尝试其他方式")
            pass
        
        # 执行代码
        exec(code, module.__dict__)
        
        # 查找策略类
        strategy_class = None
        
        # 导入所有可能的基类
        base_classes = []
        
        # 尝试导入StrategyTemplate
        if 'StrategyTemplate' not in module.__dict__:
            try:
                from ..strategy.templates.strategy_template import StrategyTemplate
                module.__dict__['StrategyTemplate'] = StrategyTemplate
                base_classes.append(StrategyTemplate)
            except ImportError:
                try:
                    # 备用导入方式
                    import src.backend.strategy.templates.strategy_template
                    StrategyTemplate = src.backend.strategy.templates.strategy_template.StrategyTemplate
                    module.__dict__['StrategyTemplate'] = StrategyTemplate
                    base_classes.append(StrategyTemplate)
                except ImportError:
                    logger.warning("无法导入StrategyTemplate")
        else:
            base_classes.append(module.__dict__['StrategyTemplate'])
        
        # 尝试导入StrategyBase
        if 'StrategyBase' not in module.__dict__:
            try:
                from ..strategy.base.strategy_base import StrategyBase
                module.__dict__['StrategyBase'] = StrategyBase
                base_classes.append(StrategyBase)
            except ImportError:
                try:
                    # 备用导入方式
                    import src.backend.strategy.base.strategy_base
                    StrategyBase = src.backend.strategy.base.strategy_base.StrategyBase
                    module.__dict__['StrategyBase'] = StrategyBase
                    base_classes.append(StrategyBase)
                except ImportError:
                    logger.warning("无法导入StrategyBase")
        else:
            base_classes.append(module.__dict__['StrategyBase'])
        
        # 查找继承自任一基类的策略类
        for name, obj in module.__dict__.items():
            if isinstance(obj, type):
                # 检查是否继承自任何一个基类
                for base_class in base_classes:
                    if obj is not base_class and issubclass(obj, base_class):
                        strategy_class = obj
                        logger.debug(f"找到策略类: {name}, 继承自: {base_class.__name__}")
                        break
                if strategy_class:
                    break
        
        if strategy_class is None:
            error_msg = "未找到策略类"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 安全实例化辅助：兼容不同 __init__ 签名，避免 'unexpected keyword' 错误
        def _safe_instantiate(cls, name, data, parameters):
            sig = None
            try:
                sig = inspect.signature(cls.__init__)
            except Exception:
                pass
        
            def _apply_aliases(kwargs):
                if sig is None:
                    return kwargs
                param_names = set(sig.parameters.keys())
                # parameters 别名
                if 'parameters' in kwargs and 'parameters' not in param_names:
                    for alias in ['params', 'config']:
                        if alias in param_names:
                            kwargs[alias] = kwargs.pop('parameters')
                            break
                # data 别名
                if 'data' in kwargs and 'data' not in param_names:
                    for alias in ['df', 'dataset']:
                        if alias in param_names:
                            kwargs[alias] = kwargs.pop('data')
                            break
                # name 别名（尽量保持 name）
                if 'name' in kwargs and 'name' not in param_names:
                    for alias in ['title', 'strategy_name']:
                        if alias in param_names:
                            kwargs[alias] = kwargs.pop('name')
                            break
                return kwargs
        
            def _try(kwargs):
                usable = _apply_aliases(dict(kwargs))
                # 无 **kwargs 时过滤无效键
                if sig is not None:
                    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if not has_var_kw:
                        usable = {k: v for k, v in usable.items() if k in sig.parameters}
                try:
                    inst = cls(**usable)
                    logger.debug(f"策略实例化成功，使用参数组合: {list(usable.keys())}")
                    return inst, set(usable.keys())
                except TypeError as e:
                    logger.debug(f"策略实例化失败组合 {list(usable.keys())}: {e}")
                    return None, set()
        
            # 构造优先级队列：先完整、再降级；如提供了 data/parameters，禁止空 {}
            provided_data = data is not None
            provided_params = parameters is not None
            candidates = []
        
            # 优先级排序
            if provided_data and provided_params:
                candidates = [
                    {"name": name, "data": data, "parameters": parameters},
                    {"data": data, "parameters": parameters},
                    {"name": name, "data": data},
                    {"name": name, "parameters": parameters},
                    {"data": data},
                    {"parameters": parameters},
                ]
            elif provided_data:
                candidates = [
                    {"name": name, "data": data},
                    {"data": data},
                    {"name": name},
                ]
            elif provided_params:
                candidates = [
                    {"name": name, "parameters": parameters},
                    {"parameters": parameters},
                    {"name": name},
                ]
            else:
                candidates = [
                    {"name": name},
                    {},
                ]
        
            # 依次尝试
            for kwargs in candidates:
                inst, used_keys = _try(kwargs)
                if inst is not None:
                    # 严格校验：若提供了关键参数但未被使用（别名映射后仍不在 used_keys），则继续尝试下一组合
                    if provided_data and not any(k in used_keys for k in ['data', 'df', 'dataset']):
                        logger.debug("实例化未包含 data，继续尝试下一组合")
                        continue
                    if provided_params and not any(k in used_keys for k in ['parameters', 'params', 'config']):
                        logger.debug("实例化未包含 parameters，继续尝试下一组合")
                        continue
                    return inst
        
            raise TypeError("策略类无法在兼容模式下实例化，请检查构造函数签名与参数名")
        
        # 首先实例化一个默认策略，用于提取默认参数规范（使用兼容模式，不强制传 name/data/parameters）
        logger.debug(f"实例化默认策略以提取参数规范: {strategy_class.__name__}")
        default_instance = strategy_class(name="动态策略", data=None, parameters=None)

        #default_instance = _safe_instantiate(strategy_class, name_value="动态策略", data_value=None, parameters_value=None)

        default_params = {}
        # 优先使用 get_strategy_info().parameters
        if hasattr(default_instance, 'get_strategy_info'):
            try:
                info = default_instance.get_strategy_info() or {}
                default_params = info.get('parameters') or {}
            except Exception:
                default_params = {}
        if not default_params:
            default_params = getattr(default_instance, 'parameters', {}) or {}
        # 进一步回退：类级/模块级默认参数约定名
        # if not default_params:
        #     for attr_name in [
        #         'DEFAULT_PARAMETERS', 'DEFAULT_PARAMS', 'PARAMETERS',
        #         'default_parameters', 'default_params', 'parameters'
        #     ]:
        #         dp = getattr(strategy_class, attr_name, None)
        #         if isinstance(dp, dict) and dp:
        #             default_params = dp
        #             break
        # if not default_params:
        #     for attr_name in [
        #         'DEFAULT_PARAMETERS', 'DEFAULT_PARAMS', 'PARAMETERS',
        #         'default_parameters', 'default_params', 'parameters'
        #     ]:
        #         dp = module.__dict__.get(attr_name)
        #         if isinstance(dp, dict) and dp:
        #             default_params = dp
        #             break

        # 进行参数一致性校验与合并
        def _type_name(v):
            if isinstance(v, bool):
                return 'boolean'
            if isinstance(v, int):
                return 'integer'
            if isinstance(v, float):
                return 'float'
            if isinstance(v, str):
                return 'string'
            if isinstance(v, list):
                return 'list'
            if isinstance(v, dict):
                return 'dict'
            return type(v).__name__

        def _is_compatible(default_v, given_v):
            # float 默认允许 int 作为兼容
            if isinstance(default_v, float) and isinstance(given_v, (int, float)):
                return True
            # 其他类型必须严格匹配（bool、int、str、list、dict）
            return isinstance(given_v, type(default_v))

        params_to_use = dict(default_params)  # 从默认开始
        if parameters:
            expected_keys = set(default_params.keys())
            given_keys = set(parameters.keys())
            unknown = given_keys - expected_keys
            if unknown:
                raise ValueError(f"传入参数包含未定义的键: {sorted(list(unknown))}")

            # 类型校验
            type_errors = []
            for k, v in parameters.items():
                if k in default_params:
                    dv = default_params[k]
                    # 如果默认值为 None，跳过类型校验
                    if dv is not None and not _is_compatible(dv, v):
                        type_errors.append(f"参数'{k}'类型不一致: 期望{_type_name(dv)}, 实际{_type_name(v)}")
            if type_errors:
                raise ValueError("; ".join(type_errors))

            # 合并到默认参数
            params_to_use.update(parameters)

        # 使用校验/合并后的参数实例化策略
        logger.debug(f"实例化策略类并应用参数: {strategy_class.__name__}, 参数: {params_to_use}")
        strategy_instance = strategy_class(name="动态策略", data=data, parameters=params_to_use)

        # strategy_instance = _safe_instantiate(strategy_class, name_value="动态策略", data_value=data, parameters_value=params_to_use)
        return strategy_instance
    
    finally:
        # 清理临时模块
        if temp_module_name in sys.modules:
            del sys.modules[temp_module_name]

def preprocess_strategy_code(code: str) -> str:
    """
    预处理策略代码，修复常见问题
    
    Args:
        code: 原始策略代码
        
    Returns:
        处理后的策略代码
    """
    # 确保 code 为字符串（有时从数据库读取为 bytes）
    if code is None:
        return ""
    if isinstance(code, (bytes, bytearray)):
        try:
            code = code.decode('utf-8')
        except Exception:
            code = code.decode('latin-1')

    # 替换相对导入为绝对导入
    import re
    
    # 记录原始代码，用于调试
    logger.debug(f"原始策略代码:\n{code[:200]}...")
    
    # 最关键的替换：确保.strategy_template的导入使用正确的路径
    if "from .strategy_template import" in code:
        logger.debug("发现策略模板导入语句，执行关键替换")
        code = code.replace(
            "from .strategy_template import",
            "from src.backend.strategy.templates.strategy_template import"
        )
    
    # 修复对strategy_template模块的导入
    if "from src.backend.strategy.strategy_template import" in code:
        code = code.replace(
            "from src.backend.strategy.strategy_template import",
            "from src.backend.strategy.templates.strategy_template import"
        )
    
    # 检测并替换from .templates 或 from ..strategy 等相对导入
    code = re.sub(
        r'from\s+\.templates\s+import', 
        r'from src.backend.strategy.templates import', 
        code
    )
    
    # 特殊情况处理
    code = re.sub(
        r'from\s+\.(templates\.strategy_template)\s+import', 
        r'from src.backend.strategy.\1 import', 
        code
    )
    
    # 一般相对导入替换
    code = re.sub(
        r'from\s+\.([a-zA-Z0-9_]+)\s+import', 
        r'from src.backend.strategy.\1 import', 
        code
    )
    
    code = re.sub(
        r'from\s+\.\.([a-zA-Z0-9_]+)\s+import', 
        r'from src.backend.\1 import', 
        code
    )
    
    # 特别处理常见导入
    if "from .templates.strategy_template import StrategyTemplate" in code:
        code = code.replace(
            "from .templates.strategy_template import StrategyTemplate",
            "from src.backend.strategy.templates.strategy_template import StrategyTemplate"
        )
    
    if "from ..strategy.templates.strategy_template import StrategyTemplate" in code:
        code = code.replace(
            "from ..strategy.templates.strategy_template import StrategyTemplate",
            "from src.backend.strategy.templates.strategy_template import StrategyTemplate"
        )
    
    # 查找并输出替换后的导入语句，用于调试
    import_lines = [line for line in code.split('\n') if line.strip().startswith('from') or line.strip().startswith('import')]
    if import_lines:
        logger.debug(f"处理后的导入语句:\n" + "\n".join(import_lines[:5]))
    
    # 最后的保险：如果仍然有".strategy_template"的导入，直接修复
    line_fixed = False
    fixed_lines = []
    for line in code.split('\n'):
        if '.strategy_template' in line and 'import' in line and not line_fixed:
            fixed_line = "from src.backend.strategy.templates.strategy_template import StrategyTemplate"
            fixed_lines.append(fixed_line)
            logger.debug(f"强制修复导入行: {line} -> {fixed_line}")
            line_fixed = True
        else:
            fixed_lines.append(line)
    
    if line_fixed:
        code = '\n'.join(fixed_lines)
    
    # 如果代码第一行包含相对导入，则添加绝对导入
    lines = code.strip().split('\n')
    if lines and (lines[0].startswith("from .") or lines[0].startswith("from ..")):
        # 添加绝对导入策略模板
        code = "from src.backend.strategy.templates.strategy_template import StrategyTemplate\n" + code
        
    return code

@router.post("/backtest")
async def backtest_strategy(request: Request, db: Session = Depends(get_db)):
    """对策略进行历史数据回测"""
    try:
        data = await request.json()
        
        # 获取请求参数
        strategy_id = data.get("strategy_id")
        symbol = data.get("symbol")
        start_date = data.get("start_date")
        end_date = data.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        initial_capital = float(data.get("initial_capital", 100000))
        commission_rate = float(data.get("commission_rate", 0.0015))
        slippage_rate = float(data.get("slippage_rate", 0.001))
        parameters = data.get("parameters", {})
        data_source = data.get("data_source", "database")  # 默认从数据库获取
        features = data.get("features", [])
        force_refresh = data.get("force_refresh", False)  # 新增强制刷新参数
        
        logger.info("=" * 80)
        logger.info(f"开始策略回测 - 股票: {symbol}, 策略: {strategy_id}")
        logger.info(f"回测参数: 日期范围={start_date}至{end_date}, 初始资金={initial_capital}")
        logger.info(f"交易成本: 手续费率={commission_rate}, 滑点率={slippage_rate}")
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

        # 在运行回测前进行策略代码验证与可加载性检查
        try:
            strategy = db.query(StrategyModel).filter(StrategyModel.id == strategy_id).first()
            if not strategy:
                raise ValueError(f"未找到ID为{strategy_id}的策略")
            raw_code = strategy.code
            if isinstance(raw_code, (bytes, bytearray)):
                try:
                    raw_code = raw_code.decode('utf-8')
                except Exception:
                    raw_code = raw_code.decode('latin-1')

            logger.info("开始在回测前验证策略代码…")
            is_valid, errors = StrategyValidator.validate_strategy_code(raw_code)
            if not is_valid:
                error_message = "策略代码验证失败:\n" + "\n".join(errors)
                logger.error(error_message)
                return {
                    "status": "error",
                    "message": error_message,
                    "data": {
                        "is_valid": False,
                        "errors": errors
                    }
                }

            # 进一步尝试加载，捕获相对导入等运行期问题
            try:
                load_ok, load_errors = StrategyValidator.test_load_strategy(raw_code)
                if not load_ok:
                    error_message = "策略代码可加载性检查失败:\n" + "\n".join(load_errors)
                    logger.error(error_message)
                    return {
                        "status": "error",
                        "message": error_message,
                        "data": {
                            "is_valid": False,
                            "errors": load_errors
                        }
                    }
            except Exception as e:
                logger.error(f"策略可加载性检查异常: {e}")
                return {
                    "status": "error",
                    "message": f"策略代码可加载性检查异常: {str(e)}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"获取或验证策略代码失败: {e}")
            return {
                "status": "error",
                "message": f"获取或验证策略代码失败: {str(e)}",
                "data": None
            }

        # 初始化回测服务
        from ..api.backtest_service import BacktestService
        backtest_service = BacktestService(db)
        
        # 运行回测
        result = backtest_service.run_backtest(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            parameters=parameters,
            data_source=data_source,
            features=features,
            force_refresh=force_refresh
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
        logger.error(f"回测策略失败: {str(e)}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"回测策略失败: {str(e)}",
            "data": None
        }