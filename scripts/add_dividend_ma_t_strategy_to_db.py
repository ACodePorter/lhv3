#!/usr/bin/env python3
"""
将“高股息MA偏离小T策略”添加到策略表，并配置默认参数与参数空间。
"""

import os
import sys
import json
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 确保可以以 `src.backend...` 形式导入
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, 'backtesting.db')

# 模型导入
StrategyModel = __import__('src.backend.models.strategy', fromlist=['Strategy']).Strategy
StrategyParameterSpace = __import__('src.backend.models.optimization', fromlist=['StrategyParameterSpace']).StrategyParameterSpace

def add_strategy_to_db():
    # 数据库连接
    engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f'连接到数据库: {DB_PATH}')

    # 读取策略代码
    strategy_file = os.path.join(PROJECT_ROOT, 'src', 'backend', 'strategy', 'dividend_ma_t_strategy.py')
    with open(strategy_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # 默认参数
    default_parameters = json.dumps({
        "long_ma": 120,
        "dev_threshold_pct": 0.08,
        "baseline_long_position": 0.6,
        "t_chunk_count": 5,
        "total_max_position": 1.0,
    }, ensure_ascii=False)

    # 查找是否存在该策略（按 template / name）
    target = session.query(StrategyModel).filter(StrategyModel.template == 'dividend_ma_t').first()
    if not target:
        target = session.query(StrategyModel).filter(StrategyModel.name == '高股息MA偏离小T策略').first()

    if not target:
        print('创建新策略: 高股息MA偏离小T策略')
        s = StrategyModel(
            name='高股息MA偏离小T策略',
            description='保留长期仓位，围绕长期均线偏离进行小额T的高股息策略',
            code=code,
            parameters=default_parameters,
            template='dividend_ma_t',
            is_template=True,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
        )
        session.add(s)
        session.flush()
        session.commit()
        strategy_id = s.id
        print('插入策略成功，ID:', strategy_id)
    else:
        print('更新现有策略，ID:', target.id)
        target.code = code
        target.description = '保留长期仓位，围绕长期均线偏离进行小额T的高股息策略'
        target.parameters = default_parameters
        target.updated_at = datetime.datetime.now()
        try:
            session.commit()
            print('更新策略成功，ID:', target.id)
        except Exception as e:
            session.rollback()
            print('更新策略失败:', e)
        strategy_id = target.id

    # 设置参数空间：先删除旧配置
    existing_spaces = session.query(StrategyParameterSpace).filter(
        StrategyParameterSpace.strategy_id == strategy_id
    ).all()
    if existing_spaces:
        print(f'删除现有的 {len(existing_spaces)} 个参数空间')
        for sp in existing_spaces:
            session.delete(sp)
        session.commit()

    spaces = [
        {
            'parameter_name': 'long_ma',
            'parameter_type': 'int',
            'min_value': 60,
            'max_value': 240,
            'step_size': 10,
            'description': '长期均线周期'
        },
        {
            'parameter_name': 'dev_threshold_pct',
            'parameter_type': 'float',
            'min_value': 0.02,
            'max_value': 0.15,
            'step_size': 0.01,
            'description': '相对长期均线的偏离百分比阈值'
        },
        {
            'parameter_name': 'baseline_long_position',
            'parameter_type': 'float',
            'min_value': 0.2,
            'max_value': 0.9,
            'step_size': 0.1,
            'description': '长期基线仓位占比'
        },
        {
            'parameter_name': 't_chunk_count',
            'parameter_type': 'int',
            'min_value': 1,
            'max_value': 10,
            'step_size': 1,
            'description': '小T分批次数'
        },
        {
            'parameter_name': 'total_max_position',
            'parameter_type': 'float',
            'min_value': 0.6,
            'max_value': 1.0,
            'step_size': 0.05,
            'description': '仓位上限（基线+小T）'
        },
    ]

    for cfg in spaces:
        sp = StrategyParameterSpace(
            strategy_id=strategy_id,
            parameter_name=cfg['parameter_name'],
            parameter_type=cfg['parameter_type'],
            min_value=cfg.get('min_value'),
            max_value=cfg.get('max_value'),
            step_size=cfg.get('step_size'),
            choices=cfg.get('choices'),
            description=cfg.get('description'),
            created_at=datetime.datetime.now(),
        )
        session.add(sp)

    try:
        session.commit()
        print(f'成功添加 {len(spaces)} 个参数空间配置')
    except Exception as e:
        session.rollback()
        print('添加参数空间失败:', e)

    session.close()
    print('完成')


if __name__ == '__main__':
    add_strategy_to_db()