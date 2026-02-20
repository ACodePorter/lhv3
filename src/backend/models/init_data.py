from sqlalchemy.orm import Session
from datetime import datetime

from .data_models import DataSource, get_db
from .strategy import AiPromptSetting


def init_default_data():
    db = next(get_db())

    try:
        existing_sources = db.query(DataSource).all()
        if not existing_sources:
            default_sources = [
                DataSource(
                    name="Yahoo Finance",
                    description="雅虎财经数据，提供美股、港股等全球市场数据",
                    created_at=datetime.now(),
                ),
                DataSource(
                    name="A股数据",
                    description="中国A股市场数据",
                    created_at=datetime.now(),
                ),
                DataSource(
                    name="用户上传",
                    description="用户自定义上传的数据",
                    created_at=datetime.now(),
                ),
            ]

            db.add_all(default_sources)
            db.commit()

        existing_prompt = (
            db.query(AiPromptSetting)
            .filter(AiPromptSetting.model_type == "deepseek")
            .filter(AiPromptSetting.scene == "ai_investment")
            .first()
        )
        if not existing_prompt:
            now = datetime.now()
            system_prompt_text = (
                "你是一个专注于股票量化交易的金融大模型，充当价格预测引擎。"
                "\n\n"
                "你的任务是根据输入的K线数据、账户状态和最近成交记录，预测下一根K线的收盘价。"
                "\n\n"
                "要求："
                "\n1）只能输出一个数字，不要任何文字、解释、符号或单位；"
                "\n2）数字应为合理的价格水平，不为负数，尽量接近当前价格量级；"
                "\n3）可以保留2到6位小数；"
                "\n4）充分利用给出的周期、买入/卖出阈值、止损和止盈参数来判断趋势和风险；"
                "\n5）只使用输入的数据进行推断，不要编造外部信息或新闻；"
                "\n6）如果历史数据较少，也要给出尽可能稳健的预测，而不是报错。"
            )
            setting = AiPromptSetting(
                model_type="deepseek",
                scene="ai_investment",
                system_prompt=system_prompt_text,
                description=None,
                created_at=now,
                updated_at=now,
            )
            db.add(setting)
            db.commit()

    except Exception as e:
        db.rollback()
        print(f"初始化数据失败: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    init_default_data()
