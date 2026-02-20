import os
from pathlib import Path

# 基础目录配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 确保数据目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/backtesting.db")

# API配置
API_KEYS = {
    "alphavantage": os.getenv("ALPHAVANTAGE_API_KEY", ""),
    "yahoo_finance": "",
    "akshare": "",
    "qwen": os.getenv("QWEN_API_KEY", ""),
    "kimi": os.getenv("KIMI_API_KEY", ""),
    "deepseek": os.getenv("DEEPSEEK_API_KEY", "sk-249cb27bf7354688bf1568958aa9db8f"),
}

# 回测配置
DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_SLIPPAGE_RATE = 0.0001

# 应用配置
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 前端配置
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000") 
