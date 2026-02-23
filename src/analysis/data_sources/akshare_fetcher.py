"""
AkShare 数据抓取器

使用 akshare 库抓取A股、美股和港股数据
"""

import pandas as pd
import akshare as ak
from typing import List, Dict
import logging
from datetime import datetime, timedelta
import re

from ..data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

class AkshareDataFetcher(DataFetcher):
    """AkShare 数据抓取器，支持A股、美股和港股"""
    
    def __init__(self, base_path: str = None):
        super().__init__("akshare", base_path)
        self.default_adjust = "qfq"
        logger.info(f"AkShare 数据抓取器初始化，默认复权模式: {self.default_adjust}")
        
        # 常用A股代码列表
        self.popular_a_stocks = [
            {'symbol': '000001', 'name': '平安银行', 'market': 'A'},
            {'symbol': '000002', 'name': '万科A', 'market': 'A'},
            {'symbol': '000858', 'name': '五粮液', 'market': 'A'},
            {'symbol': '002415', 'name': '海康威视', 'market': 'A'},
            {'symbol': '002594', 'name': 'BYD', 'market': 'A'},
            {'symbol': '300059', 'name': '东方财富', 'market': 'A'},
            {'symbol': '300750', 'name': '宁德时代', 'market': 'A'},
            {'symbol': '600000', 'name': '浦发银行', 'market': 'A'},
            {'symbol': '600036', 'name': '招商银行', 'market': 'A'},
            {'symbol': '600519', 'name': '贵州茅台', 'market': 'A'},
            {'symbol': '600887', 'name': '伊利股份', 'market': 'A'},
            {'symbol': '002304', 'name': '洋河股份', 'market': 'A'},
            {'symbol': '000568', 'name': '泸州老窖', 'market': 'A'},
            {'symbol': '600276', 'name': '恒瑞医药', 'market': 'A'},
            {'symbol': '000661', 'name': '长春高新', 'market': 'A'},
            {'symbol': '300015', 'name': '爱尔眼科', 'market': 'A'},
            {'symbol': '002142', 'name': '宁波银行', 'market': 'A'},
            {'symbol': '600031', 'name': '三一重工', 'market': 'A'},
            {'symbol': '000725', 'name': '京东方A', 'market': 'A'}
        ]
        
        # 常用美股代码列表
        self.popular_us_stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'market': 'US'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'market': 'US'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'market': 'US'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'market': 'US'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'market': 'US'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'market': 'US'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'market': 'US'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'market': 'US'},
            {'symbol': 'BABA', 'name': 'Alibaba Group Holding Limited', 'market': 'US'},
            {'symbol': 'JD', 'name': 'JD.com Inc.', 'market': 'US'},
            {'symbol': 'PDD', 'name': 'PDD Holdings Inc.', 'market': 'US'},
            {'symbol': 'BIDU', 'name': 'Baidu Inc.', 'market': 'US'},
            {'symbol': 'NIO', 'name': 'NIO Inc.', 'market': 'US'},
            {'symbol': 'XPEV', 'name': 'XPeng Inc.', 'market': 'US'},
            {'symbol': 'LI', 'name': 'Li Auto Inc.', 'market': 'US'},
            {'symbol': 'BILI', 'name': 'Bilibili Inc.', 'market': 'US'},
            {'symbol': 'TME', 'name': 'Tencent Music Entertainment Group', 'market': 'US'},
            {'symbol': 'IQ', 'name': 'iQIYI Inc.', 'market': 'US'},
            {'symbol': 'WB', 'name': 'Weibo Corporation', 'market': 'US'},
            {'symbol': 'DIDI', 'name': 'DiDi Global Inc.', 'market': 'US'}
        ]
    
    def _is_us_stock(self, symbol: str) -> bool:
        """判断是否为美股代码"""
        # 美股代码通常是字母组成，A股是数字
        return bool(re.match(r'^[A-Z]+$', symbol.upper()))
    
    def _is_hk_stock(self, symbol: str) -> bool:
        """判断是否为港股代码"""
        # 港股代码通常是5位数字，以0开头
        return bool(re.match(r'^0\d{4}$', symbol))
    
    def fetch_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从AkShare抓取股票数据（支持A股、美股和港股）
        
        Args:
            symbol: 股票代码（A股：6位数字，美股：字母代码，港股：5位数字以0开头）
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            包含股票数据的DataFrame
        """
        try:
            # 设置默认日期范围（最近1年）
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            else:
                end_date = end_date.replace('-', '')
                
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            else:
                start_date = start_date.replace('-', '')
            
            logger.info(f"从AkShare抓取数据: {symbol}, 日期范围: {start_date} 至 {end_date}, 复权模式: {self.default_adjust}")
            
            # 判断是A股、美股还是港股
            if self._is_us_stock(symbol):
                return self._fetch_us_stock_data(symbol, start_date, end_date)
            elif self._is_hk_stock(symbol):
                return self._fetch_hk_stock_data(symbol, start_date, end_date)
            else:
                return self._fetch_a_stock_data(symbol, start_date, end_date)
                
        except Exception as e:
            logger.error(f"抓取AkShare数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()
    
    def _fetch_a_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """抓取A股数据"""
        try:
            logger.info(f"调用 A 股接口 stock_zh_a_hist: symbol={symbol}, adjust={self.default_adjust}")
            stock_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=self.default_adjust,
            )
            
            if stock_data.empty:
                logger.warning(f"AkShare未返回A股数据: {symbol}")
                return pd.DataFrame()
            
            # 标准化列名
            data = pd.DataFrame({
                'date': pd.to_datetime(stock_data['日期']).dt.strftime('%Y-%m-%d'),
                'open': stock_data['开盘'].round(2),
                'high': stock_data['最高'].round(2),
                'low': stock_data['最低'].round(2),
                'close': stock_data['收盘'].round(2),
                'volume': stock_data['成交量'].astype(int),
                'adj_close': stock_data['收盘'].round(2)  # 暂时使用收盘价作为调整后价格
            })
            
            logger.info(f"成功获取A股数据: {symbol}, 行数: {len(data)}")
            return data
            
        except Exception as e:
            logger.error(f"抓取A股数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()
    
    def _fetch_us_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """抓取美股数据"""
        try:
            logger.info(f"调用美股接口 stock_us_daily: symbol={symbol}, adjust={self.default_adjust}")
            stock_data = ak.stock_us_daily(symbol=symbol, adjust=self.default_adjust)
            
            if stock_data is None or stock_data.empty:
                logger.warning(f"AkShare未返回美股数据: {symbol}")
                return pd.DataFrame()
            
            # 转换日期格式进行过滤
            start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 确保日期列是datetime格式
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            
            # 过滤日期范围
            mask = (stock_data['date'] >= start_date_formatted) & (stock_data['date'] <= end_date_formatted)
            stock_data = stock_data[mask]
            
            if stock_data.empty:
                logger.warning(f"指定日期范围内无美股数据: {symbol}, {start_date_formatted} 至 {end_date_formatted}")
                return pd.DataFrame()
            
            # 标准化数据格式
            data = pd.DataFrame({
                'date': stock_data['date'].dt.strftime('%Y-%m-%d'),
                'open': stock_data['open'].round(2),
                'high': stock_data['high'].round(2),
                'low': stock_data['low'].round(2),
                'close': stock_data['close'].round(2),
                'volume': stock_data['volume'].astype(int),
                'adj_close': stock_data['close'].round(2)  # 使用收盘价作为调整后价格
            })
            
            logger.info(f"成功获取美股数据: {symbol}, 行数: {len(data)}")
            return data
            
        except Exception as e:
            logger.error(f"抓取美股数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()
    
    def _fetch_hk_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """抓取港股数据"""
        try:
            # 尝试多种港股数据抓取方法
            stock_data = None
            
            # 方法1: 使用 stock_hk_hist_min_em (修复参数问题)
            try:
                logger.info(f"尝试使用 stock_hk_hist_min_em 抓取港股数据: {symbol}, adjust={self.default_adjust}")
                stock_data = ak.stock_hk_hist_min_em(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=self.default_adjust,
                )
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"使用 stock_hk_hist_min_em 成功获取港股数据: {symbol}")
                else:
                    logger.warning(f"stock_hk_hist_min_em 返回空数据: {symbol}")
                    stock_data = None
            except Exception as e:
                logger.warning(f"stock_hk_hist_min_em 失败: {e}")
                stock_data = None
            
            # 方法2: 使用 stock_hk_hist (东方财富历史数据)
            if stock_data is None or stock_data.empty:
                try:
                    logger.info(f"尝试使用 stock_hk_hist 抓取港股数据: {symbol}, adjust={self.default_adjust}")
                    stock_data = ak.stock_hk_hist(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self.default_adjust,
                    )
                    if stock_data is not None and not stock_data.empty:
                        logger.info(f"使用 stock_hk_hist 成功获取港股数据: {symbol}")
                    else:
                        logger.warning(f"stock_hk_hist 返回空数据: {symbol}")
                        stock_data = None
                except Exception as e:
                    logger.warning(f"stock_hk_hist 失败: {e}")
                    stock_data = None
            
            # 方法3: 使用 stock_hk_spot_em (最后尝试 - 仅用于获取当前快照)
            if stock_data is None or stock_data.empty:
                try:
                    logger.info(f"尝试使用 stock_hk_spot_em 抓取港股数据: {symbol}")
                    stock_data = ak.stock_hk_spot_em()
                    if stock_data is not None and not stock_data.empty:
                        # 过滤指定股票
                        stock_data = stock_data[stock_data['代码'] == symbol]
                        if stock_data.empty:
                            logger.warning(f"stock_hk_spot_em 未找到股票 {symbol}")
                            stock_data = None
                        else:
                            logger.warning(f"stock_hk_spot_em 只返回当前快照数据，不是历史数据: {symbol}")
                            stock_data = None  # 不使用快照数据，因为我们需要历史数据
                except Exception as e:
                    logger.warning(f"stock_hk_spot_em 失败: {e}")
                    stock_data = None
            
            
            if stock_data is None or stock_data.empty:
                logger.warning(f"所有方法都未能获取港股数据: {symbol}")
                return pd.DataFrame()
            
            # 检查数据格式
            logger.info(f"港股数据列名: {list(stock_data.columns)}")
            logger.info(f"港股数据形状: {stock_data.shape}")
            
            # 标准化数据格式
            data = self._standardize_hk_data(stock_data, symbol)
            
            if not data.empty:
                logger.info(f"成功获取港股数据: {symbol}, 行数: {len(data)}")
                return data
            else:
                logger.error(f"港股数据标准化失败: {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"抓取港股数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()
    
    def _standardize_hk_data(self, stock_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标准化港股数据格式"""
        try:
            # 尝试不同的列名映射
            date_col = None
            open_col = None
            high_col = None
            low_col = None
            close_col = None
            volume_col = None
            
            for col in stock_data.columns:
                col_lower = col.lower()
                if '日期' in col or '时间' in col or 'date' in col_lower or 'time' in col_lower:
                    date_col = col
                elif '开盘' in col or '今开' in col or 'open' in col_lower:
                    open_col = col
                elif '最高' in col or 'high' in col_lower:
                    high_col = col
                elif '最低' in col or 'low' in col_lower:
                    low_col = col
                elif '收盘' in col or '最新价' in col or 'close' in col_lower:
                    close_col = col
                elif '成交量' in col or 'volume' in col_lower:
                    volume_col = col
            
            if open_col and high_col and low_col and close_col:
                # 如果没有日期列，使用当前日期（用于快照数据）
                if date_col:
                    dates = pd.to_datetime(stock_data[date_col]).dt.strftime('%Y-%m-%d')
                else:
                    # 对于快照数据，使用当前日期
                    dates = [datetime.now().strftime('%Y-%m-%d')] * len(stock_data)
                
                data = pd.DataFrame({
                    'date': dates,
                    'open': pd.to_numeric(stock_data[open_col], errors='coerce').round(2),
                    'high': pd.to_numeric(stock_data[high_col], errors='coerce').round(2),
                    'low': pd.to_numeric(stock_data[low_col], errors='coerce').round(2),
                    'close': pd.to_numeric(stock_data[close_col], errors='coerce').round(2),
                    'volume': pd.to_numeric(stock_data[volume_col], errors='coerce').astype(int) if volume_col else 0,
                    'adj_close': pd.to_numeric(stock_data[close_col], errors='coerce').round(2)
                })
                
                # 删除包含NaN的行
                data = data.dropna()
                return data
            else:
                logger.error(f"无法识别港股数据列名: {list(stock_data.columns)}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"标准化港股数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self) -> List[Dict[str, str]]:
        """
        获取常用股票列表（A股+美股）
        
        Returns:
            股票列表
        """
        return self.popular_a_stocks.copy() + self.popular_us_stocks.copy()
    
    def get_a_stock_list(self) -> List[Dict[str, str]]:
        """
        获取常用A股列表
        
        Returns:
            A股列表
        """
        return self.popular_a_stocks.copy()
    
    def get_us_stock_list(self) -> List[Dict[str, str]]:
        """
        获取常用美股列表
        
        Returns:
            美股列表
        """
        return self.popular_us_stocks.copy()
    
    def get_all_stocks(self) -> List[Dict[str, str]]:
        """
        获取所有股票列表（A股+美股）
        
        Returns:
            所有股票列表
        """
        try:
            all_stocks = []
            
            # 获取A股列表
            try:
                a_stock_list = ak.stock_info_a_code_name()
                for _, row in a_stock_list.iterrows():
                    all_stocks.append({
                        'symbol': row['code'],
                        'name': row['name'],
                        'market': 'A'
                    })
                logger.info(f"获取A股列表成功: {len(a_stock_list)}只股票")
            except Exception as e:
                logger.error(f"获取A股列表失败: {e}")
                all_stocks.extend(self.popular_a_stocks)
            
            # 获取美股列表
            try:
                us_stock_list = ak.stock_us_spot_em()
                # 只取前1000只活跃美股，避免数据过多
                for _, row in us_stock_list.head(1000).iterrows():
                    symbol = row.get('代码', row.get('symbol', ''))
                    name = row.get('名称', row.get('name', ''))
                    if symbol and name:
                        all_stocks.append({
                            'symbol': symbol,
                            'name': name,
                            'market': 'US'
                        })
                logger.info(f"获取美股列表成功: 1000只股票")
            except Exception as e:
                logger.error(f"获取美股列表失败: {e}")
                all_stocks.extend(self.popular_us_stocks)
            
            logger.info(f"获取股票列表成功: 总计{len(all_stocks)}只股票")
            return all_stocks
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return self.get_stock_list()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票信息字典
        """
        try:
            result = {'symbol': symbol}
            
            if self._is_us_stock(symbol):
                # 美股信息获取
                try:
                    # 尝试获取美股基本信息
                    info = ak.stock_us_fundamental(symbol=symbol)
                    if not info.empty:
                        result['name'] = info.get('公司名称', symbol)
                        result['market'] = 'US'
                        result['industry'] = info.get('行业', '')
                        result['market_cap'] = info.get('市值', '')
                except Exception as e:
                    logger.warning(f"获取美股详细信息失败: {symbol}, {e}")
                    # 从常用列表中查找
                    for stock in self.popular_us_stocks:
                        if stock['symbol'] == symbol:
                            result.update(stock)
                            break
            else:
                # A股信息获取
                info = ak.stock_individual_info_em(symbol=symbol)
                result['market'] = 'A'
                
                # 解析信息
                for _, row in info.iterrows():
                    key = row['item']
                    value = row['value']
                    
                    if key == '股票简称':
                        result['name'] = value
                    elif key == '所属行业':
                        result['industry'] = value
                    elif key == '总市值':
                        result['market_cap'] = value
                    elif key == '流通市值':
                        result['float_market_cap'] = value
            
            return result
            
        except Exception as e:
            logger.error(f"获取股票信息失败: {symbol}, 错误: {e}")
            return {'symbol': symbol, 'name': '', 'error': str(e)}
    
    def search_stocks(self, query: str, market: str = 'all') -> List[Dict[str, str]]:
        """
        搜索股票
        
        Args:
            query: 搜索关键词
            market: 市场类型 ('all', 'A', 'US')
            
        Returns:
            匹配的股票列表
        """
        results = []
        
        # 确定搜索范围
        if market == 'A':
            search_list = self.popular_a_stocks
        elif market == 'US':
            search_list = self.popular_us_stocks
        else:
            search_list = self.get_stock_list()
        
        # 在常用股票中搜索
        for stock in search_list:
            if query.upper() in stock['symbol'].upper() or query in stock['name']:
                results.append(stock)
        
        # 如果没有找到，尝试从完整列表中搜索
        if not results:
            try:
                all_stocks = self.get_all_stocks()
                for stock in all_stocks:
                    if market != 'all' and stock.get('market') != market:
                        continue
                    if query.upper() in stock['symbol'].upper() or query in stock['name']:
                        results.append(stock)
                        if len(results) >= 20:  # 限制返回数量
                            break
            except Exception as e:
                logger.error(f"搜索股票失败: {e}")
        
        return results 
