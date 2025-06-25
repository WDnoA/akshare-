#2.1
# 优化得分计算和权重设置
# -*- coding: utf-8 -*-

"""
短线交易选股系统 - 优化版
专注于识别第二天可能上涨或涨停的股票，用于短线交易

系统流程:
1. 通过数据获取模块获取股票历史数据
2. 计算各类技术指标和价格模式识别
3. 综合分析各维度指标计算股票得分
4. 生成排名靠前的股票报告


主要模块:
1. Datasr: 数据获取模块，负责股票数据的获取与缓存管理
2. TechnicalAnalysis: 技术分析模块，计算各类技术指标
3. TechnicalIndicators: 技术指标总控模块，协调各类指标计算
4. calculate_stock_score: 股票得分计算函数
5. calculate_limit_up_probability: 涨停概率计算函数
6. MarketAnalyzer: 市场分析模块，分析市场趋势与情绪
7. BaseIndicator及子类: 各类技术指标计算
8. generate_report: 生成选股报告，展示排名前100的股票

输出:
- 基础信息视图: 代码、名称、得分、信号等关键信息
- 技术指标视图: RSI、MACD、KDJ等指标值
- 分项得分视图: 各维度得分明细
- 交易决策视图: 购买目标价、止损价等交易参考
- 交易建议: 强烈推荐、建议关注的股票列表
- 市场分析: 当日涨停风险分布、上涨潜力分布等
"""

import os
import sys
import time
import datetime
import pytz
from datetime import date, datetime, time as datetime_time
import json
import logging
import traceback
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

# 尝试导入 akshare 库（用于获取股票数据）
try:
    import akshare as ak
except ImportError:
    print("错误: 未安装 akshare 库，请执行 'pip install akshare' 安装")
    sys.exit(1)

# 尝试导入 TA-Lib 库（用于技术指标计算）
try:
    import talib as ta
except ImportError:
    print("错误: 未安装 TA-Lib 库，请安装: pip install ta-lib")
    sys.exit(1)

# =============================================================================
# 全局常量定义
# =============================================================================
class Constants:
    """全局常量类，集中管理所有配置参数"""
    EPSILON = 1e-10            # 避免除零错误
    DEFAULT_STRENGTH = 0.5     # 默认收盘强度
    MIN_DATA_POINTS = 20       # 最小数据点数量要求
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    BASE_DIR = "quant_data"    # 数据存放目录
    CACHE_L_MINUTES = 60 * 24  # 列表缓存
    CACHE_EXPIRY_MINUTES = float('inf')  # 股票数据缓存永不过期
    DEFAULT_FILL_VALUE = 0     # 默认填充值
    DEFAULT_RSI = 50           # 默认 RSI 初始值
    DEFAULT_TREND = 1.0        # 默认趋势值
 
# 添加一个全局变量用于记录已配置的日志
_CONFIGURED_LOGGERS = {}

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO, 
               show_in_console: bool = True) -> logging.Logger:
    """
    获取或创建一个日志记录器，确保同名日志器只配置一次
    
    参数:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        show_in_console: 是否在控制台显示日志
        
    返回:
        配置好的日志记录器
    """
    # 如果已配置过此日志器，直接返回
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 移除所有现有处理器避免重复
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    logger.setLevel(level)
    formatter = logging.Formatter(Constants.LOG_FORMAT)
    
    # 添加控制台处理器
    if show_in_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # 记录此日志器已被配置
    _CONFIGURED_LOGGERS[name] = logger
    logger.propagate = False  # 防止日志传播到根日志器

    # 添加初始化日志
    logger.info(f"Logger '{name}' initialized. Console logging: {'enabled' if show_in_console else 'disabled'}")
    return logger

# 单例元类
class Singleton(type):
    """
    单例模式元类，确保使用此元类的类只有一个实例
    使用线程锁确保线程安全
    """
    _instances = {}
    _lock = threading.RLock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# =============================================================================
# 数据获取模块
# =============================================================================
class Datasr(metaclass=Singleton):
    """
    数据获取模块（使用元类实现线程安全的单例模式）
    负责股票日线数据的获取、缓存管理与网络请求重试。
    """
    DEFAULT_CONFIG = {
        "cache_expiry_minutes": Constants.CACHE_EXPIRY_MINUTES,
        "max_retries": 1,# 最大重试次数
        "retry_delay": 2,# 重试延迟时间
        "min_data_ratio": 0.5,# 最小数据比例
        "default_days": 180   # 默认获取180天
    }

    def __init__(self, config: Optional[Dict] = None):
        # 使用属性判断是否已初始化
        if hasattr(self, '_initialized'):
            return
            
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}# 合并配置
        self.base_dir = Constants.BASE_DIR
        self.cache_dir = os.path.join(self.base_dir, "stock_r_cache")# 股票列表存放目录
        self.realtime_dir = os.path.join(self.base_dir, "stock_r_realtime")# 股票数据存放目录
        self.report_dir = os.path.join(self.base_dir, "stock_r_report")# 报告存放目录
        self.industry_dir = os.path.join(self.base_dir, "stock_r_industry")# 行业数据存放目录
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.realtime_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.industry_dir, exist_ok=True)
        
        # 使用统一的日志获取函数
        log_file = os.path.join(self.base_dir, 'stock_r.log')
        self.logger = get_logger('Datasr', log_file, show_in_console=True)
        
        self._stock_name_cache = {}# 股票名称缓存
        self._initialized = True# 初始化标志

        # 记录初始化完成
        self.logger.info(f"Datasr initialized. Config: {self.config}")
        self.logger.info(f"Data directories: {self.base_dir}")

    @lru_cache(maxsize=1)
    def get_stock_list(self) -> Dict[str, str]:
        """
        获取股票代码与名称映射表，优先使用本地缓存。
        """
        cache_file = os.path.join(self.cache_dir, "stock_list_cache.json")
        try:
            if os.path.exists(cache_file):
                cache_time = os.path.getmtime(cache_file)
                if (time.time() - cache_time) < (Constants.CACHE_L_MINUTES * 60):
                    self.logger.info("从本地缓存加载股票列表...")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            self.logger.info("正在获取 A 股股票列表...")
            stock_list = ak.stock_info_a_code_name()
            stock_dict = dict(zip(stock_list['code'], stock_list['name']))
            self.logger.info(f"成功获取 {len(stock_dict)} 只股票的信息")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(stock_dict, f, ensure_ascii=False)
            return stock_dict
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {str(e)}", exc_info=True)
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.logger.warning("使用过期的本地缓存数据")
                    return json.load(f)
            return {}

    def get_stock_name(self, symbol: str) -> str:
        """
        根据股票代码获取股票名称，若未找到则返回代码本身。
        """
        if not self._stock_name_cache:
            self._stock_name_cache = self.get_stock_list()
        return self._stock_name_cache.get(symbol, symbol)

    def _get_cache_path(self, symbol: str, days: int) -> str:
        """获取股票数据的缓存文件路径。"""
        stock_name = self.get_stock_name(symbol)
        days = days if days is not None else self.config["default_days"]
        return os.path.join(self.realtime_dir, f"{symbol}_{stock_name}_{days}_realtime.csv")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """检查缓存文件是否有效（未过期）。"""
        if not os.path.exists(cache_path):
            return False
        cache_time = os.path.getmtime(cache_path)
        cache_age_minutes = (time.time() - cache_time) / 60
        return cache_age_minutes < self.config["cache_expiry_minutes"]


    def fetch_data(self, symbol: str, days: Optional[int] = None, min_data_ratio: Optional[float] = None) -> pd.DataFrame:
        """
        获取股票日线数据，支持缓存及数据完整性检查。
        支持增量更新数据，只获取缺失的最新数据。
        
        参数:
            symbol: 股票代码
            days: 获取天数，默认使用配置中的default_days
            min_data_ratio: 最小数据比例，默认使用配置中的min_data_ratio
            
        返回:
            pd.DataFrame: 包含股票数据的DataFrame，如遇错误则返回空DataFrame
        """
        days = days if days is not None else self.config["default_days"]
        min_data_ratio = min_data_ratio if min_data_ratio is not None else self.config["min_data_ratio"]
        cache_path = self._get_cache_path(symbol, days)
        
        try:
            # 1. 加载缓存并检查是否需要更新,如果不需要更新，直接返回缓存数据
            cached_df, should_update = self._load_from_cache(symbol, cache_path)
            if not should_update and not cached_df.empty:
                self.logger.info(f"缓存数据有效，直接使用 {symbol} 的缓存数据")  # 记录缓存命中日志
                return cached_df
            
            # 2. 决定获取数据的范围
            fetch_days = 5 if not cached_df.empty else days
            self.logger.info(f"获取 {symbol} 数据 (模式: {'增量替换' if fetch_days==5 else '全量'}, 天数: {fetch_days})")
            
            # 3. 获取并处理新数据
            raw_df = self._fetch_from_source(symbol, fetch_days)
            if raw_df.empty:
                self.logger.warning(f"获取 {symbol} 数据失败")
                return cached_df if not cached_df.empty else pd.DataFrame()
            
            new_df = self._preprocess_data(raw_df)
            
            # 4. 合并数据
            result_df = self._merge_data(cached_df, new_df, days)
            
            # 5. 数据质量验证
            result_df = self._validate_data_quality(result_df, days, min_data_ratio, symbol)
            
            # 6. 保存缓存
            if not result_df.empty:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    result_df.to_csv(cache_path, encoding='utf-8-sig')
                    os.utime(cache_path, None)  # 更新修改时间
                    self.logger.debug(f"缓存已更新 {cache_path}")
                except Exception as e:
                    self.logger.error(f"保存缓存失败: {str(e)}")
            
            return result_df
        except Exception as e:
            self.logger.error(f"获取 {symbol} 数据失败: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _load_from_cache(self, symbol: str, cache_path: str) -> Tuple[pd.DataFrame, bool]:
        """从缓存加载数据并判断是否需要更新"""
        cached_df = pd.DataFrame()
        should_update = True
        
        # 检查本地缓存是否存在并有效
        if not self._is_cache_valid(cache_path):
            return cached_df, should_update
        
        # 检查缓存时间有效性 (16小时)
        cache_mod_time = os.path.getmtime(cache_path)
        cache_age_hours = (time.time() - cache_mod_time) / 3600
        
        if cache_age_hours <= 16:
            should_update = False
            self.logger.info(f"缓存有效 (age: {cache_age_hours:.1f}h <= 16h)")
        
        try:
            # 尝试从缓存文件读取数据
            cached_df = pd.read_csv(cache_path, index_col='Date', parse_dates=['Date'])
            
            # 数据完整性检查
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in cached_df.columns for col in required_columns) or cached_df.empty:
                self.logger.warning(f"缓存数据{cache_path}不完整，需要重新获取")
                return pd.DataFrame(), True
            return cached_df, should_update
        except Exception as e:
            self.logger.warning(f"读取缓存失败: {str(e)}，重新获取数据")
            return pd.DataFrame(), True
    
    def _merge_data(self, cached_df: pd.DataFrame, new_df: pd.DataFrame, days: int) -> pd.DataFrame:
        if cached_df.empty:
            return new_df.sort_index().tail(days)  # 无缓存时直接返回新数据
        
        # 替换式更新
        new_dates = new_df.index
        cached_df = cached_df[~cached_df.index.isin(new_dates)]
        
        # 合并数据并确保日期连续
        result_df = pd.concat([cached_df, new_df]).sort_index()
        result_df = result_df[~result_df.index.duplicated(keep='last')]  # 去重
        
        # 检查数据连续性
        date_diff = (result_df.index[-1] - result_df.index[0]).days
        if date_diff > days * 1.5:  # 允许50%的缓冲
            self.logger.warning(f"数据日期范围异常: {date_diff}天 (预期: {days}天)")
        
        return result_df.tail(days)
    
    def _validate_data_quality(self, df: pd.DataFrame, expected_days: int, 
                             min_data_ratio: float, symbol: str) -> pd.DataFrame:
        """验证数据质量并处理异常情况"""
        if df.empty:
            self.logger.warning(f"股票 {symbol} 数据为空")
            return df
            
        actual_days = len(df)
        data_ratio = actual_days / expected_days
        
        if data_ratio < min_data_ratio:
            self.logger.warning(f"股票 {symbol} 数据量不足 ({data_ratio:.2%}), 低于阈值 {min_data_ratio:.2%}")
            return pd.DataFrame()       
        return df

    def _get_industry_cache_path(self, symbol: str) -> str:
        """获取股票行业数据的缓存文件路径。"""
        stock_name = self.get_stock_name(symbol)
        return os.path.join(self.industry_dir, f"{symbol}_{stock_name}_industry.json")
    
    def _is_industry_cache_valid(self, cache_path: str) -> bool:
        """检查行业数据缓存文件是否有效（未过期）。"""
        if not os.path.exists(cache_path):
            return False
        cache_time = os.path.getmtime(cache_path)
        cache_age_hours = (time.time() - cache_time) / 3600
        # 行业数据缓存有效期为16小时
        return cache_age_hours <= 16
    
    def get_stock_industry_data(self, symbol: str) -> dict:
        """
        获取股票所属行业，返回行业名称和行业涨跌幅。
        支持缓存功能，缓存有效期为16小时。
        
        参数:
            symbol: 股票代码
            
        返回:
            dict: 包含行业名称(industry_name)和涨跌幅(industry_change)的字典
                如果无法获取数据，行业名称默认为"未知行业"，涨跌幅默认为0.0
        """
        # 初始化结果字典
        result = {"industry_name": "未知行业", "industry_change": 0.0}
        
        # 获取缓存路径
        cache_path = self._get_industry_cache_path(symbol)
        
        # 检查缓存是否有效
        if self._is_industry_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8-sig') as f:
                    cached_data = json.load(f)
                    self.logger.info(f"从缓存加载股票{symbol}的行业数据")
                    return cached_data
            except Exception as e:
                self.logger.warning(f"读取行业数据缓存失败: {str(e)}，重新获取数据")
        
        # 获取行业基本信息
        try:
            # akshare 文档调用东方财富接口：个股信息
            info_hy = self._fetch_with_retry(ak.stock_individual_info_em(symbol=symbol))
            # 提取行业信息
            industry_hy = info_hy[info_hy['item'] == '行业']
            # 确保行业信息存在
            if not industry_hy.empty:
                # 行业名称通常是一个字符串
                industry_name = industry_hy['value'].iloc[0]
                # 确保行业名称不为空
                result['industry_name'] = industry_name
            # 获取行业涨跌幅数据
                try:
                    # akshare 文档调用东方财富接口
                    info_hyzf = ak.stock_hsgt_board_rank_em(
                        symbol="北向资金增持行业板块排行", 
                        indicator="今日"
                    )
                    # 提取匹配的行业数据
                    industry_hyzf = info_hyzf[info_hyzf['名称'] == industry_name]['最新涨跌幅'].values
                    # 确保数据不为空
                    if len(industry_hyzf) > 0:
                        result['industry_change'] = float(industry_hyzf[0])
                except Exception as e:
                    self.logger.warning(f"获取股票{symbol}行业涨跌幅数据失败: {str(e)}")
            else:
                self.logger.warning(f"股票 {symbol} 未知行业")
        except Exception as e:
            self.logger.warning(f"获取股票{symbol}所属行业数据失败: {str(e)}")
        
        # 保存缓存
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8-sig') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.utime(cache_path, None)  # 更新修改时间
            self.logger.debug(f"行业数据缓存已更新: {cache_path}")
        except Exception as e:
            self.logger.error(f"保存行业数据缓存失败: {str(e)}")
            
        return result

    def _fetch_with_retry(self, fetch_func, *args, **kwargs) -> pd.DataFrame:
        """
        带重试机制的数据请求，采用指数退避策略。
        """
        max_retries = self.config["max_retries"]
        retry_delay = self.config["retry_delay"]
        for attempt in range(max_retries):
            try:
                return fetch_func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(f"请求失败，{wait_time}秒后重试 ({attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"达到最大重试次数，返回空数据: {str(e)}")
                    return pd.DataFrame()
        return pd.DataFrame()

    def _fetch_from_source(self, symbol: str, days: int) -> pd.DataFrame:
        """
        从 akshare 数据源获取股票日线数据，支持备用接口。
        """
        end_date = datetime.now()
        start_date_str = (end_date - timedelta(days=days)).strftime("%Y%m%d")
        self.logger.info(f"正在获取 {symbol} {days}天历史数据")
        try:
            df = self._fetch_with_retry(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start_date_str,
                end_date=end_date.strftime("%Y%m%d"),
                adjust=""
            )
            if df.empty:
                self.logger.warning("主接口数据为空")
                return pd.DataFrame()
            return df
        except Exception as e:
            self.logger.warning(f"接口异常: {str(e)}，获取数据失败")
            return pd.DataFrame()    

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对原始数据进行清洗、标准化和格式化。
        包括：检查必要列、重命名、日期转换及排序。
        """
        if df.empty:
            return pd.DataFrame()
        required_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        missing = set(required_columns) - set(df.columns)
        if missing:
            self.logger.error(f"原始数据缺失关键列: {missing}")
            return pd.DataFrame()
        result_df = df.rename(columns={
            '日期': 'Date', '开盘': 'Open', '收盘': 'Close',
            '最高': 'High', '最低': 'Low', '成交量': 'Volume'
        })
        result_df['Date'] = pd.to_datetime(result_df['Date'])
        result_df = result_df.sort_values('Date').set_index('Date')
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        result_df = result_df.dropna(subset=['Close'])
        return result_df

# =============================================================================
# 技术指标计算模块
# =============================================================================
class BaseIndicator:
    """
    技术指标计算基类。
    提供所有指标计算类的通用功能和接口。
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_length: int = 20) -> bool:
        """
        验证DataFrame是否有效且长度足够。
        
        参数:
            df: 需要验证的DataFrame
            min_length: 最小数据长度要求
            
        返回:
            bool: 数据是否有效
        """
        return df is not None and not df.empty and len(df) >= min_length
    
    @staticmethod
    def safe_talib_call(talib_func, *args, **kwargs) -> Tuple:
        """
        安全调用TA-Lib函数。
        
        参数:
            talib_func: 要调用的TA-Lib函数
            *args, **kwargs: 传递给TA-Lib函数的参数
            
        返回:
            TA-Lib函数的结果，若出错则返回None
        """
        try:
            return talib_func(*args, **kwargs)
        except Exception as e:
            logging.error(f"TA-Lib调用失败: {str(e)}")
            return None

class PriceIndicator(BaseIndicator):
    """
    价格类技术指标。
    负责计算基于价格的指标，如移动平均线、布林带等。
    """
    
    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods: List[int] = [3, 5, 10, 20, 34]) -> pd.DataFrame:
        """
        计算多周期移动平均线，优化版本
        
        参数:
            df: 包含价格数据的DataFrame
            periods: 周期列表，默认为[3, 5, 10, 20, 34]
                    新增MA3用于超短线，MA34用于中期趋势
                    
        返回:
            添加了移动平均线的DataFrame
        """
        # 避免不必要的复制，只在必要时复制
        if not BaseIndicator.validate_dataframe(df, max(periods)):
            return df
            
        # 获取收盘价数组，避免重复访问
        close_array = df['Close'].values
        
        # 使用向量化操作一次性计算所有均线，提高效率
        for period in periods:
            col_name = f'MA{period}'
            df[col_name] = ta.SMA(close_array, timeperiod=period)
                
        return df
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算指数移动平均线（EMA）。
        
        参数:
            df: 股票数据DataFrame
            periods: 均线周期列表，默认为[5, 10, 20]
            
        返回:
            添加了EMA列的DataFrame
        """
        # 避免不必要的复制，只在必要时复制
        if not BaseIndicator.validate_dataframe(df, max(periods)):
            return df
            
        # 获取收盘价数组，避免重复访问
        close_array = df['Close'].values
        
        # 计算所有EMA周期
        for period in periods:
            col_name = f'EMA{period}'
            df[col_name] = ta.EMA(close_array, timeperiod=period)
                
        return df
        
    @staticmethod
    def calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, matype: int = 0) -> pd.DataFrame:
        """
        计算布林带指标。
        
        参数:
            df: 股票数据DataFrame
            period: 布林带周期，默认20
            std_dev: 标准差倍数，默认2.0
            matype: 移动平均类型，默认0(SMA)
            
        返回:
            添加了布林带指标的DataFrame
        """
        # 避免不必要的复制，只在必要时复制
        # 检查数据量是否足够
        if not BaseIndicator.validate_dataframe(df, period):
            return df
            
        # 获取收盘价数组
        close_array = df['Close'].values
            
        # 使用TA-Lib计算布林带
        bbands_result = BaseIndicator.safe_talib_call(
            ta.BBANDS,
            close_array,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=matype
        )
        
        if bbands_result is None:
            return df
            
        upper, middle, lower = bbands_result
        
        # 一次性赋值所有布林带指标
        df['BBANDS_Upper'] = upper
        df['BBANDS_Middle'] = middle
        df['BBANDS_Lower'] = lower
        
        # 计算带宽，衡量波动性
        df['BB_Width'] = (upper - lower) / middle
        
        # 计算价格在布林带中的位置百分比 (0-100%)
        df['BB_Percent'] = (close_array - lower) / (upper - lower + Constants.EPSILON) * 100
        
        # 布林带收缩指标 - 当前带宽与N周期前带宽比较
        if len(df) > period:
            width_ma = ta.SMA(df['BB_Width'].values, timeperiod=period//2)
            df['BB_Squeeze'] = df['BB_Width'] / (width_ma + Constants.EPSILON)
        
        # 突破上轨/下轨信号
        if len(df) >= 2:
            # 获取价格和布林带上下轨数组
            close_values = df['Close'].values
            upper_values = upper
            lower_values = lower
            
            # 创建结果数组
            upper_breakout = np.zeros(len(df), dtype=bool)
            lower_breakout = np.zeros(len(df), dtype=bool)
            
            # 计算向上/向下突破
            if len(close_values) > 1:
                upper_breakout[1:] = (close_values[1:] > upper_values[1:]) & (close_values[:-1] <= upper_values[:-1])
                lower_breakout[1:] = (close_values[1:] < lower_values[1:]) & (close_values[:-1] >= lower_values[:-1])
            
            # 赋值给DataFrame
            df['BB_UpperBreakout'] = upper_breakout
            df['BB_LowerBreakout'] = lower_breakout
        
        return df

class MomentumIndicator(BaseIndicator):
    """
    动量类技术指标。
    负责计算动量相关的技术指标，如MACD、RSI、KDJ等。
    """
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        计算MACD指标，并标记金叉与死叉信号。
        
        参数:
            df: 股票数据DataFrame
            fast_period: 快线周期，默认12
            slow_period: 慢线周期，默认26
            signal_period: 信号线周期，默认9
            
        返回:
            添加了MACD指标的DataFrame
        """
        # 避免不必要的复制，只在必要时复制
        if not BaseIndicator.validate_dataframe(df, slow_period + signal_period):
            return df
            
        try:
            # 获取收盘价数组，避免重复访问
            close_array = df['Close'].values
            
            # 计算MACD基础指标
            macd_result = BaseIndicator.safe_talib_call(
                ta.MACD,
                close_array,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            
            if macd_result is None:
                return df
                
            macd, signal, hist = macd_result
            
            # 一次性赋值，减少DataFrame操作次数
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
                
            # 使用向量化操作计算金叉与死叉信号，避免循环
            if len(df) >= 2:
                # 创建信号数组
                golden_cross = np.zeros(len(df), dtype=bool)
                death_cross = np.zeros(len(df), dtype=bool)
                
                # 计算金叉信号（MACD从下方穿越信号线）
                golden_cross[1:] = (macd[:-1] < signal[:-1]) & (macd[1:] >= signal[1:])
                
                # 计算死叉信号（MACD从上方穿越信号线）
                death_cross[1:] = (macd[:-1] > signal[:-1]) & (macd[1:] <= signal[1:])
                
                # 一次性赋值
                df['MACD_GoldenCross'] = golden_cross
                df['MACD_DeathCross'] = death_cross
                
                # 计算MACD趋势（正值为上升趋势，负值为下降趋势）
                df['MACD_Trend'] = np.where(macd > signal, 1, -1)
                
                # 计算MACD零轴穿越信号
                zero_cross_up = np.zeros(len(df), dtype=bool)
                zero_cross_down = np.zeros(len(df), dtype=bool)
                
                zero_cross_up[1:] = (macd[:-1] < 0) & (macd[1:] >= 0)
                zero_cross_down[1:] = (macd[:-1] > 0) & (macd[1:] <= 0)
                
                df['MACD_ZeroCrossUp'] = zero_cross_up
                df['MACD_ZeroCrossDown'] = zero_cross_down
                
                # 计算MACD柱状图动量变化 - 修复.loc使用错误
                # 创建动量数组
                momentum = np.zeros(len(df))
                if len(hist) > 1:
                    momentum[1:] = hist[1:] - hist[:-1]
                df['MACD_Momentum'] = momentum
                
            return df
                
        except Exception as e:
            logging.error(f"计算MACD失败: {str(e)}")
            return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, rsi_high: int = 70, rsi_low: int = 30) -> pd.DataFrame:
        """
        计算相对强弱指数(RSI)，以及超买/超卖信号与趋势反转信号。
        
        参数:
            df: 股票数据DataFrame
            period: RSI周期，默认14
            rsi_high: 超买阈值，默认70
            rsi_low: 超卖阈值，默认30
            
        返回:
            添加了RSI指标的DataFrame
        """
        # 避免不必要的复制，只在需要时复制
        if not BaseIndicator.validate_dataframe(df, period):
            return df
            
        # 获取收盘价数组，避免重复访问
        close_array = df['Close'].values
        
        # 使用TA-Lib计算RSI指标
        rsi = ta.RSI(close_array, timeperiod=period)
        
        # 赋值
        df['RSI'] = rsi
            
        # 使用向量化操作计算超买超卖信号
        df['RSI_Overbought'] = df['RSI'] > rsi_high
        df['RSI_Oversold'] = df['RSI'] < rsi_low
            
        # 使用向量化操作计算RSI趋势反转信号
        if len(df) >= 2:
            # 初始化反转信号列
            rsi_overbought_exit = np.zeros(len(df), dtype=bool)
            rsi_oversold_exit = np.zeros(len(df), dtype=bool)
            
            # 计算反转信号 - 向量化操作
            if len(rsi) > 1:
                # 从超买区域跌出 - 可能是卖出信号
                rsi_overbought_exit[1:] = (rsi[:-1] > rsi_high) & (rsi[1:] <= rsi_high)
                
                # 从超卖区域升出 - 可能是买入信号
                rsi_oversold_exit[1:] = (rsi[:-1] < rsi_low) & (rsi[1:] >= rsi_low)
            
            # 设置反转信号列
            df['RSI_OverboughtExit'] = rsi_overbought_exit
            df['RSI_OversoldExit'] = rsi_oversold_exit
        
        return df
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        """
        计算KDJ随机指标。
        
        参数:
            df: 股票数据DataFrame
            period: KDJ计算周期，默认9
            
        返回:
            添加了KDJ指标的DataFrame，包含K、D、J三个值
        """
        result_df = df.copy()
        if not BaseIndicator.validate_dataframe(df, period):
            return result_df
            
        try:
            # 计算随机值RSV
            low_min = result_df['Low'].rolling(period).min()
            high_max = result_df['High'].rolling(period).max()
            rsv = (result_df['Close'] - low_min) / (high_max - low_min + Constants.EPSILON) * 100
            
            # 计算K、D、J值
            result_df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
            result_df['D'] = result_df['K'].ewm(alpha=1/3, adjust=False).mean()
            result_df['J'] = 3 * result_df['K'] - 2 * result_df['D']
            
            # 添加KDJ金叉死叉信号
            if len(result_df) >= 2:
                # 获取K和D值数组
                k_values = result_df['K'].values
                d_values = result_df['D'].values
                
                # 创建结果数组
                golden_cross = np.zeros(len(result_df), dtype=bool)
                death_cross = np.zeros(len(result_df), dtype=bool)
                
                # 使用数组索引计算信号
                golden_cross[1:] = (k_values[:-1] < d_values[:-1]) & (k_values[1:] >= d_values[1:])
                death_cross[1:] = (k_values[:-1] > d_values[:-1]) & (k_values[1:] <= d_values[1:])
                
                # 赋值给DataFrame
                result_df['KDJ_GoldenCross'] = golden_cross
                result_df['KDJ_DeathCross'] = death_cross
                
                # 添加超买超卖区域
                result_df['KDJ_Overbought'] = result_df['K'] > 80
                result_df['KDJ_Oversold'] = result_df['K'] < 20
                
        except Exception as e:
            logging.error(f"计算KDJ指标失败: {str(e)}")
            
        return result_df

class TrendIndicator(BaseIndicator):
    """
    趋势类技术指标。
    负责计算趋势相关的技术指标，如ADX、DMI等。
    """
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算平均趋向指数(ADX)，以及+DI和-DI值，用于判断趋势强度和方向。
        
        参数:
            df: 股票数据DataFrame
            period: ADX计算周期，默认14
            
        返回:
            添加了ADX指标的DataFrame
        """
        # 避免不必要的复制，只在必要时进行复制
        if not BaseIndicator.validate_dataframe(df, period * 2):
            return df
            
        # 获取价格数组
        high_array = df['High'].values
        low_array = df['Low'].values
        close_array = df['Close'].values
        
        # 使用TA-Lib计算ADX指标族
        adx_result = BaseIndicator.safe_talib_call(
            ta.ADX, high_array, low_array, close_array, timeperiod=period
        )
        
        plus_di_result = BaseIndicator.safe_talib_call(
            ta.PLUS_DI, high_array, low_array, close_array, timeperiod=period
        )
        
        minus_di_result = BaseIndicator.safe_talib_call(
            ta.MINUS_DI, high_array, low_array, close_array, timeperiod=period
        )
        
        if adx_result is None or plus_di_result is None or minus_di_result is None:
            return df
            
        # 赋值
        df['ADX'] = adx_result
        df['PLUS_DI'] = plus_di_result
        df['MINUS_DI'] = minus_di_result
            
        # 使用向量化操作计算趋势强度和方向
        # 趋势强度分类
        trend_strength = np.zeros(len(df), dtype=object)
        trend_strength[:] = 'No Trend'  # 默认无趋势
        
        # 根据ADX值确定趋势强度
        mask_weak = adx_result < 25
        mask_moderate = (adx_result >= 25) & (adx_result < 50)
        mask_strong = (adx_result >= 50) & (adx_result < 75)
        mask_very_strong = adx_result >= 75
        
        trend_strength[mask_weak] = 'Weak'
        trend_strength[mask_moderate] = 'Moderate'
        trend_strength[mask_strong] = 'Strong'
        trend_strength[mask_very_strong] = 'Very Strong'
        
        df['ADX_Strength'] = trend_strength
        
        # 使用向量化操作计算趋势方向
        df['ADX_Direction'] = np.where(plus_di_result > minus_di_result, 'Bullish', 'Bearish')
        
        # 计算+DI和-DI的交叉信号
        if len(df) >= 2:
            # 初始化信号数组
            di_crossover = np.zeros(len(df), dtype=bool)
            di_crossunder = np.zeros(len(df), dtype=bool)
            
            # 向量化操作计算交叉信号
            if len(plus_di_result) > 1 and len(minus_di_result) > 1:
                # +DI上穿-DI (看涨信号)
                di_crossover[1:] = (plus_di_result[:-1] <= minus_di_result[:-1]) & (plus_di_result[1:] > minus_di_result[1:])
                
                # +DI下穿-DI (看跌信号)
                di_crossunder[1:] = (plus_di_result[:-1] >= minus_di_result[:-1]) & (plus_di_result[1:] < minus_di_result[1:])
            
            # 设置交叉信号列
            df['DI_Crossover'] = di_crossover
            df['DI_Crossunder'] = di_crossunder
        
        return df

class VolumeIndicator(BaseIndicator):
    """
    成交量类技术指标。
    负责计算成交量相关的技术指标，如OBV、CMF等。
    """
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量相关指标。
        
        参数:
            df: 股票数据DataFrame
            
        返回:
            添加了成交量指标的DataFrame，包含:
            - Volume_MA5/10/20: 成交量移动平均
            - OBV: 能量潮指标
            - Volume_ROC: 成交量变化率
            - Volume_Trend: 成交量趋势
        """
        result_df = df.copy()
        if not BaseIndicator.validate_dataframe(df, 20):
            return result_df
            
        try:
            # 获取价格和成交量数据
            close_array = result_df['Close'].values
            volume_array = result_df['Volume'].values
            
            # 计算成交量移动平均
            for period in [5, 10, 20]:
                result_df[f'Volume_MA{period}'] = ta.SMA(volume_array, timeperiod=period)
            
            # 计算OBV (On-Balance Volume)
            result_df['OBV'] = ta.OBV(close_array, volume_array)
            
            # 计算成交量变化率 (Volume Rate of Change)
            result_df['Volume_ROC'] = ta.ROC(volume_array, timeperiod=1)
            
            # 计算成交量趋势
            result_df['Volume_Trend'] = np.where(
                result_df['Volume'] > result_df['Volume_MA20'], 
                1,  # 成交量高于20日均量
                -1  # 成交量低于20日均量
            )
            
            # 计算CMF (Chaikin Money Flow)
            if all(col in result_df.columns for col in ['High', 'Low']):
                high = result_df['High'].values
                low = result_df['Low'].values
                
                # Money Flow Multiplier
                mf_multiplier = ((close_array - low) - (high - close_array)) / (high - low + Constants.EPSILON)
                
                # Money Flow Volume
                mf_volume = mf_multiplier * volume_array
                
                # 计算20日CMF
                result_df['CMF'] = ta.SUM(mf_volume, 20) / ta.SUM(volume_array, 20)
                
            # 计算成交量强弱指标
            vol_ratio = result_df['Volume'] / result_df['Volume_MA20']
            result_df['Volume_Strength'] = np.where(
                vol_ratio > 2.0, 
                'Very Strong',
                np.where(
                    vol_ratio > 1.5, 
                    'Strong',
                    np.where(
                        vol_ratio > 1.0, 
                        'Above Average',
                        np.where(
                            vol_ratio > 0.5, 
                            'Below Average', 
                            'Weak'
                        )
                    )
                )
            )
            
            # 计算散户/机构资金流向 (基于成交量和价格变化方向)
            if len(result_df) >= 2:
                # 上涨日资金流入，下跌日资金流出
                result_df['Money_Flow'] = result_df['Volume'] * (result_df['Close'] - result_df['Close'].shift(1))
                
                # 计算10日资金流向
                result_df['Money_Flow_10d'] = result_df['Money_Flow'].rolling(10).sum()
                
                # 资金流向趋势
                result_df['Money_Flow_Trend'] = np.where(
                    result_df['Money_Flow_10d'] > 0, 
                    'Inflow',  # 资金净流入
                    'Outflow'  # 资金净流出
                )
            
        except Exception as e:
            logging.error(f"计算成交量指标失败: {str(e)}")
            
        return result_df

class TechnicalAnalysis:
    """
    整合所有技术指标的分析类。
    提供一站式技术分析接口。
    """
    
    def __init__(self):
        """初始化技术分析模块"""
        self.price_indicator = PriceIndicator()
        self.momentum_indicator = MomentumIndicator()
        self.trend_indicator = TrendIndicator()
        self.volume_indicator = VolumeIndicator()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标。
        
        参数:
            df: 股票数据DataFrame
            
        返回:
            添加了所有技术指标的DataFrame
        """
        if not BaseIndicator.validate_dataframe(df):
            return df.copy()
            
        result_df = df.copy()
        
        # 计算价格指标
        result_df = self.price_indicator.calculate_ma(result_df)
        result_df = self.price_indicator.calculate_ema(result_df)
        result_df = self.price_indicator.calculate_bollinger(result_df)
        
        # 计算动量指标
        result_df = self.momentum_indicator.calculate_macd(result_df)
        result_df = self.momentum_indicator.calculate_rsi(result_df)
        result_df = self.momentum_indicator.calculate_kdj(result_df)
        
        # 计算趋势指标
        result_df = self.trend_indicator.calculate_adx(result_df)
        
        # 计算成交量指标
        result_df = self.volume_indicator.calculate_volume_indicators(result_df)
        
        return result_df
    
    def calculate_selected_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """
        计算指定的技术指标。
        
        参数:
            df: 股票数据DataFrame
            indicators: 需要计算的指标列表，可选值包括:
                - "ma": 移动平均线
                - "ema": 指数移动平均线
                - "bollinger": 布林带
                - "macd": MACD指标
                - "rsi": RSI指标
                - "kdj": KDJ指标
                - "adx": ADX指标
                - "volume": 成交量指标
            
        返回:
            添加了指定技术指标的DataFrame
        """
        if not BaseIndicator.validate_dataframe(df):
            return df.copy()
            
        result_df = df.copy()
        
        indicator_map = {
            "ma": lambda df: self.price_indicator.calculate_ma(df),
            "ema": lambda df: self.price_indicator.calculate_ema(df),
            "bollinger": lambda df: self.price_indicator.calculate_bollinger(df),
            "macd": lambda df: self.momentum_indicator.calculate_macd(df),
            "rsi": lambda df: self.momentum_indicator.calculate_rsi(df),
            "kdj": lambda df: self.momentum_indicator.calculate_kdj(df),
            "adx": lambda df: self.trend_indicator.calculate_adx(df),
            "volume": lambda df: self.volume_indicator.calculate_volume_indicators(df),
        }
        
        for indicator in indicators:
            if indicator in indicator_map:
                result_df = indicator_map[indicator](result_df)
                
        return result_df

# =============================================================================
# 选股参数配置模块
# =============================================================================
class TradingParams:
    """交易参数类，集中管理所有交易策略参数"""
    
    def __init__(self):
        # 移动平均线参数
        self.ma_short = 5               # 短期均线周期
        self.ma_mid = 10                # 中期均线周期
        self.ma_long = 20               # 长期均线周期
        self.vol_ma_period = 5          # 成交量均线周期
        
        # MACD参数
        self.macd_fast = 12             # MACD快线周期
        self.macd_slow = 26             # MACD慢线周期
        self.macd_signal = 9            # MACD信号线周期
        
        # RSI参数
        self.rsi_period = 14            # RSI周期
        self.rsi_overbought = 70        # RSI超买阈值
        self.rsi_oversold = 30          # RSI超卖阈值
        self.short_term_rsi_range = (30, 70)  # 短线交易RSI范围
        
        # KDJ参数
        self.kdj_period = 9             # KDJ周期
        
        # ADX参数
        self.adx_period = 14            # ADX周期
        
        # CCI参数
        self.cci_period = 14            # CCI周期
        
        # 短线交易参数
        self.closing_strength_threshold = 0.6  # 收盘强度阈值
        self.short_term_volume_threshold = 1.2  # 短线交易量比阈值
        self.trend_weight = 0.6         # 趋势因子权重
        self.volume_weight = 0.4        # 量能因子权重
        self.market_weight = 0.3        # 市场因子权重
        self.next_day_up_prob_threshold = 0.9  # 次日上涨概率上限阈值

# =============================================================================
# 市场分析模块
# =============================================================================
class MarketAnalyzer:
    """
    市场分析器，计算市场趋势、情绪、强度和周期指标。
    提供全面的市场环境分析，为选股策略提供多维度的市场信息。
    """
    @staticmethod
    def get_market_trend(df: pd.DataFrame, period: int = 20) -> float:
        """
        计算市场趋势指标，基于移动平均线的变化率。
        正值表示上升趋势，负值表示下降趋势，数值大小表示趋势强度。
        """
        # 类型检查，确保df是DataFrame类型
        if not isinstance(df, pd.DataFrame):
            logging.error(f"MarketAnalyzer.get_market_trend: 输入类型错误，预期DataFrame，实际为{type(df)}")
            return 0.0
            
        if df is None or df.empty or len(df) < Constants.MIN_DATA_POINTS:
            return 0.0
        try:
            # 确保'Close'列存在
            if 'Close' not in df.columns:
                logging.error("MarketAnalyzer.get_market_trend: 数据中缺少'Close'列")
                return 0.0
                
            # 计算移动平均线
            ma_values = np.array(df['Close'].rolling(period).mean(), dtype=np.float64)
            if len(ma_values) > 3:
                # 计算移动平均线的变化率 - 忽略前3个移动平均线值
                recent_ma = ma_values[-5:]
                if not np.isnan(recent_ma).all():
                    # 计算相对变化率，而不是绝对变化
                    pct_changes = np.diff(recent_ma) / recent_ma[:-1]
                    # 过滤掉NaN值并计算平均变化率
                    valid_changes = pct_changes[~np.isnan(pct_changes)]
                    if len(valid_changes) > 0:
                        return float(np.mean(valid_changes))
            return 0.0
        except Exception as e:
            logging.error(f"MarketAnalyzer.get_market_trend: 计算出错 - {str(e)}")
            return 0.0

    @staticmethod
    def get_market_sentiment(df: pd.DataFrame) -> float:
        """
        计算市场情绪指标，基于价格波动率、成交量变化趋势和价格动量的综合分析。
        高值表示市场情绪活跃，低值表示市场情绪低迷。
        
        优化点：
        1. 考虑短期和中期价格波动率，更全面反映市场情绪
        2. 分析成交量与价格变化的协同性，识别有效放量
        3. 引入价格动量因子，捕捉短期趋势变化
        4. 动态调整各因子权重，提高短线预测准确性
        """
        # 类型检查，确保df是DataFrame类型
        if not isinstance(df, pd.DataFrame):
            logging.error(f"MarketAnalyzer.get_market_sentiment: 输入类型错误，预期DataFrame，实际为{type(df)}")
            return 0.0
            
        if df is None or df.empty or len(df) < Constants.MIN_DATA_POINTS:
            return 0.0
        try:
            # 确保必要的列存在
            if 'Close' not in df.columns or 'Volume' not in df.columns:
                logging.error("MarketAnalyzer.get_market_sentiment: 数据中缺少必要的列")
                return 0.0
                
            # 1. 计算短期和中期价格波动率 - 短期波动更能反映当前情绪
            volatility_short = df['Close'].pct_change().rolling(5).std().iloc[-1] * 100  # 5日波动率
            volatility_mid = df['Close'].pct_change().rolling(10).std().iloc[-1] * 100   # 10日波动率
            
            # 波动率因子 - 短期波动权重更高
            volatility_factor = volatility_short * 0.7 + volatility_mid * 0.3
            
            # 2. 计算成交量与价格变化的协同性 - 量价配合是有效市场的特征
            price_change = df['Close'].pct_change()
            volume_change = df['Volume'].pct_change()
            
            # 计算最近5日的量价协同性 - 正相关表示情绪一致
            recent_corr = price_change.tail(5).corr(volume_change.tail(5))
            # 处理NaN值
            volume_price_sync = 0.5 if np.isnan(recent_corr) else (recent_corr + 1) / 2  # 归一化到0-1
            
            # 3. 计算价格动量 - 短期价格趋势
            price_momentum = df['Close'].pct_change(3).iloc[-1]  # 3日价格动量
            momentum_factor = (price_momentum + 0.05) * 10  # 归一化并放大
            momentum_factor = max(0, min(momentum_factor, 2))  # 限制在0-2范围内
            
            # 4. 计算成交量趋势 - 成交量变化率
            volume_trend = df['Volume'].pct_change(5).mean()  # 5日成交量变化趋势
            volume_factor = volume_trend * 5 if volume_trend > 0 else volume_trend * 2  # 放量上涨权重更高
            volume_factor = max(-1, min(volume_factor, 2))  # 限制在-1到2范围内
            
            # 5. 综合计算市场情绪 - 动态权重分配
            # 波动率权重：30%，量价协同权重：25%，动量权重：25%，成交量趋势权重：20%
            sentiment = (volatility_factor * 0.3 + 
                        volume_price_sync * 0.25 + 
                        momentum_factor * 0.25 + 
                        volume_factor * 0.2)
            
            return sentiment
        except Exception as e:
            logging.error(f"MarketAnalyzer.get_market_sentiment: 计算出错 - {str(e)}")
            return 0.0
            
    @staticmethod
    def get_market_strength(df: pd.DataFrame, window: int = 20) -> float:
        """
        计算市场强度指标，综合考虑价格趋势、成交量和波动性。
        正值表示市场强势，负值表示市场弱势，数值大小表示强度。
        """
        # 类型检查，确保df是DataFrame类型
        if not isinstance(df, pd.DataFrame):
            logging.error(f"MarketAnalyzer.get_market_strength: 输入类型错误，预期DataFrame，实际为{type(df)}")
            return 0.0
            
        if df is None or df.empty or len(df) < Constants.MIN_DATA_POINTS:
            return 0.0
            
        try:
            # 确保必要的列存在
            if 'Close' not in df.columns or 'Volume' not in df.columns:
                logging.error("MarketAnalyzer.get_market_strength: 数据中缺少必要的列")
                return 0.0
                
            # 1. 计算价格趋势因子 - 使用短期均线斜率
            if len(df) >= window:
                # 计算短期均线
                ma_short = df['Close'].rolling(window=window//2).mean().values
                # 计算均线斜率 - 使用最近5个点
                if len(ma_short) >= 5:
                    recent_ma = ma_short[-5:]
                    # 使用线性回归计算斜率
                    x = np.arange(len(recent_ma))
                    if not np.isnan(recent_ma).all():
                        # 处理可能的NaN值
                        valid_indices = ~np.isnan(recent_ma)
                        if np.sum(valid_indices) >= 2:  # 至少需要2个点才能计算斜率
                            x_valid = x[valid_indices]
                            y_valid = recent_ma[valid_indices]
                            # 使用polyfit计算斜率
                            if len(x_valid) > 0 and len(y_valid) > 0:
                                slope = np.polyfit(x_valid, y_valid, 1)[0]
                                # 归一化斜率 - 除以均线均值
                                mean_price = np.nanmean(recent_ma)
                                if mean_price > 0:
                                    trend_factor = slope / mean_price * 100
                                else:
                                    trend_factor = 0.0
                            else:
                                trend_factor = 0.0
                        else:
                            trend_factor = 0.0
                    else:
                        trend_factor = 0.0
                else:
                    trend_factor = 0.0
            else:
                trend_factor = 0.0
                
            # 2. 计算成交量支撑因子 - 成交量变化与价格变化的协同性
            volume_support = 0.0
            if len(df) >= 5:
                # 计算最近5天的价格变化和成交量变化
                price_change = df['Close'].pct_change().tail(5).values
                volume_change = df['Volume'].pct_change().tail(5).values
                
                # 计算量价协同性 - 价格上涨时成交量增加，或价格下跌时成交量减少
                valid_indices = ~(np.isnan(price_change) | np.isnan(volume_change))
                if np.sum(valid_indices) > 0:
                    # 计算量价同向变化的比例
                    price_up = price_change[valid_indices] > 0
                    volume_up = volume_change[valid_indices] > 0
                    sync_ratio = np.mean((price_up & volume_up) | (~price_up & ~volume_up))
                    # 归一化到-1到1之间，0表示无关联
                    volume_support = (sync_ratio - 0.5) * 2
            
            # 3. 计算波动性因子 - 短期波动率相对于长期波动率
            volatility_factor = 0.0
            if len(df) >= window:
                # 计算短期和长期波动率
                volatility_short = df['Close'].pct_change().rolling(window=window//4).std().iloc[-1]
                volatility_long = df['Close'].pct_change().rolling(window=window).std().iloc[-1]
                
                # 计算相对波动率 - 短期/长期，大于1表示波动加剧
                if not np.isnan(volatility_short) and not np.isnan(volatility_long) and volatility_long > 0:
                    rel_volatility = volatility_short / volatility_long
                    # 归一化到-1到1之间，1表示短期波动明显高于长期
                    volatility_factor = np.clip((rel_volatility - 1) * 2, -1, 1)
            
            # 4. 综合计算市场强度 - 加权平均
            # 趋势因子权重最高，其次是成交量支撑，波动性权重最低
            strength = trend_factor * 0.6 + volume_support * 0.3 + volatility_factor * 0.1
            
            # 限制在合理范围内，避免极端值
            return float(np.clip(strength, -10, 10))
            
        except Exception as e:
            logging.error(f"MarketAnalyzer.get_market_strength: 计算出错 - {str(e)}")
            return 0.0
    
    @staticmethod
    def identify_market_cycle(df: pd.DataFrame) -> str:
        """
        识别市场周期，基于价格趋势、波动率和技术指标组合。
        返回市场周期阶段：上升、下降、盘整、反转。
        """
        # 类型检查，确保df是DataFrame类型
        if not isinstance(df, pd.DataFrame):
            logging.error(f"MarketAnalyzer.identify_market_cycle: 输入类型错误，预期DataFrame，实际为{type(df)}")
            return "未知"
            
        if df is None or df.empty or len(df) < Constants.MIN_DATA_POINTS:
            return "未知"
            
        try:
            # 计算市场趋势和强度
            trend = MarketAnalyzer.get_market_trend(df)
            strength = MarketAnalyzer.get_market_strength(df)
            
            # 最近的价格数据
            recent_prices = df['Close'].tail(20)
            
            # 计算短期和长期移动平均线
            short_ma = df['Close'].rolling(window=5).mean()
            long_ma = df['Close'].rolling(window=20).mean()
            
            # 计算波动率
            volatility = recent_prices.pct_change().std() * 100
            
            # 计算RSI (如果数据中没有)
            if 'RSI' not in df.columns and len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean().replace(0, Constants.EPSILON)
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                latest_rsi = rsi.iloc[-1]
            else:
                latest_rsi = df.get('RSI', pd.Series([50])).iloc[-1]
            
            # 判断市场周期
            if trend > 0.005:  # 上升趋势
                if strength > 3:
                    return "快速上升"
                else:
                    return "稳步上升"
            elif trend < -0.005:  # 下降趋势
                if strength < -3:
                    return "快速下降"
                else:
                    return "稳步下降"
            elif abs(short_ma.iloc[-1] / long_ma.iloc[-1] - 1) < 0.02:
                if latest_rsi > 50:
                    return "盘整后上行"
                else:
                    return "盘整后下行"
            elif short_ma.iloc[-5] < short_ma.iloc[-1] and short_ma.iloc[-10] > short_ma.iloc[-5]:
                return "筑底反转"
            elif short_ma.iloc[-5] > short_ma.iloc[-1] and short_ma.iloc[-10] < short_ma.iloc[-5]:
                return "见顶回落"
            else:
                return "盘整"
        except Exception as e:
            logging.error(f"MarketAnalyzer.identify_market_cycle: 计算出错 - {str(e)}")
            return "未知"

# =============================================================================
# 技术指标总控模块（门面模式）
# =============================================================================
class TechnicalIndicators:
    """
    综合技术指标计算与选股逻辑总控模块，
    协调数据获取、技术指标计算、市场分析与短线交易信号生成。
    """
    def __init__(self):
        self.base_dir = Constants.BASE_DIR
        self.s_rD = Datasr()
        # 使用新的技术分析类替代原来的计算器
        self.technical_analysis = TechnicalAnalysis()
        # 使用统一的日志设置函数
        log_file = os.path.join(self.base_dir, 'stock_r_indicators.log')
        self.logger = get_logger('TechnicalIndicators', log_file, logging.DEBUG, show_in_console=False)

    def _filter_stock_code(self, code: str) -> bool:
        """
        过滤股票代码：保留有上涨潜力的股票。
        
        参数:
            code: 股票代码
            
        返回:
            bool: 如果股票代码满足筛选条件，返回True；否则返回False
        """
        code = str(code).strip()
        
        # 1. 首先排除ST股票、退市股票和B股
        stock_name = self.s_rD.get_stock_name(code)
        if not stock_name:
            return False
            
        if ('ST' in stock_name or 'st' in stock_name or '*' in stock_name or 
            '退' in stock_name or 'B' == stock_name[0]):
            return False
        
        # 2. 根据股票代码前缀进行过滤

        valid_prefixes = ['00', '60']
        
        for prefix in valid_prefixes:
            if code.startswith(prefix):
                return True
                
        return False

    def get_target_stocks(self) -> List[List[str]]:
        """
        获取目标股票列表，返回格式为 [代码, 名称] 的二维列表，
        只包含过过滤的股票
        """
        stock_list = self.s_rD.get_stock_list()
        if not stock_list:
            return []
        filtered_stocks = [[code, name] for code, name in stock_list.items() if self._filter_stock_code(code)]
        self.logger.info(f"成功获取 {len(filtered_stocks)} 只目标股票")
        return filtered_stocks

    def calculate_all(self, df: pd.DataFrame, params: Optional[TradingParams] = None) -> pd.DataFrame:
        """
        计算所有技术指标和短线信号，包括：
         1. 预处理数据（统一列名、数值类型转换）
         2. 计算移动均线、MACD、RSI、KDJ、布林带及成交量指标
         3. 计算短线交易专用指标（如收盘强度、量比、短线买入信号）
         4. 计算市场趋势和情绪指标
         5. 计算趋势强度和反转信号
         6. 计算CCI指标和K线形态识别指标
         7. 计算涨停概率指标
         8. 填充缺失值
         
        返回添加了各项技术指标的DataFrame，用于后续选股评分。
        """
        # 类型检查，确保df是DataFrame类型
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"输入类型错误: 预期DataFrame，实际为{type(df)}")
            return pd.DataFrame()
            
        # 参数初始化
        if params is None:
            params = TradingParams()
        if df is None or df.empty:
            return pd.DataFrame()
            
        # 1. 数据预处理 - 统一列名和数据类型
        df = self._preprocess_data(df)
        
        # 定义必要指标的默认值
        essential_indicators = {
            'RSI': Constants.DEFAULT_RSI,
            'MACD': 0.0,
            'MACD_Signal': 0.0,
            'MACD_Hist': 0.0,
            'K': 50.0,
            'D': 50.0,
            'J': 50.0,
            'MA5': 0.0,
            'MA10': 0.0,
            'MA20': 0.0,
            'MA60': 0.0,
            'VOL_MA5': 0.0,
            'Trend': Constants.DEFAULT_TREND,
            'Strength': Constants.DEFAULT_STRENGTH,
            'ADX': 20.0,
            'CCI': 0.0,
            'MFI': 50.0
        }
        
        try:
            self.logger.info(f"开始计算技术指标，数据长度: {len(df)}")
            
            # 2.1 计算移动均线 - 短中长期均线系统（一次性计算所有均线）
            try:
                self.logger.info("计算移动均线...")
                self._calculate_moving_averages(df, [5, 10, 20, 60])
                self.logger.info("移动均线计算完成")
            except Exception as e:
                self.logger.error(f"计算移动均线失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.2 计算MACD指标 - 趋势指标
            try:
                self.logger.info("计算MACD指标...")
                df = self._safe_calculate(
                    self.technical_analysis.momentum_indicator.calculate_macd,
                    df,
                    "MACD",
                    fast_period=params.macd_fast,
                    slow_period=params.macd_slow,
                    signal_period=params.macd_signal
                )
                self.logger.info("MACD指标计算完成")
            except Exception as e:
                self.logger.error(f"计算MACD指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.3 计算RSI指标 - 超买超卖指标
            try:
                self.logger.info("计算RSI指标...")
                df = self._safe_calculate(
                    self.technical_analysis.momentum_indicator.calculate_rsi,
                    df,
                    "RSI",
                    period=params.rsi_period,
                    rsi_high=params.rsi_overbought,
                    rsi_low=params.rsi_oversold
                )
                self.logger.info("RSI指标计算完成")
            except Exception as e:
                self.logger.error(f"计算RSI指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.4 计算KDJ指标 - 随机指标
            try:
                self.logger.info("计算KDJ指标...")
                df = self._safe_calculate(
                    self.technical_analysis.momentum_indicator.calculate_kdj,
                    df,
                    "KDJ",
                    period=params.kdj_period
                )
                self.logger.info("KDJ指标计算完成")
            except Exception as e:
                self.logger.error(f"计算KDJ指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.5 计算布林带指标 - 波动率指标
            try:
                self.logger.info("计算布林带指标...")
                df = self._safe_calculate(
                    self.technical_analysis.price_indicator.calculate_bollinger,
                    df,
                    "布林带",
                    period=20,
                    std_dev=2.0
                )
                self.logger.info("布林带指标计算完成")
            except Exception as e:
                self.logger.error(f"计算布林带指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.6 计算成交量指标 - 量能指标
            try:
                self.logger.info("计算成交量指标...")
                self._calculate_volume_indicators(df, params.vol_ma_period)
                self.logger.info("成交量指标计算完成")
            except Exception as e:
                self.logger.error(f"计算成交量指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.7 计算ADX指标 - 趋势指标
            try:
                self.logger.info("计算ADX指标...")
                df = self._safe_calculate(
                    self.technical_analysis.trend_indicator.calculate_adx,
                    df,
                    "ADX",
                    period=params.adx_period
                )
                self.logger.info("ADX指标计算完成")
            except Exception as e:
                self.logger.error(f"计算ADX指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
            # 2.8 计算CCI指标 - 顺势指标
            try:
                self.logger.info("计算CCI指标...")
                df = self._safe_calculate(
                    self._calculate_cci,
                    df,
                    "CCI",
                    period=params.cci_period
                )
                self.logger.info("CCI指标计算完成")
            except Exception as e:
                self.logger.error(f"计算CCI指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 2.9 计算MFI指标 - 资金流量指标
            try:
                self.logger.info("计算MFI指标...")
                if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
                    high_array = df['High'].astype(np.float64).values
                    low_array = df['Low'].astype(np.float64).values
                    close_array = df['Close'].astype(np.float64).values
                    volume_array = df['Volume'].astype(np.float64).values
                    
                    # 使用TA-Lib计算MFI
                    mfi = BaseIndicator.safe_talib_call(ta.MFI, high_array, low_array, close_array, volume_array, timeperiod=14)
                    
                    if mfi is not None:
                        df['MFI'] = mfi
                        # 添加MFI指标到必要指标列表，确保填充时不为空
                        essential_indicators['MFI'] = 50.0
                    self.logger.info("MFI指标计算完成")
                else:
                    self.logger.warning("计算MFI需要High、Low、Close和Volume列")
            except Exception as e:
                self.logger.error(f"计算MFI指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

            # 3. 计算短线交易专用指标
            try:
                self.logger.info("计算短线交易指标开始")
                df = self._calculate_short_term_indicators(df, params)
                self.logger.info("短线交易指标计算完成")
            except Exception as e:
                self.logger.error(f"计算短线交易指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 设置默认值
                for col, val in {'收盘强度': 0.5, '次日上涨概率': 0.5, '短线买入信号': False}.items():
                    if col not in df.columns:
                        df[col] = val
            
            # 4. 计算市场趋势和情绪指标 - 使用MarketAnalyzer类
            try:
                self.logger.info("计算市场趋势和情绪指标...")
                self._calculate_market_indicators(df)
                self.logger.info("市场趋势和情绪指标计算完成")
            except Exception as e:
                self.logger.error(f"计算市场趋势和情绪指标失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 5. 计算趋势强度和反转信号
            try:
                self.logger.info("计算趋势强度和反转信号...")
                self._calculate_trend_and_reversal(df)
                self.logger.info("趋势强度和反转信号计算完成")
            except Exception as e:
                self.logger.error(f"计算趋势强度和反转信号失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
            # 6. 计算K线形态识别
            try:
                self.logger.info("识别K线形态...")
                df = self._calculate_candlestick_patterns(df)
                self.logger.info("K线形态识别完成")
            except Exception as e:
                self.logger.error(f"识别K线形态失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
            # 7. 计算涨停概率
            try:
                self.logger.info("计算涨停概率...")
                self._calculate_limit_up_probability(df)
                self.logger.info("涨停概率计算完成")
            except Exception as e:
                self.logger.error(f"计算涨停概率失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 确保涨停概率列存在，即使计算失败
                if '涨停概率' not in df.columns:
                    df['涨停概率'] = 0.0
                if '涨停风险' not in df.columns:
                    df['涨停风险'] = '低'
            
            # 8. 填充缺失值并返回结果
            try:
                self.logger.info("填充缺失值...")
                self._fill_missing_values(df, essential_indicators)
                self.logger.info("填充缺失值完成")
            except Exception as e:
                self.logger.error(f"填充缺失值失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
            self.logger.info(f"技术指标计算完成，共计算{len(df.columns)-5}个指标")
            # 去除DataFrame的碎片化，解决性能警告
            df = df.copy()
            return df
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            for col, val in essential_indicators.items():
                if col not in df.columns:
                    df[col] = val
            return df
            
    def _calculate_moving_averages(self, df: pd.DataFrame, periods: List[int]) -> None:
        """
        计算多个周期的移动平均线，使用TA-Lib
        """
        if df is None or df.empty:
            return
            
        # 获取收盘价数组，避免重复访问
        close_array = df['Close'].values
            
        # 使用TA-Lib一次性计算所有均线，避免循环中多次调用rolling
        for period in periods:
            if len(df) >= period:
                df[f'MA{period}'] = ta.SMA(close_array, timeperiod=period)

    def _calculate_volume_indicators(self, df: pd.DataFrame, vol_ma_period: int) -> None:
        """
        计算成交量相关指标，使用TA-Lib和向量化操作
        """
        if df is None or df.empty or 'Volume' not in df.columns or len(df) < vol_ma_period:
            return
            
        try:
            # 获取成交量数组，避免重复访问，并确保转换为double类型
            volume_array = df['Volume'].values.astype(np.float64)
                
            # 使用TA-Lib计算成交量移动平均
            df['VOL_MA5'] = ta.SMA(volume_array, timeperiod=vol_ma_period)
            df['VOL_MA20'] = ta.SMA(volume_array, timeperiod=20)
            
            # 量比计算（当日成交量/5日平均成交量）- 使用向量化操作
            df['量比'] = volume_array / (df['VOL_MA5'].replace(0, Constants.EPSILON).values)
            
            # 量能趋势 - 计算成交量变化率，使用向量化操作
            df['量能趋势'] = np.zeros(len(df))
            if len(df) > 5:
                volume_trend = (volume_array[5:] / (volume_array[:-5] + Constants.EPSILON)) - 1
                df.iloc[5:, df.columns.get_loc('量能趋势')] = volume_trend
        except Exception as e:
            self.logger.error(f"计算成交量指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _calculate_market_indicators(self, df: pd.DataFrame) -> None:
        """
        计算市场趋势、情绪、强度和周期指标
        """
        try:
            # 确保df是DataFrame类型
            if not isinstance(df, pd.DataFrame):
                self.logger.error(f"数据类型错误: 预期DataFrame，实际为{type(df)}")
                df = pd.DataFrame() if df is None else pd.DataFrame(df)
                
            # 使用MarketAnalyzer计算市场指标
            market_trend = MarketAnalyzer.get_market_trend(df)
            market_sentiment = MarketAnalyzer.get_market_sentiment(df)
            market_strength = MarketAnalyzer.get_market_strength(df)
            market_cycle = MarketAnalyzer.identify_market_cycle(df)
            
            # 添加到数据框，确保值是有效的数值
            df['MarketTrend'] = market_trend if market_trend is not None else 0.0
            df['MarketSentiment'] = market_sentiment if market_sentiment is not None else 0.0
            df['MarketStrength'] = market_strength if market_strength is not None else 0.0
            df['MarketCycle'] = market_cycle if market_cycle is not None else "未知"
            
            self.logger.info(f"市场指标 - 趋势: {market_trend:.4f}, 情绪: {market_sentiment:.4f}, "  
                            f"强度: {market_strength:.4f}, 周期: {market_cycle}")
        except Exception as e:
            self.logger.error(f"计算市场指标失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if isinstance(df, pd.DataFrame):
                df['MarketTrend'] = 0.0
                df['MarketSentiment'] = 0.0
                df['MarketStrength'] = 0.0
                df['MarketCycle'] = "未知"
            
    def _calculate_trend_and_reversal(self, df: pd.DataFrame) -> None:
        """
        计算趋势强度和反转信号 - 优化版
        
        增强功能：
        1. 多维度趋势强度评估，包括短中长期均线关系
        2. 更全面的反转信号识别，包括技术指标、K线形态和量价特征
        3. 动态权重分配，根据市场环境调整各信号权重
        4. 反转信号强度分级，提供更精细的交易决策参考
        """
        try:
            # 1. 趋势强度计算 - 多维度评估
            # 1.1 基于价格与均线的关系
            if all(f'MA{period}' in df.columns for period in [5, 10, 20]):
                # 计算价格与各周期均线的距离
                df['MA5距离'] = (df['Close'] / df['MA5'] - 1) * 100
                df['MA10距离'] = (df['Close'] / df['MA10'] - 1) * 100
                df['MA20距离'] = (df['Close'] / df['MA20'] - 1) * 100
                
                # 如果存在MA60则使用，否则跳过
                if 'MA60' in df.columns:
                    df['MA60距离'] = (df['Close'] / df['MA60'] - 1) * 100
                    # 1.2 均线系统评估 - 多头排列、空头排列
                    df['均线多头'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) & (df['MA20'] > df['MA60'])
                    df['均线空头'] = (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20']) & (df['MA20'] < df['MA60'])
                    
                    # 1.3 综合趋势强度 - 短期均线权重更高，更适合短线交易
                    df['趋势强度'] = df['MA5距离'] * 0.4 + df['MA10距离'] * 0.3 + \
                                  df['MA20距离'] * 0.2 + df['MA60距离'] * 0.1
                else:
                    # 不使用MA60的版本
                    df['均线多头'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
                    df['均线空头'] = (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20'])
                    
                    # 不使用MA60的趋势强度计算
                    df['趋势强度'] = df['MA5距离'] * 0.5 + df['MA10距离'] * 0.3 + df['MA20距离'] * 0.2
                
                # 1.4 趋势状态判断
                df['趋势状态'] = '盘整'
                df.loc[df['趋势强度'] > 2, '趋势状态'] = '上升'
                df.loc[df['趋势强度'] > 5, '趋势状态'] = '强势上升'
                df.loc[df['趋势强度'] < -2, '趋势状态'] = '下降'
                df.loc[df['趋势强度'] < -5, '趋势状态'] = '强势下降'
            
            # 2. 反转信号识别 - 多维度信号
            # 初始化反转信号列
            df['反转信号'] = False
            df['反转强度'] = 0.0
            
            if all(col in df.columns for col in ['RSI', 'MACD', 'MACD_Signal']):
                # 2.1 RSI相关反转信号
                # 获取RSI数组，避免shift操作
                rsi_values = df['RSI'].values
                
                # RSI超卖反弹 - 从超卖区回升，使用索引而非shift
                rsi_oversold_rebound = np.zeros(len(df), dtype=bool)
                if len(df) > 1:
                    rsi_oversold_rebound[1:] = (rsi_values[:-1] < 30) & (rsi_values[1:] >= 30)
                
                # RSI背离 - RSI上升但价格下跌，可能是底部信号
                rsi_divergence = np.zeros(len(df), dtype=bool)
                if len(df) > 5:
                    close_values = df['Close'].values
                    # 计算5日差值
                    if len(close_values) >= 6:
                        price_diff = np.zeros(len(df))
                        price_diff[5:] = close_values[5:] - close_values[:-5]
                        
                        rsi_diff = np.zeros(len(df))
                        rsi_diff[5:] = rsi_values[5:] - rsi_values[:-5]
                        
                        price_down = price_diff < 0
                        rsi_up = rsi_diff > 0
                        rsi_divergence = price_down & rsi_up
                
                # 2.2 MACD相关反转信号
                # 获取MACD和Signal数组
                macd_values = df['MACD'].values
                macd_signal_values = df['MACD_Signal'].values
                
                # MACD金叉 - MACD线上穿信号线，使用索引而非shift
                macd_golden = np.zeros(len(df), dtype=bool)
                if len(df) > 1:
                    macd_golden[1:] = (macd_values[:-1] < macd_signal_values[:-1]) & \
                                     (macd_values[1:] >= macd_signal_values[1:])
                
                # MACD零轴突破 - MACD线从负转正，使用索引而非shift
                macd_zero_cross = np.zeros(len(df), dtype=bool)
                if len(df) > 1:
                    macd_zero_cross[1:] = (macd_values[:-1] < 0) & (macd_values[1:] >= 0)
                
                # 2.3 价格形态反转信号
                # 价格突破均线，使用索引而非shift
                price_break_ma20 = np.zeros(len(df), dtype=bool)
                if 'MA20' in df.columns and len(df) > 1:
                    close_values = df['Close'].values
                    ma20_values = df['MA20'].values
                    price_break_ma20[1:] = (close_values[:-1] < ma20_values[:-1]) & \
                                         (close_values[1:] >= ma20_values[1:])
                
                # 锤子线形态 - 下影线长，上影线短，实体小
                hammer_pattern = np.zeros(len(df), dtype=bool)
                if '锤子线' in df.columns:
                    hammer_pattern = df['锤子线'].values
                
                # 2.4 量价配合反转信号
                # 放量突破 - 价格上涨且成交量放大，使用索引而非shift
                volume_breakout = np.zeros(len(df), dtype=bool)
                if '量比' in df.columns and len(df) > 1:
                    close_values = df['Close'].values
                    vol_ratio_values = df['量比'].values
                    
                    # 价格上涨
                    price_up = np.zeros(len(df), dtype=bool)
                    price_up[1:] = close_values[1:] > close_values[:-1]
                    
                    # 成交量放大
                    volume_up = vol_ratio_values > 1.5
                    
                    volume_breakout = price_up & volume_up
                
                # 2.5 综合反转信号 - 各信号加权
                # 短线交易中，RSI和MACD信号权重更高
                df['反转信号'] = rsi_oversold_rebound | macd_golden | price_break_ma20 | \
                               hammer_pattern | volume_breakout | rsi_divergence | macd_zero_cross
                
                # 2.6 反转强度计算 - 动态权重分配
                # 根据市场环境调整权重：下跌市场中RSI权重更高，盘整市场中MACD权重更高
                market_weight = 1.0
                if 'MarketTrend' in df.columns:
                    # 下跌市场中提高RSI权重
                    market_weight = np.where(df['MarketTrend'] < 0, 1.5, 1.0)
                
                # 计算反转强度 - 各信号加权求和
                df['反转强度'] = rsi_oversold_rebound.astype(int) * (0.4 * market_weight) + \
                               macd_golden.astype(int) * 0.3 + \
                               price_break_ma20.astype(int) * 0.2 + \
                               hammer_pattern.astype(int) * 0.2 + \
                               volume_breakout.astype(int) * 0.2 + \
                               rsi_divergence.astype(int) * 0.3 + \
                               macd_zero_cross.astype(int) * 0.2
                
                # 2.7 反转信号分级
                df['反转信号级别'] = 0
                df.loc[df['反转强度'] > 0.3, '反转信号级别'] = 1  # 弱反转信号
                df.loc[df['反转强度'] > 0.6, '反转信号级别'] = 2  # 中等反转信号
                df.loc[df['反转强度'] > 0.9, '反转信号级别'] = 3  # 强烈反转信号
            
            self.logger.info("趋势和反转指标计算完成")
        except Exception as e:
            self.logger.error(f"计算趋势强度和反转信号失败: {e}")
            
    def _fill_missing_values(self, df: pd.DataFrame, essential_indicators: Dict[str, float]) -> None:
        """
        填充缺失的指标值
        使用智能插值方法处理缺失值，根据指标特性选择合适的填充策略
        """
        if df is None or df.empty:
            return
        
        # 分类指标
        oscillator_indicators = ['RSI', 'CCI', 'MFI']  # 震荡指标通常在中间值
        trend_indicators = ['ADX', 'MarketTrend', 'DMI_Trend'] # 趋势指标
        momentum_indicators = ['MACD', 'MACD_Hist', 'MACD_Signal'] # 动量指标
        volume_indicators = ['OBV', 'CMF', 'MFI', 'Volume_MA']  # 成交量指标
            
        # 填充已存在列的缺失值
        existing_cols = [col for col in essential_indicators.keys() if col in df.columns]
        if existing_cols:
            for col in existing_cols:
                if df[col].isna().any():  # 只处理有缺失值的列
                    # 1. 尝试线性插值
                    filled_values = df[col].interpolate(method='linear')
                    
                    # 2. 处理头尾可能仍然缺失的值 - 修复废弃警告
                    filled_values = filled_values.ffill().bfill()
                    
                    # 3. 如果仍有缺失值，使用基于指标特性的智能默认值
                    if filled_values.isna().any():
                        if col in oscillator_indicators:
                            filled_values = filled_values.fillna(50)  # 震荡指标默认中性值
                        elif col in trend_indicators:
                            filled_values = filled_values.fillna(25)  # 趋势指标默认弱趋势
                        elif col in momentum_indicators:
                            filled_values = filled_values.fillna(0)  # 动量指标默认无动量
                        elif col in volume_indicators:
                            # 成交量指标使用均值填充
                            non_na_mean = df[col].dropna().mean()
                            filled_values = filled_values.fillna(non_na_mean if not np.isnan(non_na_mean) else Constants.DEFAULT_FILL_VALUE)
                        else:
                            filled_values = filled_values.fillna(essential_indicators.get(col, Constants.DEFAULT_FILL_VALUE))
                    
                    # 更新DataFrame
                    df[col] = filled_values
        
        # 添加缺失的列 - 根据指标特性设置默认值
        missing_cols = [col for col in essential_indicators.keys() if col not in df.columns]
        for col in missing_cols:
            if col in oscillator_indicators:
                df[col] = essential_indicators.get(col, 50)  # 震荡指标默认中性值
            elif col in trend_indicators:
                df[col] = essential_indicators.get(col, 25)  # 趋势指标默认弱趋势
            elif col in momentum_indicators:
                df[col] = essential_indicators.get(col, 0)   # 动量指标默认无动量
            else:
                df[col] = essential_indicators.get(col, Constants.DEFAULT_FILL_VALUE)

    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理：统一列名、转换数值类型、填充缺失值。
        """
        df = df.copy()
        column_mapping = {
            '开盘': 'Open', 'open': 'Open',
            '最高': 'High', 'high': 'High',
            '最低': 'Low', 'low': 'Low',
            '收盘': 'Close', 'close': 'Close',
            '成交量': 'Volume', 'volume': 'Volume', 'vol': 'Volume',
            '日期': 'Date', 'date': 'Date', 'time': 'Date',
            '换手率': 'Turnover', 'turnover': 'Turnover',
            '涨跌幅': 'Change', 'change': 'Change'
        }
        for orig, target in column_mapping.items():
            if orig in df.columns and target not in df.columns:
                df.rename(columns={orig: target}, inplace=True)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if not df.empty:
            existing_numeric = [col for col in numeric_cols if col in df.columns]
            if existing_numeric:
                df[existing_numeric] = df[existing_numeric].ffill().fillna(Constants.DEFAULT_FILL_VALUE)
        return df

    @staticmethod
    def _calculate_short_term_indicators(df: pd.DataFrame, params: TradingParams) -> pd.DataFrame:
        """
        短线交易指标计算 - 包含多个预测因子
        
        参数:
            df: 数据DataFrame
            params: 交易参数配置
            
        返回:
            添加了短线交易指标的DataFrame
        """
        # 如果数据不足，直接返回
        if df is None or df.empty or len(df) < Constants.MIN_DATA_POINTS:
            return df
            
        try:
            # 提取基础数据并转换为NumPy数组，提高计算效率
            high = np.ascontiguousarray(df['High'].values, dtype=np.float64)
            low = np.ascontiguousarray(df['Low'].values, dtype=np.float64)
            close = np.ascontiguousarray(df['Close'].values, dtype=np.float64)
            open_price = np.ascontiguousarray(df['Open'].values, dtype=np.float64) if 'Open' in df.columns else close
            volume = np.ascontiguousarray(df['Volume'].values, dtype=np.float64) if 'Volume' in df.columns else None
            data_len = len(close)
            
            # 确保数据长度一致
            high = high[:data_len]
            low = low[:data_len]
            open_price = open_price[:data_len]
            if volume is not None:
                volume = volume[:data_len]
            
            # 分步计算各类指标
            # 改为自行计算K线形态，而不是调用方法
            # 1.1 收盘强度 - 衡量收盘价在当日价格区间中的位置
            high_low_diff = high - low
            valid_idx = high_low_diff > Constants.EPSILON
            closing_strength = np.full_like(close, Constants.DEFAULT_STRENGTH)
            if np.any(valid_idx):
                closing_strength[valid_idx] = (close[valid_idx] - low[valid_idx]) / high_low_diff[valid_idx]
            df['收盘强度'] = closing_strength
            
            # 1.2 上下影线比例 - 衡量K线形态
            upper_shadow = np.zeros_like(close)
            lower_shadow = np.zeros_like(close)
            body_size = np.abs(close - open_price)
            
            # 计算上影线比例
            upper_idx = valid_idx & (close >= open_price)  # 阳线
            if np.any(upper_idx):
                upper_shadow[upper_idx] = (high[upper_idx] - close[upper_idx]) / high_low_diff[upper_idx]
            
            lower_idx = valid_idx & (close < open_price)  # 阴线
            if np.any(lower_idx):
                upper_shadow[lower_idx] = (high[lower_idx] - open_price[lower_idx]) / high_low_diff[lower_idx]
            
            # 计算下影线比例
            if np.any(upper_idx):
                lower_shadow[upper_idx] = (open_price[upper_idx] - low[upper_idx]) / high_low_diff[upper_idx]
            
            if np.any(lower_idx):
                lower_shadow[lower_idx] = (close[lower_idx] - low[lower_idx]) / high_low_diff[lower_idx]
            
            df['上影线比例'] = upper_shadow
            df['下影线比例'] = lower_shadow
            
            # 1.3 跳空缺口 - 计算跳空高开率
            gap_up = np.zeros_like(close)
            if data_len > 1:
                gap_up[1:] = (open_price[1:] - close[:-1]) / np.maximum(close[:-1], Constants.EPSILON) * 100
            df['跳空高开'] = gap_up
            
            # 1.4 K线形态识别 - 锤子线、吞没线等
            hammer = (body_size > 0) & (lower_shadow > 2 * body_size / np.maximum(high_low_diff, Constants.EPSILON)) & (upper_shadow < 0.1)
            df['锤子线'] = hammer
            
            # 计算其他指标
            df = TechnicalIndicators._calculate_volume_metrics(df, volume, close, data_len, params)
            df = TechnicalIndicators._calculate_technical_signals(df, close, params, data_len)
            df = TechnicalIndicators._calculate_buy_signals(df, params, data_len)
            df = TechnicalIndicators._calculate_next_day_prediction(df, close, data_len)
            
            # 去除DataFrame的碎片化，解决性能警告
            df = df.copy()
            
            return df
            
        except Exception as e:
            logging.error(f"短线指标计算异常: {str(e)}")
            return df
            
    @staticmethod
    def _calculate_volume_metrics(df: pd.DataFrame, volume: np.ndarray, close: np.ndarray, 
                                data_len: int, params: TradingParams) -> pd.DataFrame:
        """
        计算量能相关指标
        """
        if volume is None or data_len <= 5:
            return df
            
        # 2.1 量比 - 当日成交量与5日平均成交量之比
        # 使用向量化操作计算移动平均
        vol_ma5 = np.array([np.mean(volume[max(0, i-4):i+1]) for i in range(data_len)])
        vol_ratio = volume / np.maximum(vol_ma5, Constants.EPSILON)
        df['量比'] = vol_ratio
        
        # 2.2 量能趋势 - 成交量变化趋势
        vol_trend = np.zeros_like(close)
        if data_len > 3:
            vol_trend[3:] = (volume[3:] - volume[:-3]) / np.maximum(volume[:-3], Constants.EPSILON)
        df['量能趋势'] = vol_trend
        
        # 2.3 放量程度 - 与20日平均成交量相比
        if data_len > 20:
            vol_ma20 = np.array([np.mean(volume[max(0, i-19):i+1]) for i in range(data_len)])
            vol_ratio_20 = volume / np.maximum(vol_ma20, Constants.EPSILON)
            df['放量程度'] = vol_ratio_20
            
            # 2.4 量价配合 - 成交量与价格变化的协同性
            price_change = np.zeros_like(close)
            price_change[1:] = (close[1:] / close[:-1] - 1) * 100
            vol_price_match = price_change * vol_ratio
            df['量价配合'] = vol_price_match
            
        return df
    
    @staticmethod
    def _calculate_technical_signals(df: pd.DataFrame, close: np.ndarray, 
                                   params: TradingParams, data_len: int) -> pd.DataFrame:
        """
        计算技术指标信号
        使用NumPy向量化操作提高计算效率，减少指标冗余，改进信号计算
        """
        # 预分配信号数组，避免重复创建
        all_signals = {
            'RSI信号': np.zeros(data_len, dtype=bool),
            'MACD信号': np.zeros(data_len, dtype=bool),
            '均线突破信号': np.zeros(data_len, dtype=bool)
        }
        
        # =====================================================================
        # 1. RSI信号计算 - 使用NumPy向量化操作
        # =====================================================================
        if 'RSI' in df.columns:
            # 获取RSI数值数组
            rsi_values = np.ascontiguousarray(df['RSI'].values)
            
            # 计算RSI超买、超卖和金叉信号
            rsi_overbought = rsi_values > params.short_term_rsi_range[1]
            rsi_oversold = rsi_values < params.short_term_rsi_range[0]
            
            # 计算RSI金叉信号（从超卖区上穿）- 使用切片而非shift
            if data_len > 1:
                rsi_signal = np.zeros(data_len, dtype=bool)  
                rsi_signal[1:] = (rsi_values[1:] > params.short_term_rsi_range[0]) & \
                                (rsi_values[:-1] <= params.short_term_rsi_range[0])
                all_signals['RSI信号'] = rsi_signal
            
            # 保存到DataFrame
            df['RSI超买'] = rsi_overbought
            df['RSI超卖'] = rsi_oversold
            df['RSI金叉'] = all_signals['RSI信号']
        
        # =====================================================================
        # 2. MACD信号计算 - 减少冗余判断，提高效率
        # =====================================================================
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            # 获取MACD相关数组
            macd_values = np.ascontiguousarray(df['MACD'].values)
            macd_signal_values = np.ascontiguousarray(df['MACD_Signal'].values)
            
            # 计算MACD金叉信号 - 使用NumPy切片优化
            if data_len > 1:
                macd_cross = np.zeros(data_len, dtype=bool)
                macd_cross[1:] = (macd_values[1:] > macd_signal_values[1:]) & \
                                (macd_values[:-1] <= macd_signal_values[:-1])
                all_signals['MACD信号'] = macd_cross
            
            # 计算MACD柱状图由负转正信号
            if 'MACD_Hist' in df.columns and data_len > 1:
                macd_hist = np.ascontiguousarray(df['MACD_Hist'].values)
                macd_turn_positive = np.zeros(data_len, dtype=bool)
                macd_turn_positive[1:] = (macd_hist[1:] > 0) & (macd_hist[:-1] <= 0)
                df['MACD柱转正'] = macd_turn_positive
            
            # 保存MACD金叉信号
            df['MACD金叉'] = all_signals['MACD信号']
        
        # =====================================================================
        # 3. 均线突破信号计算 - 优化向量操作
        # =====================================================================
        if 'MA20' in df.columns and data_len > 1:
            # 获取MA20数组
            ma20_values = np.ascontiguousarray(df['MA20'].values)
            
            # 计算均线突破信号
            ma_break = np.zeros(data_len, dtype=bool)
            ma_break[1:] = (close[1:] > ma20_values[1:]) & (close[:-1] <= ma20_values[:-1])
            all_signals['均线突破信号'] = ma_break
            
            # 保存均线突破信号
            df['均线突破'] = all_signals['均线突破信号']
        
        # =====================================================================
        # 4. 综合信号计算 - 提高效率
        # =====================================================================
        # 计算信号综合强度 (0-3)
        signal_strength = all_signals['RSI信号'].astype(int) + \
                         all_signals['MACD信号'].astype(int) + \
                         all_signals['均线突破信号'].astype(int)
        
        # 添加信号强度列
        df['信号强度'] = signal_strength
        
        # 保存所有信号到DataFrame
        for signal_name, signal_array in all_signals.items():
            df[signal_name] = signal_array
        
        # 统一信号名称，避免冗余
        if 'RSI金叉' in df.columns and 'RSI信号' in df.columns:
            # 确保新旧列名兼容
            df['RSI信号'] = df['RSI金叉']
        
        if 'MACD金叉' in df.columns and 'MACD信号' in df.columns:
            df['MACD信号'] = df['MACD金叉']
        
        if '均线突破' in df.columns and '均线突破信号' in df.columns:
            df['均线突破信号'] = df['均线突破']
            
        return df
    
    @staticmethod
    def _calculate_buy_signals(df: pd.DataFrame, params: TradingParams, data_len: int) -> pd.DataFrame:
        """
        计算短线买入信号 - 优化版本
        
        通过综合多种技术指标和形态特征，识别第二天可能上涨的股票。
        关注点：
        1. 价格走势：收盘强度、均线关系、K线形态
        2. 量能特征：量比适中、量价配合
        3. 技术指标：RSI、MACD、KDJ信号
        4. 市场环境：行业趋势、市场情绪
        """
        # 获取之前计算的信号
        # 使用正确的列名获取信号，优先使用新列名，如果不存在则使用旧列名
        rsi_signal = df.get('RSI信号', df.get('RSI金叉', pd.Series([False] * data_len))).values
        macd_signal = df.get('MACD信号', df.get('MACD金叉', pd.Series([False] * data_len))).values
        ma_break_signal = df.get('均线突破信号', df.get('均线突破', pd.Series([False] * data_len))).values
        hammer = df.get('锤子线', pd.Series([False] * data_len)).values
        
        # =====================================================================
        # 1. 价格形态条件 - 收盘强度和K线形态
        # =====================================================================
        
        # 收盘强度高 - 表示收盘价接近最高价，上涨动能强
        closing_strength = df['收盘强度'].values
        price_condition1 = closing_strength > params.closing_strength_threshold
        
        # 下影线比例大 - 表示有支撑买盘
        lower_shadow = df.get('下影线比例', pd.Series([0.0] * data_len)).values
        price_condition2 = lower_shadow > 0.2
        
        # 价格站上短期均线 - 表示短期趋势向上
        if 'MA5' in df.columns and 'MA10' in df.columns:
            price_above_ma5 = df['Close'].values > df['MA5'].values
            ma5_above_ma10 = df['MA5'].values > df['MA10'].values
            price_condition3 = price_above_ma5 & ma5_above_ma10
        else:
            price_condition3 = np.full(data_len, True)  # 默认为True
        
        # 综合价格条件 - 至少满足两个条件
        price_conditions = price_condition1.astype(int) + price_condition2.astype(int) + price_condition3.astype(int)
        price_signal = price_conditions >= 2
        
        # =====================================================================
        # 2. 量能条件 - 成交量特征
        # =====================================================================
        
        # 量比适中 - 放量但不过度（1.2-3倍）
        volume_ratio = df.get('量比', pd.Series([1.0] * data_len)).values
        volume_condition1 = (volume_ratio > 1.2) & (volume_ratio < 3.0)
        
        # 量价配合 - 量增价升
        if '量价配合' in df.columns:
            volume_price_match = df['量价配合'].values > 0
            volume_condition2 = volume_price_match
        else:
            # 简化计算：价格上涨且成交量增加
            if 'Volume' in df.columns and data_len > 1:
                price_up = df['Close'].values[1:] > df['Close'].values[:-1]
                volume_up = df['Volume'].values[1:] > df['Volume'].values[:-1]
                # 确保数组长度一致
                if len(price_up) == len(volume_up):
                    vol_price_match = np.append(False, price_up & volume_up)
                    # 确保vol_price_match长度与data_len一致
                    if len(vol_price_match) < data_len:
                        vol_price_match = np.append(vol_price_match, [False] * (data_len - len(vol_price_match)))
                    elif len(vol_price_match) > data_len:
                        vol_price_match = vol_price_match[:data_len]
                    volume_condition2 = vol_price_match
                else:
                    # 处理长度不一致的情况
                    logging.warning(f"价格和成交量数组长度不一致: {len(price_up)} vs {len(volume_up)}")
                    volume_condition2 = np.full(data_len, False)
            else:
                volume_condition2 = np.full(data_len, True)  # 默认为True
        
        # 综合量能条件 - 满足任一条件
        volume_signal = volume_condition1 | volume_condition2
        
        # =====================================================================
        # 3. 技术指标条件 - 多指标共振
        # =====================================================================
        
        # RSI条件 - RSI金叉或RSI在理想区间(40-65)
        if 'RSI' in df.columns:
            rsi_ideal = (df['RSI'].values >= 40) & (df['RSI'].values <= 65)
            rsi_condition = rsi_signal | rsi_ideal
        else:
            rsi_condition = np.full(data_len, True)  # 默认为True
        
        # MACD条件 - MACD金叉或MACD柱状图由负转正
        if 'MACD_Hist' in df.columns and data_len > 1:
            # 获取MACD_Hist值数组
            macd_hist = df['MACD_Hist'].values
            
            # 计算MACD柱状图由负转正，使用索引而不是shift
            macd_turn_positive = np.zeros(data_len, dtype=bool)
            macd_turn_positive[1:] = (macd_hist[1:] > 0) & (macd_hist[:-1] <= 0)
            
            # 组合条件
            macd_condition = macd_signal | macd_turn_positive
        else:
            macd_condition = np.full(data_len, True)  # 默认为True
        
        # 均线突破条件
        ma_condition = ma_break_signal
        
        # K线形态条件 - 锤子线等看涨形态
        pattern_condition = hammer
        
        # 综合技术指标条件 - 至少满足两个条件
        tech_conditions = rsi_condition.astype(int) + macd_condition.astype(int) + \
                         ma_condition.astype(int) + pattern_condition.astype(int)
        tech_signal = tech_conditions >= 2
        
        # =====================================================================
        # 4. 市场环境条件 - 顺势而为
        # =====================================================================
        
        # 市场趋势向上
        if 'MarketTrend' in df.columns:
            market_up = df['MarketTrend'].values > 0
        else:
            market_up = np.full(data_len, True)  # 默认为True
        
        # =====================================================================
        # 综合买入信号 - 价格、量能、技术指标三者都满足，且市场环境有利
        # =====================================================================
        df['短线买入信号'] = price_signal & volume_signal & tech_signal & market_up
        
        # =====================================================================
        # 计算次日上涨概率 - 更精确的预测模型
        # =====================================================================
        
        # 1. K线形态因子 - 收盘强度和影线比例
        k_shape_factor = closing_strength * 0.4 + \
                        df.get('下影线比例', pd.Series([0.0] * data_len)).values * 0.3 + \
                        (1 - df.get('上影线比例', pd.Series([0.0] * data_len)).values) * 0.3
        
        # 2. 量能因子 - 量比和量价配合
        volume_factor = np.minimum(volume_ratio / params.short_term_volume_threshold, 2.0) * 0.5
        
        # 3. 技术指标因子 - 各指标信号加权
        tech_factor = (rsi_condition.astype(float) * 0.7 + \
                      macd_condition.astype(float) * 0.7 + \
                      ma_condition.astype(float) * 0.6) / 2.0
        
        # 4. 市场环境因子 - 市场趋势和情绪
        if 'MarketTrend' in df.columns and 'MarketSentiment' in df.columns:
            market_trend = np.maximum(0, np.minimum(df['MarketTrend'].values + 0.5, 1.0))
            market_sentiment = np.maximum(0, np.minimum(df['MarketSentiment'].values / 2.0, 1.0))
            market_factor = (market_trend * 0.7 + market_sentiment * 0.3)
        else:
            market_factor = np.full(data_len, 0.5)  # 默认中性
        
        # 综合计算次日上涨概率 - 各因子加权求和（修复权重计算）
        # 确保各因子都在合理范围内
        k_shape_factor = np.clip(k_shape_factor, 0.0, 1.0)
        volume_factor = np.clip(volume_factor, 0.0, 1.0)
        tech_factor = np.clip(tech_factor, 0.0, 1.0)
        market_factor = np.clip(market_factor, 0.0, 1.0)
        
        # 使用修正后的权重计算
        df['次日上涨概率'] = k_shape_factor * 0.35 + \
                        volume_factor * 0.25 + \
                        tech_factor * 0.25 + \
                        market_factor * 0.15
        
        # 确保概率在0-1之间
        df['次日上涨概率'] = np.maximum(0.0, np.minimum(df['次日上涨概率'], 1.0))
        
        return df
    
    @staticmethod
    def _calculate_next_day_prediction(df: pd.DataFrame, close: np.ndarray, data_len: int) -> pd.DataFrame:
        """
        计算次日目标价和涨幅预测 - 优化版本
        
        通过分析历史波动率、价格趋势和技术指标状态，
        更准确地预测第二天可能的目标价格和涨幅。
        """
        if data_len <= 20:
            return df
            
        # =====================================================================
        # 1. 计算历史波动特征 - 更精确的波动率计算（优化为向量化操作）
        # =====================================================================
        
        # 创建价格变化序列
        price_series = pd.Series(close)
        pct_change = price_series.pct_change()
        
        # 1.1 计算短期波动率（10日）- 使用pandas rolling
        volatility_short = pct_change.rolling(window=10).std() * 100
        
        # 1.2 计算中期波动率（20日）- 使用pandas rolling
        volatility_mid = pct_change.rolling(window=20).std() * 100
        
        # 1.3 综合波动率 - 短期波动占更大权重
        volatility = volatility_short * 0.6 + volatility_mid * 0.4
        volatility = volatility.fillna(0).values  # 转换为numpy数组并填充NaN
        
        # =====================================================================
        # 2. 计算目标涨幅 - 考虑多种因素
        # =====================================================================
        
        # 确保次日上涨概率数组长度与其他数组一致
        if '次日上涨概率' in df.columns:
            next_day_prob = df['次日上涨概率'].values
            
            # 确保长度匹配
            if len(next_day_prob) != data_len:
                # 如果长度不匹配，截取或扩展数组
                if len(next_day_prob) > data_len:
                    next_day_prob = next_day_prob[:data_len]
                else:
                    # 扩展数组，用0.5填充缺失部分
                    temp = np.full(data_len, 0.5)
                    temp[:len(next_day_prob)] = next_day_prob
                    next_day_prob = temp
        else:
            # 如果没有次日上涨概率列，使用默认值
            next_day_prob = np.full(data_len, 0.5)
        
        # 2.1 基础涨幅 - 基于上涨概率和波动率
        base_target_pct = next_day_prob * volatility * 0.6
        
        # 2.2 趋势调整因子 - 强势股可能涨幅更大
        trend_factor = np.ones(data_len)
        
        # 如果有均线数据，使用均线状态调整目标涨幅
        if 'MA5' in df.columns and 'MA10' in df.columns and 'MA20' in df.columns:
            # 获取均线数据，确保长度匹配
            ma5 = df['MA5'].values[:data_len]
            ma10 = df['MA10'].values[:data_len]
            ma20 = df['MA20'].values[:data_len]
            
            # 创建均线多头排列的掩码（布尔数组）
            mask = np.zeros(data_len, dtype=bool)
            valid_indices = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
            mask[valid_indices] = (ma5[valid_indices] > ma10[valid_indices]) & (ma10[valid_indices] > ma20[valid_indices])
            
            # 调整趋势因子
            trend_factor[mask] = 1.2  # 均线多头排列，涨幅可能更大
        
        # 2.3 应用其他调整因子（例如KDJ/RSI超买超卖、市场情绪等）
        if 'RSI' in df.columns:
            rsi = df['RSI'].values
            if len(rsi) != data_len:
                rsi = np.array([50.0] * data_len)  # 使用默认RSI值
            
            # RSI超卖区域，涨幅可能更大
            rsi_mask = (rsi < 30)
            trend_factor[rsi_mask] *= 1.1
        
        # 2.4 最终目标涨幅 - 应用所有调整因子
        target_pct = base_target_pct * trend_factor
        
        # 2.5 处理极端值和限制目标涨幅范围
        target_pct = np.maximum(0.1, np.minimum(9.8, target_pct))  # 限制在0.1%~9.8%之间
        
        # =====================================================================
        # 3. 计算目标价格 - 基于当前收盘价和目标涨幅
        # =====================================================================
        target_price = close * (1 + target_pct / 100)
        
        # 3.1 目标止损价 - 基于动态因素计算止损幅度
        # 使用多种因素计算合适的止损比例，不再使用固定值
        
        # 基础止损幅度 - 基于波动率动态调整
        base_stop_loss_pct = volatility_short * 0.3  # 使用短期波动率的30%作为基础止损幅度
        
        # 根据股票技术指标状态调整止损幅度
        risk_adjustment = np.zeros(data_len)
        
        # 1. RSI因子 - RSI高时增加止损幅度，RSI低时减少止损幅度
        if 'RSI' in df.columns:
            rsi = df['RSI'].values
            if len(rsi) == data_len:  # 确保长度匹配
                # RSI > 70: 高风险，增加止损幅度
                # RSI < 30: 低风险，减少止损幅度
                rsi_factor = np.zeros(data_len)
                rsi_factor[rsi > 70] = 0.5  # 高风险增加0.5%
                rsi_factor[rsi < 30] = -0.3  # 低风险减少0.3%
                risk_adjustment += rsi_factor
        
        # 2. MACD因子 - MACD柱状图为负且增大时，增加止损幅度
        if 'MACD_HIST' in df.columns and len(df) > 1:
            macd_hist = df['MACD_HIST'].values
            if len(macd_hist) == data_len:
                macd_factor = np.zeros(data_len)
                # 计算MACD柱状图变化 - 使用滚动差分
                macd_change = np.zeros(data_len)
                macd_change[1:] = np.diff(macd_hist)
                
                # MACD为负且继续走低，高风险，增加止损幅度
                high_risk_mask = (macd_hist < 0) & (macd_change < 0)
                macd_factor[high_risk_mask] = 0.4
                
                # MACD为正且继续走高，低风险，减少止损幅度
                low_risk_mask = (macd_hist > 0) & (macd_change > 0)
                macd_factor[low_risk_mask] = -0.2
                
                risk_adjustment += macd_factor
        
        # 3. K线形态风险因子
        if 'Pattern_Score' in df.columns:
            pattern_score = df['Pattern_Score'].values
            if len(pattern_score) == data_len:
                pattern_factor = np.zeros(data_len)
                # 负分形态（看跌）增加止损幅度
                pattern_factor[pattern_score < -2] = 0.3
                # 高分形态（看涨）减少止损幅度
                pattern_factor[pattern_score > 2] = -0.2
                risk_adjustment += pattern_factor
        
        # 4. 趋势强度因子 - 强下降趋势增加止损幅度
        if 'Trend_Strength' in df.columns:
            # 假设Trend_Strength是字符串列，我们需要转换为数值
            # 暂时跳过趋势强度调整，因为需要特殊处理字符串
            pass
        
        # 综合止损幅度计算 - 基础止损 + 风险调整
        stop_loss_pct = base_stop_loss_pct + risk_adjustment
        
        # 确保止损幅度在合理范围内 - 0.5%到3%之间
        stop_loss_pct = np.maximum(0.5, np.minimum(stop_loss_pct, 3.0))
        
        # 以下特定情况覆盖前面的计算结果
        # 1. 股票得分≥80，止损固定为2%
        if '股票得分' in df.columns:
            scores = df['股票得分'].values
            if len(scores) == data_len:
                stop_loss_pct[scores >= 80] = 2.0
        
        # 2. 高波动率时保证最低止损
        high_volatility_mask = volatility_short > 2.0
        stop_loss_pct[high_volatility_mask] = np.maximum(stop_loss_pct[high_volatility_mask], 
                                                        volatility_short[high_volatility_mask] * 0.25)
        
        # 计算止损价格
        stop_loss_price = close * (1 - stop_loss_pct / 100)
        
        # 检查并处理止损价格中的NaN值
        if np.isnan(stop_loss_price).any():
            # 如果有NaN值，使用固定百分比止损填充
            nan_mask = np.isnan(stop_loss_price)
            stop_loss_price[nan_mask] = close[nan_mask] * 0.99  # 默认1%止损
        
        # 确保没有极端值和无效值
        stop_loss_price = np.maximum(close * 0.9, np.minimum(stop_loss_price, close * 0.995))
        
        # =====================================================================
        # 4. 将计算结果添加到DataFrame
        # =====================================================================
        
        df['次日目标涨幅'] = target_pct
        df['次日目标价'] = target_price
        df['次日止损价'] = stop_loss_price
        # 计算次日止损幅度（百分比）
        df['次日止损幅度'] = (1 - df['次日止损价'] / close) * 100
        
        # 最后确保次日止损价和止损幅度中没有NaN值 - 已修改默认值为动态值
        df['次日止损价'] = df['次日止损价'].fillna(df['Close'] * 0.98)  # 默认2%止损
        # 对于没有计算出止损幅度的行，根据股票得分动态设置
        if '股票得分' in df.columns and df['次日止损幅度'].isna().any():
            # 得分越高，默认止损越小
            df.loc[df['次日止损幅度'].isna() & (df['股票得分'] >= 70), '次日止损幅度'] = 1.5
            df.loc[df['次日止损幅度'].isna() & (df['股票得分'] < 70) & (df['股票得分'] >= 50), '次日止损幅度'] = 2.0
            df.loc[df['次日止损幅度'].isna() & (df['股票得分'] < 50), '次日止损幅度'] = 2.5
        else:
            # 如果没有股票得分或其他情况，使用中等止损幅度
            df['次日止损幅度'] = df['次日止损幅度'].fillna(2.0)
        
        return df

    def _calculate_cci(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算CCI（顺势指标），用于判断超买超卖和趋势反转
        
        CCI高于+100进入超买区，低于-100进入超卖区
        CCI从下向上穿越0轴是买入信号
        CCI从上向下穿越0轴是卖出信号
        
        参数:
            df: 股票数据DataFrame
            period: CCI计算周期，默认14
            
        返回:
            添加了CCI指标的DataFrame
        """
        if df is None or df.empty or len(df) < period:
            return df
            
        try:
            # 检查必要的数据列
            if not all(col in df.columns for col in ['High', 'Low', 'Close']):
                self.logger.warning("计算CCI指标缺少必要的价格数据列")
                return df
                
            # 使用TA-Lib计算CCI指标
            high_array = df['High'].values
            low_array = df['Low'].values
            close_array = df['Close'].values
            
            # 确保CCI计算正确并填充NaN值
            df['CCI'] = ta.CCI(high_array, low_array, close_array, timeperiod=period)
            df['CCI'] = df['CCI'].fillna(0)  # 填充NaN值，避免报告中显示为空
            
            # 添加CCI超买超卖状态
            df['CCI_Overbought'] = df['CCI'] > 100
            df['CCI_Oversold'] = df['CCI'] < -100
            
            # 添加CCI零轴穿越信号
            if len(df) >= 2:
                # CCI金叉：CCI从下方穿越0
                df['CCI_GoldenCross'] = (df['CCI'].shift(1) < 0) & (df['CCI'] >= 0)
                
                # CCI死叉：CCI从上方穿越0
                df['CCI_DeathCross'] = (df['CCI'].shift(1) > 0) & (df['CCI'] <= 0)
                
                # CCI超买区反转：CCI从超买区下穿100
                df['CCI_OverboughtExit'] = (df['CCI'].shift(1) > 100) & (df['CCI'] <= 100)
                
                # CCI超卖区反转：CCI从超卖区上穿-100
                df['CCI_OversoldExit'] = (df['CCI'].shift(1) < -100) & (df['CCI'] >= -100)
            
            # 添加CCI强度描述
            df['CCI_Strength'] = 'Neutral'  # 中性
            df.loc[df['CCI'] > 100, 'CCI_Strength'] = 'Strong'  # 强势
            df.loc[df['CCI'] > 200, 'CCI_Strength'] = 'Very Strong'  # 非常强势
            df.loc[df['CCI'] < -100, 'CCI_Strength'] = 'Weak'  # 弱势
            df.loc[df['CCI'] < -200, 'CCI_Strength'] = 'Very Weak'  # 非常弱势
            
            self.logger.info(f"CCI指标计算完成，范围: {df['CCI'].min():.2f} 到 {df['CCI'].max():.2f}")
                
        except Exception as e:
            self.logger.error(f"计算CCI指标失败: {str(e)}")
            # 确保CCI列存在，即使计算失败
            if 'CCI' not in df.columns:
                df['CCI'] = 0.0
            
        return df

    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用TA-Lib识别常见K线形态，增强短线交易信号
        """
        if df is None or df.empty or len(df) < 10:
            return df
            
        try:
            # 检查必要的数据列
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                self.logger.warning("识别K线形态缺少必要的OHLC数据列")
                return df
                
            # 提取数据数组
            open_array = df['Open'].values
            high_array = df['High'].values
            low_array = df['Low'].values
            close_array = df['Close'].values
            
            # 使用TA-Lib识别各种K线形态 - 一次批量计算所有形态
            candle_patterns = {
                'Pattern_Hammer': ta.CDLHAMMER(open_array, high_array, low_array, close_array),
                'Pattern_HangingMan': ta.CDLHANGINGMAN(open_array, high_array, low_array, close_array),
                'Pattern_Engulfing': ta.CDLENGULFING(open_array, high_array, low_array, close_array),
                'Pattern_Doji': ta.CDLDOJI(open_array, high_array, low_array, close_array),
                'Pattern_DragonflyDoji': ta.CDLDRAGONFLYDOJI(open_array, high_array, low_array, close_array),
                'Pattern_GravestoneDoji': ta.CDLGRAVESTONEDOJI(open_array, high_array, low_array, close_array),
                'Pattern_MorningStar': ta.CDLMORNINGSTAR(open_array, high_array, low_array, close_array),
                'Pattern_EveningStar': ta.CDLEVENINGSTAR(open_array, high_array, low_array, close_array),
                'Pattern_3WhiteSoldiers': ta.CDL3WHITESOLDIERS(open_array, high_array, low_array, close_array),
                'Pattern_3BlackCrows': ta.CDL3BLACKCROWS(open_array, high_array, low_array, close_array),
                'Pattern_ShootingStar': ta.CDLSHOOTINGSTAR(open_array, high_array, low_array, close_array),
                'Pattern_InvertedHammer': ta.CDLINVERTEDHAMMER(open_array, high_array, low_array, close_array),
                'Pattern_Harami': ta.CDLHARAMI(open_array, high_array, low_array, close_array)
            }
            
            # 一次性添加所有K线形态结果 - 使用向量化操作替代循环
            for pattern_name, pattern_result in candle_patterns.items():
                df[pattern_name] = pattern_result != 0
            
            # 添加买入信号 - 使用向量化操作替代多个或条件
            # 创建初始全零数组
            bullish_signals = np.zeros(len(df), dtype=bool)
            bearish_signals = np.zeros(len(df), dtype=bool)
            
            # 更新看涨信号
            bullish_signals |= (candle_patterns['Pattern_Hammer'] > 0)
            bullish_signals |= (candle_patterns['Pattern_MorningStar'] > 0)
            bullish_signals |= (candle_patterns['Pattern_3WhiteSoldiers'] > 0)
            bullish_signals |= (candle_patterns['Pattern_InvertedHammer'] > 0)
            bullish_signals |= ((candle_patterns['Pattern_Engulfing'] > 0) & (close_array > open_array))
            
            # 更新看跌信号
            bearish_signals |= (candle_patterns['Pattern_HangingMan'] > 0)
            bearish_signals |= (candle_patterns['Pattern_EveningStar'] > 0)
            bearish_signals |= (candle_patterns['Pattern_3BlackCrows'] > 0)
            bearish_signals |= (candle_patterns['Pattern_ShootingStar'] > 0)
            bearish_signals |= ((candle_patterns['Pattern_Engulfing'] < 0) & (close_array < open_array))
            
            # 赋值给DataFrame
            df['Pattern_BullishSignal'] = bullish_signals
            df['Pattern_BearishSignal'] = bearish_signals
            
            # 添加K线形态综合评分 (针对次日可能涨停的预测) - 使用向量化操作
            pattern_score = np.zeros(len(df))
            
            # 看涨形态加分 - 使用向量化操作
            pattern_score += df['Pattern_Hammer'].astype(int) * 2
            pattern_score += df['Pattern_MorningStar'].astype(int) * 3
            pattern_score += df['Pattern_3WhiteSoldiers'].astype(int) * 4
            pattern_score += df['Pattern_InvertedHammer'].astype(int) * 1
            pattern_score += ((df['Pattern_Engulfing']) & (df['Close'] > df['Open'])).astype(int) * 3
            pattern_score += df['Pattern_DragonflyDoji'].astype(int) * 1
            
            # 看跌形态减分 - 使用向量化操作
            pattern_score -= df['Pattern_HangingMan'].astype(int) * 2
            pattern_score -= df['Pattern_EveningStar'].astype(int) * 3
            pattern_score -= df['Pattern_3BlackCrows'].astype(int) * 4
            pattern_score -= df['Pattern_ShootingStar'].astype(int) * 2
            pattern_score -= ((df['Pattern_Engulfing']) & (df['Close'] < df['Open'])).astype(int) * 3
            pattern_score -= df['Pattern_GravestoneDoji'].astype(int) * 1
            
            # 一次性赋值
            df['Pattern_Score'] = pattern_score
            
            self.logger.info("K线形态识别完成")
            
        except Exception as e:
            self.logger.error(f"识别K线形态失败: {str(e)}")
            
        return df

    def _calculate_limit_up_probability(self, df: pd.DataFrame) -> None:
        """
        计算股票次日涨停概率 - 优化版本
        
        涨停概率指标使用以下因素:
        1. K线形态指标 - 特定形态对涨停有较强预测力 (25%)
        2. 技术指标信号 - RSI、MACD、ADX、CCI等指标组合 (25%)
        3. 成交量特征 - 放量突破、缩量回调等模式 (20%)
        4. 价格趋势 - 均线系统、趋势状态等 (10%)
        5. 反转信号 - 超跌反弹、突破形态等 (10%)
        6. 市场热点因素 - 行业板块轮动、概念题材 (10%)
        
        参数:
            df: DataFrame, 包含计算好的各类技术指标
        """
        if df is None or df.empty or len(df) < 20:
            return
            
        try:
            # 初始化涨停概率列
            df['涨停概率'] = 0.0
            
            # 为提高性能，创建统一的列名映射表
            column_mapping = {
                'pattern_score': ['Pattern_Score', 'pattern_score'],
                'bullish_signal': ['Pattern_BullishSignal', 'Bullish_Pattern'],
                'hammer': ['Pattern_Hammer', 'hammer'],
                'morning_star': ['Pattern_MorningStar', 'morning_star'],
                'three_white_soldiers': ['Pattern_3WhiteSoldiers', '3_white_soldiers'],
                'bullish_engulfing': ['Pattern_BullishEngulfing', 'bullish_engulfing'],
                'piercing': ['Pattern_Piercing', 'piercing_pattern'],
                'rsi': ['RSI', 'rsi'],
                'rsi_golden_cross': ['RSI_GoldenCross', 'rsi_golden_cross'],
                'rsi_oversold_entry': ['RSI_OversoldEntry', 'rsi_oversold_entry'],
                'macd': ['MACD', 'macd'],
                'macd_signal': ['MACD_Signal', 'macd_signal'],
                'macd_hist': ['MACD_Hist', 'macd_hist', 'MACD_HIST'],
                'macd_golden_cross': ['MACD_GoldenCross', 'macd_golden_cross'],
                'kdj_k': ['K', 'KDJ_K', 'kdj_k'],
                'kdj_j': ['J', 'KDJ_J', 'kdj_j'],
                'adx': ['ADX', 'adx'],
                'plus_di': ['PLUS_DI', 'plus_di'],
                'minus_di': ['MINUS_DI', 'minus_di'],
                'di_bullish_cross': ['DI_BullishCross', 'di_bullish_cross'],
                'cci': ['CCI', 'cci'],
                'cci_golden_cross': ['CCI_GoldenCross', 'cci_golden_cross'],
                'volume_ratio': ['量比', 'Volume_Ratio']
            }
            
            # 辅助函数 - 获取特定指标的列名
            def get_column(key):
                if key not in column_mapping:
                    return None
                for col_name in column_mapping[key]:
                    if col_name in df.columns:
                        return col_name
                return None
            
            # =====================================================================
            # 1. K线形态因子 (25% 权重) - 向量化处理
            # =====================================================================
            pattern_score = np.zeros(len(df))
            
            # 获取形态评分列
            pattern_score_col = get_column('pattern_score')
            if pattern_score_col:
                # 获取并规范化K线形态评分
                pattern_score = np.clip(df[pattern_score_col].fillna(0).values, -5, 5) + 5
            
            # 获取看涨信号列
            bullish_signal_col = get_column('bullish_signal')
            if bullish_signal_col:
                pattern_score += (df[bullish_signal_col].fillna(0) > 0).astype(int) * 3
            
            # 形态权重字典 - 更高效的查找
            pattern_weights = {
                'hammer': 2,
                'morning_star': 3,
                'three_white_soldiers': 2,
                'bullish_engulfing': 3,
                'piercing': 2
            }
            
            # 一次性处理所有形态
            for pattern_key, weight in pattern_weights.items():
                col_name = get_column(pattern_key)
                if col_name:
                    pattern_score += (df[col_name].fillna(0) > 0).astype(int) * weight
            
            # 归一化到0-1范围 (最大可能得分为18)
            pattern_factor = np.clip(pattern_score / 18, 0, 1)
            
            # =====================================================================
            # 2. 技术指标因子 (25% 权重) - 使用NumPy向量化操作
            # =====================================================================
            data_len = len(df)
            tech_score = np.zeros(data_len)
            
            # RSI指标信号处理
            rsi_col = get_column('rsi')
            if rsi_col:
                # 获取RSI值数组
                rsi_values = df[rsi_col].values
                
                # RSI在30-50之间且向上，是潜在的超跌反弹信号 - 使用NumPy操作
                if data_len > 1:
                    rsi_signal = np.zeros(data_len, dtype=bool)
                    rsi_signal[1:] = ((rsi_values[1:] > 30) & (rsi_values[1:] < 50) & 
                                     (rsi_values[1:] > rsi_values[:-1]))
                    tech_score += rsi_signal.astype(int) * 2
                
                # RSI金叉信号
                golden_cross_col = get_column('rsi_golden_cross')
                if golden_cross_col:
                    tech_score += (df[golden_cross_col].fillna(0) > 0).astype(int) * 2
                
                # RSI超卖区反弹
                oversold_col = get_column('rsi_oversold_entry')
                if oversold_col:
                    tech_score += (df[oversold_col].fillna(0) > 0).astype(int) * 3
            
            # MACD指标信号处理
            macd_col = get_column('macd')
            macd_signal_col = get_column('macd_signal')
            
            if macd_col and macd_signal_col:
                # MACD金叉
                golden_cross_col = get_column('macd_golden_cross')
                if golden_cross_col:
                    tech_score += (df[golden_cross_col].fillna(0) > 0).astype(int) * 3
                
                # MACD柱状图相关信号
                macd_hist_col = get_column('macd_hist')
                if macd_hist_col and data_len > 2:
                    # 获取MACD柱状图数据
                    macd_hist = df[macd_hist_col].values
                    
                    # 柱状图由负转正信号 - 使用NumPy操作
                    macd_turn_positive = np.zeros(data_len, dtype=bool)
                    macd_turn_positive[1:] = (macd_hist[1:] > 0) & (macd_hist[:-1] <= 0)
                    tech_score += macd_turn_positive.astype(int) * 2
                    
                    # 柱状图连续放大信号 - 使用NumPy操作
                    macd_hist_growing = np.zeros(data_len, dtype=bool)
                    if data_len > 2:
                        macd_hist_growing[2:] = ((macd_hist[2:] > 0) &
                                               (macd_hist[2:] > macd_hist[1:-1]) &
                                               (macd_hist[1:-1] > macd_hist[:-2]))
                    tech_score += macd_hist_growing.astype(int) * 2
            
            # KDJ指标信号处理
            k_col = get_column('kdj_k')
            j_col = get_column('kdj_j')
            
            if k_col and j_col and data_len > 1:
                # 获取KDJ数据
                k_values = df[k_col].values
                j_values = df[j_col].values
                
                # KDJ金叉信号 - 使用NumPy操作
                kdj_golden_cross = np.zeros(data_len, dtype=bool)
                kdj_golden_cross[1:] = ((k_values[1:] > k_values[:-1]) &
                                      (j_values[1:] > j_values[:-1]) &
                                      (k_values[:-1] < j_values[:-1]) &
                                      (k_values[1:] > j_values[1:]))
                tech_score += kdj_golden_cross.astype(int) * 3  # 提高金叉权重
                
                # J线从超卖区回升 - 使用NumPy操作
                j_oversold_exit = np.zeros(data_len, dtype=bool)
                j_oversold_exit[1:] = (j_values[1:] > 20) & (j_values[:-1] <= 20)
                tech_score += j_oversold_exit.astype(int) * 4  # 提高J线超卖反弹权重
                
                # 新增：J值低位区间评分 - 对短线更敏感
                j_low_range = (j_values >= 0) & (j_values < 20)  # J值在超卖区
                tech_score += j_low_range.astype(int) * 2
                
                # 新增：J值快速上升判断
                j_fast_rising = np.zeros(data_len, dtype=bool)
                if data_len > 2:
                    j_fast_rising[2:] = ((j_values[2:] > j_values[1:-1]) & 
                                       (j_values[1:-1] > j_values[:-2]) &
                                       (j_values[2:] - j_values[:-2] > 10))  # J值两天上升超过10
                tech_score += j_fast_rising.astype(int) * 2
            
            # ADX趋势强度信号处理
            adx_col = get_column('adx')
            plus_di_col = get_column('plus_di')
            minus_di_col = get_column('minus_di')
            
            if adx_col and plus_di_col and minus_di_col:
                # 强趋势 + 多头趋势方向
                adx_values = df[adx_col].values
                plus_di_values = df[plus_di_col].values
                minus_di_values = df[minus_di_col].values
                
                adx_signal = (adx_values > 25) & (plus_di_values > minus_di_values)
                tech_score += adx_signal.astype(int) * 2
                
                # 多头信号：+DI上穿-DI
                bullish_cross_col = get_column('di_bullish_cross')
                if bullish_cross_col:
                    tech_score += (df[bullish_cross_col].fillna(0) > 0).astype(int) * 2
            
            # CCI指标信号处理
            cci_col = get_column('cci')
            if cci_col and data_len > 1:
                # 获取CCI数据
                cci_values = df[cci_col].values
                
                # CCI从超卖区回升 - 使用NumPy操作
                cci_signal = np.zeros(data_len, dtype=bool)
                cci_signal[1:] = (cci_values[1:] > -100) & (cci_values[:-1] <= -100)
                tech_score += cci_signal.astype(int) * 2
                
                # CCI金叉
                cci_golden_cross_col = get_column('cci_golden_cross')
                if cci_golden_cross_col:
                    tech_score += (df[cci_golden_cross_col].fillna(0) > 0).astype(int) * 1
            
            # 归一化到0-1范围 (最大可能得分为22)
            tech_factor = np.clip(tech_score / 22, 0, 1)
            
            # =====================================================================
            # 3. 成交量因子 (20% 权重) - 优化向量化计算
            # =====================================================================
            volume_score = np.zeros(data_len)
            
            # 量比因子处理
            volume_ratio_col = get_column('volume_ratio')
            if volume_ratio_col:
                # 获取量比数组
                vol_ratio = df[volume_ratio_col].values
                
                # 量比在1.5-2.5之间是理想状态 - 使用NumPy操作
                ideal_ratio = (vol_ratio >= 1.5) & (vol_ratio <= 2.5)
                volume_score += ideal_ratio.astype(int) * 2
                
                # 温和放量(1.2-1.5) - 使用NumPy操作
                mild_ratio = (vol_ratio >= 1.2) & (vol_ratio < 1.5)
                volume_score += mild_ratio.astype(int) * 1
                
                # 近期成交量变化
                if 'Volume' in df.columns and data_len > 2:
                    volume_values = df['Volume'].values
                    
                    # 连续两天成交量增加 - 使用NumPy操作
                    vol_increase = np.zeros(data_len, dtype=bool)
                    vol_increase[2:] = (volume_values[2:] > volume_values[1:-1]) & (volume_values[1:-1] > volume_values[:-2])
                    volume_score += vol_increase.astype(int) * 1
                    
                    # 温和放量但不过度 - 使用NumPy操作
                    moderate_increase = np.zeros(data_len, dtype=bool)
                    moderate_increase[1:] = (volume_values[1:] > volume_values[:-1]) & (volume_values[1:] < volume_values[:-1] * 2)
                    volume_score += moderate_increase.astype(int) * 1
            
            # 资金流向指标处理
            if 'MFI' in df.columns and data_len > 1:
                # 获取MFI数据
                mfi_values = df['MFI'].values
                
                # MFI从超卖区回升 - 使用NumPy操作
                mfi_signal = np.zeros(data_len, dtype=bool)
                mfi_signal[1:] = (mfi_values[1:] > 20) & (mfi_values[:-1] <= 20)
                volume_score += mfi_signal.astype(int) * 2
                
                # MFI区间评分 - 使用NumPy操作
                mfi_ideal = (mfi_values > 20) & (mfi_values < 70)
                volume_score += mfi_ideal.astype(int) * 1
            
            # MFI资金流向指标 - 资金流入是上涨的重要支撑
            if 'MFI' in df.columns:
                # 优化：重点关注MFI超卖区间反弹，这是强势信号
                vol_flow_score = np.zeros(data_len)
                vol_flow_score += np.where(
                    (df['MFI'] > 20) & (df['MFI'] < 70),  # MFI在理想区间
                    3,  # 中性区间得分适中
                    np.where(
                        (df['MFI'] <= 20) & (df['MFI'] > df['MFI'].shift(1)),  # MFI在超卖区且回升
                        6,  # 超卖反弹得分提高
                        np.where(
                            (df['MFI'] <= 20),  # 超卖但未反弹
                            1,
                            np.where(
                                (df['MFI'] >= 70) & (df['MFI'] < 80),  # 轻度超买
                                1,
                                np.where(
                                    df['MFI'] >= 80,  # 严重超买
                                    -1,  # 给予负分
                                    0
                                )
                            )
                        )
                    )
                )
                
                # 新增：MFI金叉死叉判断
                mfi_golden_cross = (df['MFI'] > 20) & (df['MFI'].shift(1) <= 20)  # MFI金叉20
                vol_flow_score += np.where(mfi_golden_cross, 2, 0)  # MFI金叉额外加分
            
            # 资金流量指标处理
            if 'CMF' in df.columns and data_len > 1:
                # 获取CMF数据
                cmf_values = df['CMF'].values
                
                # CMF由负转正 - 使用NumPy操作
                cmf_turn_positive = np.zeros(data_len, dtype=bool)
                cmf_turn_positive[1:] = (cmf_values[1:] > 0) & (cmf_values[:-1] <= 0)
                volume_score += cmf_turn_positive.astype(int) * 2
                
                # CMF为正且上升 - 使用NumPy操作
                cmf_increasing = np.zeros(data_len, dtype=bool)
                cmf_increasing[1:] = (cmf_values[1:] > 0) & (cmf_values[1:] > cmf_values[:-1])
                volume_score += cmf_increasing.astype(int) * 1
            
            # 归一化到0-1范围 (最大可能得分为10)
            volume_factor = np.clip(volume_score / 10, 0, 1)
            
            # =====================================================================
            # 4. 价格趋势因子 (10% 权重) - 优化向量化计算
            # =====================================================================
            price_score = np.zeros(data_len)
            
            # 均线系统检查
            if all(col in df.columns for col in ['MA5', 'MA10', 'MA20', 'MA60']):
                # 获取均线数据
                ma5 = df['MA5'].values
                ma10 = df['MA10'].values
                ma20 = df['MA20'].values
                ma60 = df['MA60'].values
                close = df['Close'].values
                
                # 计算均线多头排列得分
                ma_bull_alignment = ((ma5 > ma10) & (ma10 > ma20)).astype(int) * 2
                
                # 计算价格站上均线得分
                price_above_ma = ((close > ma20) & (close > ma60)).astype(int) * 2
                
                # 计算均线系统转多头得分
                ma_turn_bull = np.zeros(data_len, dtype=bool)
                if data_len > 1:
                    ma_turn_bull[1:] = ((ma5[1:] > ma10[1:]) & (ma5[:-1] <= ma10[:-1]))
                
                # 合并均线系统得分
                price_score += ma_bull_alignment + price_above_ma + ma_turn_bull.astype(int) * 3
            
            # 规范化价格趋势因子 (最大可能得分约为7)
            price_factor = np.clip(price_score / 7, 0, 1)
            
            # =====================================================================
            # 5. 合并所有因子，计算最终涨停概率
            # =====================================================================
            # 各因子权重
            weights = {
                'pattern': 0.35,  # K线形态 - 提高权重，K线形态对涨停预测更有价值
                'tech': 0.30,     # 技术指标 - 保持不变
                'volume': 0.20,   # 成交量特征 - 保持不变
                'price': 0.08,    # 价格趋势 - 略微降低
                'market': 0.04,   # 市场因素 - 略微降低
                'reversal': 0.03  # 反转信号 - 略微降低
            }
            
            # 获取其他可能已经计算的因子
            market_factor = df['市场因素得分'].values / 10 if '市场因素得分' in df.columns else np.ones(data_len) * 0.5
            reversal_factor = df['反转信号得分'].values / 10 if '反转信号得分' in df.columns else np.ones(data_len) * 0.5
            
            # 确保因子范围在0-1之间
            market_factor = np.clip(market_factor, 0, 1)
            reversal_factor = np.clip(reversal_factor, 0, 1)
            
            # 计算加权涨停概率 - 使用NumPy向量化
            limit_up_prob = (
                pattern_factor * weights['pattern'] +
                tech_factor * weights['tech'] +
                volume_factor * weights['volume'] +
                price_factor * weights['price'] +
                market_factor * weights['market'] +
                reversal_factor * weights['reversal']
            ) * 100  # 转换为百分比
            
            # 保存涨停概率到DataFrame
            df['涨停概率'] = np.round(limit_up_prob, 1)
            
            # 保存各因子贡献分到DataFrame (可选)
            df['形态因子得分'] = np.round(pattern_factor * 10, 1)
            df['技术指标因子得分'] = np.round(tech_factor * 10, 1)
            df['成交量因子得分'] = np.round(volume_factor * 10, 1)
            df['价格趋势因子得分'] = np.round(price_factor * 10, 1)
            
        except Exception as e:
            logging.error(f"涨停概率计算出错: {str(e)}")
            traceback.print_exc()

    def _safe_calculate(self, calculation_func, df: pd.DataFrame, indicator_name: str, *args, **kwargs) -> pd.DataFrame:
        """
        安全地执行指标计算，统一处理异常
        
        参数:
            calculation_func: 计算函数
            df: 股票数据DataFrame
            indicator_name: 指标名称（用于日志记录）
            *args, **kwargs: 传递给计算函数的参数
            
        返回:
            处理后的DataFrame
        """
        try:
            self.logger.info(f"计算{indicator_name}指标...")
            result_df = calculation_func(df, *args, **kwargs)
            self.logger.info(f"{indicator_name}指标计算完成")
            return result_df
        except Exception as e:
            self.logger.error(f"计算{indicator_name}指标失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df

# =============================================================================
# 股票得分计算函数
# =============================================================================
def calculate_stock_score(df: pd.DataFrame, params: TradingParams) -> pd.DataFrame:
    """
    计算股票综合得分，综合考虑价格趋势、量能、技术指标、反转信号等因素。
    优化版专注于短线交易，预测第二天可能上涨或涨停的股票。
    
    参数:
        df: 包含技术指标的DataFrame
        params: 交易参数配置
        
    返回:
        添加了各项得分的DataFrame
    """
    try:
        # 数据验证 - 确保输入数据有效
        if df is None or df.empty or len(df) < Constants.MIN_DATA_POINTS:
            logging.warning("数据不足，无法计算股票得分")
            return df
        
        # 优化：一次性检查所有必要列是否存在
        required_columns = ['Close', 'Open', 'High', 'Low']
        if not all(col in df.columns for col in required_columns):
            logging.warning("缺少必要的价格数据列，无法计算股票得分")
            return df
        
        # 确保基础指标已计算 - 使用向量化操作提高效率
        if '收盘强度' not in df.columns:
            df['收盘强度'] = (df['Close'] - df['Low']) / np.maximum(df['High'] - df['Low'], Constants.EPSILON)
        
        if '量比' not in df.columns and 'Volume' in df.columns:
            vol_ma5 = df['Volume'].rolling(5).mean()
            df['量比'] = df['Volume'] / np.maximum(vol_ma5, Constants.EPSILON)
        
        # 初始化得分组件 - 使用一次性赋值提高效率
        score_columns = ['价格趋势得分', '量能得分', '技术指标得分', '短线信号得分', 
                        '市场因素得分', '反转信号得分', '涨停概率得分', '连续上涨得分', 
                        '价格动量得分', '支撑突破得分']  # 新增两个得分维度
        
        # 优化：使用字典初始化所有得分列，一次性添加到DataFrame
        scores_dict = {col: np.zeros(len(df)) for col in score_columns}
        scores_dict['股票得分'] = np.zeros(len(df))
        df = pd.concat([df, pd.DataFrame(scores_dict, index=df.index)], axis=1)
        
        # =====================================================================
        # 1. 价格趋势评分 (最高25分) - 评估价格走势和K线形态
        # =====================================================================
        
        # 1.1 收盘强度评分 (0-8分) - 收盘位置靠近最高价更有利于次日上涨
        strength_score = np.minimum((df['收盘强度'] - params.closing_strength_threshold) * 100, 8)
        strength_score = np.maximum(strength_score, 0)  # 确保不为负数
        
        # 1.2 均线关系评分 (0-8分) - 价格站上均线系统是强势信号
        ma_score = np.zeros(len(df))
        ma_columns = ['MA5', 'MA10', 'MA20', 'MA60']
        if all(col in df.columns for col in ma_columns[:3]):  # 至少需要MA5、MA10、MA20
            # 使用布尔数组的向量化操作，避免循环
            price_above_ma5 = df['Close'] > df['MA5']  # 价格站上短期均线
            ma5_above_ma10 = df['MA5'] > df['MA10']    # 短期均线站上中期均线
            ma10_above_ma20 = df['MA10'] > df['MA20']  # 中期均线站上长期均线
            
            # 均线多头排列得分（最高6分）- 优化：增加均线角度因素
            ma_score = price_above_ma5.astype(float) * 1.5 + \
                       ma5_above_ma10.astype(float) * 1.5 + \
                       ma10_above_ma20.astype(float) * 1
                       
            # 新增：均线角度评分 - 均线向上倾斜更有利于上涨
            if len(df) >= 5:
                ma5_angle = (df['MA5'] - df['MA5'].shift(3)) / np.maximum(df['MA5'].shift(3), Constants.EPSILON) * 100
                ma5_angle_score = np.where(ma5_angle > 0, 
                                         np.minimum(ma5_angle, 3), # 最高3分
                                         0)
                ma_score += ma5_angle_score
            
            # 均线拐头向上更有利于短线上涨
            ma5_turning_up = (df['MA5'] > df['MA5'].shift(1)) & (df['MA5'].shift(1) <= df['MA5'].shift(2))
            ma_score += ma5_turning_up.astype(float) * 2
            
            # 新增：均线距离评分 - 防止均线粘合后急跌
            if 'MA5' in df.columns and 'MA10' in df.columns:
                ma_distance = (df['MA5'] - df['MA10']) / np.maximum(df['MA10'], Constants.EPSILON) * 100
                # 均线距离在1%-3%之间最佳，过大意味着短期过度上涨
                ma_distance_score = np.where(
                    (ma_distance > 0.5) & (ma_distance < 3), 
                    1, 
                    np.where(ma_distance >= 3, -1, 0))  # 过大距离反而减分
                ma_score += ma_distance_score
            
            ma_score = np.minimum(ma_score, 8)  # 最高8分
        
        # 1.3 K线形态评分 (0-7分) - 使用TA-Lib识别的K线形态
        pattern_score = np.zeros(len(df))
        
        # 如果有K线形态评分，直接使用
        if 'Pattern_Score' in df.columns:
            pattern_score = np.minimum(df['Pattern_Score'], 7)
        else:
            # 增加更多K线形态权重，特别是强烈看涨形态
            # 锤子线形态 - 下跌趋势中出现锤子线是反转信号
            if 'Pattern_Hammer' in df.columns:
                pattern_score += df['Pattern_Hammer'].astype(float) * 3
            
            # 晨星形态 - 非常强的反转信号
            if 'Pattern_MorningStar' in df.columns:
                pattern_score += df['Pattern_MorningStar'].astype(float) * 5
                
            # 三白兵 - 强势上涨信号
            if 'Pattern_3WhiteSoldiers' in df.columns:
                pattern_score += df['Pattern_3WhiteSoldiers'].astype(float) * 5
                
            # 吞没形态 - 重要的反转信号
            if 'Pattern_Engulfing' in df.columns:
                bullish_engulfing = (df['Pattern_Engulfing'] > 0) & (df['Close'] > df['Open'])
                pattern_score += bullish_engulfing.astype(float) * 4
                
            # 刺形态 - 看涨反转信号
            if 'Pattern_Piercing' in df.columns:
                pattern_score += df['Pattern_Piercing'].astype(float) * 3
                
            pattern_score = np.minimum(pattern_score, 7)
        
        # 1.4 价格动量得分 (0-6分) - 新增单独维度
        momentum_score = np.zeros(len(df))
        if 'Close' in df.columns:
            # 计算最近3日涨幅
            recent_gain = (df['Close'] / df['Close'].shift(3) - 1) * 100
            # 温和上涨(1%-3%)最适合次日继续上涨
            momentum_score = np.where(
                recent_gain < 0, 
                np.where(recent_gain > -2, 1, 0),  # 微跌处理：抗跌信号
                
                np.where(
                    recent_gain <= 1,  # 0-1%区间
                    recent_gain * 3,   # 线性递增：0%→0分，1%→3分
                    
                    np.where(
                        recent_gain <= 3,  # 1%-3%核心区间
                        6 - np.abs(recent_gain - 2) * 1,  # 2%→6分，1%或3%→5分
                        
                        np.where(
                            recent_gain <= 5,  # 3%-5%区间
                            4 - (recent_gain - 3),  # 快速递减
                            
                            np.where(
                                recent_gain <= 7,  # 5%-7%区间
                                2 - (recent_gain - 5) * 0.5,  # 缓速递减
                                0  # >7%过度上涨
                            )
                        )
                    )
                )
            )
            
            # 新增：高开走强形态评分
            # 高开走强是强势特征，尤其是连续高开
            if all(col in df.columns for col in ['Open', 'Close']):
                # 昨日收盘 vs 今日开盘
                gap_up = df['Open'] > df['Close'].shift(1)
                # 今日高开走强（开盘>昨收，收盘>开盘）
                strong_open = gap_up & (df['Close'] > df['Open'])
                # 连续两天高开走强
                consecutive_strong = strong_open & strong_open.shift(1)
                
                momentum_score += strong_open.astype(float) * 1
                momentum_score += consecutive_strong.astype(float) * 2
            
            momentum_score = np.minimum(momentum_score, 10)
        
        # 将动量得分单独保存
        df['价格动量得分'] = momentum_score
        
        # 1.5 波动率评分 (0-4分) - 适度波动性有利于上涨
        volatility_score = np.zeros(len(df))
        if 'ATR' in df.columns and 'Close' in df.columns:
            # 相对波动率 = ATR / 收盘价
            rel_volatility = df['ATR'] / df['Close'] * 100
            
            # 适度波动(1%-3%)最有利于短线交易
            volatility_score = np.where(
                (rel_volatility >= 1) & (rel_volatility <= 3),
                4,  # 理想波动区间
                np.where(
                    (rel_volatility > 3) & (rel_volatility <= 5),
                    2,  # 较高波动
                    np.where(
                        rel_volatility > 5,
                        0,  # 过高波动不加分
                        np.where(
                            rel_volatility < 0.5,
                            0,  # 过低波动不加分
                            2   # 低波动区间
                        )
                    )
                )
            )
        
        # 综合价格趋势得分 - 确保不超过25分
        df['价格趋势得分'] = np.minimum(strength_score + ma_score + pattern_score + volatility_score, 25)
        
        # =====================================================================
        # 2. 量能评分 (最高20分) - 评估成交量特征
        # =====================================================================
        
        # 2.1 量比评分 (0-5分) - 放量但不过度
        # 量比是短线交易中判断主力资金进出的重要指标
        vol_score = np.zeros(len(df))
        if '量比' in df.columns:
            # 优化：量比理想区间调整为1.1-2.0，轻微放量更有利于次日上涨
            vol_ratio = df['量比'].values
            vol_ratio_score = np.zeros_like(vol_ratio)
            
            # 分段处理不同量比区间
            # 小于1.1的量比
            mask_low = vol_ratio < 1.1
            vol_ratio_score[mask_low] = (vol_ratio[mask_low] / 1.1) * 2
            
            # 1.1-2.0的理想量比区间
            mask_ideal = (vol_ratio >= 1.1) & (vol_ratio <= 2.0)
            vol_ratio_score[mask_ideal] = 3 + (vol_ratio[mask_ideal] - 1.1) * 2
            
            # 2.0-2.5的量比，得分开始下降
            mask_high = (vol_ratio > 2.0) & (vol_ratio <= 2.5)
            vol_ratio_score[mask_high] = 5 - (vol_ratio[mask_high] - 2.0) * 2
            
            # 大于2.5的量比可能是出货，给予负分
            mask_very_high = vol_ratio > 2.5
            vol_ratio_score[mask_very_high] = -2
            
            vol_score = np.maximum(-2, np.minimum(vol_ratio_score, 5))
        
        # 2.2 资金流向评分 (0-9分) - 使用高级资金流向指标
        vol_flow_score = np.zeros(len(df))
        
        # MFI资金流向指标 - 资金流入是上涨的重要支撑
        if 'MFI' in df.columns:
            mfi = df['MFI'].values
            mfi_score = np.zeros_like(mfi)
            
            # 超卖区且回升 - 最强买入信号
            mask_oversold_rising = (mfi <= 20) & (mfi > np.roll(mfi, 1))
            mfi_score[mask_oversold_rising] = 5
            
            # 超卖区但未回升
            mask_oversold = (mfi <= 20)
            mfi_score[~mask_oversold_rising & mask_oversold] = 1
            
            # 中性区间 - 理想区间
            mask_neutral = (mfi > 20) & (mfi < 70)
            mfi_score[mask_neutral & ~mask_oversold] = 3
            
            # 轻度超买区间
            mask_overbought = (mfi >= 70) & (mfi < 80)
            mfi_score[mask_overbought] = 1
            
            # 严重超买区间 - 给予负分
            mask_very_overbought = mfi >= 80
            mfi_score[mask_very_overbought] = -1
            
            # MFI金叉20，重要买点
            if len(mfi) > 1:
                mfi_golden_cross = (mfi > 20) & (np.roll(mfi, 1) <= 20)
                # 避免处理第一个元素
                mfi_golden_cross[0] = False
                mfi_score[mfi_golden_cross] += 2
            
            vol_flow_score += mfi_score
            
            # 新增：MFI底背离识别 - 价格创新低但MFI未创新低是强烈买入信号
            if len(df) >= 5 and 'Low' in df.columns:
                # 计算近5天的最低价
                rolling_min = df['Low'].rolling(5).min()
                # 最低价创新低
                new_low = df['Low'] == rolling_min
                
                # 找出MFI未创新低，但价格创新低的点
                if len(df) > 5:
                    mfi_5d_min = df['MFI'].rolling(5).min()
                    mfi_not_new_low = df['MFI'] > mfi_5d_min
                    
                    # MFI底背离 - 价格创新低但MFI未创新低
                    mfi_divergence = new_low & mfi_not_new_low
                    vol_flow_score[mfi_divergence] += 3
        
        # CMF钱德勒资金流量 - 正值表示资金流入
        if 'CMF' in df.columns:
            cmf = df['CMF'].values
            cmf_score = np.zeros_like(cmf)
            
            if len(cmf) > 1:
                # 资金由流出转为流入 - 重要的转折信号
                cmf_turn_positive = (cmf > 0) & (np.roll(cmf, 1) <= 0)
                # 避免处理第一个元素
                cmf_turn_positive[0] = False
                cmf_score[cmf_turn_positive] = 3
                
                # 持续资金流入
                mask_positive = cmf > 0
                # 流入加速
                mask_accelerating = mask_positive & (cmf > np.roll(cmf, 1))
                cmf_score[mask_accelerating & ~cmf_turn_positive] = 2
                # 流入减速
                cmf_score[mask_positive & ~mask_accelerating & ~cmf_turn_positive] = 1
            
            vol_flow_score += cmf_score
        
        # 新增：OBV能量潮指标 - 成交量先行
        if 'OBV' in df.columns:
            obv = df['OBV'].values
            obv_score = np.zeros_like(obv)
            
            if len(obv) > 5:
                # OBV 5日均线
                obv_ma5 = pd.Series(obv).rolling(5).mean().values
                
                # OBV站上5日均线 - 买入信号
                obv_above_ma = obv > obv_ma5
                obv_score[obv_above_ma] = 1
                
                # OBV快速上升 - 强势特征
                if len(obv) > 1:
                    obv_rising = obv > np.roll(obv, 1)
                    # 避免处理第一个元素
                    obv_rising[0] = False
                    obv_score[obv_rising & obv_above_ma] += 1
            
            vol_flow_score += obv_score
        
        # 将资金流向得分限制在0-9分范围内
        vol_flow_score = np.minimum(vol_flow_score, 9)
        vol_flow_score = np.maximum(vol_flow_score, 0)  # 确保非负
        
        # 2.3 量价配合评分 (0-6分) - 价量同步是强势特征
        vol_price_score = np.zeros(len(df))
        
        # 成交量震荡指标
        if 'ADOSC' in df.columns:
            adosc = df['ADOSC'].values
            adosc_score = np.zeros_like(adosc)
            
            if len(adosc) > 1:
                # 震荡指标由负转正 - 买方占优的重要信号
                adosc_turn_positive = (adosc > 0) & (np.roll(adosc, 1) <= 0)
                # 避免处理第一个元素
                adosc_turn_positive[0] = False
                adosc_score[adosc_turn_positive] = 3
                
                # 震荡指标为正且上升
                mask_positive = adosc > 0
                mask_rising = mask_positive & (adosc > np.roll(adosc, 1))
                adosc_score[mask_rising & ~adosc_turn_positive] = 2
                
                # 震荡指标为正但下降
                adosc_score[mask_positive & ~mask_rising & ~adosc_turn_positive] = 1
            
            vol_price_score += adosc_score
        
        # 成交量趋势与价格趋势配合
        if all(col in df.columns for col in ['Close', 'Volume']):
            close = df['Close'].values
            volume = df['Volume'].values
            
            if len(close) > 1 and len(volume) > 1:
                # 价格上涨
                price_up = close > np.roll(close, 1)
                # 成交量增加
                vol_up = volume > np.roll(volume, 1)
                # 避免处理第一个元素
                price_up[0] = False
                vol_up[0] = False
                
                # 量价同步上涨 - 强势特征
                vol_price_match = price_up & vol_up
                vol_price_score[vol_price_match] += 2
                
                # 缩量上涨形态 - 持筹者坚定
                if len(volume) >= 5:
                    vol_ma5 = pd.Series(volume).rolling(5).mean().values
                    sufficient_vol = volume > vol_ma5 * 0.7
                    shrinking_vol_up = price_up & ~vol_up & sufficient_vol
                    vol_price_score[shrinking_vol_up] += 1
                    
                    # 新增：连续缩量上涨 - 强势整理形态
                    if len(price_up) > 2:
                        consecutive_up = price_up & np.roll(price_up, 1)
                        consecutive_shrink = consecutive_up & ~vol_up & np.roll(~vol_up, 1)
                        vol_price_score[consecutive_shrink] += 1
        
        # 新增：量价分歧评分 - 识别潜在趋势转折点
        if 'Close' in df.columns and 'Volume' in df.columns and len(df) >= 5:
            # 计算5日价格变化率和成交量变化率
            price_change = (df['Close'] / df['Close'].shift(5) - 1) * 100
            volume_change = (df['Volume'] / df['Volume'].shift(5) - 1) * 100
            
            # 量增价跌 - 潜在见底信号
            vol_up_price_down = (volume_change > 20) & (price_change < -5)
            vol_price_score[vol_up_price_down] += 1
            
            # 量增价平 - 蓄势待发信号
            vol_up_price_flat = (volume_change > 30) & (price_change.abs() < 1)
            vol_price_score[vol_up_price_flat] += 2
        
        # 限制在0-6分范围内
        vol_price_score = np.minimum(vol_price_score, 6)
        
        # 综合量能得分 - 确保不超过20分
        df['量能得分'] = np.minimum(vol_score + vol_flow_score + vol_price_score, 20)
        
        # =====================================================================
        # 3. 技术指标评分 (最高30分) - 评估多种技术指标
        # =====================================================================
        
        # 3.1 RSI指标评分 (0-10分) - 关注超买超卖和金叉
        # RSI是判断超买超卖的重要指标，也是短线交易的重要参考
        rsi_score = np.zeros(len(df))
        if 'RSI' in df.columns:
            # 优化：RSI理想区间调整为40-60，这是动能充沛但未超买的最佳区间
            rsi_score = np.where(
                (df['RSI'] >= 40) & (df['RSI'] <= 60), 7,  # 理想区间，上涨动能强
                np.where(
                    (df['RSI'] > 30) & (df['RSI'] < 40), 6,  # 刚脱离超卖区，反弹概率大
                    np.where(
                        (df['RSI'] > 60) & (df['RSI'] <= 70), 4,  # 接近超买但未超买
                        np.where(
                            (df['RSI'] <= 30) & (df['RSI'] > df['RSI'].shift(1)), 5,  # 超卖区回升
                            np.where(df['RSI'] > 70, 1, 2)  # 超买区得分低，超卖区得分适中
                        )
                    )
                )
            )
            
            # RSI金叉额外加分 - RSI金叉是强势信号
            if 'RSI_GoldenCross' in df.columns:
                rsi_score += df['RSI_GoldenCross'].astype(int) * 1
                
            # 超卖区反弹额外加分
            if 'RSI_OversoldEntry' in df.columns:
                rsi_score += df['RSI_OversoldEntry'].astype(int) * 2
                
            # 限制在0-8分范围内
            rsi_score = np.minimum(rsi_score, 10)
        
        # 3.2 MACD指标评分 (0-9分)
        # MACD是判断中长期趋势的重要指标
        macd_score = np.zeros(len(df))
        if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            # MACD柱状图为正且增长 - 强势特征
            macd_hist_positive = df['MACD_Hist'] > 0
            macd_hist_growing = df['MACD_Hist'] > df['MACD_Hist'].shift(1)
            
            # MACD位于零轴上方 - 多头市场特征
            macd_above_zero = df['MACD'] > 0
            
            # 优化：MACD柱状图由负转正是重要买点
            macd_hist_turn_positive = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
            
            # 综合MACD状态评分
            macd_score = macd_hist_positive.astype(int) * 1.5 + \
                        macd_hist_growing.astype(int) * 1.5 + \
                        macd_above_zero.astype(int) * 1 + \
                        macd_hist_turn_positive.astype(int) * 3  # 柱状图由负转正权重高
            
            # MACD金叉额外加分 - MACD金叉是买入信号
            if 'MACD_GoldenCross' in df.columns:
                macd_score += df['MACD_GoldenCross'].astype(int) * 2
                
            # 限制在0-7分范围内
            macd_score = np.minimum(macd_score, 9)
        
        # 3.3 KDJ指标评分 (0-6分) 
        kdj_score = np.zeros(len(df))
        if all(col in df.columns for col in ['K', 'D', 'J']):
            # KDJ金叉
            k_cross_d = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
            kdj_score += k_cross_d.astype(int) * 3
            
            # J线由超卖区回升
            j_up_from_oversold = (df['J'] > 20) & (df['J'].shift(1) <= 20)
            kdj_score += j_up_from_oversold.astype(int) * 2
            
            # KDJ三线向上发散
            kdj_bullish = (df['K'] > df['K'].shift(1)) & (df['D'] > df['D'].shift(1)) & (df['J'] > df['J'].shift(1))
            kdj_score += kdj_bullish.astype(int) * 1
            
            # 限制在0-5分范围内
            kdj_score = np.minimum(kdj_score, 6)
        
        # 3.4 ADX趋势强度指标评分 (0-5分)
        # ADX是判断趋势强度的重要指标
        adx_score = np.zeros(len(df))
        if all(col in df.columns for col in ['ADX', 'PLUS_DI', 'MINUS_DI']):
            # ADX大于25表示强趋势，短线更容易跟随趋势获利
            adx_score = np.where(
                df['ADX'] > 25,  # 强趋势
                np.where(
                    df['ADX'] > 30,  # 非常强的趋势
                    3,
                    2
                ),
                1  # 弱趋势
            )
            
            # 多头趋势方向加分
            plus_di_dominant = df['PLUS_DI'] > df['MINUS_DI']
            adx_score += plus_di_dominant.astype(int) * 2
            
            # 限制在0-5分范围内
            adx_score = np.minimum(adx_score, 5)
        
        # 综合技术指标得分 - 确保不超过30分
        # RSI和MACD指标权重最高，KDJ次之，ADX权重较低
        df['技术指标得分'] = np.minimum(rsi_score + macd_score + kdj_score + adx_score, 30)
        
        # =====================================================================
        # 4. 短线信号评分 (最高20分) - 评估短线交易信号
        # =====================================================================
        
        # 4.1 短线买入信号 (0-12分) - 系统生成的买入信号
        # 短线买入信号是综合多个指标生成的信号，可靠性较高
        signal_score = np.zeros(len(df))
        if '短线买入信号' in df.columns:
            signal_score = np.where(df['短线买入信号'], 12, 0)
        
        # 4.2 次日上涨概率 (0-8分) - 基于历史数据的统计概率
        # 次日上涨概率是基于历史数据统计的概率，可靠性较高
        prob_score = np.zeros(len(df))
        if '次日上涨概率' in df.columns:
            # 优化：次日上涨概率阈值降低到60%，增加候选股票池
            next_day_up_threshold = 0.6  
            prob_score = np.where(
                df['次日上涨概率'] > next_day_up_threshold,
                (df['次日上涨概率'] - next_day_up_threshold) * 10 * 2 + 6,  # 增加权重
                0
            )
            prob_score = np.minimum(prob_score, 8)  # 最高8分
        
        # 综合短线信号得分 - 确保不超过20分
        df['短线信号得分'] = np.minimum(signal_score + prob_score, 20)
        
        # =====================================================================
        # 5. 市场与行业因素评分 (最高10分) - 评估外部环境
        # =====================================================================
        
        # 5.1 市场趋势评分 (0-5分) - 大盘趋势
        # 市场趋势是影响个股走势的重要因素，顺势而为是短线交易的基本原则
        market_trend_score = np.zeros(len(df))
        if 'MarketTrend' in df.columns:
            market_trend = df['MarketTrend'].values
            
            # 强势上涨市场
            mask_strong_up = market_trend > 0.01
            market_trend_score[mask_strong_up] = 5
            
            # 温和上涨市场
            mask_mild_up = (market_trend > 0) & (market_trend <= 0.01)
            market_trend_score[mask_mild_up] = 3
            
            # 盘整市场
            mask_flat = (market_trend >= -0.01) & (market_trend <= 0)
            market_trend_score[mask_flat] = 1
            
            # 下跌市场不加分
        
        # 5.2 市场情绪评分 (0-5分) - 市场情绪
        # 市场情绪是影响短线交易的重要因素，适度活跃的市场更有利于短线交易
        market_sentiment_score = np.zeros(len(df))
        if 'MarketSentiment' in df.columns:
            market_sentiment = df['MarketSentiment'].values
            
            # 适度活跃，最佳状态
            mask_ideal = (market_sentiment > 0.5) & (market_sentiment < 2)
            market_sentiment_score[mask_ideal] = 5
            
            # 低活跃度但有上升迹象
            mask_rising = (market_sentiment > 0.3) & (market_sentiment <= 0.5)
            market_sentiment_score[mask_rising] = 3
            
            # 市场过热，风险增加
            mask_hot = market_sentiment >= 2
            market_sentiment_score[mask_hot] = 2
            
            # 市场过冷不加分
        
        # 新增：行业趋势评分 (0-5分) - 行业整体趋势
        industry_trend_score = np.zeros(len(df))
        if 'IndustryTrend' in df.columns:
            industry_trend = df['IndustryTrend'].values
            
            # 行业强势上涨
            mask_ind_strong = industry_trend > 0.03
            industry_trend_score[mask_ind_strong] = 5
            
            # 行业温和上涨
            mask_ind_mild = (industry_trend > 0) & (industry_trend <= 0.03)
            industry_trend_score[mask_ind_mild] = 3
            
            # 行业盘整
            mask_ind_flat = (industry_trend >= -0.01) & (industry_trend <= 0)
            industry_trend_score[mask_ind_flat] = 1
            
            # 行业下跌不加分
        elif 'Industry' in df.columns and 'IndustryRank' in df.columns:
            # 如果有行业排名信息，也可以作为参考
            industry_rank = df['IndustryRank'].values
            
            # 行业排名前30%
            mask_top_rank = industry_rank <= 30
            industry_trend_score[mask_top_rank] = 3
            
            # 行业排名30%-60%
            mask_mid_rank = (industry_rank > 30) & (industry_rank <= 60)
            industry_trend_score[mask_mid_rank] = 1
        
        # 综合市场因素得分 - 确保不超过10分
        # 新算法：将行业趋势纳入考量，但设置上限
        df['市场因素得分'] = np.minimum(market_trend_score + market_sentiment_score + industry_trend_score * 0.6, 10)
        
        # =====================================================================
        # 6. 反转信号得分 (最高8分) - 评估潜在反转机会 - 提高权重
        # =====================================================================
        
        # 6.1 超跌反弹信号 (0-4分) - RSI超卖反弹
        # 超跌反弹是短线交易中的重要机会，RSI从超卖区反弹是强烈买入信号
        oversold_score = np.zeros(len(df))
        if 'RSI' in df.columns:
            rsi = df['RSI'].values
            
            # RSI从超卖区反弹 - 连续两天在超卖区后回升
            if len(rsi) > 2:
                oversold_rebound = (rsi > 30) & (np.roll(rsi, 1) <= 35) & (np.roll(rsi, 2) <= 30)
                # 避免处理前两个元素
                oversold_rebound[:2] = False
                oversold_score[oversold_rebound] = 4
                
                # 新增：更敏感的超跌反弹识别 - RSI触及极低值后回升
                extreme_oversold = (rsi > 25) & (np.roll(rsi, 1) <= 25)
                extreme_oversold[0] = False
                oversold_score[extreme_oversold & ~oversold_rebound] = 3
        
        # 6.2 突破信号 (0-4分) - 价格突破关键均线或阻力位
        # 突破信号是短线交易中的重要机会，价格突破关键均线是买入信号
        breakout_score = np.zeros(len(df))
        if '均线突破' in df.columns:
            breakout_score = np.where(df['均线突破'], 4, 0)
        else:
            # 如果没有均线突破指标，使用MA10和MA20作为关键均线判断
            if all(col in df.columns for col in ['Close', 'MA10', 'MA20']):
                close = df['Close'].values
                ma10 = df['MA10'].values
                ma20 = df['MA20'].values
                
                if len(close) > 1 and len(ma10) > 1:
                    # MA10突破 - 短期突破
                    ma10_breakout = (close > ma10) & (np.roll(close, 1) <= np.roll(ma10, 1))
                    ma10_breakout[0] = False
                    breakout_score[ma10_breakout] += 2
                
                if len(close) > 1 and len(ma20) > 1:
                    # MA20突破 - 中期突破，更重要
                    ma20_breakout = (close > ma20) & (np.roll(close, 1) <= np.roll(ma20, 1))
                    ma20_breakout[0] = False
                    breakout_score[ma20_breakout] += 3
                    
                    # MA10和MA20同时突破 - 强烈信号
                    dual_breakout = ma10_breakout & ma20_breakout
                    breakout_score[dual_breakout] = 4  # 最高分
        
        # 综合反转信号得分 - 确保不超过8分
        df['反转信号得分'] = np.minimum(oversold_score + breakout_score, 8)
        
        # =====================================================================
        # 7. 涨停概率得分 (最高10分) - 基于涨停概率指标 - 优化权重
        # =====================================================================
        limit_up_score = np.zeros(len(df))
        if '涨停概率' in df.columns:
            # 降低涨停概率阈值到12%，更灵敏地捕捉潜在涨停股
            prob_values = df['涨停概率'].fillna(0).values
            
            # 概率越高得分越高，采用非线性映射
            limit_up_score = np.where(
                prob_values > 12,
                np.minimum(prob_values / 7, 10),  # 12%得1.7分，30%得4.3分，70%得10分
                0
            )
            
            # 涨停风险评估
            if '涨停风险' in df.columns:
                risk_levels = df['涨停风险'].values
                risk_adjustment = np.zeros_like(risk_levels, dtype=float)
                
                # 高风险涨停股减分
                mask_high_risk = df['涨停风险'] == '高'
                risk_adjustment[mask_high_risk] = -2
                
                # 低风险涨停股加分
                mask_low_risk = df['涨停风险'] == '低'
                risk_adjustment[mask_low_risk] = 1
                
                # 应用风险调整
                limit_up_score = limit_up_score + risk_adjustment
            
            # 新增：涨停板效应评分
            if 'LimitUpEffect' in df.columns:
                effect_values = df['LimitUpEffect'].values
                effect_score = np.zeros_like(effect_values, dtype=float)
                
                # 强势涨停板效应加分
                mask_strong_effect = df['LimitUpEffect'] == '强势'
                effect_score[mask_strong_effect] = 2
                
                # 普通涨停板效应小幅加分
                mask_normal_effect = df['LimitUpEffect'] == '普通'
                effect_score[mask_normal_effect] = 1
                
                # 应用涨停板效应调整
                limit_up_score = limit_up_score + effect_score
            
            # 确保在0-10分范围内
            limit_up_score = np.maximum(0, np.minimum(limit_up_score, 10))
        
        df['涨停概率得分'] = limit_up_score
        
        # =====================================================================
        # 8. 连续上涨得分 (最高5分) - 评估连续上涨趋势
        # =====================================================================
        consecutive_up_score = np.zeros(len(df))
        if 'Close' in df.columns:
            # 计算连续上涨天数
            close = df['Close'].values
            up_days = np.zeros_like(close, dtype=int)
            
            # 优化：使用向量化操作计算连续上涨天数
            if len(close) > 1:
                # 初始化每日涨跌数组
                daily_up = np.zeros_like(close, dtype=bool)
                daily_up[1:] = close[1:] > close[:-1]
                
                # 计算连续上涨天数
                for i in range(1, len(close)):
                    if daily_up[i]:
                        up_days[i] = up_days[i-1] + 1
            
            # 连续上涨评分优化：连续2天上涨最有可能继续上涨，连续4天以上可能即将回调
            consecutive_up_score = np.zeros_like(close, dtype=float)
            
            # 连续2天上涨，满分
            mask_2days = up_days == 2
            consecutive_up_score[mask_2days] = 5
            
            # 刚开始上涨，中等分数
            mask_1day = up_days == 1
            consecutive_up_score[mask_1day] = 3
            
            # 连续3天上涨，低分
            mask_3days = up_days == 3
            consecutive_up_score[mask_3days] = 2
            
            # 连续4天以上，最低分
            mask_4plus_days = up_days >= 4
            consecutive_up_score[mask_4plus_days] = 1
            
            # 上涨幅度因素，温和上涨更可能持续
            if len(close) >= 3:
                # 计算最近2天的累计涨幅
                two_day_gain = np.zeros_like(close, dtype=float)
                two_day_gain[2:] = (close[2:] / close[:-2] - 1) * 100
                
                gain_score = np.zeros_like(close, dtype=float)
                
                # 温和上涨(1%-4%)更可能持续
                mask_ideal_gain = (two_day_gain > 1) & (two_day_gain <= 4)
                gain_score[mask_ideal_gain] = 5
                
                # 较大涨幅(4%-7%)，小幅加分
                mask_larger_gain = (two_day_gain > 4) & (two_day_gain <= 7)
                gain_score[mask_larger_gain] = 3
                
                # 过度上涨(>7%)，减分
                mask_excessive_gain = two_day_gain > 7
                gain_score[mask_excessive_gain] = -2
                
                # 合并上涨天数得分和涨幅得分
                consecutive_up_score = consecutive_up_score + gain_score
                
                # 确保在0-5分范围内
                consecutive_up_score = np.maximum(0, np.minimum(consecutive_up_score, 5))
        
        df['连续上涨得分'] = consecutive_up_score
        
        # =====================================================================
        # 9. 价格动量得分 (0-10分) - 评估短期价格动能
        # =====================================================================
        momentum_score = np.zeros(len(df))
        if 'Close' in df.columns:
            # 计算最近3日涨幅
            recent_gain = (df['Close'] / df['Close'].shift(3) - 1) * 100
            # 温和上涨(1%-3%)最适合次日继续上涨
            momentum_score = np.where(
                recent_gain < 0, 
                np.where(recent_gain > -2, 1, 0),  # 微跌处理：抗跌信号
                
                np.where(
                    recent_gain <= 1,  # 0-1%区间
                    recent_gain * 3,   # 线性递增：0%→0分，1%→3分
                    
                    np.where(
                        recent_gain <= 3,  # 1%-3%核心区间
                        6 - np.abs(recent_gain - 2) * 1,  # 2%→6分，1%或3%→5分
                        
                        np.where(
                            recent_gain <= 5,  # 3%-5%区间
                            4 - (recent_gain - 3),  # 快速递减
                            
                            np.where(
                                recent_gain <= 7,  # 5%-7%区间
                                2 - (recent_gain - 5) * 0.5,  # 缓速递减
                                0  # >7%过度上涨
                            )
                        )
                    )
                )
            )
            
            # 新增：高开走强形态评分
            # 高开走强是强势特征，尤其是连续高开
            if all(col in df.columns for col in ['Open', 'Close']):
                # 昨日收盘 vs 今日开盘
                gap_up = df['Open'] > df['Close'].shift(1)
                # 今日高开走强（开盘>昨收，收盘>开盘）
                strong_open = gap_up & (df['Close'] > df['Open'])
                # 连续两天高开走强
                consecutive_strong = strong_open & strong_open.shift(1)
                
                momentum_score += strong_open.astype(float) * 1
                momentum_score += consecutive_strong.astype(float) * 2
            
            momentum_score = np.minimum(momentum_score, 10)
        
        # 将动量得分单独保存
        df['价格动量得分'] = momentum_score
        
        # =====================================================================
        # 10. 支撑突破得分 (0-6分) - 新增单独维度
        # =====================================================================
        support_breakout_score = np.zeros(len(df))
        if 'Close' in df.columns and 'MA5' in df.columns and 'MA10' in df.columns:
            # 计算支撑突破条件
            support_breakout = (df['Close'] > df['MA5']) & (df['Close'] > df['MA10'])
            support_breakout_score = np.where(support_breakout, 6, 0)
        
        # 将支撑突破得分单独保存
        df['支撑突破得分'] = support_breakout_score
        
        # =====================================================================
        # 计算综合得分 - 各维度得分加权求和（优化版）
        # =====================================================================
        
        # 各维度得分加权求和
        # 短线交易优化权重分配 - 重视短线交易相关指标
        # 确保各维度得分都存在，如果不存在则使用0
        price_score = df['价格趋势得分']
        volume_score = df['量能得分']
        tech_score = df['技术指标得分']
        signal_score = df['短线信号得分']
        market_score = df['市场因素得分']
        reversal_score = df['反转信号得分']
        limit_up_score = df['涨停概率得分']
        consecutive_up_score = df['连续上涨得分']
        momentum_score = df['价格动量得分']
        support_score = df['支撑突破得分']
        
        # 优化后的权重计算 - 提高技术指标和趋势信号权重
        df['股票得分'] = 30 + (
            price_score * 1.2 +         # 价格趋势
            volume_score * 1.0 +        #量能
            tech_score * 1.2 +          # 技术指标
            signal_score * 1.0 +        #短线信号
            market_score * 0.8 +        # 市场因素
            reversal_score * 1.5 +      # 反转信号
            limit_up_score * 0.6 +      #涨停概率
            consecutive_up_score * 0.5 +   #连续上涨
            momentum_score * 1.0 +      # 价格动量
            support_score * 1.2         # 支撑突破
        )
        
        # 确保得分在0-150之间，并处理可能的NaN值
        df['股票得分'] = df['股票得分'].fillna(20.0).clip(0, 150)
        
        # 计算次日上涨概率评级（优化版）
        df['上涨潜力'] = '低'
        df.loc[df['股票得分'] > 105, '上涨潜力'] = '中'    # 降低中等潜力的阈值
        df.loc[df['股票得分'] > 120, '上涨潜力'] = '高'   # 降低高潜力的阈值
        df.loc[df['股票得分'] > 135, '上涨潜力'] = '极高'  # 降低极高潜力的阈值
        
        # 新增：计算买入建议价格区间
        if all(col in df.columns for col in ['Close', 'Open', 'Low', 'High']):
            # 买入区间：根据近期波动确定合理买入价格区间
            if 'ATR' in df.columns:
                atr = df['ATR'].values
                df['建议买入下限'] = df['Low'] - 0.3 * atr
                df['建议买入上限'] = df['Close'] + 0.1 * atr
            else:
                # 如果没有ATR，使用当日价格区间
                df['建议买入下限'] = df['Low'] * 0.99
                df['建议买入上限'] = df['Close']
            
            # 预期目标价：基于得分和当前价格的合理上涨目标
            score_based_target = 1.0 + (df['股票得分'] - 80) / 200  # 80分以上每增加20分，目标价上涨10%
            df['目标价'] = df['Close'] * np.maximum(score_based_target, 1.02)  # 至少2%的上涨目标
            
            # 止损价：基于支撑位和风险控制
            if 'MA10' in df.columns:
                # 使用10日均线作为主要支撑
                df['止损价'] = np.minimum(df['Low'] * 0.97, df['MA10'] * 0.98)
            else:
                # 没有均线数据时使用近期最低价
                df['止损价'] = df['Low'] * 0.97
    
    except Exception as e:
        logging.error(f"计算股票得分失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # 出错时确保返回原始DataFrame
    
    return df

# =============================================================================
# 报告生成函数 
# =============================================================================
def generate_report(target_stocks: List[List[str]], tech: TechnicalIndicators, params: TradingParams):
    """
    遍历目标股票，获取数据、计算技术指标和综合得分，
    最后根据得分排名，生成得分前50的股票报告（CSV文件）。
    
    报告中包含：
      - 基本信息：代码、名称、行业、行业涨幅%
      - 价格信息：最新收盘价、收盘强度、量比、换手率
      - 技术指标：RSI、MACD、ADX、CCI等
      - 交易信号：得分、短线买入信号、信号强度、涨停概率
      - 交易参考：触发价、次日目标价、次日目标涨幅%、次日止损价、止损幅度%
      - K线形态：主要看涨形态识别结果
    """
    # 初始化报告列表和线程锁
    report = []
    report_lock = threading.Lock()
    s_rD = tech.s_rD
    
    # 记录处理开始时间
    start_time = time.time()
    print(f"开始处理{len(target_stocks)}只目标股票...")
    
    # 定义处理单个股票的函数
    def process_stock(idx, stock_item):
        # 确保stock_item是列表或元组，并且包含两个元素
        if not isinstance(stock_item, (list, tuple)) or len(stock_item) < 2:
            print(f"警告: 股票项 #{idx} 格式错误，跳过处理")
            return None
            
        code, name = stock_item[0], stock_item[1]
        
        # 每处理30只股票显示一次进度（使用锁确保输出不混乱）
        if idx % 30 == 0 and idx > 0:
            with report_lock:
                elapsed = time.time() - start_time
                print(f"已处理 {idx}/{len(target_stocks)} 只股票，耗时 {elapsed:.1f}秒")
        
        # 1. 获取股票历史数据
        df = s_rD.fetch_data(code)
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"警告: 股票 {code}({name}) 数据获取失败或为空")
            return None
            
        # 2. 计算技术指标
        try:
            df = tech.calculate_all(df, params)
            if not isinstance(df, pd.DataFrame):
                print(f"警告: 股票 {code}({name}) 技术指标计算结果非DataFrame类型")
                return None
        except Exception as e:
            print(f"错误: 股票 {code}({name}) 技术指标计算失败 - {str(e)}")
            return None
            
        # 3. 计算综合得分
        try:
            df = calculate_stock_score(df, params)
        except Exception as e:
            print(f"错误: 股票 {code}({name}) 得分计算失败 - {str(e)}")
            return None
        
        # 4. 获取最新一天数据
        latest = df.iloc[-1]
        
        # 5. 获取股票所属行业和行业涨幅
        try:
            industry_data = s_rD.get_stock_industry_data(code)
            industry_name = industry_data["industry_name"]
            industry_change = industry_data["industry_change"]
        except Exception as e:
            print(f"警告: 获取股票{code}({name})行业数据失败 - {str(e)}")
            industry_name = "未知行业"
            industry_change = 0.0
        
        # 6. 计算信号强度 - 综合多个指标判断信号可靠性
        signal_strength = 0
        if latest.get('短线买入信号', False):
            # 基础分5分
            signal_strength = 5
            
            # 技术指标加分
            if latest.get('RSI', 0) > 40 and latest.get('RSI', 0) < 65:
                signal_strength += 1  # RSI在理想区间
            if latest.get('MACD_Hist', 0) > 0:
                signal_strength += 1  # MACD柱状图为正
            if latest.get('次日上涨概率', 0) > 0.7:
                signal_strength += 2  # 上涨概率高
            
            # 价格形态加分
            if latest.get('收盘强度', 0) > 0.7:
                signal_strength += 1  # 收盘强度高
                
            # ADX和CCI指标加分
            if latest.get('ADX', 0) > 25 and latest.get('PLUS_DI', 0) > latest.get('MINUS_DI', 0):
                signal_strength += 1  # ADX趋势强且多头方向
            if latest.get('CCI', 0) > -100 and latest.get('CCI', 0) < 100:
                signal_strength += 1  # CCI在理想区间
                
            # K线形态加分
            pattern_bullish = False
            for pattern in ['Pattern_Hammer', 'Pattern_MorningStar', 'Pattern_3WhiteSoldiers', 'Pattern_InvertedHammer']:
                if latest.get(pattern, False):
                    pattern_bullish = True
                    break
            if pattern_bullish:
                signal_strength += 2  # 存在看涨K线形态
                
            # 资金流指标加分
            if latest.get('MFI', 0) > 20 and latest.get('MFI', 0) < 70:
                signal_strength += 1  # MFI在理想区间
            if latest.get('CMF', 0) > 0:
                signal_strength += 1  # CMF为正，资金流入
        
        # 7. 提取关键K线形态
        bullish_patterns = []
        for pattern_name, display_name in [
            ('Pattern_Hammer', '锤子线'), 
            ('Pattern_MorningStar', '晨星'), 
            ('Pattern_3WhiteSoldiers', '三白兵'),
            ('Pattern_Engulfing', '吞没'), 
            ('Pattern_InvertedHammer', '倒锤'),
            ('Pattern_DragonflyDoji', '蜻蜓十字')
        ]:
            if latest.get(pattern_name, False):
                bullish_patterns.append(display_name)
        
        pattern_text = '、'.join(bullish_patterns) if bullish_patterns else '无'
            
        # 8. 构建结果字典 - 包含所有关键信息
        result = {
            # 基本信息
            "代码": code,
            "名称": name,
            "行业": industry_name,
            "行业涨幅%": round(industry_change, 2),
            
            # 价格信息
            "最新收盘价": round(latest['Close'], 2),
            "收盘强度": round(latest.get('收盘强度', 0), 2),
            "量比": round(latest.get('量比', 0), 2),
            "换手率": round(latest.get('Turnover', 0), 2),
            
            # 技术指标
            "RSI": round(latest.get('RSI', 0), 1),
            "MACD": round(latest.get('MACD', 0), 3),
            "ADX": round(latest.get('ADX', 0), 1),
            "KDJ_J": round(latest.get('J', 0), 1) if 'J' in latest else None,
            "MFI": round(latest.get('MFI', 0), 1) if 'MFI' in latest else None,
            
            # 交易信号
            "得分": round(latest.get('股票得分', 0), 1),
            "上涨潜力": latest.get('上涨潜力', '低'),  # 新增上涨潜力评级
            "短线买入信号": latest.get('短线买入信号', False),
            "信号强度": signal_strength,
            "次日上涨概率": round(latest.get('次日上涨概率', 0), 2),
            "涨停概率": round(latest.get('涨停概率', 0), 1),
            "涨停风险": latest.get('涨停风险', '低'),
            
            # 分项得分
            "价格趋势得分": round(latest.get('价格趋势得分', 0), 1),
            "量能得分": round(latest.get('量能得分', 0), 1),
            "技术指标得分": round(latest.get('技术指标得分', 0), 1),
            "短线信号得分": round(latest.get('短线信号得分', 0), 1),
            "市场因素得分": round(latest.get('市场因素得分', 0), 1),
            "反转信号得分": round(latest.get('反转信号得分', 0), 1),
            "涨停概率得分": round(latest.get('涨停概率得分', 0), 1),
            "连续上涨得分": round(latest.get('连续上涨得分', 0), 1),  # 新增连续上涨得分
            
            # 交易参考
            "触发价": round(latest['Close'], 2),
            "次日目标价": round(latest.get('次日目标价', latest['Close'] * 1.02), 2),
            "次日目标涨幅%": round(latest.get('次日目标涨幅', 2.0), 2),
            "次日止损价": round(latest.get('次日止损价', latest['Close'] * 0.99) if not pd.isna(latest.get('次日止损价', None)) else latest['Close'] * 0.99, 2),
            "止损幅度%": round(latest.get('次日止损幅度', 1.0) if not pd.isna(latest.get('次日止损幅度', None)) else 1.0, 2),
            
            # K线形态
            "看涨形态": pattern_text,
            "K线形态评分": int(latest.get('Pattern_Score', 0)),
            
            # 资金流向
            "资金流向": "流入" if latest.get('CMF', 0) > 0 else "流出",
            
            # 趋势与反转
            "趋势强度": latest.get('趋势强度', 'Unknown') if '趋势强度' in latest else None,
            "反转信号": latest.get('反转信号', False),
            
            # 均线系统
            "均线多头排列": (latest.get('MA5', 0) > latest.get('MA10', 0) and latest.get('MA10', 0) > latest.get('MA20', 0)) if all(col in latest for col in ['MA5', 'MA10', 'MA20']) else False,
            "站上均线系统": (latest.get('Close', 0) > latest.get('MA5', 0) and latest.get('Close', 0) > latest.get('MA10', 0) and latest.get('Close', 0) > latest.get('MA20', 0)) if all(col in latest for col in ['MA5', 'MA10', 'MA20']) else False,
        }
        
        return result
    
    # 确定线程数量 - 使用CPU核心数的2倍，但不超过8个线程
    max_workers = min(8, os.cpu_count() * 2 or 8)
    print(f"使用{max_workers}个线程并行处理数据...")
    
    # 使用线程池并行处理股票数据
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(process_stock, idx, stock_item): idx 
                         for idx, stock_item in enumerate(target_stocks)}
        
        # 收集结果
        for future in as_completed(future_to_idx):
            result = future.result()
            if result is not None:
                with report_lock:
                    report.append(result)
    
    # 处理完成，显示耗时
    elapsed = time.time() - start_time
    print(f"处理完成，共处理{len(report)}/{len(target_stocks)}只股票，总耗时 {elapsed:.1f}秒")
    
    # 8. 转换为DataFrame并排序
    if not report:
        print("没有符合条件的股票！")
        return
        
    report_df = pd.DataFrame(report)
    
    # 9. 按得分排序，取前100只股票
    report_df = report_df.sort_values(by="得分", ascending=False).head(100)
    
    # 10. 生成涨停风险评级和上涨潜力分布
    risk_distribution = report_df['涨停风险'].value_counts()
    risk_high_count = risk_distribution.get('高', 0) + risk_distribution.get('极高', 0)
    potential_distribution = report_df['上涨潜力'].value_counts()
    high_potential_count = potential_distribution.get('高', 0) + potential_distribution.get('极高', 0)
    
    # 11. 保存报告到CSV文件
    current_date = time.strftime("%Y%m%d")
    report_file = os.path.join(Constants.BASE_DIR,"stock_r_report", f"stock_r_top100_{current_date}.csv")
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
    
    # 12. 输出结果
    print(f"\n短线交易选股报告已生成：{report_file}")
    print("\n前100只推荐股票：")
    
    # 创建不同的展示视图
    # 12.1. 基础信息视图
    basic_columns = ["代码", "名称", "得分", "上涨潜力", "短线买入信号", "次日上涨概率", "涨停概率"]
    print("\n=== 基础信息视图 ===")
    print(report_df[basic_columns].to_string(index=False))
    
    # 12.2. 技术指标视图
    tech_columns = ["代码", "名称", "RSI", "MACD", "KDJ_J", "ADX", "趋势强度", "看涨形态"]
    print("\n=== 技术指标视图 ===")
    print(report_df[tech_columns].to_string(index=False))
    
    # 12.3. 分项得分视图 - 新增
    score_columns = ["代码", "名称", "价格趋势得分", "量能得分", "技术指标得分", "短线信号得分", "涨停概率得分", "连续上涨得分"]
    print("\n=== 分项得分视图 ===")
    print(report_df[score_columns].to_string(index=False))
    
    # 12.4. 交易决策视图
    trade_columns = ["代码", "名称", "得分", "上涨潜力", "触发价", "次日目标价", "次日目标涨幅%", "次日止损价"]
    print("\n=== 交易决策视图 ===")
    print(report_df[trade_columns].to_string(index=False))
    
    # 13. 输出高级交易建议
    print("\n=== 交易建议 ===")
    
    # 13.1. 强烈信号股票 - 优化选股标准
    strong_signals = report_df[(report_df["信号强度"] >= 8) | (report_df["上涨潜力"] == "极高") | ((report_df["信号强度"] >= 7) & (report_df["涨停概率"] >= 50))]
    if not strong_signals.empty:
        print(f"🔥 强烈推荐（{len(strong_signals)}只）：{', '.join([f'{code}({name})' for code, name in zip(strong_signals['代码'], strong_signals['名称'])])}")
    
    # 13.2. 不错信号股票 - 优化选股标准
    medium_signals = report_df[(report_df["上涨潜力"] == "高") | ((report_df["信号强度"] >= 6) & (report_df["涨停概率"] >= 30)) & ~report_df.index.isin(strong_signals.index)]
    if not medium_signals.empty:
        print(f"👍 建议关注（{len(medium_signals)}只）：{', '.join([f'{code}({name})' for code, name in zip(medium_signals['代码'], medium_signals['名称'])])}")
    
    # 13.3. K线形态推荐
    pattern_signals = report_df[report_df["看涨形态"] != "无"]
    if not pattern_signals.empty:
        pattern_signals = pattern_signals.sort_values(by="K线形态评分", ascending=False).head(5)
        print(f"📊 K线形态良好（{len(pattern_signals)}只）：{', '.join([f'{code}({name})-{pattern}' for code, name, pattern in zip(pattern_signals['代码'], pattern_signals['名称'], pattern_signals['看涨形态'])])}")
    
    # 13.4. 涨停风险最高股票
    limit_up_risks = report_df[report_df["涨停风险"].isin(['高', '极高'])].sort_values(by="涨停概率", ascending=False).head(5)
    if not limit_up_risks.empty:
        print(f"🚀 涨停风险高（{len(limit_up_risks)}只）：{', '.join([f'{code}({name})-{prob}%' for code, name, prob in zip(limit_up_risks['代码'], limit_up_risks['名称'], limit_up_risks['涨停概率'])])}")
    
    # 14. 输出市场统计分析
    print("\n=== 市场分析 ===")
    print(f"今日涨停风险分布: 极高({risk_distribution.get('极高', 0)}只), 高({risk_distribution.get('高', 0)}只), 中({risk_distribution.get('中', 0)}只), 低({risk_distribution.get('低', 0)}只)")
    print(f"上涨潜力分布: 极高({potential_distribution.get('极高', 0)}只), 高({potential_distribution.get('高', 0)}只), 中({potential_distribution.get('中', 0)}只), 低({potential_distribution.get('低', 0)}只)")
    print(f"短线交易机会指数: {high_potential_count / len(report_df) * 100:.1f}% (高上涨潜力股票占比)")
    
    return report_df

# =============================================================================
# 主程序 - 短线交易选股系统入口
# =============================================================================
if __name__ == "__main__":
    print("\n======== 短线交易选股系统 - 优化版 ========")
    print("专注于识别第二天可能上涨的股票，适合短线交易")
    print("================================================\n")
    
    # 1. 初始化技术指标计算器
    tech_indicators = TechnicalIndicators()
    
    # 2. 初始化交易参数（可根据需要调整参数）
    params = TradingParams()
    
    # 3. 获取目标股票（仅保留以 "" 开头的股票）
    print("正在获取目标股票列表...")
    target_stocks = tech_indicators.get_target_stocks()
    
    # 4. 生成选股报告
    if not target_stocks:
        print("未获取到目标股票列表！请检查网络连接或数据源。")
    else:
        print(f"成功获取{len(target_stocks)}只目标股票，开始分析...\n")
        # 生成报告并获取结果DataFrame
        result_df = generate_report(target_stocks, tech_indicators, params)
        
        # 5. 输出总结信息
        if result_df is not None and not result_df.empty:
            buy_signals = result_df[result_df['短线买入信号'] == True]
            high_potential = result_df[result_df['上涨潜力'].isin(['高', '极高'])]
            print(f"\n总结: 共有{len(buy_signals)}只股票触发买入信号，{len(high_potential)}只股票上涨潜力高")
            print(f"平均目标涨幅: {buy_signals['次日目标涨幅%'].mean():.2f}%")
            print("\n选股完成! 祝交易顺利!")
        
    print("\n================================================")
    print("提示: 本系统仅供参考，实际交易请结合市场情况和个人风险承受能力")
    print("================================================\n")
