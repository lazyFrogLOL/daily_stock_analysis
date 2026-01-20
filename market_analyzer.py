# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd

from config import get_config
from search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块
    
    # ========== 增强数据维度 ==========
    
    # 融资融券
    margin_balance: float = 0.0         # 融资余额（亿元）
    margin_buy: float = 0.0             # 融资买入额（亿元）
    short_balance: float = 0.0          # 融券余额（亿元）
    
    # 龙虎榜
    lhb_stocks: List[Dict] = field(default_factory=list)      # 龙虎榜股票
    lhb_net_buy: float = 0.0            # 龙虎榜净买入（亿元）
    lhb_org_buy_count: int = 0          # 机构买入次数
    lhb_org_sell_count: int = 0         # 机构卖出次数
    
    # 大宗交易
    block_trade_amount: float = 0.0     # 大宗交易成交额（亿元）
    block_trade_premium_ratio: float = 0.0   # 溢价成交占比(%)
    block_trade_discount_ratio: float = 0.0  # 折价成交占比(%)
    
    # 概念板块热点
    top_concepts: List[Dict] = field(default_factory=list)    # 涨幅前5概念
    bottom_concepts: List[Dict] = field(default_factory=list) # 跌幅前5概念
    
    # 市场情绪指标
    avg_turnover_rate: float = 0.0      # 平均换手率(%)
    high_turnover_count: int = 0        # 高换手率(>10%)股票数
    new_high_count: int = 0             # 创60日新高股票数
    new_low_count: int = 0              # 创60日新低股票数
    
    # 涨停分析
    limit_up_broken_count: int = 0      # 炸板数（曾涨停后打开）
    continuous_limit_up_count: int = 0  # 连板股票数
    first_limit_up_count: int = 0       # 首板股票数


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 获取北向资金流向
    5. 获取融资融券数据
    6. 获取龙虎榜数据
    7. 获取大宗交易数据
    8. 获取概念板块热点
    9. 搜索市场新闻
    10. 生成大盘复盘报告
    """
    
    # 主要指数代码
    MAIN_INDICES = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
        'sh000688': '科创50',
        'sh000016': '上证50',
        'sh000300': '沪深300',
        'sh000015': '红利指数',
        'sh000905': '中证500',
        'sh000906': '中证800',
        'sz399012': '创业300',
        'sz399303': '国证2000',
        'sz399372': '大盘成长',
        'sz399373': '大盘价值',
        'sz399374': '中盘成长',
        'sz399375': '中盘价值',
        'sz399376': '小盘成长',
        'sz399377': '小盘价值'
    }
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        初始化大盘分析器
        
        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        
    def get_market_overview(self, target_date: Optional[str] = None) -> MarketOverview:
        """
        获取市场概览数据（增强版，支持指定日期）
        
        Args:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为今天
            
        Returns:
            MarketOverview: 市场概览数据对象
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # 判断是否为历史日期
        is_historical = target_date != datetime.now().strftime('%Y-%m-%d')
        
        overview = MarketOverview(date=target_date)
        
        if is_historical:
            logger.info(f"[大盘] 获取历史数据: {target_date}")
            # 历史数据模式
            overview.indices = self._get_main_indices_hist(target_date)
            self._get_market_statistics_hist(overview, target_date)
        else:
            # 实时数据模式
            overview.indices = self._get_main_indices()
            self._get_market_statistics(overview)
        
        # 以下数据支持历史查询
        self._get_sector_rankings(overview)
        self._get_concept_rankings(overview)
        self._get_margin_data(overview, target_date)
        self._get_lhb_data(overview, target_date)
        self._get_block_trade_data(overview)
        
        return overview

    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 2):
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[大盘] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 5))
        logger.error(f"[大盘] {name} 最终失败: {last_error}")
        return None
    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []
        
        try:
            logger.info("[大盘] 获取主要指数实时行情...")
            
            # 使用 akshare 获取指数行情（新浪财经接口，包含深市指数）
            df = self._call_akshare_with_retry(ak.stock_zh_index_spot_sina, "指数行情", attempts=2)
            
            if df is not None and not df.empty:
                for code, name in self.MAIN_INDICES.items():
                    # 查找对应指数
                    row = df[df['代码'] == code]
                    if row.empty:
                        # 尝试带前缀查找
                        row = df[df['代码'].str.contains(code)]
                    
                    if not row.empty:
                        row = row.iloc[0]
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=float(row.get('最新价', 0) or 0),
                            change=float(row.get('涨跌额', 0) or 0),
                            change_pct=float(row.get('涨跌幅', 0) or 0),
                            open=float(row.get('今开', 0) or 0),
                            high=float(row.get('最高', 0) or 0),
                            low=float(row.get('最低', 0) or 0),
                            prev_close=float(row.get('昨收', 0) or 0),
                            volume=float(row.get('成交量', 0) or 0),
                            amount=float(row.get('成交额', 0) or 0),
                        )
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        indices.append(index)
                        
                logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")
                
        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")
        
        return indices
    
    def _get_main_indices_hist(self, target_date: str) -> List[MarketIndex]:
        """获取主要指数历史行情"""
        indices = []
        
        try:
            logger.info(f"[大盘] 获取指数历史行情: {target_date}...")
            
            # 转换日期格式
            date_str = target_date.replace('-', '')
            
            for code, name in self.MAIN_INDICES.items():
                try:
                    # 使用 index_zh_a_hist 获取历史数据
                    # 代码格式需要转换：sh000001 -> 000001
                    symbol = code[2:] if code.startswith(('sh', 'sz')) else code
                    
                    df = ak.index_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=date_str,
                        end_date=date_str
                    )
                    
                    if df is not None and not df.empty:
                        row = df.iloc[0]
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=float(row.get('收盘', 0) or 0),
                            change=float(row.get('收盘', 0) or 0) - float(row.get('开盘', 0) or 0),
                            change_pct=float(row.get('涨跌幅', 0) or 0),
                            open=float(row.get('开盘', 0) or 0),
                            high=float(row.get('最高', 0) or 0),
                            low=float(row.get('最低', 0) or 0),
                            volume=float(row.get('成交量', 0) or 0),
                            amount=float(row.get('成交额', 0) or 0),
                        )
                        # 计算振幅
                        if index.open > 0:
                            index.amplitude = (index.high - index.low) / index.open * 100
                        indices.append(index)
                        
                except Exception as e:
                    logger.warning(f"[大盘] 获取 {name} 历史数据失败: {e}")
                    continue
                    
            logger.info(f"[大盘] 获取到 {len(indices)} 个指数历史行情")
                
        except Exception as e:
            logger.error(f"[大盘] 获取指数历史行情失败: {e}")
        
        return indices
    
    def _get_market_statistics_hist(self, overview: MarketOverview, target_date: str):
        """获取历史市场涨跌统计（简化版，部分数据不可用）"""
        try:
            logger.info(f"[大盘] 获取历史涨跌统计: {target_date}...")
            
            # 历史数据模式下，部分实时数据不可用
            # 可以从指数成交额估算两市成交额
            total_amount = 0.0
            for idx in overview.indices:
                if idx.code in ['sh000001', 'sz399001']:  # 上证+深证
                    total_amount += idx.amount / 1e8  # 转为亿元
            
            overview.total_amount = total_amount
            
            logger.info(f"[大盘] 历史成交额(估算): {overview.total_amount:.0f}亿")
            logger.warning("[大盘] 历史模式下涨跌家数等数据不可用")
                
        except Exception as e:
            logger.error(f"[大盘] 获取历史涨跌统计失败: {e}")
    
    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计（含情绪指标）"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")
            
            # 获取全部A股实时行情
            df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, "A股实时行情", attempts=2)
            
            if df is not None and not df.empty:
                # 涨跌统计
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    overview.up_count = len(df[df[change_col] > 0])
                    overview.down_count = len(df[df[change_col] < 0])
                    overview.flat_count = len(df[df[change_col] == 0])
                    
                    # 涨停跌停统计（涨跌幅 >= 9.9% 或 <= -9.9%）
                    overview.limit_up_count = len(df[df[change_col] >= 9.9])
                    overview.limit_down_count = len(df[df[change_col] <= -9.9])
                
                # 两市成交额
                amount_col = '成交额'
                if amount_col in df.columns:
                    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                    overview.total_amount = df[amount_col].sum() / 1e8  # 转为亿元
                
                # ========== 情绪指标 ==========
                
                # 平均换手率
                turnover_col = '换手率'
                if turnover_col in df.columns:
                    df[turnover_col] = pd.to_numeric(df[turnover_col], errors='coerce')
                    overview.avg_turnover_rate = df[turnover_col].mean()
                    # 高换手率股票数（>10%）
                    overview.high_turnover_count = len(df[df[turnover_col] > 10])
                
                # 创60日新高/新低
                change_60d_col = '60日涨跌幅'
                high_col = '最高'
                low_col = '最低'
                if change_60d_col in df.columns:
                    df[change_60d_col] = pd.to_numeric(df[change_60d_col], errors='coerce')
                    # 简化判断：60日涨幅>30%且今日创新高
                    overview.new_high_count = len(df[(df[change_60d_col] > 30) & (df[change_col] > 3)])
                    # 60日跌幅>30%且今日创新低
                    overview.new_low_count = len(df[(df[change_60d_col] < -30) & (df[change_col] < -3)])
                
                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿 "
                          f"平均换手:{overview.avg_turnover_rate:.2f}%")
                
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")
    
    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")
            
            # 获取行业板块行情
            df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, "行业板块行情", attempts=2)
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5
                    top = df.nlargest(5, change_col)
                    overview.top_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    def _get_concept_rankings(self, overview: MarketOverview):
        """获取概念板块热点"""
        try:
            logger.info("[大盘] 获取概念板块热点...")
            
            # 获取概念板块行情
            df = self._call_akshare_with_retry(ak.stock_board_concept_name_em, "概念板块行情", attempts=2)
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5概念
                    top = df.nlargest(5, change_col)
                    overview.top_concepts = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5概念
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_concepts = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] 热门概念: {[s['name'] for s in overview.top_concepts]}")
                    
        except Exception as e:
            logger.error(f"[大盘] 获取概念板块失败: {e}")
    
    def _get_margin_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """获取融资融券数据"""
        try:
            logger.info("[大盘] 获取融资融券数据...")
            
            # 获取两融账户信息
            df = self._call_akshare_with_retry(ak.stock_margin_account_info, "融资融券", attempts=2)
            
            if df is not None and not df.empty:
                # 如果指定了日期，尝试找到对应日期的数据
                if target_date and '日期' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
                    date_match = df[df['日期'] == target_date]
                    if not date_match.empty:
                        latest = date_match.iloc[0]
                    else:
                        # 找不到指定日期，使用最近的数据
                        latest = df.iloc[-1]
                        logger.warning(f"[大盘] 未找到 {target_date} 的融资融券数据，使用最新数据")
                else:
                    latest = df.iloc[-1]
                
                # 融资余额（亿元）
                if '融资余额' in df.columns:
                    overview.margin_balance = float(latest.get('融资余额', 0) or 0)
                # 融资买入额（亿元）
                if '融资买入额' in df.columns:
                    overview.margin_buy = float(latest.get('融资买入额', 0) or 0)
                # 融券余额（亿元）
                if '融券余额' in df.columns:
                    overview.short_balance = float(latest.get('融券余额', 0) or 0)
                
                logger.info(f"[大盘] 融资余额: {overview.margin_balance:.0f}亿 "
                          f"融资买入: {overview.margin_buy:.2f}亿 融券余额: {overview.short_balance:.2f}亿")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取融资融券数据失败: {e}")
    
    def _get_lhb_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """
        获取龙虎榜数据
        
        注意：龙虎榜数据通常在收盘后才更新，当天数据可能不可用
        如果当天数据获取失败，会自动尝试获取前几个交易日的数据
        """
        try:
            logger.info("[大盘] 获取龙虎榜数据...")
            
            # 使用指定日期或今天
            if target_date:
                base_date = datetime.strptime(target_date, '%Y-%m-%d')
            else:
                base_date = datetime.now()
            
            df = None
            actual_date = None
            
            # 尝试获取最近几天的数据（当天数据可能还没更新）
            for days_ago in range(0, 5):
                try_date = base_date - pd.Timedelta(days=days_ago)
                date_str = try_date.strftime('%Y%m%d')
                
                try:
                    df = ak.stock_lhb_detail_em(start_date=date_str, end_date=date_str)
                    if df is not None and not df.empty:
                        actual_date = try_date.strftime('%Y-%m-%d')
                        if days_ago > 0:
                            logger.info(f"[大盘] 使用 {actual_date} 的龙虎榜数据（{days_ago}天前）")
                        break
                except Exception as e:
                    if days_ago == 0:
                        logger.debug(f"[大盘] 当天龙虎榜数据暂不可用: {e}")
                    continue
            
            if df is not None and not df.empty:
                # 龙虎榜股票列表
                overview.lhb_stocks = []
                for _, row in df.head(10).iterrows():
                    overview.lhb_stocks.append({
                        'code': row.get('代码', ''),
                        'name': row.get('名称', ''),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'net_buy': float(row.get('龙虎榜净买额', 0) or 0) / 1e8,  # 转为亿元
                        'reason': row.get('上榜原因', ''),
                    })
                
                # 龙虎榜净买入总额
                if '龙虎榜净买额' in df.columns:
                    df['龙虎榜净买额'] = pd.to_numeric(df['龙虎榜净买额'], errors='coerce')
                    overview.lhb_net_buy = df['龙虎榜净买额'].sum() / 1e8
                
                logger.info(f"[大盘] 龙虎榜({actual_date}): {len(df)}只股票上榜, 净买入: {overview.lhb_net_buy:.2f}亿")
            else:
                logger.warning("[大盘] 未能获取到龙虎榜数据")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取龙虎榜数据失败: {e}")
    
    def _get_block_trade_data(self, overview: MarketOverview):
        """获取大宗交易数据"""
        try:
            logger.info("[大盘] 获取大宗交易数据...")
            
            # 获取大宗交易市场统计
            df = self._call_akshare_with_retry(ak.stock_dzjy_sctj, "大宗交易", attempts=2)
            
            if df is not None and not df.empty:
                # 取最新一条数据
                latest = df.iloc[0]  # 按日期降序，第一条是最新
                
                # 大宗交易成交额（元转亿元）
                if '大宗交易成交总额' in df.columns:
                    overview.block_trade_amount = float(latest.get('大宗交易成交总额', 0) or 0) / 1e8
                # 溢价成交占比
                if '溢价成交总额占比' in df.columns:
                    overview.block_trade_premium_ratio = float(latest.get('溢价成交总额占比', 0) or 0)
                # 折价成交占比
                if '折价成交总额占比' in df.columns:
                    overview.block_trade_discount_ratio = float(latest.get('折价成交总额占比', 0) or 0)
                
                logger.info(f"[大盘] 大宗交易: {overview.block_trade_amount:.2f}亿 "
                          f"溢价占比:{overview.block_trade_premium_ratio:.1f}% "
                          f"折价占比:{overview.block_trade_discount_ratio:.1f}%")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取大宗交易数据失败: {e}")

    
    def search_market_news(self, use_smart_search: bool = True) -> List[Dict]:
        """
        搜索市场新闻
        
        Args:
            use_smart_search: 是否使用 LLM 智能搜索优化
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []
        today = datetime.now()
        month_str = f"{today.year}年{today.month}月"
        
        # 尝试使用 LLM 生成智能搜索词
        search_queries = None
        
        if use_smart_search and self.analyzer and self.analyzer.is_available():
            try:
                search_queries = self._generate_market_search_queries()
                if search_queries:
                    logger.info(f"[大盘] 使用 LLM 生成的智能搜索词: {search_queries}")
            except Exception as e:
                logger.warning(f"[大盘] LLM 生成搜索词失败: {e}")
        
        # 如果智能搜索失败，使用默认搜索词
        if not search_queries:
            search_queries = [
                f"A股 大盘 复盘 {month_str}",
                f"股市 行情 分析 今日 {month_str}",
                f"A股 市场 热点 板块 {month_str}",
            ]
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            for query in search_queries:
                # 使用 custom_query 直接传递完整的搜索词
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name="大盘",
                    max_results=3,
                    custom_query=query  # 直接使用完整的搜索词
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def _generate_market_search_queries(self) -> Optional[List[str]]:
        """
        使用 LLM 生成大盘分析的智能搜索词
        
        Returns:
            搜索词列表，失败返回 None
        """
        if not self.analyzer or not self.analyzer.is_available():
            return None
        
        current_date = datetime.now().strftime('%Y年%m月%d日')
        current_month = datetime.now().strftime('%Y年%m月')
        
        prompt = f"""你是一位专业的A股市场分析师，请为今日大盘复盘生成精准的搜索关键词。

## 任务
生成用于搜索今日A股市场新闻和分析的关键词

## 当前日期
{current_date}

## 搜索目的
1. 获取今日大盘走势分析
2. 了解市场热点板块和概念
3. 获取重要政策和消息面信息
4. 了解资金流向和市场情绪

## 要求
1. 生成 4-5 个搜索关键词/短语
2. 关键词要具体、精准，能搜索到有价值的市场信息
3. 包含时间限定词（如"{current_month}"、"今日"、"最新"）
4. 覆盖不同维度：大盘走势、板块热点、政策消息、资金动向

## 输出格式
请直接输出搜索关键词，每行一个，不要编号，不要解释：

示例输出：
A股 大盘 今日 走势分析
股市 热点板块 {current_month}
A股 政策 利好 最新消息
市场 资金流向 主力动向
"""
        
        try:
            generation_config = {
                'temperature': 0.3,
                'max_output_tokens': 300,
            }
            
            response = self.analyzer._call_openai_api(prompt, generation_config)
            
            if not response:
                return None
            
            # 解析搜索词
            queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('示例'):
                    # 移除可能的编号
                    if line[0].isdigit() and '.' in line[:3]:
                        line = line.split('.', 1)[1].strip()
                    if line:
                        queries.append(line)
            
            return queries[:5] if queries else None
            
        except Exception as e:
            logger.warning(f"[大盘] LLM 生成搜索词异常: {e}")
            return None
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        try:
            logger.info("[大盘] 调用大模型生成复盘报告...")
            
            generation_config = {
                'temperature': 0.7,
            }
            
            # 使用 OpenAI 兼容 API
            review = self.analyzer._call_openai_api(prompt, generation_config)
            
            if review:
                logger.info(f"[大盘] 复盘报告生成成功，长度: {len(review)} 字符")
                return review
            else:
                logger.warning("[大盘] 大模型返回为空")
                return self._generate_template_review(overview, news)
                
        except Exception as e:
            logger.error(f"[大盘] 大模型生成复盘报告失败: {e}")
            return self._generate_template_review(overview, news)
    
    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt（增强版，包含多维度数据）"""
        # 指数行情信息
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 行业板块信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]])
        
        # 概念板块信息
        top_concepts_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_concepts[:5]]) if overview.top_concepts else "暂无数据"
        bottom_concepts_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_concepts[:5]]) if overview.bottom_concepts else "暂无数据"
        
        # 龙虎榜股票
        lhb_text = ""
        for stock in overview.lhb_stocks[:5]:
            lhb_text += f"- {stock['name']}({stock['code']}): {stock['change_pct']:+.2f}%, 净买入{stock['net_buy']:.2f}亿, {stock['reason']}\n"
        
        # 新闻信息
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        # 计算涨跌比
        total_stocks = overview.up_count + overview.down_count + overview.flat_count
        up_ratio = overview.up_count / total_stocks * 100 if total_stocks > 0 else 0
        
        prompt = f"""你是一位专业的A股市场分析师，请根据提供的多维度数据生成一份深度大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式和代码块
- emoji 仅在标题处少量使用

---

# 今日市场数据（{overview.date}）

## 一、主要指数
{indices_text}

## 二、市场概况
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} (涨跌比 {up_ratio:.1f}%) |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |
| 平均换手率 | {overview.avg_turnover_rate:.2f}% |
| 高换手(>10%)股票数 | {overview.high_turnover_count} |

## 三、资金流向

### 融资融券
- 融资余额: {overview.margin_balance:.0f}亿
- 融资买入额: {overview.margin_buy:.2f}亿
- 融券余额: {overview.short_balance:.2f}亿

### 大宗交易
- 成交总额: {overview.block_trade_amount:.2f}亿
- 溢价成交占比: {overview.block_trade_premium_ratio:.1f}%
- 折价成交占比: {overview.block_trade_discount_ratio:.1f}%

## 四、板块表现

### 行业板块
- 领涨: {top_sectors_text}
- 领跌: {bottom_sectors_text}

### 概念板块
- 热门: {top_concepts_text}
- 冷门: {bottom_concepts_text}

## 五、龙虎榜（净买入: {overview.lhb_net_buy:.2f}亿）
{lhb_text if lhb_text else "今日无龙虎榜数据"}

## 六、市场新闻
{news_text if news_text else "暂无相关新闻"}

---

# 分析要点

请重点关注：
1. 融资余额变化：杠杆资金是加仓还是减仓？
2. 大宗交易折溢价：折价成交多说明大股东/机构在出货
3. 龙虎榜机构动向：机构席位买入的板块往往是中期主线
4. 概念板块轮动：哪些概念在持续发酵？哪些在退潮？
5. 涨跌比与成交额：赚钱效应如何？量能是否配合？

---

# 输出格式

## 📊 {overview.date} 大盘复盘

### 一、市场总结
（概括今日市场表现、赚钱效应、成交量变化）

### 二、指数点评
（分析各指数走势特点，大小盘风格切换）

### 三、资金动向解读
（综合分析融资融券、大宗交易的信号含义）

### 四、热点解读
（分析板块和概念背后的逻辑，判断持续性）

### 五、龙虎榜点评
（分析主力资金动向，机构偏好的方向）

### 六、后市展望
（给出明日市场预判和操作建议）

### 七、风险提示
（需要关注的风险点）

---

请直接输出复盘报告内容。
"""
        return prompt

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        
        # 判断市场走势
        sh_index = next((idx for idx in overview.indices if idx.code == '000001'), None)
        if sh_index:
            if sh_index.change_pct > 1:
                market_mood = "强势上涨"
            elif sh_index.change_pct > 0:
                market_mood = "小幅上涨"
            elif sh_index.change_pct > -1:
                market_mood = "小幅下跌"
            else:
                market_mood = "明显下跌"
        else:
            market_mood = "震荡整理"
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_text = "、".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = "、".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        report = f"""## 📊 {overview.date} 大盘复盘

### 一、市场总结
今日A股市场整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text}

### 三、涨跌统计
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |

### 四、板块表现
- **领涨**: {top_text}
- **领跌**: {bottom_text}

### 五、风险提示
市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self, target_date: Optional[str] = None, include_opportunity: bool = True) -> str:
        """
        执行每日大盘复盘流程
        
        Args:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为今天
            include_opportunity: 是否包含板块机会分析
        
        Returns:
            复盘报告文本
        """
        date_display = target_date or datetime.now().strftime('%Y-%m-%d')
        logger.info(f"========== 开始大盘复盘分析 ({date_display}) ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview(target_date)
        
        # 2. 搜索市场新闻（历史日期时可能搜索不到相关新闻）
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        # 4. 添加板块机会分析
        if include_opportunity:
            try:
                opportunity_analyzer = SectorOpportunityAnalyzer(self.search_service, self.analyzer)
                opportunities = opportunity_analyzer.find_opportunity_sectors(fast_mode=True)
                opportunity_report = opportunity_analyzer.generate_opportunity_report(opportunities)
                report += "\n\n" + opportunity_report
            except Exception as e:
                logger.warning(f"[大盘] 板块机会分析失败: {e}")
        
        logger.info("========== 大盘复盘分析完成 ==========")
        
        return report


# ============================================================
# 板块埋伏机会分析模块
# ============================================================

@dataclass
class SectorOpportunity:
    """
    板块埋伏机会
    
    埋伏逻辑：必须同时满足以下三个条件中的至少两个，胜率才高
    1. 够便宜（安全垫）：经历了长时间调整，机构仓位低，散户绝望，估值在历史底部
    2. 有催化（导火索）：未来3-6个月内有确定的政策预期、技术突破或产品落地
    3. 有反转（基本面）：行业供需格局改善，从"杀估值"转向"杀业绩"结束，进入业绩修复期
    """
    sector_name: str                    # 板块名称
    sector_code: str                    # 板块代码（申万行业代码）
    
    # ========== 够便宜（安全垫）==========
    current_pe: float = 0.0             # 当前PE
    current_pb: float = 0.0             # 当前PB
    pe_percentile: float = 100.0        # PE历史分位数 (0-100, 越低越便宜)
    pb_percentile: float = 100.0        # PB历史分位数
    price_percentile: float = 100.0     # 价格历史分位数（基于3年数据）
    dividend_yield: float = 0.0         # 股息率
    cheap_score: int = 0                # 便宜得分 (0-4，增加筹码维度)
    cheap_reasons: List[str] = field(default_factory=list)  # 便宜原因
    
    # ========== 筹码集中度（安全垫补充）==========
    avg_chip_concentration: float = 0.0  # 板块平均筹码集中度（90%集中度）
    avg_profit_ratio: float = 0.0        # 板块平均获利比例
    leader_chip_concentration: float = 0.0  # 龙头股筹码集中度
    leader_profit_ratio: float = 0.0     # 龙头股获利比例
    leader_stock_name: str = ""          # 龙头股名称
    chip_analysis: str = ""              # 筹码分析结论
    
    # ========== 有催化（导火索）==========
    recent_news: List[str] = field(default_factory=list)    # 相关新闻标题
    policy_keywords: List[str] = field(default_factory=list)  # 政策关键词
    concept_heat: float = 0.0           # 相关概念热度
    catalyst_score: int = 0             # 催化得分 (0-3)
    catalyst_reasons: List[str] = field(default_factory=list)  # 催化原因
    
    # ========== 有反转（基本面）==========
    recent_5d_change: float = 0.0       # 近5日涨跌幅
    recent_20d_change: float = 0.0      # 近20日涨跌幅
    zt_count: int = 0                   # 近期涨停股数量
    lhb_net_buy: float = 0.0            # 龙虎榜净买入（亿元）
    volume_ratio: float = 0.0           # 成交量比（相对20日均量）
    reversal_score: int = 0             # 反转得分 (0-3)
    reversal_reasons: List[str] = field(default_factory=list)  # 反转原因
    
    # ========== 综合评估 ==========
    total_score: int = 0                # 总分 (0-10)
    recommendation: str = ""            # 推荐理由
    risk_warning: str = ""              # 风险提示
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sector_name': self.sector_name,
            'sector_code': self.sector_code,
            'current_pe': self.current_pe,
            'current_pb': self.current_pb,
            'pe_percentile': self.pe_percentile,
            'price_percentile': self.price_percentile,
            'dividend_yield': self.dividend_yield,
            'cheap_score': self.cheap_score,
            'catalyst_score': self.catalyst_score,
            'reversal_score': self.reversal_score,
            'total_score': self.total_score,
            'recommendation': self.recommendation,
        }


class SectorOpportunityAnalyzer:
    """
    板块埋伏机会分析器
    
    核心逻辑：
    1. 够便宜：PE/PB历史分位数 < 30%，或价格处于3年低位
    2. 有催化：相关概念近期活跃，或有政策/技术催化预期
    3. 有反转：近期有资金流入迹象，涨停股增多，龙虎榜净买入
    
    支持 LLM 深度分析：
    - 传入 analyzer 参数后，可调用 generate_ai_opportunity_report() 生成 AI 深度分析报告
    - AI 会综合分析所有数据，给出更专业的埋伏建议
    """
    
    # 申万一级行业代码映射（用于历史数据查询）
    SW_INDUSTRY_CODES = {
        '801010.SI': '农林牧渔',
        '801030.SI': '基础化工',
        '801040.SI': '钢铁',
        '801050.SI': '有色金属',
        '801080.SI': '电子',
        '801110.SI': '家用电器',
        '801120.SI': '食品饮料',
        '801130.SI': '纺织服饰',
        '801140.SI': '轻工制造',
        '801150.SI': '医药生物',
        '801160.SI': '公用事业',
        '801170.SI': '交通运输',
        '801180.SI': '房地产',
        '801200.SI': '商贸零售',
        '801210.SI': '社会服务',
        '801230.SI': '综合',
        '801710.SI': '建筑材料',
        '801720.SI': '建筑装饰',
        '801730.SI': '电力设备',
        '801740.SI': '国防军工',
        '801750.SI': '计算机',
        '801760.SI': '传媒',
        '801770.SI': '通信',
        '801780.SI': '银行',
        '801790.SI': '非银金融',
        '801880.SI': '汽车',
        '801890.SI': '机械设备',
        '801950.SI': '煤炭',
        '801960.SI': '石油石化',
        '801970.SI': '环保',
        '801980.SI': '美容护理',
    }
    
    # 政策催化关键词（用于新闻匹配）
    POLICY_KEYWORDS = {
        '农林牧渔': ['粮食安全', '乡村振兴', '种业', '农业现代化'],
        '基础化工': ['新材料', '碳中和', '化工新材料'],
        '钢铁': ['基建', '特钢', '钢铁重组'],
        '有色金属': ['新能源', '稀土', '锂电', '铜'],
        '电子': ['半导体', '芯片', '国产替代', 'AI芯片'],
        '家用电器': ['消费复苏', '家电下乡', '智能家居'],
        '食品饮料': ['消费升级', '白酒', '预制菜'],
        '医药生物': ['创新药', '医保', '集采', '中药'],
        '房地产': ['房地产政策', '保交楼', '城中村'],
        '银行': ['降息', '利率', '金融改革'],
        '非银金融': ['注册制', '券商', '保险'],
        '电力设备': ['新能源', '光伏', '储能', '电网'],
        '国防军工': ['军工', '国防', '航空航天'],
        '计算机': ['人工智能', 'AI', '信创', '数字经济'],
        '传媒': ['游戏', 'AI应用', '短视频', '文化'],
        '通信': ['5G', '6G', '算力', '数据中心'],
        '汽车': ['新能源汽车', '智能驾驶', '汽车出海'],
        '机械设备': ['工业母机', '机器人', '高端装备'],
        '煤炭': ['能源安全', '煤炭保供'],
        '石油石化': ['油价', '能源安全', '炼化'],
        '环保': ['碳中和', '环保督察', '垃圾处理'],
        '美容护理': ['国货美妆', '消费复苏'],
    }
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        初始化板块机会分析器
        
        Args:
            search_service: 搜索服务实例（用于搜索催化剂新闻）
            analyzer: AI分析器实例（用于调用LLM生成深度分析）
        """
        self.search_service = search_service
        self.analyzer = analyzer
        self._sw_info_cache: Optional[pd.DataFrame] = None
        self._industry_hist_cache: Dict[str, pd.DataFrame] = {}
        self._em_industry_mapping: Optional[Dict[str, str]] = None  # 东财行业板块名称->代码映射缓存
        
    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 2):
        """带重试的akshare调用"""
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[板块机会] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 5))
        logger.error(f"[板块机会] {name} 最终失败: {last_error}")
        return None
    
    def _get_em_industry_mapping(self) -> Dict[str, str]:
        """
        获取东财行业板块名称到代码的映射
        
        Returns:
            {板块名称: 板块代码} 字典，如 {'小金属': 'BK1027', '银行': 'BK0475'}
        """
        if not hasattr(self, '_em_industry_mapping') or self._em_industry_mapping is None:
            try:
                df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, "东财行业板块列表")
                if df is not None and not df.empty:
                    # 构建名称到代码的映射
                    self._em_industry_mapping = {}
                    for _, row in df.iterrows():
                        name = str(row.get('板块名称', ''))
                        code = str(row.get('板块代码', ''))
                        if name and code:
                            self._em_industry_mapping[name] = code
                    logger.info(f"[板块机会] 缓存东财行业板块映射: {len(self._em_industry_mapping)} 个")
                else:
                    self._em_industry_mapping = {}
            except Exception as e:
                logger.warning(f"[板块机会] 获取东财行业板块映射失败: {e}")
                self._em_industry_mapping = {}
        
        return self._em_industry_mapping
    
    def _find_em_sector(self, sector_name: str) -> Optional[str]:
        """
        根据申万行业名称查找对应的东财板块名称或代码
        
        Args:
            sector_name: 申万行业名称（如"银行"、"有色金属"）
            
        Returns:
            东财板块名称或代码，找不到返回 None
        """
        mapping = self._get_em_industry_mapping()
        if not mapping:
            return None
        
        # 1. 精确匹配
        if sector_name in mapping:
            return sector_name
        
        # 2. 模糊匹配（申万名称可能与东财名称略有不同）
        for em_name in mapping.keys():
            # 包含关系匹配
            if sector_name in em_name or em_name in sector_name:
                logger.debug(f"[板块机会] 板块名称匹配: {sector_name} -> {em_name}")
                return em_name
            # 前两个字匹配
            if len(sector_name) >= 2 and len(em_name) >= 2 and sector_name[:2] == em_name[:2]:
                logger.debug(f"[板块机会] 板块名称前缀匹配: {sector_name} -> {em_name}")
                return em_name
        
        return None
    
    def _get_sector_constituents(self, sector_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        获取板块成分股（按市值排序取前N只）
        
        优化策略：
        1. 先从缓存的东财板块映射中查找正确的板块名称
        2. 使用正确的板块名称查询成分股
        3. 支持传入板块代码（如 BK1027）
        
        Args:
            sector_name: 板块名称（申万或东财行业板块名称）
            top_n: 获取前N只股票
            
        Returns:
            成分股列表，每个元素包含 code, name, market_cap
        """
        try:
            logger.debug(f"[板块机会] 获取 {sector_name} 成分股...")
            
            # 查找正确的东财板块名称
            em_sector = self._find_em_sector(sector_name)
            query_name = em_sector if em_sector else sector_name
            
            # 获取成分股
            df = self._call_akshare_with_retry(
                lambda: ak.stock_board_industry_cons_em(symbol=query_name),
                f"{query_name}成分股"
            )
            
            if df is None or df.empty:
                logger.warning(f"[板块机会] {sector_name} 成分股数据为空")
                return []
            
            # 按总市值排序（如果有该列）
            if '总市值' in df.columns:
                df['总市值'] = pd.to_numeric(df['总市值'], errors='coerce')
                df = df.sort_values('总市值', ascending=False)
            
            # 取前N只
            result = []
            for _, row in df.head(top_n).iterrows():
                result.append({
                    'code': str(row.get('代码', '')),
                    'name': str(row.get('名称', '')),
                    'market_cap': float(row.get('总市值', 0) or 0),
                })
            
            logger.debug(f"[板块机会] {sector_name} 获取到 {len(result)} 只成分股")
            return result
            
        except Exception as e:
            logger.warning(f"[板块机会] 获取 {sector_name} 成分股失败: {e}")
            return []
    
    def _analyze_sector_chips(self, opp: SectorOpportunity, em_sector_name: Optional[str] = None) -> None:
        """
        分析板块筹码集中度
        
        策略：
        1. 获取板块前3-5只龙头股（按市值）
        2. 获取每只股票的筹码分布数据
        3. 计算板块平均筹码集中度和获利比例
        4. 识别龙头股的筹码状态
        
        Args:
            opp: 板块机会对象
            em_sector_name: 东财板块名称（用于获取成分股）
        """
        try:
            # 使用东财板块名称获取成分股
            sector_name = em_sector_name or opp.sector_name
            
            # 获取板块龙头股（前5只）
            constituents = self._get_sector_constituents(sector_name, top_n=10)
            
            if not constituents:
                logger.debug(f"[板块机会] {opp.sector_name} 无法获取成分股，跳过筹码分析")
                return
            
            # 导入 AkshareFetcher 获取筹码数据
            from data_provider.akshare_fetcher import AkshareFetcher
            fetcher = AkshareFetcher(sleep_min=1, sleep_max=1.5)  # 减少等待时间
            
            chip_data_list = []
            leader_chip = None
            leader_name = ""
            
            for i, stock in enumerate(constituents[:5]):  # 只分析前3只，避免API调用过多
                code = stock['code']
                name = stock['name']
                
                try:
                    chip = fetcher.get_chip_distribution(code)
                    if chip:
                        chip_data_list.append({
                            'code': code,
                            'name': name,
                            'concentration_90': chip.concentration_90,
                            'profit_ratio': chip.profit_ratio,
                            'avg_cost': chip.avg_cost,
                        })
                        
                        # 第一只就是龙头股
                        if i == 0:
                            leader_chip = chip
                            leader_name = name
                            
                except Exception as e:
                    logger.debug(f"[板块机会] 获取 {code} 筹码数据失败: {e}")
                    continue
                
                time.sleep(0.3)  # 避免请求过快
            
            if not chip_data_list:
                logger.debug(f"[板块机会] {opp.sector_name} 无有效筹码数据")
                return
            
            # 计算板块平均值
            avg_concentration = sum(d['concentration_90'] for d in chip_data_list) / len(chip_data_list)
            avg_profit = sum(d['profit_ratio'] for d in chip_data_list) / len(chip_data_list)
            
            # 更新板块机会对象
            opp.avg_chip_concentration = avg_concentration
            opp.avg_profit_ratio = avg_profit
            
            if leader_chip:
                opp.leader_chip_concentration = leader_chip.concentration_90
                opp.leader_profit_ratio = leader_chip.profit_ratio
                opp.leader_stock_name = leader_name
            
            # 生成筹码分析结论
            analysis_parts = []
            
            # 筹码集中度分析
            if avg_concentration < 0.10:
                analysis_parts.append("筹码高度集中")
            elif avg_concentration < 0.15:
                analysis_parts.append("筹码较集中")
            elif avg_concentration < 0.25:
                analysis_parts.append("筹码分散度中等")
            else:
                analysis_parts.append("筹码较分散")
            
            # 获利比例分析
            if avg_profit < 0.30:
                analysis_parts.append("套牢盘较重(获利<30%)")
            elif avg_profit < 0.50:
                analysis_parts.append("获利盘中等(30-50%)")
            elif avg_profit < 0.70:
                analysis_parts.append("获利盘较多(50-70%)")
            else:
                analysis_parts.append("获利盘极高(>70%)")
            
            opp.chip_analysis = "，".join(analysis_parts)
            
            logger.info(f"[板块机会] {opp.sector_name} 筹码分析: 平均集中度={avg_concentration:.1%}, "
                       f"平均获利比例={avg_profit:.1%}, 龙头={leader_name}")
            
        except Exception as e:
            logger.warning(f"[板块机会] {opp.sector_name} 筹码分析失败: {e}")
    
    def _get_sw_industry_info(self) -> Optional[pd.DataFrame]:
        """获取申万一级行业当前估值信息"""
        if self._sw_info_cache is not None:
            return self._sw_info_cache
        
        try:
            logger.info("[板块机会] 获取申万行业估值信息...")
            df = self._call_akshare_with_retry(ak.sw_index_first_info, "申万行业信息")
            if df is not None:
                self._sw_info_cache = df
                logger.info(f"[板块机会] 获取到 {len(df)} 个申万一级行业")
            return df
        except Exception as e:
            logger.error(f"[板块机会] 获取申万行业信息失败: {e}")
            return None
    
    def _get_industry_hist(self, symbol: str, days: int = 750) -> Optional[pd.DataFrame]:
        """
        获取申万行业指数历史数据（约3年）
        
        Args:
            symbol: 申万行业代码，如 '801030'
            days: 获取天数
        """
        if symbol in self._industry_hist_cache:
            return self._industry_hist_cache[symbol]
        
        try:
            df = self._call_akshare_with_retry(
                lambda: ak.index_hist_sw(symbol=symbol, period='day'),
                f"申万指数历史({symbol})"
            )
            if df is not None and not df.empty:
                # 只保留最近N天
                df = df.tail(days).copy()
                self._industry_hist_cache[symbol] = df
            return df
        except Exception as e:
            logger.warning(f"[板块机会] 获取 {symbol} 历史数据失败: {e}")
            return None
    
    def _calculate_price_percentile(self, symbol: str) -> float:
        """
        计算价格历史分位数
        
        Args:
            symbol: 申万行业代码
            
        Returns:
            分位数 (0-100)，越低表示当前价格越便宜
        """
        # 为了提高效率，使用简化的估算方法
        # 基于当前PE/PB与历史均值的比较
        return 50.0  # 默认中位数，后续可通过批量获取优化
    
    def _calculate_price_percentile_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        批量计算价格历史分位数（优化版本）
        
        Args:
            symbols: 申万行业代码列表
            
        Returns:
            {symbol: percentile} 字典
        """
        result = {}
        for symbol in symbols:
            df = self._get_industry_hist(symbol)
            if df is None or df.empty:
                result[symbol] = 50.0
                continue
            
            try:
                current_price = float(df['收盘'].iloc[-1])
                all_prices = df['收盘'].astype(float)
                percentile = (all_prices < current_price).sum() / len(all_prices) * 100
                result[symbol] = percentile
            except Exception as e:
                logger.warning(f"[板块机会] 计算 {symbol} 价格分位数失败: {e}")
                result[symbol] = 50.0
        
        return result
    
    def _get_em_industry_realtime(self) -> Optional[pd.DataFrame]:
        """获取东财行业板块实时行情"""
        try:
            df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, "东财行业板块")
            return df
        except Exception as e:
            logger.error(f"[板块机会] 获取东财行业板块失败: {e}")
            return None
    
    def _get_zt_pool_by_industry(self, date: Optional[str] = None) -> Dict[str, int]:
        """
        获取各行业涨停股数量
        
        Returns:
            {行业名称: 涨停数量}
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        industry_zt_count: Dict[str, int] = {}
        
        try:
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_em(date=date),
                "涨停股池"
            )
            if df is not None and not df.empty and '所属行业' in df.columns:
                industry_zt_count = df['所属行业'].value_counts().to_dict()
                logger.info(f"[板块机会] 涨停股池: {len(df)} 只，涉及 {len(industry_zt_count)} 个行业")
        except Exception as e:
            logger.warning(f"[板块机会] 获取涨停股池失败: {e}")
        
        return industry_zt_count
    
    def _get_fund_position(self) -> float:
        """获取当前股票型基金仓位"""
        try:
            df = self._call_akshare_with_retry(ak.fund_stock_position_lg, "基金仓位")
            if df is not None and not df.empty:
                return float(df['position'].iloc[-1])
        except Exception as e:
            logger.warning(f"[板块机会] 获取基金仓位失败: {e}")
        return 90.0  # 默认值
    
    def _analyze_cheap(self, opp: SectorOpportunity, sw_row: pd.Series, 
                       price_percentiles: Optional[Dict[str, float]] = None) -> None:
        """
        分析"够便宜"维度
        
        评分标准：
        - PE分位数 < 20%: +1分
        - PB分位数 < 20%: +1分  
        - 价格分位数 < 30%: +1分
        - 股息率 > 3%: +1分（额外加分）
        - 筹码集中度低 + 获利比例低: +1分（筹码维度）
        """
        score = 0
        reasons = []
        
        # 获取当前估值
        opp.current_pe = float(sw_row.get('TTM(滚动)市盈率', 0) or 0)
        opp.current_pb = float(sw_row.get('市净率', 0) or 0)
        opp.dividend_yield = float(sw_row.get('静态股息率', 0) or 0)
        
        # 获取价格分位数
        code = str(sw_row.get('行业代码', '')).replace('.SI', '')
        if price_percentiles and code in price_percentiles:
            opp.price_percentile = price_percentiles[code]
        else:
            opp.price_percentile = 50.0  # 默认值
        
        # 简化的PE/PB分位数估算（基于价格分位数）
        opp.pe_percentile = opp.price_percentile
        opp.pb_percentile = opp.price_percentile
        
        # 评分
        if opp.price_percentile < 20:
            score += 1
            reasons.append(f"价格处于历史{opp.price_percentile:.0f}%分位（极低）")
        elif opp.price_percentile < 30:
            score += 1
            reasons.append(f"价格处于历史{opp.price_percentile:.0f}%分位（较低）")
        
        # PE评估（低PE行业更便宜）
        if opp.current_pe > 0 and opp.current_pe < 15:
            score += 1
            reasons.append(f"PE仅{opp.current_pe:.1f}倍（低估值）")
        
        # 高股息
        if opp.dividend_yield > 3:
            score += 1
            reasons.append(f"股息率{opp.dividend_yield:.2f}%（高分红）")
        
        # 筹码维度评分（如果有筹码数据）
        if opp.avg_chip_concentration > 0:
            # 筹码集中度低（<15%）且获利比例低（<40%）表示卖压已释放
            if opp.avg_chip_concentration < 0.15 and opp.avg_profit_ratio < 0.40:
                score += 1
                reasons.append(f"筹码集中({opp.avg_chip_concentration:.0%})且套牢盘重({opp.avg_profit_ratio:.0%})，卖压释放")
            elif opp.avg_chip_concentration < 0.12:
                score += 1
                reasons.append(f"筹码高度集中({opp.avg_chip_concentration:.0%})，主力控盘")
            elif opp.avg_profit_ratio < 0.30:
                score += 1
                reasons.append(f"获利盘极低({opp.avg_profit_ratio:.0%})，抛压枯竭")
        
        opp.cheap_score = min(score, 4)  # 最高4分（增加筹码维度）
        opp.cheap_reasons = reasons
    
    def _analyze_catalyst(self, opp: SectorOpportunity, concept_df: Optional[pd.DataFrame] = None,
                          search_result: Optional[Dict[str, Any]] = None) -> None:
        """
        分析"有催化"维度
        
        评分标准：
        - 相关概念近5日涨幅 > 5%: +1分
        - 有政策关键词匹配: +1分
        - 近期有相关新闻/政策: +1分（通过智能搜索获取）
        
        Args:
            opp: 板块机会对象
            concept_df: 概念板块数据
            search_result: 智能搜索结果（可选）
        """
        score = 0
        reasons = []
        
        # 获取相关政策关键词
        keywords = self.POLICY_KEYWORDS.get(opp.sector_name, [])
        opp.policy_keywords = keywords
        
        # 检查概念板块热度（使用传入的缓存数据）
        if concept_df is not None and not concept_df.empty:
            try:
                # 查找与行业相关的概念
                for keyword in keywords:
                    matched = concept_df[concept_df['板块名称'].str.contains(keyword, na=False)]
                    if not matched.empty:
                        change = float(matched['涨跌幅'].iloc[0])
                        if change > 3:
                            score += 1
                            reasons.append(f"相关概念'{keyword}'今日涨{change:.1f}%")
                            opp.concept_heat = change
                            break
            except Exception as e:
                logger.warning(f"[板块机会] 检查概念热度失败: {e}")
        
        # 使用智能搜索结果（如果有）
        if search_result and search_result.get('success'):
            catalyst_results = search_result.get('catalyst', {})
            if catalyst_results.get('results'):
                score += 1
                # 提取搜索到的催化剂信息
                top_news = catalyst_results['results'][:2]
                news_titles = [r.title[:30] for r in top_news]
                reasons.append(f"搜索到催化剂: {'; '.join(news_titles)}")
                opp.recent_news = [r.title for r in catalyst_results['results'][:5]]
            
            # 如果有 LLM 摘要，添加到原因中
            if catalyst_results.get('summary'):
                reasons.append(f"AI分析: {catalyst_results['summary'][:100]}")
        
        # 政策关键词本身就是催化信号
        if keywords and score == 0:
            score += 1
            reasons.append(f"关注催化: {', '.join(keywords[:3])}")
        
        opp.catalyst_score = min(score, 3)
        opp.catalyst_reasons = reasons
    
    def _analyze_reversal(self, opp: SectorOpportunity, em_row: Optional[pd.Series], 
                          zt_count_map: Dict[str, int]) -> None:
        """
        分析"有反转"维度
        
        评分标准：
        - 近5日涨幅 > 3%: +1分（资金开始关注）
        - 行业涨停股 >= 3只: +1分（赚钱效应）
        - 换手率放大: +1分（量能配合）
        """
        score = 0
        reasons = []
        
        # 从东财数据获取近期表现
        if em_row is not None:
            try:
                opp.recent_5d_change = float(em_row.get('涨跌幅', 0) or 0)
                
                # 近期涨幅判断
                if opp.recent_5d_change > 5:
                    score += 1
                    reasons.append(f"今日涨{opp.recent_5d_change:.1f}%（强势）")
                elif opp.recent_5d_change > 2:
                    score += 1
                    reasons.append(f"今日涨{opp.recent_5d_change:.1f}%（走强）")
                
                # 换手率
                turnover = float(em_row.get('换手率', 0) or 0)
                if turnover > 2:
                    score += 1
                    reasons.append(f"换手率{turnover:.1f}%（活跃）")
                    opp.volume_ratio = turnover
            except Exception as e:
                logger.warning(f"[板块机会] 解析东财数据失败: {e}")
        
        # 涨停股数量
        # 需要匹配行业名称（东财和申万命名可能不同）
        for industry_name, count in zt_count_map.items():
            if opp.sector_name in industry_name or industry_name in opp.sector_name:
                opp.zt_count = count
                if count >= 5:
                    score += 1
                    reasons.append(f"涨停{count}只（赚钱效应强）")
                elif count >= 3:
                    score += 1
                    reasons.append(f"涨停{count}只（有赚钱效应）")
                break
        
        opp.reversal_score = min(score, 3)
        opp.reversal_reasons = reasons
    
    def _generate_recommendation(self, opp: SectorOpportunity) -> None:
        """生成推荐理由和风险提示"""
        opp.total_score = opp.cheap_score + opp.catalyst_score + opp.reversal_score
        
        # 统计满足条件数（便宜维度阈值调整为>=2，因为最高4分）
        conditions_met = sum([
            opp.cheap_score >= 2,
            opp.catalyst_score >= 1,
            opp.reversal_score >= 1
        ])
        
        # 生成推荐理由
        all_reasons = opp.cheap_reasons + opp.catalyst_reasons + opp.reversal_reasons
        
        # 添加筹码分析结论
        if opp.chip_analysis:
            all_reasons.append(f"筹码: {opp.chip_analysis}")
        
        if conditions_met >= 2:
            opp.recommendation = f"【推荐埋伏】满足{conditions_met}/3条件。" + "；".join(all_reasons[:4])
        elif conditions_met == 1:
            opp.recommendation = f"【观察】仅满足1个条件。" + "；".join(all_reasons[:3])
        else:
            opp.recommendation = f"【暂不推荐】条件不足。"
        
        # 风险提示
        risks = []
        if opp.cheap_score == 0:
            risks.append("估值不便宜")
        if opp.catalyst_score == 0:
            risks.append("缺乏催化剂")
        if opp.reversal_score == 0:
            risks.append("尚无反转信号")
        # 筹码风险提示
        if opp.avg_profit_ratio > 0.80:
            risks.append("获利盘过高，注意抛压")
        opp.risk_warning = "；".join(risks) if risks else "暂无明显风险"
    
    def find_opportunity_sectors(self, fast_mode: bool = True, use_smart_search: bool = True, 
                                   analyze_chips: bool = True) -> List[SectorOpportunity]:
        """
        寻找符合埋伏条件的板块
        
        Args:
            fast_mode: 快速模式，跳过耗时的历史数据计算
            use_smart_search: 是否使用智能搜索获取催化剂信息
            analyze_chips: 是否分析筹码集中度（会增加API调用）
        
        Returns:
            按总分排序的板块机会列表
        """
        logger.info("========== 开始板块机会分析 ==========")
        
        opportunities: List[SectorOpportunity] = []
        
        # 1. 获取申万行业估值数据
        sw_df = self._get_sw_industry_info()
        if sw_df is None or sw_df.empty:
            logger.error("[板块机会] 无法获取申万行业数据")
            return opportunities
        
        # 2. 获取东财行业板块实时数据
        em_df = self._get_em_industry_realtime()
        
        # 3. 获取涨停股池按行业统计
        zt_count_map = self._get_zt_pool_by_industry()
        
        # 4. 获取概念板块数据（一次性获取，避免重复调用）
        concept_df = self._call_akshare_with_retry(ak.stock_board_concept_name_em, "概念板块")
        
        # 5. 批量计算价格分位数（如果不是快速模式）
        price_percentiles: Optional[Dict[str, float]] = None
        if not fast_mode:
            logger.info("[板块机会] 计算历史价格分位数（耗时较长）...")
            symbols = [str(row['行业代码']).replace('.SI', '') for _, row in sw_df.iterrows()]
            price_percentiles = self._calculate_price_percentile_batch(symbols)
        
        # 6. 初始化智能搜索服务（如果启用）
        smart_search = None
        sector_search_results: Dict[str, Dict[str, Any]] = {}
        
        if use_smart_search and self.search_service and self.analyzer:
            try:
                from search_service import SmartSearchService
                # 创建智能搜索服务
                smart_search = SmartSearchService(
                    bocha_keys=getattr(self.search_service, '_providers', []),
                    analyzer=self.analyzer
                )
                # 复用现有的搜索引擎
                smart_search._providers = self.search_service._providers
                logger.info("[板块机会] 已启用智能搜索服务")
            except Exception as e:
                logger.warning(f"[板块机会] 初始化智能搜索失败: {e}")
        
        # 7. 对高潜力板块进行智能搜索（只搜索估值较低的前10个板块）
        if smart_search and smart_search.is_available:
            # 先快速筛选出可能有价值的板块
            potential_sectors = []
            for _, sw_row in sw_df.iterrows():
                sector_name = str(sw_row.get('行业名称', ''))
                pe = float(sw_row.get('TTM(滚动)市盈率', 100) or 100)
                dividend = float(sw_row.get('静态股息率', 0) or 0)
                # 低PE或高股息的板块优先搜索
                if pe < 20 or dividend > 3:
                    potential_sectors.append(sector_name)
            
            # 限制搜索数量（避免API调用过多）
            sectors_to_search = potential_sectors[:5]
            
            if sectors_to_search:
                logger.info(f"[板块机会] 对 {len(sectors_to_search)} 个高潜力板块进行智能搜索...")
                
                for sector_name in sectors_to_search:
                    try:
                        keywords = self.POLICY_KEYWORDS.get(sector_name, [])
                        result = smart_search.search_sector_comprehensive(
                            sector_name, 
                            policy_keywords=keywords,
                            use_llm=True
                        )
                        sector_search_results[sector_name] = result
                        logger.info(f"[板块机会] {sector_name} 智能搜索完成")
                        time.sleep(0.5)  # 避免请求过快
                    except Exception as e:
                        logger.warning(f"[板块机会] {sector_name} 智能搜索失败: {e}")
        
        # 8. 遍历每个行业进行分析
        for _, sw_row in sw_df.iterrows():
            sector_name = str(sw_row.get('行业名称', ''))
            sector_code = str(sw_row.get('行业代码', ''))
            
            if not sector_name:
                continue
            
            opp = SectorOpportunity(
                sector_name=sector_name,
                sector_code=sector_code
            )
            
            # 匹配东财数据（提前匹配，用于筹码分析）
            em_row = None
            em_sector_name = None
            if em_df is not None and not em_df.empty:
                matched = em_df[em_df['板块名称'].str.contains(sector_name[:2], na=False)]
                if not matched.empty:
                    em_row = matched.iloc[0]
                    em_sector_name = str(em_row.get('板块名称', ''))
            
            # 分析筹码集中度（在 _analyze_cheap 之前，因为筹码数据会影响便宜得分）
            if analyze_chips:
                # 只对低估值板块进行筹码分析（减少API调用）
                pe = float(sw_row.get('TTM(滚动)市盈率', 100) or 100)
                dividend = float(sw_row.get('静态股息率', 0) or 0)
                if pe < 25 or dividend > 2.5:
                    self._analyze_sector_chips(opp, em_sector_name)
            
            # 分析三个维度
            self._analyze_cheap(opp, sw_row, price_percentiles)
            
            # 获取该板块的智能搜索结果（如果有）
            search_result = sector_search_results.get(sector_name)
            self._analyze_catalyst(opp, concept_df, search_result)
            
            self._analyze_reversal(opp, em_row, zt_count_map)
            
            # 生成推荐
            self._generate_recommendation(opp)
            
            opportunities.append(opp)
        
        # 9. 按总分排序
        opportunities.sort(key=lambda x: (x.total_score, x.cheap_score), reverse=True)
        
        # 10. 输出分析结果
        logger.info(f"[板块机会] 分析完成，共 {len(opportunities)} 个行业")
        
        # 输出推荐的板块
        recommended = [o for o in opportunities if o.total_score >= 4]
        if recommended:
            logger.info(f"[板块机会] 推荐埋伏 {len(recommended)} 个板块:")
            for opp in recommended[:5]:
                logger.info(f"  - {opp.sector_name}: 总分{opp.total_score} "
                          f"(便宜:{opp.cheap_score} 催化:{opp.catalyst_score} 反转:{opp.reversal_score})")
        
        logger.info("========== 板块机会分析完成 ==========")
        
        return opportunities
    
    def _build_opportunity_prompt(self, opportunities: List[SectorOpportunity]) -> str:
        """
        构建板块机会分析的 LLM 提示词
        
        将所有板块数据整理成结构化的提示词，供 LLM 进行深度分析
        
        Args:
            opportunities: 板块机会列表
            
        Returns:
            格式化的提示词
        """
        # 分类板块
        recommended = [o for o in opportunities if o.total_score >= 4]
        watching = [o for o in opportunities if o.total_score == 3]
        cheapest = sorted(opportunities, key=lambda x: x.price_percentile)[:10]
        hottest = sorted(opportunities, key=lambda x: x.reversal_score, reverse=True)[:10]
        
        # 构建数据表格
        def format_sector_data(opp: SectorOpportunity) -> str:
            return (f"| {opp.sector_name} | {opp.total_score} | {opp.cheap_score} | "
                   f"{opp.catalyst_score} | {opp.reversal_score} | "
                   f"{opp.current_pe:.1f} | {opp.current_pb:.1f} | "
                   f"{opp.dividend_yield:.1f}% | {opp.price_percentile:.0f}% |")
        
        # 推荐板块详情
        recommended_details = ""
        for i, opp in enumerate(recommended[:8], 1):
            # 筹码信息
            chip_info = ""
            if opp.avg_chip_concentration > 0:
                chip_info = f"""
**筹码分析**：
- 板块平均筹码集中度: {opp.avg_chip_concentration:.1%}
- 板块平均获利比例: {opp.avg_profit_ratio:.1%}
- 龙头股: {opp.leader_stock_name}（集中度{opp.leader_chip_concentration:.1%}，获利{opp.leader_profit_ratio:.1%}）
- 筹码结论: {opp.chip_analysis}
"""
            
            recommended_details += f"""
### {i}. {opp.sector_name}（总分 {opp.total_score}/10）

**估值数据**：
- PE: {opp.current_pe:.1f}倍
- PB: {opp.current_pb:.1f}倍
- 股息率: {opp.dividend_yield:.1f}%
- 价格分位数: {opp.price_percentile:.0f}%（3年历史）
{chip_info}
**够便宜分析**（得分 {opp.cheap_score}/4）：
{chr(10).join('- ' + r for r in opp.cheap_reasons) if opp.cheap_reasons else '- 暂无明显便宜信号'}

**有催化分析**（得分 {opp.catalyst_score}/3）：
- 关注催化关键词: {', '.join(opp.policy_keywords[:5]) if opp.policy_keywords else '无'}
{chr(10).join('- ' + r for r in opp.catalyst_reasons) if opp.catalyst_reasons else '- 暂无明显催化剂'}

**有反转分析**（得分 {opp.reversal_score}/3）：
- 今日涨跌幅: {opp.recent_5d_change:+.1f}%
- 涨停股数量: {opp.zt_count}只
- 换手率: {opp.volume_ratio:.1f}%
{chr(10).join('- ' + r for r in opp.reversal_reasons) if opp.reversal_reasons else '- 暂无反转信号'}

"""
        
        # 估值最低板块
        cheapest_table = "| 板块 | PE | PB | 股息率 | 价格分位 | 筹码集中度 | 获利比例 |\n|------|-----|-----|--------|----------|------------|----------|\n"
        for opp in cheapest:
            chip_conc = f"{opp.avg_chip_concentration:.0%}" if opp.avg_chip_concentration > 0 else "-"
            profit_ratio = f"{opp.avg_profit_ratio:.0%}" if opp.avg_profit_ratio > 0 else "-"
            cheapest_table += f"| {opp.sector_name} | {opp.current_pe:.1f} | {opp.current_pb:.1f} | {opp.dividend_yield:.1f}% | {opp.price_percentile:.0f}% | {chip_conc} | {profit_ratio} |\n"
        
        # 今日最活跃板块
        hottest_table = "| 板块 | 涨跌幅 | 涨停数 | 换手率 | 反转得分 |\n|------|--------|--------|--------|----------|\n"
        for opp in hottest:
            hottest_table += f"| {opp.sector_name} | {opp.recent_5d_change:+.1f}% | {opp.zt_count} | {opp.volume_ratio:.1f}% | {opp.reversal_score}/3 |\n"
        
        prompt = f"""你是一位专业的A股行业分析师，擅长发现板块埋伏机会。请根据以下数据进行深度分析。您也可以基于您的理解，进行网络搜寻。

# 板块埋伏机会分析

## 核心埋伏逻辑

在A股埋伏板块，必须同时满足以下三个条件中的至少两个，胜率才高：

1. **够便宜（安全垫）**（最高4分）：
   - 经历了长时间调整，估值在历史底部
   - PE/PB处于历史低位（分位数<30%）
   - 机构仓位低，散户绝望
   - 高股息率提供安全边际
   - **筹码集中度低（<15%）表示主力控盘**
   - **获利比例低（<40%）表示抛压已释放**

2. **有催化（导火索）**（最高3分）：
   - 未来3-6个月内有确定的政策预期（如"十五五"规划）
   - 技术突破或产品落地
   - 行业重大事件或政策利好

3. **有反转（基本面）**（最高3分）：
   - 行业供需格局改善
   - 从"杀估值"转向"杀业绩"结束
   - 资金开始流入，涨停股增多
   - 龙虎榜机构净买入

---

## 当前市场数据

### 一、推荐埋伏板块（总分≥4）

共有 **{len(recommended)}** 个板块符合推荐条件：

{recommended_details if recommended_details else '暂无符合条件的板块'}

### 二、观察板块（总分=3）

共有 **{len(watching)}** 个板块处于观察状态。

### 三、估值最低板块 TOP10（含筹码数据）

{cheapest_table}

### 四、今日最活跃板块 TOP10

{hottest_table}

---

## 筹码分析说明

筹码集中度和获利比例是判断板块安全边际的重要指标：
- **筹码集中度**：90%筹码的价格区间占比，越低表示筹码越集中，主力控盘程度越高
- **获利比例**：当前价格下的获利筹码占比，越低表示套牢盘越重，但也意味着抛压已释放

理想的埋伏标的：筹码集中度<15%（主力控盘）+ 获利比例<40%（抛压枯竭）

---

## 分析任务

请基于以上数据，生成一份专业的【板块埋伏机会深度分析报告】，包含：

### 输出格式要求

请直接输出 Markdown 格式的分析报告，包含以下章节：

## 🎯 板块埋伏机会深度分析

### 一、核心推荐（最值得埋伏的2-3个板块）

对于每个推荐板块，请分析：
1. 为什么便宜？（估值分析、历史对比、筹码状态）
2. 催化剂是什么？（政策、技术、事件）
3. 反转信号有哪些？（资金、量能、赚钱效应）
4. 具体埋伏策略（时机、仓位、止损）

### 二、潜力观察（值得关注但时机未到的板块）

分析哪些板块虽然暂时不满足条件，但可能即将满足

### 三、风险警示（需要回避的板块）

哪些板块看似便宜但有陷阱？特别关注获利盘过高的板块

### 四、操作建议

1. 短期（1-2周）：哪些板块可以开始建仓？
2. 中期（1-3月）：哪些板块值得持续跟踪？
3. 仓位建议：如何分配资金？

### 五、风险提示

当前市场环境下的主要风险点

---

请直接输出分析报告，不要输出 JSON 格式。
"""
        return prompt
    
    def generate_ai_opportunity_report(self, opportunities: List[SectorOpportunity]) -> Optional[str]:
        """
        使用 LLM 生成板块机会深度分析报告
        
        Args:
            opportunities: 板块机会列表
            
        Returns:
            AI 生成的深度分析报告，如果 LLM 不可用则返回 None
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[板块机会] AI分析器未配置或不可用，无法生成深度分析")
            return None
        
        try:
            logger.info("[板块机会] 开始调用 LLM 生成深度分析报告...")
            
            # 构建提示词
            prompt = self._build_opportunity_prompt(opportunities)
            logger.info(f"[板块机会] Prompt 长度: {len(prompt)} 字符")
            
            # 调用 LLM
            generation_config = {
                'temperature': 0.7,
            }
            
            report = self.analyzer._call_openai_api(prompt, generation_config)
            
            if report:
                logger.info(f"[板块机会] AI 深度分析报告生成成功，长度: {len(report)} 字符")
                return report
            else:
                logger.warning("[板块机会] LLM 返回为空")
                return None
                
        except Exception as e:
            logger.error(f"[板块机会] LLM 生成深度分析失败: {e}")
            return None
    
    def generate_opportunity_report(self, opportunities: List[SectorOpportunity], use_ai: bool = True) -> str:
        """
        生成板块机会报告
        
        Args:
            opportunities: 板块机会列表
            use_ai: 是否使用 AI 生成深度分析（默认 True）
            
        Returns:
            Markdown格式的报告
        """
        # 如果有 AI 分析器且启用 AI，尝试生成深度分析
        if use_ai and self.analyzer:
            ai_report = self.generate_ai_opportunity_report(opportunities)
            if ai_report:
                # 添加数据摘要头部
                recommended = [o for o in opportunities if o.total_score >= 4]
                header = f"""## 🎯 板块埋伏机会分析（AI 深度版）

> 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> 分析板块: {len(opportunities)} 个申万一级行业
> 推荐埋伏: {len(recommended)} 个板块

---

"""
                return header + ai_report
        
        # 降级到模板报告
        return self._generate_template_opportunity_report(opportunities)
    
    def _generate_template_opportunity_report(self, opportunities: List[SectorOpportunity]) -> str:
        """
        使用模板生成板块机会报告（无 LLM 时的备选方案）
        
        Args:
            opportunities: 板块机会列表
            
        Returns:
            Markdown格式的报告
        """
        report = f"""## 🎯 板块埋伏机会分析

> 埋伏逻辑：必须同时满足以下三个条件中的至少两个
> 1. **够便宜**：估值在历史底部，机构仓位低，筹码集中度低
> 2. **有催化**：未来有政策预期、技术突破或产品落地
> 3. **有反转**：行业供需改善，资金开始流入

---

### 📊 推荐埋伏板块

"""
        # 推荐板块（总分>=4）
        recommended = [o for o in opportunities if o.total_score >= 4]
        
        if recommended:
            for i, opp in enumerate(recommended[:5], 1):
                # 筹码信息
                chip_info = ""
                if opp.avg_chip_concentration > 0:
                    chip_info = f"筹码集中度:{opp.avg_chip_concentration:.0%} 获利比例:{opp.avg_profit_ratio:.0%}"
                    if opp.leader_stock_name:
                        chip_info += f" 龙头:{opp.leader_stock_name}"
                
                report += f"""#### {i}. {opp.sector_name} ⭐ 总分: {opp.total_score}/10

| 维度 | 得分 | 说明 |
|------|------|------|
| 够便宜 | {opp.cheap_score}/4 | PE:{opp.current_pe:.1f} PB:{opp.current_pb:.1f} 股息率:{opp.dividend_yield:.1f}% |
| 有催化 | {opp.catalyst_score}/3 | {', '.join(opp.catalyst_reasons[:2]) if opp.catalyst_reasons else '暂无明显催化'} |
| 有反转 | {opp.reversal_score}/3 | {', '.join(opp.reversal_reasons[:2]) if opp.reversal_reasons else '暂无反转信号'} |

"""
                if chip_info:
                    report += f"**筹码分析**: {chip_info}\n\n"
                if opp.chip_analysis:
                    report += f"**筹码结论**: {opp.chip_analysis}\n\n"
                
                report += f"""**推荐理由**: {opp.recommendation}

**风险提示**: {opp.risk_warning}

---

"""
        else:
            report += "暂无符合条件的推荐板块。\n\n"
        
        # 观察板块（总分3）
        watching = [o for o in opportunities if o.total_score == 3]
        if watching:
            report += "### 👀 观察板块\n\n"
            for opp in watching[:5]:
                chip_note = f" 筹码:{opp.avg_chip_concentration:.0%}" if opp.avg_chip_concentration > 0 else ""
                report += f"- **{opp.sector_name}**: 总分{opp.total_score} (便宜:{opp.cheap_score} 催化:{opp.catalyst_score} 反转:{opp.reversal_score}){chip_note}\n"
            report += "\n"
        
        # 估值最低板块（含筹码数据）
        cheapest = sorted(opportunities, key=lambda x: x.price_percentile)[:5]
        report += "### 💰 估值最低板块（价格分位数）\n\n"
        report += "| 板块 | 价格分位 | PE | PB | 筹码集中度 | 获利比例 |\n"
        report += "|------|----------|-----|-----|------------|----------|\n"
        for opp in cheapest:
            chip_conc = f"{opp.avg_chip_concentration:.0%}" if opp.avg_chip_concentration > 0 else "-"
            profit_ratio = f"{opp.avg_profit_ratio:.0%}" if opp.avg_profit_ratio > 0 else "-"
            report += f"| {opp.sector_name} | {opp.price_percentile:.0f}% | {opp.current_pe:.1f} | {opp.current_pb:.1f} | {chip_conc} | {profit_ratio} |\n"
        
        report += f"\n---\n*分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    # 测试板块机会分析
    print("=== 测试板块机会分析 ===")
    opportunity_analyzer = SectorOpportunityAnalyzer()
    opportunities = opportunity_analyzer.find_opportunity_sectors(fast_mode=True)
    
    print(f"\n共分析 {len(opportunities)} 个行业")
    print("\n前5个推荐板块:")
    for i, opp in enumerate(opportunities[:5], 1):
        print(f"{i}. {opp.sector_name}: 总分{opp.total_score} "
              f"(便宜:{opp.cheap_score} 催化:{opp.catalyst_score} 反转:{opp.reversal_score})")
        print(f"   PE:{opp.current_pe:.1f} PB:{opp.current_pb:.1f} 股息率:{opp.dividend_yield:.1f}%")
        if opp.cheap_reasons:
            print(f"   便宜: {', '.join(opp.cheap_reasons[:2])}")
        if opp.reversal_reasons:
            print(f"   反转: {', '.join(opp.reversal_reasons[:2])}")
    
    # 生成报告
    print("\n=== 生成板块机会报告 ===")
    report = opportunity_analyzer.generate_opportunity_report(opportunities)
    print(report[:1500] + "..." if len(report) > 1500 else report)
