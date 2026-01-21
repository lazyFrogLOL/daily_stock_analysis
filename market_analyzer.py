# -*- coding: utf-8 -*-
"""
===================================
大盘数据获取模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板等）
2. 获取市场涨跌统计、板块数据、资金数据等
3. 搜索市场新闻

注意：
- 本模块只负责数据获取和存储
- LLM 分析和报告生成已移至 llm_mapreduce.py
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
    lhb_org_net_buy: float = 0.0        # 机构净买入金额（亿元）
    lhb_seat_detail: List[Dict] = field(default_factory=list) # 龙虎榜席位明细
    
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
    
    # ========== 新增数据维度（埋伏分析和复盘）==========
    
    # 板块异动详情
    board_changes: List[Dict] = field(default_factory=list)  # 板块异动列表
    board_change_count: int = 0         # 板块异动总次数
    
    # 盘口异动
    pankou_changes: Dict[str, List[Dict]] = field(default_factory=dict)  # 盘口异动分类数据
    big_buy_count: int = 0              # 大笔买入次数
    big_sell_count: int = 0             # 大笔卖出次数
    limit_up_seal_count: int = 0        # 封涨停板次数
    limit_down_seal_count: int = 0      # 封跌停板次数
    rocket_launch_count: int = 0        # 火箭发射次数
    high_dive_count: int = 0            # 高台跳水次数
    
    # 财新内容精选
    caixin_news: List[Dict] = field(default_factory=list)  # 财新新闻列表
    
    # ========== 涨停板行情数据 ==========
    
    # 涨停股池
    zt_pool: List[Dict] = field(default_factory=list)       # 涨停股池列表
    zt_pool_count: int = 0              # 涨停股数量
    zt_total_amount: float = 0.0        # 涨停股总成交额（亿元）
    zt_avg_turnover: float = 0.0        # 涨停股平均换手率
    zt_first_board_count: int = 0       # 首板数量
    zt_continuous_count: int = 0        # 连板数量（连板数>=2）
    zt_max_continuous: int = 0          # 最高连板数
    
    # 昨日涨停股池（今日表现）
    previous_zt_pool: List[Dict] = field(default_factory=list)  # 昨日涨停股今日表现
    previous_zt_count: int = 0          # 昨日涨停数量
    previous_zt_avg_change: float = 0.0 # 昨日涨停今日平均涨跌幅（溢价率）
    previous_zt_up_count: int = 0       # 昨日涨停今日上涨数量
    previous_zt_down_count: int = 0     # 昨日涨停今日下跌数量
    
    # 强势股池
    strong_pool: List[Dict] = field(default_factory=list)   # 强势股池列表
    strong_pool_count: int = 0          # 强势股数量
    strong_new_high_count: int = 0      # 60日新高数量
    strong_multi_zt_count: int = 0      # 近期多次涨停数量
    
    # 炸板股池
    zb_pool: List[Dict] = field(default_factory=list)       # 炸板股池列表
    zb_pool_count: int = 0              # 炸板股数量
    zb_total_count: int = 0             # 炸板总次数
    zb_rate: float = 0.0                # 炸板率（炸板数/涨停数）
    
    # 跌停股池
    dt_pool: List[Dict] = field(default_factory=list)       # 跌停股池列表
    dt_pool_count: int = 0              # 跌停股数量
    dt_continuous_count: int = 0        # 连续跌停数量
    
    # ========== 千股千评数据 ==========
    
    # 市场整体评分
    comment_avg_score: float = 0.0      # 市场平均综合得分
    comment_high_score_count: int = 0   # 高分股票数量（>=80分）
    comment_low_score_count: int = 0    # 低分股票数量（<=40分）
    comment_top_stocks: List[Dict] = field(default_factory=list)  # 综合得分TOP10
    comment_bottom_stocks: List[Dict] = field(default_factory=list)  # 综合得分最低10
    comment_high_attention: List[Dict] = field(default_factory=list)  # 关注指数TOP10
    
    # ========== 分析师指数数据 ==========
    
    analyst_top_list: List[Dict] = field(default_factory=list)  # 分析师排行TOP10
    analyst_top_stocks: List[Dict] = field(default_factory=list)  # 分析师推荐股票
    
    # ========== 潜力股发现（资金流入但热度不高）==========
    
    # 低热度资金流入股票
    hidden_inflow_stocks: List[Dict] = field(default_factory=list)  # 资金流入但热度低的股票
    hidden_inflow_analysis: str = ""    # AI分析结论
    
    # ========== 板块埋伏机会数据 ==========
    
    # 申万行业估值数据
    sector_opportunities: List[Dict] = field(default_factory=list)  # 板块机会列表（按总分排序）
    sector_cheap_list: List[Dict] = field(default_factory=list)     # 估值最低板块TOP5
    sector_catalyst_list: List[Dict] = field(default_factory=list)  # 有催化板块TOP5
    sector_reversal_list: List[Dict] = field(default_factory=list)  # 有反转信号板块TOP5
    sector_recommended: List[Dict] = field(default_factory=list)    # 推荐埋伏板块（总分>=4）


class MarketAnalyzer:
    """
    大盘数据获取器
    
    职责：只负责获取市场数据，不生成报告
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 获取融资融券数据
    5. 获取龙虎榜数据
    6. 获取大宗交易数据
    7. 获取概念板块热点
    8. 获取板块异动详情
    9. 获取盘口异动数据
    10. 获取财新内容精选
    11. 获取涨停板行情数据
    12. 获取千股千评数据
    13. 获取分析师指数
    14. 挖掘潜力股（资金流入但热度不高）
    15. 搜索市场新闻
    
    注意：
    - 北向资金数据已于2024年停止更新，不再获取
    - 报告生成请使用 llm_mapreduce.py
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
        初始化大盘数据获取器
        
        Args:
            search_service: 搜索服务实例（用于搜索新闻）
            analyzer: AI分析器实例（已废弃，保留参数兼容性，LLM分析请使用 llm_mapreduce.py）
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer  # 保留兼容性，但不再使用
        
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
        
        # 新增数据维度（实时数据，仅当天有效）
        if not is_historical:
            self._get_board_change_data(overview)
            self._get_pankou_change_data(overview)
            self._get_caixin_news(overview)
            self._get_zt_pool_data(overview, target_date)
            self._get_previous_zt_pool_data(overview, target_date)
            self._get_strong_pool_data(overview, target_date)
            self._get_zb_pool_data(overview, target_date)
            self._get_dt_pool_data(overview, target_date)
            self._get_comment_data(overview)
            self._get_analyst_data(overview)
            
            # 潜力股挖掘：资金流入但热度不高的股票
            overview.hidden_inflow_stocks = self._find_hidden_inflow_stocks(overview)
            # 注意：LLM 分析已移至 llm_mapreduce.py，这里只获取数据
            
            # 板块埋伏机会数据
            self._get_sector_opportunity_data(overview)
        
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
                    time.sleep(min(5 ** attempt, 20))
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
        """
        获取板块涨跌榜
        
        数据来源：同花顺-同花顺行业一览表
        https://q.10jqka.com.cn/thshy/
        
        API: stock_board_industry_summary_ths
        
        输出字段：
        - 板块、涨跌幅、总成交量(万手)、总成交额(亿元)、净流入(亿元)
        - 上涨家数、下跌家数、均价、领涨股、领涨股-最新价、领涨股-涨跌幅
        """
        try:
            logger.info("[大盘] 获取板块涨跌榜（同花顺）...")
            
            # 使用同花顺行业一览表接口
            df = self._call_akshare_with_retry(ak.stock_board_industry_summary_ths, "同花顺行业板块", attempts=2)
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5
                    top = df.nlargest(5, change_col)
                    overview.top_sectors = [
                        {
                            'name': row['板块'],
                            'change_pct': row[change_col],
                            'net_inflow': float(row.get('净流入', 0) or 0),  # 净流入（亿元）
                            'amount': float(row.get('总成交额', 0) or 0),  # 成交额（亿元）
                            'up_count': int(row.get('上涨家数', 0) or 0),
                            'down_count': int(row.get('下跌家数', 0) or 0),
                            'leader_stock': str(row.get('领涨股', '')),
                            'leader_change': float(row.get('领涨股-涨跌幅', 0) or 0),
                        }
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_sectors = [
                        {
                            'name': row['板块'],
                            'change_pct': row[change_col],
                            'net_inflow': float(row.get('净流入', 0) or 0),
                            'amount': float(row.get('总成交额', 0) or 0),
                            'up_count': int(row.get('上涨家数', 0) or 0),
                            'down_count': int(row.get('下跌家数', 0) or 0),
                            'leader_stock': str(row.get('领涨股', '')),
                            'leader_change': float(row.get('领涨股-涨跌幅', 0) or 0),
                        }
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    def _get_concept_rankings(self, overview: MarketOverview):
        """
        获取概念板块热点
        
        数据来源：同花顺
        - stock_board_concept_name_ths: 获取概念名称列表
        - stock_board_concept_info_ths: 获取单个概念详情（含涨跌幅）
        
        注意：由于需要逐个调用获取涨跌幅，为提高效率只获取部分概念
        """
        try:
            logger.info("[大盘] 获取概念板块热点（同花顺）...")
            
            # 1. 获取概念名称列表
            name_df = self._call_akshare_with_retry(ak.stock_board_concept_name_ths, "同花顺概念名称", attempts=2)
            
            if name_df is None or name_df.empty:
                logger.warning("[大盘] 概念板块名称列表为空")
                return
            
            # 2. 获取部分概念的详情（限制数量避免请求过多）
            # 随机抽取或取前N个概念
            concept_names = name_df['name'].tolist() # 只取前50个概念
            
            concept_data = []
            for concept_name in concept_names:
                try:
                    info_df = ak.stock_board_concept_info_ths(symbol=concept_name)
                    if info_df is not None and not info_df.empty:
                        # 解析数据
                        info_dict = dict(zip(info_df['项目'], info_df['值']))
                        change_pct_str = str(info_dict.get('板块涨幅', '0%'))
                        # 解析涨跌幅（格式如 "-4.96%"）
                        change_pct = float(change_pct_str.replace('%', '')) if change_pct_str else 0.0
                        
                        # 解析涨跌家数（格式如 "25/222"）
                        up_down_str = str(info_dict.get('涨跌家数', '0/0'))
                        up_down_parts = up_down_str.split('/')
                        up_count = int(up_down_parts[0]) if len(up_down_parts) > 0 else 0
                        down_count = int(up_down_parts[1]) if len(up_down_parts) > 1 else 0
                        
                        concept_data.append({
                            'name': concept_name,
                            'change_pct': change_pct,
                            'net_inflow': float(info_dict.get('资金净流入(亿)', 0) or 0),
                            'amount': float(info_dict.get('成交额(亿)', 0) or 0),
                            'up_count': up_count,
                            'down_count': down_count,
                        })
                except Exception as e:
                    logger.debug(f"[大盘] 获取概念 {concept_name} 详情失败: {e}")
                    continue
                
                # 避免请求过快
                time.sleep(0.1)
            
            if not concept_data:
                logger.warning("[大盘] 未获取到概念板块数据")
                return
            
            # 3. 按涨跌幅排序
            concept_data.sort(key=lambda x: x['change_pct'], reverse=True)
            
            # 涨幅前5
            overview.top_concepts = concept_data[:10]
            
            # 跌幅前5
            overview.bottom_concepts = concept_data[-10:][::-1]  # 倒序取最后5个
            
            logger.info(f"[大盘] 热门概念: {[s['name'] for s in overview.top_concepts]}")
                    
        except Exception as e:
            logger.warning(f"[大盘] 获取概念板块失败: {e}")
    
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
        获取龙虎榜数据（增强版：含机构席位统计）
        
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
            date_str = None
            
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
                
                # ========== 获取机构买卖每日统计（新增）==========
                self._get_lhb_org_stats(overview, date_str)
                
                # ========== 获取龙虎榜席位明细（新增）==========
                self._get_lhb_seat_detail(overview, df, date_str)
                
            else:
                logger.warning("[大盘] 未能获取到龙虎榜数据")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取龙虎榜数据失败: {e}")
    
    def _get_lhb_org_stats(self, overview: MarketOverview, date_str: str):
        """
        获取龙虎榜机构买卖每日统计
        
        数据来源：东方财富网-数据中心-龙虎榜单-机构买卖每日统计
        https://data.eastmoney.com/stock/jgmmtj.html
        """
        try:
            logger.info("[大盘] 获取龙虎榜机构买卖统计...")
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_lhb_jgmmtj_em(start_date=date_str, end_date=date_str),
                "机构买卖统计", attempts=2
            )
            
            if df is not None and not df.empty:
                # 统计机构买卖次数
                if '买方机构数' in df.columns:
                    df['买方机构数'] = pd.to_numeric(df['买方机构数'], errors='coerce')
                    overview.lhb_org_buy_count = int(df['买方机构数'].sum())
                
                if '卖方机构数' in df.columns:
                    df['卖方机构数'] = pd.to_numeric(df['卖方机构数'], errors='coerce')
                    overview.lhb_org_sell_count = int(df['卖方机构数'].sum())
                
                # 机构净买入总额
                org_net_buy = 0.0
                if '机构买入净额' in df.columns:
                    df['机构买入净额'] = pd.to_numeric(df['机构买入净额'], errors='coerce')
                    org_net_buy = df['机构买入净额'].sum() / 1e8  # 转为亿元
                
                # 存储机构净买入金额
                overview.lhb_org_net_buy = org_net_buy
                
                logger.info(f"[大盘] 机构买卖: 买入{overview.lhb_org_buy_count}次, "
                           f"卖出{overview.lhb_org_sell_count}次, 净买入{org_net_buy:.2f}亿")
                
        except Exception as e:
            logger.debug(f"[大盘] 获取机构买卖统计失败: {e}")
    
    def _get_lhb_seat_detail(self, overview: MarketOverview, lhb_df, date_str: str):
        """
        获取龙虎榜席位明细
        
        数据来源：东方财富网-数据中心-龙虎榜单-个股龙虎榜详情
        https://data.eastmoney.com/stock/lhb/{symbol}.html
        
        API: stock_lhb_stock_detail_em
        
        获取上榜股票的买入和卖出席位明细，包括：
        - 营业部名称（可判断是机构还是游资）
        - 买入金额
        - 卖出金额
        - 净额
        """
        try:
            logger.info("[大盘] 获取龙虎榜席位明细...")
            
            seat_details = []
            
            # 从龙虎榜股票中获取前10只股票的席位明细
            stock_codes = lhb_df['代码'].head(10).tolist() if '代码' in lhb_df.columns else []
            
            for code in stock_codes:
                try:
                    # 获取买入席位
                    buy_df = ak.stock_lhb_stock_detail_em(symbol=code, date=date_str, flag="买入")
                    if buy_df is not None and not buy_df.empty:
                        stock_name = lhb_df[lhb_df['代码'] == code]['名称'].iloc[0] if '名称' in lhb_df.columns else code
                        for _, row in buy_df.iterrows():
                            trader_name = str(row.get('交易营业部名称', ''))
                            buy_amount = float(row.get('买入金额', 0) or 0) / 1e8  # 转为亿元
                            sell_amount = float(row.get('卖出金额', 0) or 0) / 1e8
                            net_amount = float(row.get('净额', 0) or 0) / 1e8
                            
                            seat_details.append({
                                'stock_code': code,
                                'stock_name': stock_name,
                                'trader_name': trader_name,
                                'buy_amount': buy_amount,
                                'sell_amount': sell_amount,
                                'net_amount': net_amount,
                                'direction': '买入',
                            })
                    
                    # 获取卖出席位
                    sell_df = ak.stock_lhb_stock_detail_em(symbol=code, date=date_str, flag="卖出")
                    if sell_df is not None and not sell_df.empty:
                        stock_name = lhb_df[lhb_df['代码'] == code]['名称'].iloc[0] if '名称' in lhb_df.columns else code
                        for _, row in sell_df.iterrows():
                            trader_name = str(row.get('交易营业部名称', ''))
                            buy_amount = float(row.get('买入金额', 0) or 0) / 1e8
                            sell_amount = float(row.get('卖出金额', 0) or 0) / 1e8
                            net_amount = float(row.get('净额', 0) or 0) / 1e8
                            
                            seat_details.append({
                                'stock_code': code,
                                'stock_name': stock_name,
                                'trader_name': trader_name,
                                'buy_amount': buy_amount,
                                'sell_amount': sell_amount,
                                'net_amount': net_amount,
                                'direction': '卖出',
                            })
                            
                except Exception as e:
                    logger.debug(f"[大盘] 获取 {code} 席位明细失败: {e}")
                    continue
            
            # 按净额绝对值排序，取前20条
            seat_details.sort(key=lambda x: abs(x['net_amount']), reverse=True)
            overview.lhb_seat_detail = seat_details[:20]
            
            logger.info(f"[大盘] 龙虎榜席位明细: 获取 {len(overview.lhb_seat_detail)} 条记录")
                
        except Exception as e:
            logger.debug(f"[大盘] 获取龙虎榜席位明细失败: {e}")
    
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

    def _get_board_change_data(self, overview: MarketOverview):
        """
        获取板块异动详情
        
        数据来源：东方财富-行情中心-当日板块异动详情
        https://quote.eastmoney.com/changes/
        """
        try:
            logger.info("[大盘] 获取板块异动详情...")
            
            df = self._call_akshare_with_retry(ak.stock_board_change_em, "板块异动详情", attempts=2)
            
            if df is not None and not df.empty:
                # 板块异动总次数
                if '板块异动总次数' in df.columns:
                    overview.board_change_count = int(df['板块异动总次数'].sum())
                
                # 提取板块异动列表（按异动次数排序，取前10）
                df_sorted = df.sort_values('板块异动总次数', ascending=False)
                for _, row in df_sorted.head(10).iterrows():
                    overview.board_changes.append({
                        'name': str(row.get('板块名称', '')),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'main_net_inflow': float(row.get('主力净流入', 0) or 0) / 1e8,  # 转为亿元
                        'change_count': int(row.get('板块异动总次数', 0) or 0),
                        'top_stock_code': str(row.get('板块异动最频繁个股及所属类型-股票代码', '')),
                        'top_stock_name': str(row.get('板块异动最频繁个股及所属类型-股票名称', '')),
                        'top_stock_direction': str(row.get('板块异动最频繁个股及所属类型-买卖方向', '')),
                    })
                
                logger.info(f"[大盘] 板块异动: {len(df)}个板块, 总异动{overview.board_change_count}次")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取板块异动详情失败: {e}")
    
    def _get_pankou_change_data(self, overview: MarketOverview):
        """
        获取盘口异动数据
        
        数据来源：东方财富-行情中心-盘口异动
        https://quote.eastmoney.com/changes/
        
        异动类型包括：
        - 大笔买入/大笔卖出
        - 封涨停板/封跌停板
        - 火箭发射/高台跳水
        - 快速反弹/加速下跌
        等
        """
        try:
            logger.info("[大盘] 获取盘口异动数据...")
            
            # 定义需要获取的异动类型（用于埋伏分析和复盘）
            change_types = {
                '大笔买入': 'big_buy',
                '大笔卖出': 'big_sell',
                '封涨停板': 'limit_up_seal',
                '封跌停板': 'limit_down_seal',
                '火箭发射': 'rocket_launch',
                '高台跳水': 'high_dive',
                '快速反弹': 'quick_rebound',
                '加速下跌': 'accelerate_down',
            }
            
            for cn_name, en_key in change_types.items():
                try:
                    df = ak.stock_changes_em(symbol=cn_name)
                    if df is not None and not df.empty:
                        # 存储异动数据
                        change_list = []
                        for _, row in df.head(20).iterrows():  # 每类取前20条
                            change_list.append({
                                'time': str(row.get('时间', '')),
                                'code': str(row.get('代码', '')),
                                'name': str(row.get('名称', '')),
                                'sector': str(row.get('板块', '')),
                                'info': str(row.get('相关信息', '')),
                            })
                        overview.pankou_changes[cn_name] = change_list
                        
                        # 更新统计计数
                        count = len(df)
                        if en_key == 'big_buy':
                            overview.big_buy_count = count
                        elif en_key == 'big_sell':
                            overview.big_sell_count = count
                        elif en_key == 'limit_up_seal':
                            overview.limit_up_seal_count = count
                        elif en_key == 'limit_down_seal':
                            overview.limit_down_seal_count = count
                        elif en_key == 'rocket_launch':
                            overview.rocket_launch_count = count
                        elif en_key == 'high_dive':
                            overview.high_dive_count = count
                        
                except Exception as e:
                    logger.debug(f"[大盘] 获取{cn_name}异动失败: {e}")
                    continue
            
            logger.info(f"[大盘] 盘口异动: 大笔买入{overview.big_buy_count}次 大笔卖出{overview.big_sell_count}次 "
                       f"封涨停{overview.limit_up_seal_count}次 封跌停{overview.limit_down_seal_count}次 "
                       f"火箭发射{overview.rocket_launch_count}次 高台跳水{overview.high_dive_count}次")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取盘口异动数据失败: {e}")
    
    def _get_caixin_news(self, overview: MarketOverview):
        """
        获取财新内容精选
        
        数据来源：财新网-财新数据通
        https://cxdata.caixin.com/pc/
        """
        try:
            logger.info("[大盘] 获取财新内容精选...")
            
            df = self._call_akshare_with_retry(ak.stock_news_main_cx, "财新内容精选", attempts=2)
            
            if df is not None and not df.empty:
                # 提取财新新闻列表（取前20条）
                for _, row in df.head(20).iterrows():
                    overview.caixin_news.append({
                        'tag': str(row.get('tag', '')),
                        'summary': str(row.get('summary', '')),
                        'url': str(row.get('url', '')),
                    })
                
                logger.info(f"[大盘] 财新内容精选: 获取{len(overview.caixin_news)}条")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取财新内容精选失败: {e}")

    def _get_zt_pool_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """
        获取涨停股池数据
        
        数据来源：东方财富网-行情中心-涨停板行情-涨停股池
        https://quote.eastmoney.com/ztb/detail#type=ztgc
        
        包含：涨停股列表、连板数、封板资金、炸板次数等
        """
        try:
            logger.info("[大盘] 获取涨停股池数据...")
            
            date_str = target_date.replace('-', '') if target_date else datetime.now().strftime('%Y%m%d')
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_em(date=date_str),
                "涨停股池", attempts=2
            )
            
            if df is not None and not df.empty:
                overview.zt_pool_count = len(df)
                
                # 统计连板情况
                if '连板数' in df.columns:
                    df['连板数'] = pd.to_numeric(df['连板数'], errors='coerce')
                    overview.zt_first_board_count = len(df[df['连板数'] == 1])
                    overview.zt_continuous_count = len(df[df['连板数'] >= 2])
                    overview.zt_max_continuous = int(df['连板数'].max()) if not df['连板数'].isna().all() else 0
                
                # 统计成交额和换手率
                if '成交额' in df.columns:
                    df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
                    overview.zt_total_amount = df['成交额'].sum() / 1e8  # 转为亿元
                if '换手率' in df.columns:
                    df['换手率'] = pd.to_numeric(df['换手率'], errors='coerce')
                    overview.zt_avg_turnover = df['换手率'].mean()
                
                # 提取涨停股列表（按连板数排序，取前15）
                df_sorted = df.sort_values('连板数', ascending=False) if '连板数' in df.columns else df
                for _, row in df_sorted.head(15).iterrows():
                    overview.zt_pool.append({
                        'code': str(row.get('代码', '')),
                        'name': str(row.get('名称', '')),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'amount': float(row.get('成交额', 0) or 0) / 1e8,
                        'turnover': float(row.get('换手率', 0) or 0),
                        'continuous': int(row.get('连板数', 1) or 1),
                        'first_time': str(row.get('首次封板时间', '')),
                        'last_time': str(row.get('最后封板时间', '')),
                        'zb_count': int(row.get('炸板次数', 0) or 0),
                        'seal_amount': float(row.get('封板资金', 0) or 0) / 1e8,
                        'industry': str(row.get('所属行业', '')),
                        'zt_stat': str(row.get('涨停统计', '')),
                    })
                
                logger.info(f"[大盘] 涨停股池: {overview.zt_pool_count}只, 首板{overview.zt_first_board_count}只, "
                           f"连板{overview.zt_continuous_count}只, 最高{overview.zt_max_continuous}板")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取涨停股池数据失败: {e}")
    
    def _get_previous_zt_pool_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """
        获取昨日涨停股池数据（今日表现）
        
        数据来源：东方财富网-行情中心-涨停板行情-昨日涨停股池
        https://quote.eastmoney.com/ztb/detail#type=zrzt
        
        用于分析涨停溢价率
        """
        try:
            logger.info("[大盘] 获取昨日涨停股池数据...")
            
            date_str = target_date.replace('-', '') if target_date else datetime.now().strftime('%Y%m%d')
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_previous_em(date=date_str),
                "昨日涨停股池", attempts=2
            )
            
            if df is not None and not df.empty:
                overview.previous_zt_count = len(df)
                
                # 统计今日涨跌情况
                if '涨跌幅' in df.columns:
                    df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                    overview.previous_zt_avg_change = df['涨跌幅'].mean()
                    overview.previous_zt_up_count = len(df[df['涨跌幅'] > 0])
                    overview.previous_zt_down_count = len(df[df['涨跌幅'] < 0])
                
                # 提取昨日涨停股列表（按涨跌幅排序，取前10）
                df_sorted = df.sort_values('涨跌幅', ascending=False) if '涨跌幅' in df.columns else df
                for _, row in df_sorted.head(10).iterrows():
                    overview.previous_zt_pool.append({
                        'code': str(row.get('代码', '')),
                        'name': str(row.get('名称', '')),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'amount': float(row.get('成交额', 0) or 0) / 1e8,
                        'turnover': float(row.get('换手率', 0) or 0),
                        'yesterday_continuous': int(row.get('昨日连板数', 1) or 1),
                        'industry': str(row.get('所属行业', '')),
                    })
                
                logger.info(f"[大盘] 昨日涨停: {overview.previous_zt_count}只, "
                           f"今日平均涨跌{overview.previous_zt_avg_change:.2f}%(溢价率), "
                           f"上涨{overview.previous_zt_up_count}只 下跌{overview.previous_zt_down_count}只")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取昨日涨停股池数据失败: {e}")
    
    def _get_strong_pool_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """
        获取强势股池数据
        
        数据来源：东方财富网-行情中心-涨停板行情-强势股池
        https://quote.eastmoney.com/ztb/detail#type=qsgc
        
        包含：60日新高或近期多次涨停的股票
        """
        try:
            logger.info("[大盘] 获取强势股池数据...")
            
            date_str = target_date.replace('-', '') if target_date else datetime.now().strftime('%Y%m%d')
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_strong_em(date=date_str),
                "强势股池", attempts=2
            )
            
            if df is not None and not df.empty:
                overview.strong_pool_count = len(df)
                
                # 统计入选理由
                if '入选理由' in df.columns:
                    overview.strong_new_high_count = len(df[df['入选理由'].str.contains('60日新高', na=False)])
                    overview.strong_multi_zt_count = len(df[df['入选理由'].str.contains('多次涨停', na=False)])
                
                # 提取强势股列表（按涨跌幅排序，取前10）
                if '涨跌幅' in df.columns:
                    df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                    df_sorted = df.sort_values('涨跌幅', ascending=False)
                else:
                    df_sorted = df
                    
                for _, row in df_sorted.head(10).iterrows():
                    overview.strong_pool.append({
                        'code': str(row.get('代码', '')),
                        'name': str(row.get('名称', '')),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'is_new_high': str(row.get('是否新高', '')),
                        'reason': str(row.get('入选理由', '')),
                        'industry': str(row.get('所属行业', '')),
                    })
                
                logger.info(f"[大盘] 强势股池: {overview.strong_pool_count}只, "
                           f"60日新高{overview.strong_new_high_count}只, 多次涨停{overview.strong_multi_zt_count}只")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取强势股池数据失败: {e}")
    
    def _get_zb_pool_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """
        获取炸板股池数据
        
        数据来源：东方财富网-行情中心-涨停板行情-炸板股池
        https://quote.eastmoney.com/ztb/detail#type=zbgc
        
        注意：只能获取最近30个交易日的数据
        """
        try:
            logger.info("[大盘] 获取炸板股池数据...")
            
            date_str = target_date.replace('-', '') if target_date else datetime.now().strftime('%Y%m%d')
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_zbgc_em(date=date_str),
                "炸板股池", attempts=2
            )
            
            if df is not None and not df.empty:
                overview.zb_pool_count = len(df)
                
                # 统计炸板总次数
                if '炸板次数' in df.columns:
                    df['炸板次数'] = pd.to_numeric(df['炸板次数'], errors='coerce')
                    overview.zb_total_count = int(df['炸板次数'].sum())
                
                # 计算炸板率
                total_touched_zt = overview.zt_pool_count + overview.zb_pool_count
                if total_touched_zt > 0:
                    overview.zb_rate = overview.zb_pool_count / total_touched_zt * 100
                
                # 提取炸板股列表（按炸板次数排序，取前10）
                df_sorted = df.sort_values('炸板次数', ascending=False) if '炸板次数' in df.columns else df
                for _, row in df_sorted.head(10).iterrows():
                    overview.zb_pool.append({
                        'code': str(row.get('代码', '')),
                        'name': str(row.get('名称', '')),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'zb_count': int(row.get('炸板次数', 0) or 0),
                        'first_time': str(row.get('首次封板时间', '')),
                        'industry': str(row.get('所属行业', '')),
                    })
                
                logger.info(f"[大盘] 炸板股池: {overview.zb_pool_count}只, 炸板{overview.zb_total_count}次, "
                           f"炸板率{overview.zb_rate:.1f}%")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取炸板股池数据失败: {e}")
    
    def _get_dt_pool_data(self, overview: MarketOverview, target_date: Optional[str] = None):
        """
        获取跌停股池数据
        
        数据来源：东方财富网-行情中心-涨停板行情-跌停股池
        https://quote.eastmoney.com/ztb/detail#type=dtgc
        
        注意：只能获取最近30个交易日的数据
        """
        try:
            logger.info("[大盘] 获取跌停股池数据...")
            
            date_str = target_date.replace('-', '') if target_date else datetime.now().strftime('%Y%m%d')
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_dtgc_em(date=date_str),
                "跌停股池", attempts=2
            )
            
            if df is not None and not df.empty:
                overview.dt_pool_count = len(df)
                
                # 统计连续跌停
                if '连续跌停' in df.columns:
                    df['连续跌停'] = pd.to_numeric(df['连续跌停'], errors='coerce')
                    overview.dt_continuous_count = len(df[df['连续跌停'] >= 2])
                
                # 提取跌停股列表（按连续跌停排序，取前10）
                df_sorted = df.sort_values('连续跌停', ascending=False) if '连续跌停' in df.columns else df
                for _, row in df_sorted.head(10).iterrows():
                    overview.dt_pool.append({
                        'code': str(row.get('代码', '')),
                        'name': str(row.get('名称', '')),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'continuous': int(row.get('连续跌停', 1) or 1),
                        'seal_amount': float(row.get('封单资金', 0) or 0) / 1e8,
                        'open_count': int(row.get('开板次数', 0) or 0),
                        'industry': str(row.get('所属行业', '')),
                    })
                
                logger.info(f"[大盘] 跌停股池: {overview.dt_pool_count}只, 连续跌停{overview.dt_continuous_count}只")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取跌停股池数据失败: {e}")
    
    def _get_comment_data(self, overview: MarketOverview):
        """
        获取千股千评数据
        
        数据来源：东方财富网-数据中心-特色数据-千股千评
        https://data.eastmoney.com/stockcomment/
        
        API 返回字段：
        - 代码、名称、最新价、涨跌幅、换手率、市盈率
        - 主力成本、机构参与度、综合得分、上升、目前排名、关注指数、交易日
        """
        try:
            logger.info("[大盘] 获取千股千评数据...")
            
            df = self._call_akshare_with_retry(ak.stock_comment_em, "千股千评", attempts=2)
            
            if df is not None and not df.empty:
                # 计算市场平均得分
                if '综合得分' in df.columns:
                    df['综合得分'] = pd.to_numeric(df['综合得分'], errors='coerce')
                    overview.comment_avg_score = df['综合得分'].mean()
                    overview.comment_high_score_count = len(df[df['综合得分'] >= 80])
                    overview.comment_low_score_count = len(df[df['综合得分'] <= 40])
                    
                    # 综合得分TOP10
                    top_df = df.nlargest(10, '综合得分')
                    for _, row in top_df.iterrows():
                        overview.comment_top_stocks.append({
                            'code': str(row.get('代码', '')),
                            'name': str(row.get('名称', '')),
                            'score': float(row.get('综合得分', 0) or 0),
                            'rank': int(row.get('目前排名', 0) or 0),
                            'rank_change': int(row.get('上升', 0) or 0),  # 排名变化（正=上升）
                            'change_pct': float(row.get('涨跌幅', 0) or 0),
                            'turnover_rate': float(row.get('换手率', 0) or 0),
                            'org_participate': float(row.get('机构参与度', 0) or 0),
                            'main_cost': float(row.get('主力成本', 0) or 0),
                        })
                    
                    # 综合得分最低10
                    bottom_df = df.nsmallest(10, '综合得分')
                    for _, row in bottom_df.iterrows():
                        overview.comment_bottom_stocks.append({
                            'code': str(row.get('代码', '')),
                            'name': str(row.get('名称', '')),
                            'score': float(row.get('综合得分', 0) or 0),
                            'rank': int(row.get('目前排名', 0) or 0),
                            'rank_change': int(row.get('上升', 0) or 0),
                            'change_pct': float(row.get('涨跌幅', 0) or 0),
                        })
                
                # 关注指数TOP10
                if '关注指数' in df.columns:
                    df['关注指数'] = pd.to_numeric(df['关注指数'], errors='coerce')
                    attention_df = df.nlargest(10, '关注指数')
                    for _, row in attention_df.iterrows():
                        overview.comment_high_attention.append({
                            'code': str(row.get('代码', '')),
                            'name': str(row.get('名称', '')),
                            'attention': float(row.get('关注指数', 0) or 0),
                            'score': float(row.get('综合得分', 0) or 0),
                            'change_pct': float(row.get('涨跌幅', 0) or 0),
                            'org_participate': float(row.get('机构参与度', 0) or 0),
                        })
                
                logger.info(f"[大盘] 千股千评: 平均得分{overview.comment_avg_score:.1f}, "
                           f"高分(>=80){overview.comment_high_score_count}只, "
                           f"低分(<=40){overview.comment_low_score_count}只")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取千股千评数据失败: {e}")
    
    def _get_analyst_data(self, overview: MarketOverview):
        """
        获取分析师指数数据
        
        数据来源：东方财富网-数据中心-研究报告-东方财富分析师指数
        https://data.eastmoney.com/invest/invest/list.html
        
        包含：分析师排行、推荐股票等
        """
        try:
            logger.info("[大盘] 获取分析师指数数据...")
            
            current_year = str(datetime.now().year)
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_analyst_rank_em(year=current_year),
                "分析师指数", attempts=2
            )
            
            if df is not None and not df.empty:
                # 分析师排行TOP10
                for _, row in df.head(10).iterrows():
                    year_yield_col = f'{current_year}年收益率'
                    stock_name_col = f'{current_year}最新个股评级-股票名称'
                    stock_code_col = f'{current_year}最新个股评级-股票代码'
                    
                    overview.analyst_top_list.append({
                        'name': str(row.get('分析师名称', '')),
                        'company': str(row.get('分析师单位', '')),
                        'index': float(row.get('年度指数', 0) or 0),
                        'year_yield': float(row.get(year_yield_col, 0) or 0),
                        'stock_count': int(row.get('成分股个数', 0) or 0),
                        'latest_stock': str(row.get(stock_name_col, '')),
                        'latest_code': str(row.get(stock_code_col, '')),
                        'industry': str(row.get('行业', '')),
                    })
                    
                    # 收集分析师推荐的股票
                    if row.get(stock_name_col) and row.get(stock_code_col):
                        overview.analyst_top_stocks.append({
                            'code': str(row.get(stock_code_col, '')),
                            'name': str(row.get(stock_name_col, '')),
                            'analyst': str(row.get('分析师名称', '')),
                            'company': str(row.get('分析师单位', '')),
                        })
                
                logger.info(f"[大盘] 分析师指数: 获取TOP{len(overview.analyst_top_list)}分析师")
                
        except Exception as e:
            logger.warning(f"[大盘] 获取分析师指数数据失败: {e}")

    def _find_hidden_inflow_stocks(self, overview: MarketOverview) -> List[Dict]:
        """
        获取有资金流入的股票原始数据（供 LLM 分析）
        
        职责：只负责数据获取和整合，不做筛选判断
        
        数据来源：
        1. 盘口异动-大笔买入：统计每只股票的大笔买入次数
        2. 千股千评：获取关注指数、综合得分等指标
        3. A股实时行情：补充市值、换手率等信息
        
        注意：
        - 不做任何规则筛选，所有筛选逻辑由 llm_mapreduce.py 的 HiddenInflowAnalyst 完成
        - 返回所有有大笔买入的股票数据，供 LLM 智能分析
        
        Returns:
            股票数据列表，每个元素包含股票详细信息
        """
        all_stocks = []
        
        try:
            logger.info("[大盘] 获取资金流入股票数据...")
            
            # 1. 统计大笔买入股票的出现次数
            big_buy_stocks: Dict[str, Dict] = {}  # {code: {name, count, times, sector, info_list}}
            
            if '大笔买入' in overview.pankou_changes:
                for item in overview.pankou_changes['大笔买入']:
                    code = item.get('code', '')
                    if not code:
                        continue
                    
                    if code not in big_buy_stocks:
                        big_buy_stocks[code] = {
                            'code': code,
                            'name': item.get('name', ''),
                            'sector': item.get('sector', ''),
                            'count': 0,
                            'times': [],
                            'info_list': [],
                        }
                    
                    big_buy_stocks[code]['count'] += 1
                    big_buy_stocks[code]['times'].append(item.get('time', ''))
                    big_buy_stocks[code]['info_list'].append(item.get('info', ''))
            
            if not big_buy_stocks:
                logger.info("[大盘] 无大笔买入数据")
                return []
            
            logger.info(f"[大盘] 大笔买入股票: {len(big_buy_stocks)}只")
            
            # 2. 获取千股千评数据（用于关注指数和综合得分）
            comment_df = self._call_akshare_with_retry(ak.stock_comment_em, "千股千评", attempts=2)
            
            # 计算市场平均关注指数（供 LLM 参考）
            avg_attention = 50.0
            if comment_df is not None and not comment_df.empty and '关注指数' in comment_df.columns:
                comment_df['关注指数'] = pd.to_numeric(comment_df['关注指数'], errors='coerce')
                avg_attention = comment_df['关注指数'].mean()
            
            logger.info(f"[大盘] 市场平均关注指数: {avg_attention:.1f}")
            
            # 3. 整合所有大笔买入股票的数据（不做筛选）
            for code, stock_info in big_buy_stocks.items():
                stock_data = {
                    'code': code,
                    'name': stock_info['name'],
                    'sector': stock_info['sector'],
                    'big_buy_count': stock_info['count'],
                    'big_buy_times': stock_info['times'][:5],  # 最多保留5个时间点
                    'big_buy_info': stock_info['info_list'][:3],  # 最多保留3条信息
                    # 以下字段从千股千评获取，默认值供 LLM 判断
                    'attention': 0.0,
                    'avg_attention': avg_attention,  # 市场平均值，供 LLM 对比
                    'score': 0.0,
                    'change_pct': 0.0,
                    'rank': 0,
                    'org_participate': 0.0,
                }
                
                # 查找千股千评数据
                if comment_df is not None and not comment_df.empty:
                    stock_comment = comment_df[comment_df['代码'] == code]
                    if not stock_comment.empty:
                        row = stock_comment.iloc[0]
                        stock_data['attention'] = float(row.get('关注指数', 0) or 0)
                        stock_data['score'] = float(row.get('综合得分', 0) or 0)
                        stock_data['change_pct'] = float(row.get('涨跌幅', 0) or 0)
                        stock_data['rank'] = int(row.get('目前排名', 0) or 0)
                        stock_data['org_participate'] = float(row.get('机构参与度', 0) or 0)
                
                all_stocks.append(stock_data)
            
            # 4. 按大笔买入次数排序（次数多的排前面，供 LLM 优先分析）
            all_stocks.sort(key=lambda x: x['big_buy_count'], reverse=True)
            
            # 5. 获取补充信息（行业、市值等）
            if all_stocks:
                try:
                    spot_df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, "A股实时行情", attempts=1)
                    if spot_df is not None and not spot_df.empty:
                        for stock in all_stocks:
                            stock_spot = spot_df[spot_df['代码'] == stock['code']]
                            if not stock_spot.empty:
                                row = stock_spot.iloc[0]
                                stock['market_cap'] = float(row.get('总市值', 0) or 0) / 1e8  # 亿元
                                stock['turnover_rate'] = float(row.get('换手率', 0) or 0)
                                stock['amount'] = float(row.get('成交额', 0) or 0) / 1e8  # 亿元
                                stock['industry'] = str(row.get('所属行业', ''))
                except Exception as e:
                    logger.debug(f"[大盘] 获取补充信息失败: {e}")
            
            logger.info(f"[大盘] 获取到 {len(all_stocks)} 只有资金流入的股票数据")
            
            return all_stocks[:30]  # 返回前30只供 LLM 分析
            
        except Exception as e:
            logger.warning(f"[大盘] 获取资金流入股票数据失败: {e}")
            return []

    def _get_sector_opportunity_data(self, overview: MarketOverview):
        """
        获取板块埋伏机会数据
        
        使用 SectorOpportunityAnalyzer 获取申万行业估值、筹码等数据，
        将结果存入 overview 供 llm_mapreduce 的 SectorOpportunityAnalyst 使用。
        
        数据包括：
        - sector_opportunities: 所有板块机会列表（按总分排序）
        - sector_cheap_list: 估值最低板块TOP5
        - sector_catalyst_list: 有催化板块TOP5
        - sector_reversal_list: 有反转信号板块TOP5
        - sector_recommended: 推荐埋伏板块（总分>=4）
        """
        try:
            logger.info("[大盘] 获取板块埋伏机会数据...")
            
            # 创建板块机会分析器（不传入 analyzer，只获取数据不做 AI 分析）
            opportunity_analyzer = SectorOpportunityAnalyzer(
                search_service=self.search_service,
                analyzer=None  # 不需要 AI 分析，数据会传给 llm_mapreduce
            )
            
            # 获取板块机会数据（快速模式，不分析筹码以节省时间）
            opportunities = opportunity_analyzer.find_opportunity_sectors(
                fast_mode=True,
                use_smart_search=False,  # 不使用智能搜索
                analyze_chips=False  # 不分析筹码（太慢）
            )
            
            if not opportunities:
                logger.warning("[大盘] 未获取到板块机会数据")
                return
            
            logger.info(f"[大盘] 获取到 {len(opportunities)} 个板块机会数据")
            
            # 转换为字典格式存入 overview
            for opp in opportunities:
                opp_dict = {
                    'sector_name': opp.sector_name,
                    'sector_code': opp.sector_code,
                    # 估值数据
                    'current_pe': opp.current_pe,
                    'current_pb': opp.current_pb,
                    'pe_percentile': opp.pe_percentile,
                    'pb_percentile': opp.pb_percentile,
                    'price_percentile': opp.price_percentile,
                    'dividend_yield': opp.dividend_yield,
                    # 评分
                    'cheap_score': opp.cheap_score,
                    'catalyst_score': opp.catalyst_score,
                    'reversal_score': opp.reversal_score,
                    'total_score': opp.total_score,
                    # 原因
                    'cheap_reasons': opp.cheap_reasons,
                    'catalyst_reasons': opp.catalyst_reasons,
                    'reversal_reasons': opp.reversal_reasons,
                    # 反转信号数据
                    'recent_5d_change': opp.recent_5d_change,
                    'zt_count': opp.zt_count,
                    'volume_ratio': opp.volume_ratio,
                    # 催化剂
                    'policy_keywords': opp.policy_keywords,
                    'recent_news': opp.recent_news,
                    # 推荐
                    'recommendation': opp.recommendation,
                    'risk_warning': opp.risk_warning,
                }
                overview.sector_opportunities.append(opp_dict)
            
            # 按不同维度分类
            # 估值最低TOP5（过滤掉无数据的，-1表示无数据）
            valid_opps = [o for o in overview.sector_opportunities if o.get('price_percentile', -1) >= 0]
            overview.sector_cheap_list = sorted(
                valid_opps,
                key=lambda x: x['price_percentile']
            )[:5]
            
            # 有催化TOP5
            overview.sector_catalyst_list = sorted(
                overview.sector_opportunities,
                key=lambda x: x['catalyst_score'],
                reverse=True
            )[:5]
            
            # 有反转信号TOP5
            overview.sector_reversal_list = sorted(
                overview.sector_opportunities,
                key=lambda x: x['reversal_score'],
                reverse=True
            )[:5]
            
            # 推荐埋伏（总分>=4）
            overview.sector_recommended = [
                opp for opp in overview.sector_opportunities
                if opp['total_score'] >= 4
            ]
            
            logger.info(f"[大盘] 板块机会: 推荐{len(overview.sector_recommended)}个, "
                       f"估值低{len(overview.sector_cheap_list)}个, "
                       f"有催化{len(overview.sector_catalyst_list)}个, "
                       f"有反转{len(overview.sector_reversal_list)}个")
            
        except Exception as e:
            logger.warning(f"[大盘] 获取板块埋伏机会数据失败: {e}")
    
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
    
    def run_daily_review(self, target_date: Optional[str] = None) -> Optional[str]:
        """
        执行大盘复盘分析（完整流程）
        
        流程：
        1. 获取市场概览数据
        2. 搜索市场新闻
        3. 调用 LLM Map-Reduce 分析框架生成报告
        
        Args:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为今天
            
        Returns:
            复盘报告文本（Markdown 格式），失败返回 None
        """
        from llm_mapreduce import generate_market_review
        
        date_display = target_date or datetime.now().strftime('%Y-%m-%d')
        logger.info(f"[大盘] ========== 开始大盘复盘 ({date_display}) ==========")
        
        try:
            # 1. 获取市场概览数据
            logger.info("[大盘] 步骤1: 获取市场概览数据...")
            overview = self.get_market_overview(target_date)
            
            # 2. 搜索市场新闻
            logger.info("[大盘] 步骤2: 搜索市场新闻...")
            news = self.search_market_news()
            
            # 3. 调用 LLM Map-Reduce 分析框架生成报告
            logger.info("[大盘] 步骤3: 调用 LLM 分析框架生成报告...")
            report = generate_market_review(overview, news, self.analyzer)
            
            logger.info(f"[大盘] ========== 大盘复盘完成 ==========")
            
            return report
            
        except Exception as e:
            logger.error(f"[大盘] 大盘复盘失败: {e}")
            return None
    
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
    pe_percentile: float = -1.0         # PE历史分位数 (0-100, 越低越便宜, -1表示无数据)
    pb_percentile: float = -1.0         # PB历史分位数 (-1表示无数据)
    price_percentile: float = -1.0      # 价格历史分位数（基于3年数据, -1表示无数据）
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
    板块埋伏机会分析器（数据获取）
    
    核心逻辑：
    1. 够便宜：PE/PB历史分位数 < 30%，或价格处于3年低位
    2. 有催化：相关概念近期活跃，或有政策/技术催化预期
    3. 有反转：近期有资金流入迹象，涨停股增多，龙虎榜净买入
    
    数据来源：
    - 申万一级行业估值数据（sw_index_first_info）
    - 同花顺行业板块实时数据（stock_board_industry_summary_ths）
    - 同花顺行业指数历史数据（stock_board_industry_index_ths）
    
    注意：
    - 本类只负责数据获取和基础分析
    - LLM 深度分析已移至 llm_mapreduce.py 的 SectorOpportunityAnalyst
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
    
    # 同花顺行业名称映射（申万名称 -> 同花顺名称）
    THS_INDUSTRY_MAPPING = {
        '农林牧渔': '农林牧渔',
        '基础化工': '化工',
        '钢铁': '钢铁',
        '有色金属': '有色',
        '电子': '电子元件',
        '家用电器': '家用电器',
        '食品饮料': '食品饮料',
        '纺织服饰': '纺织服装',
        '轻工制造': '轻工制造',
        '医药生物': '医药',
        '公用事业': '公用事业',
        '交通运输': '交通运输',
        '房地产': '房地产',
        '商贸零售': '商业百货',
        '社会服务': '酒店及餐饮',
        '建筑材料': '建筑材料',
        '建筑装饰': '建筑',
        '电力设备': '电气设备',
        '国防军工': '国防军工',
        '计算机': '计算机应用',
        '传媒': '传媒',
        '通信': '通信服务',
        '银行': '银行',
        '非银金融': '保险',
        '汽车': '汽车',
        '机械设备': '机械',
        '煤炭': '煤炭',
        '石油石化': '石油',
        '环保': '环保',
        '美容护理': '日用化工',
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
        self._ths_industry_cache: Optional[pd.DataFrame] = None  # 同花顺行业板块缓存
        
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
    
    def _get_ths_industry_summary(self) -> Optional[pd.DataFrame]:
        """
        获取同花顺行业板块一览表
        
        数据来源：同花顺-同花顺行业一览表
        https://q.10jqka.com.cn/thshy/
        
        API: stock_board_industry_summary_ths
        
        输出字段：
        - 板块、涨跌幅、总成交量(万手)、总成交额(亿元)、净流入(亿元)
        - 上涨家数、下跌家数、均价、领涨股、领涨股-最新价、领涨股-涨跌幅
        
        Returns:
            同花顺行业板块 DataFrame
        """
        if self._ths_industry_cache is not None:
            return self._ths_industry_cache
        
        try:
            logger.info("[板块机会] 获取同花顺行业板块数据...")
            df = self._call_akshare_with_retry(ak.stock_board_industry_summary_ths, "同花顺行业板块")
            if df is not None and not df.empty:
                self._ths_industry_cache = df
                logger.info(f"[板块机会] 获取到 {len(df)} 个同花顺行业板块")
            return df
        except Exception as e:
            logger.error(f"[板块机会] 获取同花顺行业板块失败: {e}")
            return None
    
    def _get_ths_industry_index_hist(self, symbol: str, days: int = 750) -> Optional[pd.DataFrame]:
        """
        获取同花顺行业指数历史数据
        
        数据来源：同花顺-板块-行业板块-指数日频率数据
        https://q.10jqka.com.cn/thshy/detail/code/881270/
        
        API: stock_board_industry_index_ths
        
        Args:
            symbol: 同花顺行业名称（如"元件"、"银行"）
            days: 获取天数（约3年）
            
        Returns:
            历史数据 DataFrame，包含：日期、开盘价、最高价、最低价、收盘价、成交量、成交额
        """
        cache_key = f"ths_{symbol}"
        if cache_key in self._industry_hist_cache:
            return self._industry_hist_cache[cache_key]
        
        try:
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y%m%d')
            
            df = self._call_akshare_with_retry(
                lambda: ak.stock_board_industry_index_ths(symbol=symbol, start_date=start_date, end_date=end_date),
                f"同花顺指数历史({symbol})"
            )
            if df is not None and not df.empty:
                self._industry_hist_cache[cache_key] = df
                logger.debug(f"[板块机会] {symbol} 获取到 {len(df)} 条历史数据")
            return df
        except Exception as e:
            logger.warning(f"[板块机会] 获取 {symbol} 历史数据失败: {e}")
            return None
    
    def _find_ths_sector(self, sector_name: str) -> Optional[str]:
        """
        根据申万行业名称查找对应的同花顺板块名称
        
        Args:
            sector_name: 申万行业名称（如"银行"、"有色金属"）
            
        Returns:
            同花顺板块名称，找不到返回 None
        """
        # 1. 使用预定义映射
        if sector_name in self.THS_INDUSTRY_MAPPING:
            return self.THS_INDUSTRY_MAPPING[sector_name]
        
        # 2. 从同花顺数据中模糊匹配
        ths_df = self._get_ths_industry_summary()
        if ths_df is None or ths_df.empty:
            return None
        
        # 精确匹配
        if sector_name in ths_df['板块'].values:
            return sector_name
        
        # 模糊匹配
        for ths_name in ths_df['板块'].values:
            if sector_name in str(ths_name) or str(ths_name) in sector_name:
                return str(ths_name)
            # 前两个字匹配
            if len(sector_name) >= 2 and len(str(ths_name)) >= 2 and sector_name[:2] == str(ths_name)[:2]:
                return str(ths_name)
        
        return None
    
    def _get_sector_constituents(self, sector_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        获取板块领涨股信息（从同花顺数据获取）
        
        注意：同花顺行业一览表接口直接提供领涨股信息
        
        Args:
            sector_name: 板块名称
            top_n: 获取前N只股票（同花顺只提供领涨股）
            
        Returns:
            成分股列表
        """
        try:
            ths_df = self._get_ths_industry_summary()
            if ths_df is None or ths_df.empty:
                return []
            
            # 查找对应板块
            ths_sector = self._find_ths_sector(sector_name)
            if not ths_sector:
                return []
            
            matched = ths_df[ths_df['板块'] == ths_sector]
            if matched.empty:
                return []
            
            row = matched.iloc[0]
            leader_stock = str(row.get('领涨股', ''))
            leader_price = float(row.get('领涨股-最新价', 0) or 0)
            leader_change = float(row.get('领涨股-涨跌幅', 0) or 0)
            
            if leader_stock:
                return [{
                    'code': '',  # 同花顺接口不提供代码
                    'name': leader_stock,
                    'price': leader_price,
                    'change_pct': leader_change,
                    'is_leader': True,
                }]
            
            return []
            
        except Exception as e:
            logger.warning(f"[板块机会] 获取 {sector_name} 成分股失败: {e}")
            return []
    
    def _analyze_sector_chips(self, opp: SectorOpportunity, ths_sector_name: Optional[str] = None) -> None:
        """
        获取板块筹码集中度数据（简化版）
        
        注意：由于同花顺接口不直接提供成分股代码，筹码分析功能简化为获取领涨股信息
        
        Args:
            opp: 板块机会对象
            ths_sector_name: 同花顺板块名称
        """
        try:
            ths_df = self._get_ths_industry_summary()
            if ths_df is None or ths_df.empty:
                return
            
            sector_name = ths_sector_name or opp.sector_name
            ths_sector = self._find_ths_sector(sector_name)
            if not ths_sector:
                return
            
            matched = ths_df[ths_df['板块'] == ths_sector]
            if matched.empty:
                return
            
            row = matched.iloc[0]
            leader_stock = str(row.get('领涨股', ''))
            leader_change = float(row.get('领涨股-涨跌幅', 0) or 0)
            
            if leader_stock:
                opp.leader_stock_name = leader_stock
                opp.chip_analysis = f"领涨股{leader_stock}涨幅{leader_change:+.1f}%"
                
        except Exception as e:
            logger.warning(f"[板块机会] {opp.sector_name} 筹码数据获取失败: {e}")
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
    
    def _calculate_price_percentile_ths(self, sector_name: str) -> float:
        """
        使用同花顺历史数据计算价格分位数
        
        Args:
            sector_name: 同花顺行业名称
            
        Returns:
            分位数 (0-100)，越低表示当前价格越便宜
        """
        try:
            df = self._get_ths_industry_index_hist(sector_name)
            if df is None or df.empty:
                return 50.0
            
            current_price = float(df['收盘价'].iloc[-1])
            all_prices = df['收盘价'].astype(float)
            percentile = (all_prices < current_price).sum() / len(all_prices) * 100
            return percentile
        except Exception as e:
            logger.warning(f"[板块机会] 计算 {sector_name} 价格分位数失败: {e}")
            return 50.0
    
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
                       price_percentiles: Optional[Dict[str, float]] = None,
                       ths_sector_name: Optional[str] = None) -> None:
        """
        提取"够便宜"维度的数据
        
        注意：只提取数据，不做评分判断，评分由 llm_mapreduce 中的 LLM 完成
        """
        # 获取当前估值
        opp.current_pe = float(sw_row.get('TTM(滚动)市盈率', 0) or 0)
        opp.current_pb = float(sw_row.get('市净率', 0) or 0)
        opp.dividend_yield = float(sw_row.get('静态股息率', 0) or 0)
        
        # 获取价格分位数（使用同花顺板块名称作为key）
        if price_percentiles and ths_sector_name and ths_sector_name in price_percentiles:
            opp.price_percentile = price_percentiles[ths_sector_name]
        else:
            # 没有历史数据时，不设置默认值50%，而是设为-1表示无数据
            opp.price_percentile = -1.0
        
        # PE/PB分位数暂不计算（需要历史估值数据）
        opp.pe_percentile = -1.0
        opp.pb_percentile = -1.0
        
        # 收集原因供 LLM 参考
        reasons = []
        if opp.price_percentile >= 0 and opp.price_percentile < 30:
            reasons.append(f"价格处于历史{opp.price_percentile:.0f}%分位")
        if opp.current_pe > 0 and opp.current_pe < 15:
            reasons.append(f"PE {opp.current_pe:.1f}倍")
        if opp.dividend_yield > 3:
            reasons.append(f"股息率{opp.dividend_yield:.2f}%")
        if opp.avg_chip_concentration > 0:
            reasons.append(f"筹码集中度{opp.avg_chip_concentration:.0%}，获利比例{opp.avg_profit_ratio:.0%}")
        
        opp.cheap_score = 0  # 不再规则评分，由 LLM 判断
        opp.cheap_reasons = reasons
    
    def _analyze_catalyst(self, opp: SectorOpportunity, concept_df: Optional[pd.DataFrame] = None,
                          search_result: Optional[Dict[str, Any]] = None) -> None:
        """
        提取"有催化"维度的数据
        
        注意：概念板块热度检查已移除（同花顺接口需要逐个调用，效率低）
        改为依赖智能搜索结果和政策关键词
        
        注意：只提取数据，不做评分判断，评分由 llm_mapreduce 中的 LLM 完成
        """
        reasons = []
        
        # 获取相关政策关键词
        keywords = self.POLICY_KEYWORDS.get(opp.sector_name, [])
        opp.policy_keywords = keywords
        
        # 使用智能搜索结果
        if search_result and search_result.get('success'):
            catalyst_results = search_result.get('catalyst', {})
            if catalyst_results.get('results'):
                top_news = catalyst_results['results'][:2]
                news_titles = [r.title[:30] for r in top_news]
                reasons.append(f"相关新闻: {'; '.join(news_titles)}")
                opp.recent_news = [r.title for r in catalyst_results['results'][:5]]
        
        if keywords:
            reasons.append(f"关注催化关键词: {', '.join(keywords[:3])}")
        
        opp.catalyst_score = 0  # 不再规则评分，由 LLM 判断
        opp.catalyst_reasons = reasons
    
    def _analyze_reversal(self, opp: SectorOpportunity, ths_row: Optional[pd.Series], 
                          zt_count_map: Dict[str, int]) -> None:
        """
        提取"有反转"维度的数据
        
        数据来源：同花顺行业板块实时数据
        
        注意：只提取数据，不做评分判断，评分由 llm_mapreduce 中的 LLM 完成
        """
        reasons = []
        
        # 从同花顺数据获取近期表现
        if ths_row is not None:
            try:
                opp.recent_5d_change = float(ths_row.get('涨跌幅', 0) or 0)
                reasons.append(f"今日涨跌{opp.recent_5d_change:+.1f}%")
                
                # 净流入
                net_inflow = float(ths_row.get('净流入', 0) or 0)
                if net_inflow != 0:
                    reasons.append(f"净流入{net_inflow:.2f}亿")
                    opp.volume_ratio = net_inflow  # 复用字段存储净流入
                
                # 上涨/下跌家数
                up_count = int(ths_row.get('上涨家数', 0) or 0)
                down_count = int(ths_row.get('下跌家数', 0) or 0)
                if up_count > 0 or down_count > 0:
                    reasons.append(f"涨{up_count}/跌{down_count}")
                
                # 领涨股
                leader = str(ths_row.get('领涨股', ''))
                leader_change = float(ths_row.get('领涨股-涨跌幅', 0) or 0)
                if leader:
                    reasons.append(f"领涨股{leader}({leader_change:+.1f}%)")
                    opp.leader_stock_name = leader
                    
            except Exception as e:
                logger.warning(f"[板块机会] 解析同花顺数据失败: {e}")
        
        # 涨停股数量
        for industry_name, count in zt_count_map.items():
            if opp.sector_name in industry_name or industry_name in opp.sector_name:
                opp.zt_count = count
                reasons.append(f"涨停{count}只")
                break
        
        opp.reversal_score = 0  # 不再规则评分，由 LLM 判断
        opp.reversal_reasons = reasons
    
    def _generate_recommendation(self, opp: SectorOpportunity) -> None:
        """
        汇总数据，生成基础推荐信息
        
        注意：只汇总数据，不做评分判断，深度分析由 llm_mapreduce 中的 LLM 完成
        """
        # 汇总所有原因供 LLM 参考
        all_reasons = opp.cheap_reasons + opp.catalyst_reasons + opp.reversal_reasons
        
        if opp.chip_analysis:
            all_reasons.append(f"筹码: {opp.chip_analysis}")
        
        opp.total_score = 0  # 不再规则评分
        opp.recommendation = "；".join(all_reasons[:5]) if all_reasons else "暂无数据"
        
        # 基础风险提示
        risks = []
        if opp.avg_profit_ratio > 0.80:
            risks.append("获利盘过高，注意抛压")
        opp.risk_warning = "；".join(risks) if risks else ""
    
    def find_opportunity_sectors(self, fast_mode: bool = True, use_smart_search: bool = True, 
                                   analyze_chips: bool = True) -> List[SectorOpportunity]:
        """
        寻找符合埋伏条件的板块
        
        数据来源：
        - 申万一级行业估值数据（sw_index_first_info）
        - 同花顺行业板块实时数据（stock_board_industry_summary_ths）
        - 同花顺行业指数历史数据（stock_board_industry_index_ths）
        
        Args:
            fast_mode: 快速模式，跳过耗时的历史数据计算
            use_smart_search: 是否使用智能搜索获取催化剂信息
            analyze_chips: 是否分析筹码集中度（简化版，获取领涨股信息）
        
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
        
        # 2. 获取同花顺行业板块实时数据
        ths_df = self._get_ths_industry_summary()
        
        # 3. 获取涨停股池按行业统计
        zt_count_map = self._get_zt_pool_by_industry()
        
        # 4. 概念板块数据已在 _get_concept_rankings 中获取，这里不再单独获取
        # （同花顺概念板块需要逐个调用，效率低，改为在催化剂分析中使用智能搜索）
        
        # 5. 批量计算价格分位数（使用同花顺历史数据）
        price_percentiles: Dict[str, float] = {}
        if not fast_mode and ths_df is not None and not ths_df.empty:
            logger.info("[板块机会] 计算历史价格分位数（同花顺数据）...")
            for _, row in ths_df.iterrows():
                sector_name = str(row.get('板块', ''))
                if sector_name:
                    percentile = self._calculate_price_percentile_ths(sector_name)
                    price_percentiles[sector_name] = percentile
                    time.sleep(0.5)  # 避免请求过快
            logger.info(f"[板块机会] 完成 {len(price_percentiles)} 个板块的价格分位数计算")
        
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
                        time.sleep(2)  # 避免请求过快
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
            
            # 匹配同花顺数据
            ths_row = None
            ths_sector_name = None
            if ths_df is not None and not ths_df.empty:
                ths_sector = self._find_ths_sector(sector_name)
                if ths_sector:
                    matched = ths_df[ths_df['板块'] == ths_sector]
                    if not matched.empty:
                        ths_row = matched.iloc[0]
                        ths_sector_name = ths_sector
            
            # 分析筹码/领涨股信息
            if analyze_chips:
                # 只对低估值板块进行分析（减少API调用）
                pe = float(sw_row.get('TTM(滚动)市盈率', 100) or 100)
                dividend = float(sw_row.get('静态股息率', 0) or 0)
                if pe < 25 or dividend > 2.5:
                    self._analyze_sector_chips(opp, ths_sector_name)
            
            # 分析三个维度
            self._analyze_cheap(opp, sw_row, price_percentiles, ths_sector_name)
            
            # 获取该板块的智能搜索结果（如果有）
            search_result = sector_search_results.get(sector_name)
            self._analyze_catalyst(opp, search_result=search_result)
            
            self._analyze_reversal(opp, ths_row, zt_count_map)
            
            # 生成推荐
            self._generate_recommendation(opp)
            
            opportunities.append(opp)
        
        # 9. 按价格分位数排序（估值最低的排前面，无数据的排最后）
        opportunities.sort(key=lambda x: x.price_percentile if x.price_percentile >= 0 else 999)
        
        # 10. 输出分析结果
        logger.info(f"[板块机会] 分析完成，共 {len(opportunities)} 个行业")
        
        # 输出估值最低的板块
        valid_opps = [o for o in opportunities if o.price_percentile >= 0]
        if valid_opps:
            logger.info(f"[板块机会] 估值最低的5个板块:")
            for opp in valid_opps[:5]:
                logger.info(f"  - {opp.sector_name}: 价格分位{opp.price_percentile:.0f}% "
                          f"PE:{opp.current_pe:.1f} 股息率:{opp.dividend_yield:.1f}%")
        else:
            logger.info("[板块机会] 未获取到有效的价格分位数据")
        
        logger.info("========== 板块机会分析完成 ==========")
        
        return opportunities
    

# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    # 测试板块机会分析
    print("=== 测试板块数据获取 ===")
    opportunity_analyzer = SectorOpportunityAnalyzer()
    opportunities = opportunity_analyzer.find_opportunity_sectors(fast_mode=True)
    
    print(f"\n共获取 {len(opportunities)} 个行业数据")
    print("\n估值最低的5个板块:")
    for i, opp in enumerate(opportunities[:5], 1):
        print(f"{i}. {opp.sector_name}: 价格分位{opp.price_percentile:.0f}%")
        print(f"   PE:{opp.current_pe:.1f} PB:{opp.current_pb:.1f} 股息率:{opp.dividend_yield:.1f}%")
        if opp.cheap_reasons:
            print(f"   数据: {', '.join(opp.cheap_reasons[:2])}")
    
    # 生成数据报告
    print("\n=== 生成板块数据报告 ===")
    report = opportunity_analyzer.generate_template_report(opportunities)
    print(report[:1500] + "..." if len(report) > 1500 else report)
