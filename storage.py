# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 存储层
===================================

职责：
1. 管理 SQLite 数据库连接（单例模式）
2. 定义 ORM 数据模型
3. 提供数据存取接口
4. 实现智能更新逻辑（断点续传）
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Date,
    DateTime,
    Integer,
    Index,
    UniqueConstraint,
    select,
    and_,
    desc,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
)
from sqlalchemy.exc import IntegrityError

from config import get_config

logger = logging.getLogger(__name__)

# SQLAlchemy ORM 基类
Base = declarative_base()


# === 数据模型定义 ===

class StockDaily(Base):
    """
    股票日线数据模型
    
    存储每日行情数据和计算的技术指标
    支持多股票、多日期的唯一约束
    """
    __tablename__ = 'stock_daily'
    
    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 股票代码（如 600519, 000001）
    code = Column(String(10), nullable=False, index=True)
    
    # 交易日期
    date = Column(Date, nullable=False, index=True)
    
    # OHLC 数据
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    
    # 成交数据
    volume = Column(Float)  # 成交量（股）
    amount = Column(Float)  # 成交额（元）
    pct_chg = Column(Float)  # 涨跌幅（%）
    
    # 技术指标
    ma5 = Column(Float)
    ma10 = Column(Float)
    ma20 = Column(Float)
    volume_ratio = Column(Float)  # 量比
    
    # 数据来源
    data_source = Column(String(50))  # 记录数据来源（如 AkshareFetcher）
    
    # 更新时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 唯一约束：同一股票同一日期只能有一条数据
    __table_args__ = (
        UniqueConstraint('code', 'date', name='uix_code_date'),
        Index('ix_code_date', 'code', 'date'),
    )
    
    def __repr__(self):
        return f"<StockDaily(code={self.code}, date={self.date}, close={self.close})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount,
            'pct_chg': self.pct_chg,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'volume_ratio': self.volume_ratio,
            'data_source': self.data_source,
        }


class DatabaseManager:
    """
    数据库管理器 - 单例模式
    
    职责：
    1. 管理数据库连接池
    2. 提供 Session 上下文管理
    3. 封装数据存取操作
    """
    
    _instance: Optional['DatabaseManager'] = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_url: Optional[str] = None):
        """
        初始化数据库管理器
        
        Args:
            db_url: 数据库连接 URL（可选，默认从配置读取）
        """
        if self._initialized:
            return
        
        if db_url is None:
            config = get_config()
            db_url = config.get_db_url()
        
        # 创建数据库引擎
        self._engine = create_engine(
            db_url,
            echo=False,  # 设为 True 可查看 SQL 语句
            pool_pre_ping=True,  # 连接健康检查
        )
        
        # 创建 Session 工厂
        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )
        
        # 创建所有表
        Base.metadata.create_all(self._engine)
        
        self._initialized = True
        logger.info(f"数据库初始化完成: {db_url}")
    
    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（用于测试）"""
        if cls._instance is not None:
            cls._instance._engine.dispose()
            cls._instance = None
    
    def get_session(self) -> Session:
        """
        获取数据库 Session
        
        使用示例:
            with db.get_session() as session:
                # 执行查询
                session.commit()  # 如果需要
        """
        session = self._SessionLocal()
        try:
            return session
        except Exception:
            session.close()
            raise
    
    def has_today_data(self, code: str, target_date: Optional[date] = None) -> bool:
        """
        检查是否已有指定日期的数据
        
        用于断点续传逻辑：如果已有数据则跳过网络请求
        
        Args:
            code: 股票代码
            target_date: 目标日期（默认今天）
            
        Returns:
            是否存在数据
        """
        if target_date is None:
            target_date = date.today()
        
        with self.get_session() as session:
            result = session.execute(
                select(StockDaily).where(
                    and_(
                        StockDaily.code == code,
                        StockDaily.date == target_date
                    )
                )
            ).scalar_one_or_none()
            
            return result is not None
    
    def get_latest_data(
        self, 
        code: str, 
        days: int = 2
    ) -> List[StockDaily]:
        """
        获取最近 N 天的数据
        
        用于计算"相比昨日"的变化
        
        Args:
            code: 股票代码
            days: 获取天数
            
        Returns:
            StockDaily 对象列表（按日期降序）
        """
        with self.get_session() as session:
            results = session.execute(
                select(StockDaily)
                .where(StockDaily.code == code)
                .order_by(desc(StockDaily.date))
                .limit(days)
            ).scalars().all()
            
            return list(results)
    
    def get_data_range(
        self, 
        code: str, 
        start_date: date, 
        end_date: date
    ) -> List[StockDaily]:
        """
        获取指定日期范围的数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            StockDaily 对象列表
        """
        with self.get_session() as session:
            results = session.execute(
                select(StockDaily)
                .where(
                    and_(
                        StockDaily.code == code,
                        StockDaily.date >= start_date,
                        StockDaily.date <= end_date
                    )
                )
                .order_by(StockDaily.date)
            ).scalars().all()
            
            return list(results)
    
    def save_daily_data(
        self, 
        df: pd.DataFrame, 
        code: str,
        data_source: str = "Unknown"
    ) -> int:
        """
        保存日线数据到数据库
        
        策略：
        - 使用 UPSERT 逻辑（存在则更新，不存在则插入）
        - 跳过已存在的数据，避免重复
        
        Args:
            df: 包含日线数据的 DataFrame
            code: 股票代码
            data_source: 数据来源名称
            
        Returns:
            新增/更新的记录数
        """
        if df is None or df.empty:
            logger.warning(f"保存数据为空，跳过 {code}")
            return 0
        
        saved_count = 0
        
        with self.get_session() as session:
            try:
                for _, row in df.iterrows():
                    # 解析日期
                    row_date = row.get('date')
                    if isinstance(row_date, str):
                        row_date = datetime.strptime(row_date, '%Y-%m-%d').date()
                    elif isinstance(row_date, datetime):
                        row_date = row_date.date()
                    elif isinstance(row_date, pd.Timestamp):
                        row_date = row_date.date()
                    
                    # 检查是否已存在
                    existing = session.execute(
                        select(StockDaily).where(
                            and_(
                                StockDaily.code == code,
                                StockDaily.date == row_date
                            )
                        )
                    ).scalar_one_or_none()
                    
                    if existing:
                        # 更新现有记录
                        existing.open = row.get('open')
                        existing.high = row.get('high')
                        existing.low = row.get('low')
                        existing.close = row.get('close')
                        existing.volume = row.get('volume')
                        existing.amount = row.get('amount')
                        existing.pct_chg = row.get('pct_chg')
                        existing.ma5 = row.get('ma5')
                        existing.ma10 = row.get('ma10')
                        existing.ma20 = row.get('ma20')
                        existing.volume_ratio = row.get('volume_ratio')
                        existing.data_source = data_source
                        existing.updated_at = datetime.now()
                    else:
                        # 创建新记录
                        record = StockDaily(
                            code=code,
                            date=row_date,
                            open=row.get('open'),
                            high=row.get('high'),
                            low=row.get('low'),
                            close=row.get('close'),
                            volume=row.get('volume'),
                            amount=row.get('amount'),
                            pct_chg=row.get('pct_chg'),
                            ma5=row.get('ma5'),
                            ma10=row.get('ma10'),
                            ma20=row.get('ma20'),
                            volume_ratio=row.get('volume_ratio'),
                            data_source=data_source,
                        )
                        session.add(record)
                        saved_count += 1
                
                session.commit()
                logger.info(f"保存 {code} 数据成功，新增 {saved_count} 条")
                
            except Exception as e:
                session.rollback()
                logger.error(f"保存 {code} 数据失败: {e}")
                raise
        
        return saved_count
    
    def get_analysis_context(
        self, 
        code: str,
        target_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取分析所需的上下文数据
        
        返回今日数据 + 昨日数据的对比信息
        
        Args:
            code: 股票代码
            target_date: 目标日期（默认今天）
            
        Returns:
            包含今日数据、昨日对比等信息的字典
        """
        if target_date is None:
            target_date = date.today()
        
        # 获取最近2天数据
        recent_data = self.get_latest_data(code, days=2)
        
        if not recent_data:
            logger.warning(f"未找到 {code} 的数据")
            return None
        
        today_data = recent_data[0]
        yesterday_data = recent_data[1] if len(recent_data) > 1 else None
        
        context = {
            'code': code,
            'date': today_data.date.isoformat(),
            'today': today_data.to_dict(),
        }
        
        if yesterday_data:
            context['yesterday'] = yesterday_data.to_dict()
            
            # 计算相比昨日的变化
            if yesterday_data.volume and yesterday_data.volume > 0:
                context['volume_change_ratio'] = round(
                    today_data.volume / yesterday_data.volume, 2
                )
            
            if yesterday_data.close and yesterday_data.close > 0:
                context['price_change_ratio'] = round(
                    (today_data.close - yesterday_data.close) / yesterday_data.close * 100, 2
                )
            
            # 均线形态判断
            context['ma_status'] = self._analyze_ma_status(today_data)
        
        return context
    
    def _analyze_ma_status(self, data: StockDaily) -> str:
        """
        分析均线形态
        
        判断条件：
        - 多头排列：close > ma5 > ma10 > ma20
        - 空头排列：close < ma5 < ma10 < ma20
        - 震荡整理：其他情况
        """
        close = data.close or 0
        ma5 = data.ma5 or 0
        ma10 = data.ma10 or 0
        ma20 = data.ma20 or 0
        
        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        elif close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        elif close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        elif close < ma5 and ma5 < ma10:
            return "短期走弱 🔽"
        else:
            return "震荡整理 ↔️"


# 便捷函数
def get_db() -> DatabaseManager:
    """获取数据库管理器实例的快捷方式"""
    return DatabaseManager.get_instance()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    db = get_db()
    
    print("=== 数据库测试 ===")
    print(f"数据库初始化成功")
    
    # 测试检查今日数据
    has_data = db.has_today_data('600519')
    print(f"茅台今日是否有数据: {has_data}")
    
    # 测试保存数据
    test_df = pd.DataFrame({
        'date': [date.today()],
        'open': [1800.0],
        'high': [1850.0],
        'low': [1780.0],
        'close': [1820.0],
        'volume': [10000000],
        'amount': [18200000000],
        'pct_chg': [1.5],
        'ma5': [1810.0],
        'ma10': [1800.0],
        'ma20': [1790.0],
        'volume_ratio': [1.2],
    })
    
    saved = db.save_daily_data(test_df, '600519', 'TestSource')
    print(f"保存测试数据: {saved} 条")
    
    # 测试获取上下文
    context = db.get_analysis_context('600519')
    print(f"分析上下文: {context}")


# === 市场历史数据模型 ===

class MarketDaily(Base):
    """
    市场日度数据模型
    
    存储每日市场整体指标，用于历史对比分析
    """
    __tablename__ = 'market_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # 涨跌统计
    up_count = Column(Integer)          # 上涨家数
    down_count = Column(Integer)        # 下跌家数
    flat_count = Column(Integer)        # 平盘家数
    limit_up_count = Column(Integer)    # 涨停家数
    limit_down_count = Column(Integer)  # 跌停家数
    
    # 涨停板数据
    zt_first_board = Column(Integer)    # 首板数量
    zt_continuous = Column(Integer)     # 连板数量
    zt_max_height = Column(Integer)     # 最高连板数
    zb_count = Column(Integer)          # 炸板数量
    
    # 成交数据
    total_amount = Column(Float)        # 两市成交额（亿元）
    avg_turnover = Column(Float)        # 平均换手率
    
    # 两融数据
    margin_balance = Column(Float)      # 融资余额（亿元）
    margin_buy = Column(Float)          # 融资买入额（亿元）
    
    # 龙虎榜数据
    lhb_count = Column(Integer)         # 龙虎榜股票数
    lhb_org_net_buy = Column(Float)     # 机构净买入（亿元）
    
    # 指数数据
    sh_index = Column(Float)            # 上证指数
    sh_change_pct = Column(Float)       # 上证涨跌幅
    sz_index = Column(Float)            # 深证成指
    sz_change_pct = Column(Float)       # 深证涨跌幅
    cyb_index = Column(Float)           # 创业板指
    cyb_change_pct = Column(Float)      # 创业板涨跌幅
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'up_count': self.up_count,
            'down_count': self.down_count,
            'flat_count': self.flat_count,
            'limit_up_count': self.limit_up_count,
            'limit_down_count': self.limit_down_count,
            'zt_first_board': self.zt_first_board,
            'zt_continuous': self.zt_continuous,
            'zt_max_height': self.zt_max_height,
            'zb_count': self.zb_count,
            'total_amount': self.total_amount,
            'avg_turnover': self.avg_turnover,
            'margin_balance': self.margin_balance,
            'margin_buy': self.margin_buy,
            'lhb_count': self.lhb_count,
            'lhb_org_net_buy': self.lhb_org_net_buy,
            'sh_index': self.sh_index,
            'sh_change_pct': self.sh_change_pct,
            'sz_index': self.sz_index,
            'sz_change_pct': self.sz_change_pct,
            'cyb_index': self.cyb_index,
            'cyb_change_pct': self.cyb_change_pct,
        }


class SectorDaily(Base):
    """
    板块日度数据模型
    
    存储每日板块涨跌幅，用于板块轮动分析
    """
    __tablename__ = 'sector_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    sector_name = Column(String(50), nullable=False, index=True)
    
    change_pct = Column(Float)          # 涨跌幅
    net_inflow = Column(Float)          # 净流入（亿元）
    amount = Column(Float)              # 成交额（亿元）
    up_count = Column(Integer)          # 上涨家数
    down_count = Column(Integer)        # 下跌家数
    leader_stock = Column(String(20))   # 领涨股
    leader_change = Column(Float)       # 领涨股涨跌幅
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('date', 'sector_name', name='uix_sector_date'),
        Index('ix_sector_date', 'date', 'sector_name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'sector_name': self.sector_name,
            'change_pct': self.change_pct,
            'net_inflow': self.net_inflow,
            'amount': self.amount,
            'up_count': self.up_count,
            'down_count': self.down_count,
            'leader_stock': self.leader_stock,
            'leader_change': self.leader_change,
        }


# === 新增：概念板块日度数据 ===

class ConceptDaily(Base):
    """概念板块日度数据"""
    __tablename__ = 'concept_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    concept_name = Column(String(50), nullable=False, index=True)
    
    change_pct = Column(Float)          # 涨跌幅
    net_inflow = Column(Float)          # 净流入（亿元）
    amount = Column(Float)              # 成交额（亿元）
    up_count = Column(Integer)          # 上涨家数
    down_count = Column(Integer)        # 下跌家数
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('date', 'concept_name', name='uix_concept_date'),
        Index('ix_concept_date', 'date', 'concept_name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.concept_name,
            'change_pct': self.change_pct,
            'net_inflow': self.net_inflow,
            'amount': self.amount,
            'up_count': self.up_count,
            'down_count': self.down_count,
        }


# === 新增：融资融券日度数据 ===

class MarginDaily(Base):
    """融资融券日度数据"""
    __tablename__ = 'margin_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    margin_balance = Column(Float)      # 融资余额（亿元）
    margin_buy = Column(Float)          # 融资买入额（亿元）
    short_balance = Column(Float)       # 融券余额（亿元）
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'margin_balance': self.margin_balance,
            'margin_buy': self.margin_buy,
            'short_balance': self.short_balance,
        }


# === 新增：龙虎榜日度数据 ===

class LhbDaily(Base):
    """龙虎榜日度汇总数据"""
    __tablename__ = 'lhb_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    lhb_count = Column(Integer)         # 上榜股票数
    lhb_net_buy = Column(Float)         # 龙虎榜净买入（亿元）
    org_buy_count = Column(Integer)     # 机构买入次数
    org_sell_count = Column(Integer)    # 机构卖出次数
    org_net_buy = Column(Float)         # 机构净买入（亿元）
    
    # JSON 存储详细数据
    stocks_json = Column(String(10000)) # 龙虎榜股票列表 JSON
    seat_detail_json = Column(String(20000))  # 席位明细 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'lhb_count': self.lhb_count,
            'lhb_net_buy': self.lhb_net_buy,
            'org_buy_count': self.org_buy_count,
            'org_sell_count': self.org_sell_count,
            'org_net_buy': self.org_net_buy,
        }


# === 新增：大宗交易日度数据 ===

class BlockTradeDaily(Base):
    """大宗交易日度数据"""
    __tablename__ = 'block_trade_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    amount = Column(Float)              # 成交额（亿元）
    premium_ratio = Column(Float)       # 溢价成交占比(%)
    discount_ratio = Column(Float)      # 折价成交占比(%)
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'amount': self.amount,
            'premium_ratio': self.premium_ratio,
            'discount_ratio': self.discount_ratio,
        }


# === 新增：涨停股池日度数据 ===

class ZtPoolDaily(Base):
    """涨停股池日度数据"""
    __tablename__ = 'zt_pool_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    zt_count = Column(Integer)          # 涨停股数量
    total_amount = Column(Float)        # 涨停股总成交额（亿元）
    avg_turnover = Column(Float)        # 平均换手率
    first_board_count = Column(Integer) # 首板数量
    continuous_count = Column(Integer)  # 连板数量
    max_continuous = Column(Integer)    # 最高连板数
    
    # JSON 存储详细数据
    stocks_json = Column(String(20000)) # 涨停股列表 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'zt_count': self.zt_count,
            'total_amount': self.total_amount,
            'avg_turnover': self.avg_turnover,
            'first_board_count': self.first_board_count,
            'continuous_count': self.continuous_count,
            'max_continuous': self.max_continuous,
        }


# === 新增：昨日涨停股池数据 ===

class PreviousZtPoolDaily(Base):
    """昨日涨停股池数据（今日表现）"""
    __tablename__ = 'previous_zt_pool_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    count = Column(Integer)             # 昨日涨停数量
    avg_change = Column(Float)          # 今日平均涨跌幅（溢价率）
    up_count = Column(Integer)          # 今日上涨数量
    down_count = Column(Integer)        # 今日下跌数量
    
    stocks_json = Column(String(10000)) # 股票列表 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'count': self.count,
            'avg_change': self.avg_change,
            'up_count': self.up_count,
            'down_count': self.down_count,
        }


# === 新增：炸板股池日度数据 ===

class ZbPoolDaily(Base):
    """炸板股池日度数据"""
    __tablename__ = 'zb_pool_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    zb_count = Column(Integer)          # 炸板股数量
    total_zb_times = Column(Integer)    # 炸板总次数
    zb_rate = Column(Float)             # 炸板率
    
    stocks_json = Column(String(10000)) # 股票列表 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'zb_count': self.zb_count,
            'total_zb_times': self.total_zb_times,
            'zb_rate': self.zb_rate,
        }


# === 新增：跌停股池日度数据 ===

class DtPoolDaily(Base):
    """跌停股池日度数据"""
    __tablename__ = 'dt_pool_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    dt_count = Column(Integer)          # 跌停股数量
    continuous_count = Column(Integer)  # 连续跌停数量
    
    stocks_json = Column(String(10000)) # 股票列表 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'dt_count': self.dt_count,
            'continuous_count': self.continuous_count,
        }


# === 新增：强势股池日度数据 ===

class StrongPoolDaily(Base):
    """强势股池日度数据"""
    __tablename__ = 'strong_pool_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    count = Column(Integer)             # 强势股数量
    new_high_count = Column(Integer)    # 60日新高数量
    multi_zt_count = Column(Integer)    # 近期多次涨停数量
    
    stocks_json = Column(String(10000)) # 股票列表 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'count': self.count,
            'new_high_count': self.new_high_count,
            'multi_zt_count': self.multi_zt_count,
        }


# === 新增：千股千评日度数据 ===

class CommentDaily(Base):
    """千股千评日度汇总数据"""
    __tablename__ = 'comment_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    avg_score = Column(Float)           # 市场平均综合得分
    high_score_count = Column(Integer)  # 高分股票数量（>=80分）
    low_score_count = Column(Integer)   # 低分股票数量（<=40分）
    
    top_stocks_json = Column(String(5000))      # 综合得分TOP10 JSON
    bottom_stocks_json = Column(String(5000))   # 综合得分最低10 JSON
    high_attention_json = Column(String(5000))  # 关注指数TOP10 JSON
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'avg_score': self.avg_score,
            'high_score_count': self.high_score_count,
            'low_score_count': self.low_score_count,
        }


class MarketHistoryManager:
    """
    市场历史数据管理器
    
    职责：
    1. 存储每日市场数据
    2. 提供历史对比查询
    3. 计算时序指标
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db()
        # 确保表已创建
        Base.metadata.create_all(self.db._engine)
    
    def save_market_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        """
        保存市场日度数据
        
        Args:
            data: 市场数据字典
            target_date: 目标日期
            
        Returns:
            是否保存成功
        """
        if target_date is None:
            target_date = date.today()
        
        with self.db.get_session() as session:
            try:
                # 检查是否已存在
                existing = session.execute(
                    select(MarketDaily).where(MarketDaily.date == target_date)
                ).scalar_one_or_none()
                
                if existing:
                    # 更新
                    for key, value in data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.now()
                else:
                    # 新增
                    record = MarketDaily(date=target_date, **data)
                    session.add(record)
                
                session.commit()
                logger.info(f"保存市场数据成功: {target_date}")
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"保存市场数据失败: {e}")
                return False
    
    def save_sector_daily(self, sectors: List[Dict[str, Any]], target_date: Optional[date] = None) -> int:
        """
        保存板块日度数据
        
        Args:
            sectors: 板块数据列表
            target_date: 目标日期
            
        Returns:
            保存的记录数
        """
        if target_date is None:
            target_date = date.today()
        
        saved_count = 0
        
        with self.db.get_session() as session:
            try:
                for sector in sectors:
                    sector_name = sector.get('name') or sector.get('sector_name')
                    if not sector_name:
                        continue
                    
                    # 检查是否已存在
                    existing = session.execute(
                        select(SectorDaily).where(
                            and_(
                                SectorDaily.date == target_date,
                                SectorDaily.sector_name == sector_name
                            )
                        )
                    ).scalar_one_or_none()
                    
                    if existing:
                        existing.change_pct = sector.get('change_pct')
                        existing.net_inflow = sector.get('net_inflow')
                        existing.amount = sector.get('amount')
                        existing.up_count = sector.get('up_count')
                        existing.down_count = sector.get('down_count')
                        existing.leader_stock = sector.get('leader_stock')
                        existing.leader_change = sector.get('leader_change')
                    else:
                        record = SectorDaily(
                            date=target_date,
                            sector_name=sector_name,
                            change_pct=sector.get('change_pct'),
                            net_inflow=sector.get('net_inflow'),
                            amount=sector.get('amount'),
                            up_count=sector.get('up_count'),
                            down_count=sector.get('down_count'),
                            leader_stock=sector.get('leader_stock'),
                            leader_change=sector.get('leader_change'),
                        )
                        session.add(record)
                        saved_count += 1
                
                session.commit()
                logger.info(f"保存板块数据成功: {target_date}, {saved_count} 条")
                
            except Exception as e:
                session.rollback()
                logger.error(f"保存板块数据失败: {e}")
        
        return saved_count
    
    def get_market_history(self, days: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近N天的市场数据
        
        Args:
            days: 获取天数
            
        Returns:
            市场数据列表（按日期降序）
        """
        with self.db.get_session() as session:
            results = session.execute(
                select(MarketDaily)
                .order_by(desc(MarketDaily.date))
                .limit(days)
            ).scalars().all()
            
            return [r.to_dict() for r in results]
    
    def get_sector_history(self, sector_name: str, days: int = 10) -> List[Dict[str, Any]]:
        """
        获取指定板块的历史数据
        
        Args:
            sector_name: 板块名称
            days: 获取天数
            
        Returns:
            板块数据列表
        """
        with self.db.get_session() as session:
            results = session.execute(
                select(SectorDaily)
                .where(SectorDaily.sector_name == sector_name)
                .order_by(desc(SectorDaily.date))
                .limit(days)
            ).scalars().all()
            
            return [r.to_dict() for r in results]
    
    def get_historical_context(self, days: int = 5) -> Dict[str, Any]:
        """
        获取历史对比上下文
        
        计算各指标的时序变化，供 LLM 分析师参考
        
        Args:
            days: 对比天数
            
        Returns:
            历史上下文字典
        """
        history = self.get_market_history(days + 1)  # 多取一天用于计算变化
        
        if len(history) < 2:
            return {'has_history': False, 'message': '历史数据不足'}
        
        today = history[0]
        yesterday = history[1]
        
        context = {
            'has_history': True,
            'today': today,
            'yesterday': yesterday,
            'history_days': len(history),
        }
        
        # 计算涨停数量趋势
        zt_counts = [h.get('limit_up_count', 0) or 0 for h in history]
        if zt_counts:
            context['zt_trend'] = {
                'today': zt_counts[0],
                'yesterday': zt_counts[1] if len(zt_counts) > 1 else 0,
                'avg_5d': sum(zt_counts[:5]) / min(5, len(zt_counts)),
                'trend': self._calc_trend(zt_counts[:5]),
                'values': zt_counts[:5],
            }
        
        # 计算成交额趋势
        amounts = [h.get('total_amount', 0) or 0 for h in history]
        if amounts:
            context['amount_trend'] = {
                'today': amounts[0],
                'yesterday': amounts[1] if len(amounts) > 1 else 0,
                'avg_5d': sum(amounts[:5]) / min(5, len(amounts)),
                'trend': self._calc_trend(amounts[:5]),
                'values': amounts[:5],
            }
        
        # 计算连板数量趋势
        continuous = [h.get('zt_continuous', 0) or 0 for h in history]
        if continuous:
            context['continuous_trend'] = {
                'today': continuous[0],
                'yesterday': continuous[1] if len(continuous) > 1 else 0,
                'avg_5d': sum(continuous[:5]) / min(5, len(continuous)),
                'trend': self._calc_trend(continuous[:5]),
                'values': continuous[:5],
            }
        
        # 计算炸板数量趋势
        zb_counts = [h.get('zb_count', 0) or 0 for h in history]
        if zb_counts:
            context['zb_trend'] = {
                'today': zb_counts[0],
                'yesterday': zb_counts[1] if len(zb_counts) > 1 else 0,
                'avg_5d': sum(zb_counts[:5]) / min(5, len(zb_counts)),
                'trend': self._calc_trend(zb_counts[:5]),
                'values': zb_counts[:5],
            }
        
        # 计算两融余额趋势
        margins = [h.get('margin_balance', 0) or 0 for h in history]
        if margins and margins[0] > 0:
            context['margin_trend'] = {
                'today': margins[0],
                'yesterday': margins[1] if len(margins) > 1 else 0,
                'change': margins[0] - margins[1] if len(margins) > 1 else 0,
                'trend': self._calc_trend(margins[:5]),
                'values': margins[:5],
            }
        
        # 计算机构净买入趋势
        org_buys = [h.get('lhb_org_net_buy', 0) or 0 for h in history]
        if org_buys:
            context['org_buy_trend'] = {
                'today': org_buys[0],
                'yesterday': org_buys[1] if len(org_buys) > 1 else 0,
                'sum_5d': sum(org_buys[:5]),
                'trend': self._calc_trend(org_buys[:5]),
                'values': org_buys[:5],
            }
        
        return context
    
    def _calc_trend(self, values: List[float]) -> str:
        """
        计算趋势方向
        
        Args:
            values: 数值列表（从新到旧）
            
        Returns:
            趋势描述
        """
        if len(values) < 2:
            return "数据不足"
        
        # 计算连续上升/下降天数
        up_days = 0
        down_days = 0
        
        for i in range(len(values) - 1):
            if values[i] > values[i + 1]:
                up_days += 1
            elif values[i] < values[i + 1]:
                down_days += 1
        
        if up_days >= 3:
            return f"连续{up_days}天上升 📈"
        elif down_days >= 3:
            return f"连续{down_days}天下降 📉"
        elif up_days > down_days:
            return "震荡上行 🔼"
        elif down_days > up_days:
            return "震荡下行 🔽"
        else:
            return "横盘整理 ↔️"
    
    def has_today_data(self, target_date: Optional[date] = None) -> bool:
        """检查是否已有指定日期的市场数据"""
        if target_date is None:
            target_date = date.today()
        
        with self.db.get_session() as session:
            result = session.execute(
                select(MarketDaily).where(MarketDaily.date == target_date)
            ).scalar_one_or_none()
            
            return result is not None
    
    # ========== 新增：概念板块数据存取 ==========
    
    def save_concept_daily(self, concepts: List[Dict[str, Any]], target_date: Optional[date] = None) -> int:
        """保存概念板块日度数据"""
        if target_date is None:
            target_date = date.today()
        
        saved_count = 0
        with self.db.get_session() as session:
            try:
                for concept in concepts:
                    concept_name = concept.get('name')
                    if not concept_name:
                        continue
                    
                    existing = session.execute(
                        select(ConceptDaily).where(
                            and_(ConceptDaily.date == target_date, ConceptDaily.concept_name == concept_name)
                        )
                    ).scalar_one_or_none()
                    
                    if existing:
                        existing.change_pct = concept.get('change_pct')
                        existing.net_inflow = concept.get('net_inflow')
                        existing.amount = concept.get('amount')
                        existing.up_count = concept.get('up_count')
                        existing.down_count = concept.get('down_count')
                    else:
                        record = ConceptDaily(
                            date=target_date, concept_name=concept_name,
                            change_pct=concept.get('change_pct'), net_inflow=concept.get('net_inflow'),
                            amount=concept.get('amount'), up_count=concept.get('up_count'),
                            down_count=concept.get('down_count'),
                        )
                        session.add(record)
                        saved_count += 1
                session.commit()
                logger.info(f"保存概念板块数据成功: {target_date}, {saved_count} 条")
            except Exception as e:
                session.rollback()
                logger.error(f"保存概念板块数据失败: {e}")
        return saved_count
    
    def get_concept_daily(self, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """获取指定日期的概念板块数据"""
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            results = session.execute(
                select(ConceptDaily).where(ConceptDaily.date == target_date).order_by(desc(ConceptDaily.change_pct))
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def has_concept_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(ConceptDaily).where(ConceptDaily.date == target_date).limit(1)).scalar_one_or_none() is not None
    
    # ========== 新增：融资融券数据存取 ==========
    
    def save_margin_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(MarginDaily).where(MarginDaily.date == target_date)).scalar_one_or_none()
                if existing:
                    existing.margin_balance = data.get('margin_balance')
                    existing.margin_buy = data.get('margin_buy')
                    existing.short_balance = data.get('short_balance')
                else:
                    session.add(MarginDaily(date=target_date, margin_balance=data.get('margin_balance'),
                        margin_buy=data.get('margin_buy'), short_balance=data.get('short_balance')))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存融资融券数据失败: {e}")
                return False
    
    def get_margin_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(MarginDaily).where(MarginDaily.date == target_date)).scalar_one_or_none()
            return result.to_dict() if result else None
    
    def has_margin_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(MarginDaily).where(MarginDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：龙虎榜数据存取 ==========
    
    def save_lhb_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(LhbDaily).where(LhbDaily.date == target_date)).scalar_one_or_none()
                stocks_json = json.dumps(data.get('stocks', []), ensure_ascii=False) if data.get('stocks') else None
                seat_json = json.dumps(data.get('seat_detail', []), ensure_ascii=False) if data.get('seat_detail') else None
                if existing:
                    existing.lhb_count = data.get('lhb_count')
                    existing.lhb_net_buy = data.get('lhb_net_buy')
                    existing.org_buy_count = data.get('org_buy_count')
                    existing.org_sell_count = data.get('org_sell_count')
                    existing.org_net_buy = data.get('org_net_buy')
                    existing.stocks_json = stocks_json
                    existing.seat_detail_json = seat_json
                else:
                    session.add(LhbDaily(date=target_date, lhb_count=data.get('lhb_count'), lhb_net_buy=data.get('lhb_net_buy'),
                        org_buy_count=data.get('org_buy_count'), org_sell_count=data.get('org_sell_count'),
                        org_net_buy=data.get('org_net_buy'), stocks_json=stocks_json, seat_detail_json=seat_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存龙虎榜数据失败: {e}")
                return False
    
    def get_lhb_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(LhbDaily).where(LhbDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['stocks'] = json.loads(result.stocks_json) if result.stocks_json else []
                data['seat_detail'] = json.loads(result.seat_detail_json) if result.seat_detail_json else []
                return data
            return None
    
    def has_lhb_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(LhbDaily).where(LhbDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：大宗交易数据存取 ==========
    
    def save_block_trade_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(BlockTradeDaily).where(BlockTradeDaily.date == target_date)).scalar_one_or_none()
                if existing:
                    existing.amount = data.get('amount')
                    existing.premium_ratio = data.get('premium_ratio')
                    existing.discount_ratio = data.get('discount_ratio')
                else:
                    session.add(BlockTradeDaily(date=target_date, amount=data.get('amount'),
                        premium_ratio=data.get('premium_ratio'), discount_ratio=data.get('discount_ratio')))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存大宗交易数据失败: {e}")
                return False
    
    def get_block_trade_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(BlockTradeDaily).where(BlockTradeDaily.date == target_date)).scalar_one_or_none()
            return result.to_dict() if result else None
    
    def has_block_trade_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(BlockTradeDaily).where(BlockTradeDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：涨停股池数据存取 ==========
    
    def save_zt_pool_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(ZtPoolDaily).where(ZtPoolDaily.date == target_date)).scalar_one_or_none()
                stocks_json = json.dumps(data.get('stocks', []), ensure_ascii=False) if data.get('stocks') else None
                if existing:
                    existing.zt_count = data.get('zt_count')
                    existing.total_amount = data.get('total_amount')
                    existing.avg_turnover = data.get('avg_turnover')
                    existing.first_board_count = data.get('first_board_count')
                    existing.continuous_count = data.get('continuous_count')
                    existing.max_continuous = data.get('max_continuous')
                    existing.stocks_json = stocks_json
                else:
                    session.add(ZtPoolDaily(date=target_date, zt_count=data.get('zt_count'), total_amount=data.get('total_amount'),
                        avg_turnover=data.get('avg_turnover'), first_board_count=data.get('first_board_count'),
                        continuous_count=data.get('continuous_count'), max_continuous=data.get('max_continuous'), stocks_json=stocks_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存涨停股池数据失败: {e}")
                return False
    
    def get_zt_pool_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(ZtPoolDaily).where(ZtPoolDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['stocks'] = json.loads(result.stocks_json) if result.stocks_json else []
                return data
            return None
    
    def has_zt_pool_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(ZtPoolDaily).where(ZtPoolDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：昨日涨停股池数据存取 ==========
    
    def save_previous_zt_pool_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(PreviousZtPoolDaily).where(PreviousZtPoolDaily.date == target_date)).scalar_one_or_none()
                stocks_json = json.dumps(data.get('stocks', []), ensure_ascii=False) if data.get('stocks') else None
                if existing:
                    existing.count = data.get('count')
                    existing.avg_change = data.get('avg_change')
                    existing.up_count = data.get('up_count')
                    existing.down_count = data.get('down_count')
                    existing.stocks_json = stocks_json
                else:
                    session.add(PreviousZtPoolDaily(date=target_date, count=data.get('count'), avg_change=data.get('avg_change'),
                        up_count=data.get('up_count'), down_count=data.get('down_count'), stocks_json=stocks_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存昨日涨停股池数据失败: {e}")
                return False
    
    def get_previous_zt_pool_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(PreviousZtPoolDaily).where(PreviousZtPoolDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['stocks'] = json.loads(result.stocks_json) if result.stocks_json else []
                return data
            return None
    
    def has_previous_zt_pool_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(PreviousZtPoolDaily).where(PreviousZtPoolDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：炸板股池数据存取 ==========
    
    def save_zb_pool_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(ZbPoolDaily).where(ZbPoolDaily.date == target_date)).scalar_one_or_none()
                stocks_json = json.dumps(data.get('stocks', []), ensure_ascii=False) if data.get('stocks') else None
                if existing:
                    existing.zb_count = data.get('zb_count')
                    existing.total_zb_times = data.get('total_zb_times')
                    existing.zb_rate = data.get('zb_rate')
                    existing.stocks_json = stocks_json
                else:
                    session.add(ZbPoolDaily(date=target_date, zb_count=data.get('zb_count'),
                        total_zb_times=data.get('total_zb_times'), zb_rate=data.get('zb_rate'), stocks_json=stocks_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存炸板股池数据失败: {e}")
                return False
    
    def get_zb_pool_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(ZbPoolDaily).where(ZbPoolDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['stocks'] = json.loads(result.stocks_json) if result.stocks_json else []
                return data
            return None
    
    def has_zb_pool_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(ZbPoolDaily).where(ZbPoolDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：跌停股池数据存取 ==========
    
    def save_dt_pool_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(DtPoolDaily).where(DtPoolDaily.date == target_date)).scalar_one_or_none()
                stocks_json = json.dumps(data.get('stocks', []), ensure_ascii=False) if data.get('stocks') else None
                if existing:
                    existing.dt_count = data.get('dt_count')
                    existing.continuous_count = data.get('continuous_count')
                    existing.stocks_json = stocks_json
                else:
                    session.add(DtPoolDaily(date=target_date, dt_count=data.get('dt_count'),
                        continuous_count=data.get('continuous_count'), stocks_json=stocks_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存跌停股池数据失败: {e}")
                return False
    
    def get_dt_pool_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(DtPoolDaily).where(DtPoolDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['stocks'] = json.loads(result.stocks_json) if result.stocks_json else []
                return data
            return None
    
    def has_dt_pool_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(DtPoolDaily).where(DtPoolDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：强势股池数据存取 ==========
    
    def save_strong_pool_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(StrongPoolDaily).where(StrongPoolDaily.date == target_date)).scalar_one_or_none()
                stocks_json = json.dumps(data.get('stocks', []), ensure_ascii=False) if data.get('stocks') else None
                if existing:
                    existing.count = data.get('count')
                    existing.new_high_count = data.get('new_high_count')
                    existing.multi_zt_count = data.get('multi_zt_count')
                    existing.stocks_json = stocks_json
                else:
                    session.add(StrongPoolDaily(date=target_date, count=data.get('count'),
                        new_high_count=data.get('new_high_count'), multi_zt_count=data.get('multi_zt_count'), stocks_json=stocks_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存强势股池数据失败: {e}")
                return False
    
    def get_strong_pool_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(StrongPoolDaily).where(StrongPoolDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['stocks'] = json.loads(result.stocks_json) if result.stocks_json else []
                return data
            return None
    
    def has_strong_pool_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(StrongPoolDaily).where(StrongPoolDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：千股千评数据存取 ==========
    
    def save_comment_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(CommentDaily).where(CommentDaily.date == target_date)).scalar_one_or_none()
                top_json = json.dumps(data.get('top_stocks', []), ensure_ascii=False) if data.get('top_stocks') else None
                bottom_json = json.dumps(data.get('bottom_stocks', []), ensure_ascii=False) if data.get('bottom_stocks') else None
                attention_json = json.dumps(data.get('high_attention', []), ensure_ascii=False) if data.get('high_attention') else None
                if existing:
                    existing.avg_score = data.get('avg_score')
                    existing.high_score_count = data.get('high_score_count')
                    existing.low_score_count = data.get('low_score_count')
                    existing.top_stocks_json = top_json
                    existing.bottom_stocks_json = bottom_json
                    existing.high_attention_json = attention_json
                else:
                    session.add(CommentDaily(date=target_date, avg_score=data.get('avg_score'),
                        high_score_count=data.get('high_score_count'), low_score_count=data.get('low_score_count'),
                        top_stocks_json=top_json, bottom_stocks_json=bottom_json, high_attention_json=attention_json))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存千股千评数据失败: {e}")
                return False
    
    def get_comment_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        import json
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(CommentDaily).where(CommentDaily.date == target_date)).scalar_one_or_none()
            if result:
                data = result.to_dict()
                data['top_stocks'] = json.loads(result.top_stocks_json) if result.top_stocks_json else []
                data['bottom_stocks'] = json.loads(result.bottom_stocks_json) if result.bottom_stocks_json else []
                data['high_attention'] = json.loads(result.high_attention_json) if result.high_attention_json else []
                return data
            return None
    
    def has_comment_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(CommentDaily).where(CommentDaily.date == target_date)).scalar_one_or_none() is not None


def get_market_history_manager() -> MarketHistoryManager:
    """获取市场历史数据管理器实例"""
    return MarketHistoryManager()
