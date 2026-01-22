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
    func,
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
    
    # ========== 新增：指数技术面数据存取 ==========
    
    def save_index_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        """保存指数日度技术指标数据"""
        if target_date is None:
            target_date = date.today()
        code = data.get('code')
        if not code:
            return False
        with self.db.get_session() as session:
            try:
                existing = session.execute(
                    select(IndexDaily).where(and_(IndexDaily.date == target_date, IndexDaily.code == code))
                ).scalar_one_or_none()
                if existing:
                    for key, value in data.items():
                        if hasattr(existing, key) and key not in ('date', 'code'):
                            setattr(existing, key, value)
                else:
                    session.add(IndexDaily(date=target_date, **data))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存指数数据失败: {e}")
                return False
    
    def save_index_daily_batch(self, data_list: List[Dict[str, Any]], target_date: Optional[date] = None) -> int:
        """批量保存指数日度数据"""
        if target_date is None:
            target_date = date.today()
        saved_count = 0
        for data in data_list:
            if self.save_index_daily(data, target_date):
                saved_count += 1
        return saved_count
    
    def get_index_history(self, code: str, days: int = 60) -> List[Dict[str, Any]]:
        """获取指数历史数据"""
        with self.db.get_session() as session:
            results = session.execute(
                select(IndexDaily).where(IndexDaily.code == code).order_by(desc(IndexDaily.date)).limit(days)
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def get_index_history_range(self, code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        获取指定日期范围的指数历史数据
        
        Args:
            code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            按日期升序排列的历史数据列表
        """
        with self.db.get_session() as session:
            results = session.execute(
                select(IndexDaily).where(
                    and_(
                        IndexDaily.code == code,
                        IndexDaily.date >= start_date,
                        IndexDaily.date <= end_date
                    )
                ).order_by(IndexDaily.date)
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def get_index_data_count(self, code: str, start_date: date, end_date: date) -> int:
        """
        获取指定日期范围内的数据条数
        
        用于判断是否需要从 API 获取数据
        """
        with self.db.get_session() as session:
            count = session.execute(
                select(func.count(IndexDaily.id)).where(
                    and_(
                        IndexDaily.code == code,
                        IndexDaily.date >= start_date,
                        IndexDaily.date <= end_date
                    )
                )
            ).scalar()
            return count or 0
    
    def save_index_history_batch(self, code: str, name: str, df) -> int:
        """
        批量保存指数历史数据（从 DataFrame）
        
        Args:
            code: 指数代码
            name: 指数名称
            df: 包含历史数据的 DataFrame（来自 ak.index_zh_a_hist）
            
        Returns:
            保存的记录数
        """
        if df is None or df.empty:
            return 0
        
        saved_count = 0
        with self.db.get_session() as session:
            try:
                for _, row in df.iterrows():
                    # 解析日期
                    date_val = row.get('日期')
                    if isinstance(date_val, str):
                        target_date = date.fromisoformat(date_val)
                    elif hasattr(date_val, 'date'):
                        target_date = date_val.date()
                    else:
                        target_date = date_val
                    
                    # 检查是否已存在
                    existing = session.execute(
                        select(IndexDaily).where(
                            and_(IndexDaily.date == target_date, IndexDaily.code == code)
                        )
                    ).scalar_one_or_none()
                    
                    data = {
                        'code': code,
                        'name': name,
                        'open': float(row.get('开盘', 0) or 0),
                        'close': float(row.get('收盘', 0) or 0),
                        'high': float(row.get('最高', 0) or 0),
                        'low': float(row.get('最低', 0) or 0),
                        'volume': float(row.get('成交量', 0) or 0),
                        'amount': float(row.get('成交额', 0) or 0),
                        'change_pct': float(row.get('涨跌幅', 0) or 0),
                        'amplitude': float(row.get('振幅', 0) or 0),
                        'turnover': float(row.get('换手率', 0) or 0),
                    }
                    
                    if existing:
                        # 更新现有记录（只更新 OHLCV，不覆盖技术指标）
                        for key in ['open', 'close', 'high', 'low', 'volume', 'amount', 'change_pct', 'amplitude', 'turnover']:
                            setattr(existing, key, data[key])
                    else:
                        session.add(IndexDaily(date=target_date, **data))
                    
                    saved_count += 1
                
                session.commit()
                logger.info(f"批量保存指数历史数据成功: {code}, {saved_count} 条")
                return saved_count
                
            except Exception as e:
                session.rollback()
                logger.error(f"批量保存指数历史数据失败: {e}")
                return 0
    
    def has_index_data(self, code: str, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(
                select(IndexDaily).where(and_(IndexDaily.date == target_date, IndexDaily.code == code))
            ).scalar_one_or_none() is not None
    
    # ========== 新增：情绪周期数据存取 ==========
    
    def save_sentiment_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        """保存情绪周期日度数据"""
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(SentimentDaily).where(SentimentDaily.date == target_date)).scalar_one_or_none()
                if existing:
                    for key, value in data.items():
                        if hasattr(existing, key) and key != 'date':
                            setattr(existing, key, value)
                else:
                    session.add(SentimentDaily(date=target_date, **data))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存情绪数据失败: {e}")
                return False
    
    def get_sentiment_history(self, days: int = 20) -> List[Dict[str, Any]]:
        """获取情绪周期历史数据"""
        with self.db.get_session() as session:
            results = session.execute(
                select(SentimentDaily).order_by(desc(SentimentDaily.date)).limit(days)
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def has_sentiment_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(SentimentDaily).where(SentimentDaily.date == target_date)).scalar_one_or_none() is not None
    
    # ========== 新增：板块轮动数据存取 ==========
    
    def save_sector_rotation_daily(self, data_list: List[Dict[str, Any]], target_date: Optional[date] = None) -> int:
        """保存板块轮动日度数据"""
        if target_date is None:
            target_date = date.today()
        saved_count = 0
        with self.db.get_session() as session:
            try:
                for data in data_list:
                    sector_name = data.get('sector_name')
                    if not sector_name:
                        continue
                    existing = session.execute(
                        select(SectorRotationDaily).where(
                            and_(SectorRotationDaily.date == target_date, SectorRotationDaily.sector_name == sector_name)
                        )
                    ).scalar_one_or_none()
                    if existing:
                        for key, value in data.items():
                            if hasattr(existing, key) and key not in ('date', 'sector_name'):
                                setattr(existing, key, value)
                    else:
                        session.add(SectorRotationDaily(date=target_date, **data))
                        saved_count += 1
                session.commit()
                logger.info(f"保存板块轮动数据成功: {target_date}, {saved_count} 条")
            except Exception as e:
                session.rollback()
                logger.error(f"保存板块轮动数据失败: {e}")
        return saved_count
    
    def get_sector_rotation_history(self, sector_name: str, days: int = 20) -> List[Dict[str, Any]]:
        """获取板块轮动历史数据"""
        with self.db.get_session() as session:
            results = session.execute(
                select(SectorRotationDaily).where(SectorRotationDaily.sector_name == sector_name)
                .order_by(desc(SectorRotationDaily.date)).limit(days)
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def get_sector_rotation_daily(self, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """获取指定日期的板块轮动数据"""
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            results = session.execute(
                select(SectorRotationDaily).where(SectorRotationDaily.date == target_date)
                .order_by(desc(SectorRotationDaily.change_pct))
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def has_sector_rotation_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(
                select(SectorRotationDaily).where(SectorRotationDaily.date == target_date).limit(1)
            ).scalar_one_or_none() is not None
    
    # ========== 新增：板块轮动原始数据获取 ==========
    
    def get_sector_rotation_raw_data(self, days: int = 10) -> Dict[str, Any]:
        """
        获取板块轮动原始数据，供 LLM 分析
        
        不做规则判断，只提供原始数据：
        1. 每个板块近N天的涨跌幅序列
        2. 每个板块近N天的资金流向序列
        3. 板块排名变化
        
        Returns:
            原始数据字典，供 LLM 分析判断
        """
        result = {
            'has_data': False,
            'analysis_days': 0,
            'dates': [],
            'sector_history': {},  # {板块名: [{date, change_pct, rank, net_inflow}, ...]}
            'daily_rankings': {},  # {日期: [排名列表]}
        }
        
        with self.db.get_session() as session:
            # 获取最近N天有数据的日期
            dates = session.execute(
                select(SectorRotationDaily.date).distinct()
                .order_by(desc(SectorRotationDaily.date)).limit(days)
            ).scalars().all()
            
            if len(dates) < 2:
                return result
            
            result['has_data'] = True
            result['analysis_days'] = len(dates)
            result['dates'] = [d.isoformat() for d in sorted(dates, reverse=True)]
            
            # 获取所有板块数据
            for d in dates:
                day_data = session.execute(
                    select(SectorRotationDaily).where(SectorRotationDaily.date == d)
                    .order_by(desc(SectorRotationDaily.change_pct))
                ).scalars().all()
                
                # 记录当日排名
                result['daily_rankings'][d.isoformat()] = [
                    {'name': r.sector_name, 'change': r.change_pct, 'rank': i+1}
                    for i, r in enumerate(day_data)
                ]
                
                # 记录每个板块的历史
                for rank, r in enumerate(day_data, 1):
                    name = r.sector_name
                    if name not in result['sector_history']:
                        result['sector_history'][name] = []
                    
                    result['sector_history'][name].append({
                        'date': d.isoformat(),
                        'change_pct': r.change_pct,
                        'rank': rank,
                        'net_inflow': r.net_inflow,
                        'change_5d': r.change_5d,
                        'change_10d': r.change_10d,
                    })
            
            return result
    
    def get_sector_sustainability_data(self, days: int = 10) -> Dict[str, Any]:
        """
        获取板块持续性判断的原始数据
        
        核心维度：
        1. 板块涨跌序列 + 排名变化
        2. 资金流向序列（净流入连续性）
        3. 成交额变化（放量/缩量）
        4. 板块内涨停股数量变化
        
        Returns:
            供 LLM 判断板块持续性的原始数据
        """
        result = {
            'has_data': False,
            'analysis_days': 0,
            'dates': [],
            'sectors': {},  # {板块名: 详细数据}
        }
        
        with self.db.get_session() as session:
            # 获取最近N天有数据的日期
            dates = session.execute(
                select(SectorRotationDaily.date).distinct()
                .order_by(desc(SectorRotationDaily.date)).limit(days)
            ).scalars().all()
            
            if len(dates) < 2:
                return result
            
            result['has_data'] = True
            result['analysis_days'] = len(dates)
            result['dates'] = [d.isoformat() for d in sorted(dates, reverse=True)]
            
            # 收集每个板块的详细数据
            all_sectors = set()
            daily_data = {}
            
            for d in dates:
                day_data = session.execute(
                    select(SectorRotationDaily).where(SectorRotationDaily.date == d)
                    .order_by(desc(SectorRotationDaily.change_pct))
                ).scalars().all()
                
                daily_data[d] = {}
                for rank, r in enumerate(day_data, 1):
                    all_sectors.add(r.sector_name)
                    daily_data[d][r.sector_name] = {
                        'change_pct': r.change_pct or 0,
                        'rank': rank,
                        'net_inflow': r.net_inflow or 0,
                        'amount': r.amount or 0,
                        'amount_ratio': r.amount_ratio or 0,
                    }
            
            # 分析每个板块的持续性特征
            sorted_dates = sorted(dates, reverse=True)
            
            for sector_name in all_sectors:
                sector_data = {
                    'name': sector_name,
                    'daily_changes': [],      # 每日涨跌幅序列
                    'daily_ranks': [],        # 每日排名序列
                    'daily_inflows': [],      # 每日资金流入序列
                    'daily_amounts': [],      # 每日成交额序列
                    
                    # 统计指标（原始数据，不做判断）
                    'up_days': 0,             # 上涨天数
                    'down_days': 0,           # 下跌天数
                    'inflow_days': 0,         # 资金流入天数
                    'outflow_days': 0,        # 资金流出天数
                    'total_change': 0,        # 累计涨跌幅
                    'total_inflow': 0,        # 累计资金流入
                    'avg_rank': 0,            # 平均排名
                    'rank_trend': '',         # 排名趋势（原始数据）
                    'best_rank': 999,         # 最好排名
                    'worst_rank': 0,          # 最差排名
                }
                
                for d in sorted_dates:
                    if sector_name in daily_data[d]:
                        data = daily_data[d][sector_name]
                        change = data['change_pct']
                        rank = data['rank']
                        inflow = data['net_inflow']
                        amount = data['amount']
                        
                        sector_data['daily_changes'].append(change)
                        sector_data['daily_ranks'].append(rank)
                        sector_data['daily_inflows'].append(inflow)
                        sector_data['daily_amounts'].append(amount)
                        
                        sector_data['total_change'] += change
                        sector_data['total_inflow'] += inflow
                        
                        if change > 0:
                            sector_data['up_days'] += 1
                        elif change < 0:
                            sector_data['down_days'] += 1
                        
                        if inflow > 0:
                            sector_data['inflow_days'] += 1
                        elif inflow < 0:
                            sector_data['outflow_days'] += 1
                        
                        sector_data['best_rank'] = min(sector_data['best_rank'], rank)
                        sector_data['worst_rank'] = max(sector_data['worst_rank'], rank)
                
                # 计算平均排名
                if sector_data['daily_ranks']:
                    sector_data['avg_rank'] = sum(sector_data['daily_ranks']) / len(sector_data['daily_ranks'])
                    
                    # 排名趋势（前半段 vs 后半段）
                    mid = len(sector_data['daily_ranks']) // 2
                    if mid > 0:
                        recent_avg = sum(sector_data['daily_ranks'][:mid]) / mid
                        older_avg = sum(sector_data['daily_ranks'][mid:]) / (len(sector_data['daily_ranks']) - mid)
                        sector_data['rank_trend'] = f"近期平均{recent_avg:.0f}名 vs 早期平均{older_avg:.0f}名"
                
                result['sectors'][sector_name] = sector_data
            
            return result
    
    def analyze_sector_rotation_pattern(self, days: int = 10) -> Dict[str, Any]:
        """
        分析板块轮动特征，挖掘"一日游"等规律
        
        核心分析维度：
        1. 一日游特征：昨日领涨今日领跌的板块
        2. 持续性特征：连续上涨/下跌的板块
        3. 反转信号：连续下跌后开始反弹的板块
        4. 高低切换：资金从高位板块流向低位板块
        
        Returns:
            包含各种轮动特征的分析结果
        """
        result = {
            'has_data': False,
            'analysis_days': 0,
            'one_day_wonder': [],      # 一日游板块（昨日领涨今日领跌）
            'one_day_crash': [],       # 一日崩板块（昨日领跌今日领涨）
            'continuous_hot': [],      # 持续热门（连涨>=3天）
            'continuous_cold': [],     # 持续冷门（连跌>=3天）
            'reversal_up': [],         # 反转向上（连跌后反弹）
            'reversal_down': [],       # 反转向下（连涨后回调）
            'high_low_switch': {},     # 高低切换分析
            'rotation_pattern': '',    # 轮动模式判断
            'rotation_speed': '',      # 轮动速度
            'sector_correlation': [],  # 板块相关性
        }
        
        with self.db.get_session() as session:
            # 获取最近N天有数据的日期
            dates = session.execute(
                select(SectorRotationDaily.date).distinct()
                .order_by(desc(SectorRotationDaily.date)).limit(days)
            ).scalars().all()
            
            if len(dates) < 2:
                return result
            
            result['has_data'] = True
            result['analysis_days'] = len(dates)
            
            # 获取所有板块数据
            all_data = {}
            for d in dates:
                day_data = session.execute(
                    select(SectorRotationDaily).where(SectorRotationDaily.date == d)
                    .order_by(desc(SectorRotationDaily.change_pct))
                ).scalars().all()
                all_data[d] = {r.sector_name: r for r in day_data}
            
            # 按日期排序（最新在前）
            sorted_dates = sorted(dates, reverse=True)
            today = sorted_dates[0]
            yesterday = sorted_dates[1] if len(sorted_dates) > 1 else None
            
            if not yesterday:
                return result
            
            today_data = all_data.get(today, {})
            yesterday_data = all_data.get(yesterday, {})
            
            # 1. 分析一日游特征
            result['one_day_wonder'], result['one_day_crash'] = self._analyze_one_day_pattern(
                today_data, yesterday_data
            )
            
            # 2. 分析持续性特征
            result['continuous_hot'], result['continuous_cold'] = self._analyze_continuous_pattern(
                all_data, sorted_dates
            )
            
            # 3. 分析反转信号
            result['reversal_up'], result['reversal_down'] = self._analyze_reversal_pattern(
                all_data, sorted_dates
            )
            
            # 4. 分析高低切换
            result['high_low_switch'] = self._analyze_high_low_switch(
                all_data, sorted_dates
            )
            
            # 5. 判断轮动模式和速度
            result['rotation_pattern'] = self._judge_rotation_pattern(result)
            result['rotation_speed'] = self._judge_rotation_speed(result)
            
            return result
    
    def _analyze_one_day_pattern(self, today_data: Dict, yesterday_data: Dict) -> tuple:
        """分析一日游特征"""
        one_day_wonder = []  # 昨日领涨今日领跌
        one_day_crash = []   # 昨日领跌今日领涨
        
        # 获取昨日涨幅前10和跌幅前10
        yesterday_sorted = sorted(yesterday_data.values(), 
                                 key=lambda x: x.change_pct or 0, reverse=True)
        yesterday_top10 = yesterday_sorted[:10]
        yesterday_bottom10 = yesterday_sorted[-10:] if len(yesterday_sorted) >= 10 else []
        
        # 检查昨日领涨板块今日表现
        for sector in yesterday_top10:
            name = sector.sector_name
            if name in today_data:
                today_change = today_data[name].change_pct or 0
                yesterday_change = sector.change_pct or 0
                
                # 昨日涨幅>1%，今日跌幅>0.5%，视为一日游
                if yesterday_change > 1 and today_change < -0.5:
                    one_day_wonder.append({
                        'sector_name': name,
                        'yesterday_change': yesterday_change,
                        'today_change': today_change,
                        'reversal_magnitude': yesterday_change - today_change,
                        'yesterday_rank': yesterday_sorted.index(sector) + 1,
                    })
        
        # 检查昨日领跌板块今日表现
        for sector in yesterday_bottom10:
            name = sector.sector_name
            if name in today_data:
                today_change = today_data[name].change_pct or 0
                yesterday_change = sector.change_pct or 0
                
                # 昨日跌幅>1%，今日涨幅>0.5%，视为反弹
                if yesterday_change < -1 and today_change > 0.5:
                    one_day_crash.append({
                        'sector_name': name,
                        'yesterday_change': yesterday_change,
                        'today_change': today_change,
                        'reversal_magnitude': today_change - yesterday_change,
                    })
        
        # 按反转幅度排序
        one_day_wonder.sort(key=lambda x: x['reversal_magnitude'], reverse=True)
        one_day_crash.sort(key=lambda x: x['reversal_magnitude'], reverse=True)
        
        return one_day_wonder, one_day_crash
    
    def _analyze_continuous_pattern(self, all_data: Dict, sorted_dates: List) -> tuple:
        """分析持续性特征"""
        continuous_hot = []
        continuous_cold = []
        
        # 统计每个板块的连续涨跌天数
        sector_stats = {}
        
        for d in sorted_dates:
            day_data = all_data.get(d, {})
            for name, sector in day_data.items():
                if name not in sector_stats:
                    sector_stats[name] = {
                        'up_days': 0,
                        'down_days': 0,
                        'total_change': 0,
                        'changes': [],
                    }
                
                change = sector.change_pct or 0
                sector_stats[name]['changes'].append(change)
                sector_stats[name]['total_change'] += change
        
        # 计算连续涨跌天数
        for name, stats in sector_stats.items():
            changes = stats['changes']
            
            # 计算连涨天数（从今天往前数）
            up_days = 0
            for c in changes:
                if c > 0:
                    up_days += 1
                else:
                    break
            
            # 计算连跌天数
            down_days = 0
            for c in changes:
                if c < 0:
                    down_days += 1
                else:
                    break
            
            stats['up_days'] = up_days
            stats['down_days'] = down_days
            
            if up_days >= 3:
                continuous_hot.append({
                    'sector_name': name,
                    'continuous_days': up_days,
                    'total_change': stats['total_change'],
                    'avg_daily_change': stats['total_change'] / len(changes) if changes else 0,
                })
            
            if down_days >= 3:
                continuous_cold.append({
                    'sector_name': name,
                    'continuous_days': down_days,
                    'total_change': stats['total_change'],
                    'avg_daily_change': stats['total_change'] / len(changes) if changes else 0,
                })
        
        continuous_hot.sort(key=lambda x: x['continuous_days'], reverse=True)
        continuous_cold.sort(key=lambda x: x['continuous_days'], reverse=True)
        
        return continuous_hot[:10], continuous_cold[:10]
    
    def _analyze_reversal_pattern(self, all_data: Dict, sorted_dates: List) -> tuple:
        """分析反转信号"""
        reversal_up = []
        reversal_down = []
        
        if len(sorted_dates) < 3:
            return reversal_up, reversal_down
        
        today = sorted_dates[0]
        today_data = all_data.get(today, {})
        
        # 统计每个板块近期走势
        for name, today_sector in today_data.items():
            today_change = today_sector.change_pct or 0
            
            # 收集历史涨跌
            history_changes = []
            for d in sorted_dates[1:6]:  # 前5天
                if d in all_data and name in all_data[d]:
                    history_changes.append(all_data[d][name].change_pct or 0)
            
            if len(history_changes) < 3:
                continue
            
            # 检查反转向上：前3天都跌，今天涨
            if all(c < 0 for c in history_changes[:3]) and today_change > 0.5:
                reversal_up.append({
                    'sector_name': name,
                    'today_change': today_change,
                    'prev_3d_change': sum(history_changes[:3]),
                    'reversal_strength': today_change - sum(history_changes[:3]) / 3,
                })
            
            # 检查反转向下：前3天都涨，今天跌
            if all(c > 0 for c in history_changes[:3]) and today_change < -0.5:
                reversal_down.append({
                    'sector_name': name,
                    'today_change': today_change,
                    'prev_3d_change': sum(history_changes[:3]),
                    'reversal_strength': sum(history_changes[:3]) / 3 - today_change,
                })
        
        reversal_up.sort(key=lambda x: x['reversal_strength'], reverse=True)
        reversal_down.sort(key=lambda x: x['reversal_strength'], reverse=True)
        
        return reversal_up[:5], reversal_down[:5]
    
    def _analyze_high_low_switch(self, all_data: Dict, sorted_dates: List) -> Dict:
        """分析高低切换"""
        if len(sorted_dates) < 5:
            return {}
        
        # 计算每个板块的5日累计涨幅
        sector_5d_change = {}
        for name in all_data.get(sorted_dates[0], {}).keys():
            total = 0
            for d in sorted_dates[:5]:
                if d in all_data and name in all_data[d]:
                    total += all_data[d][name].change_pct or 0
            sector_5d_change[name] = total
        
        # 按5日涨幅排序
        sorted_sectors = sorted(sector_5d_change.items(), key=lambda x: x[1], reverse=True)
        
        # 高位板块（5日涨幅前10）
        high_sectors = sorted_sectors[:10]
        # 低位板块（5日涨幅后10）
        low_sectors = sorted_sectors[-10:]
        
        # 检查今日资金流向
        today = sorted_dates[0]
        today_data = all_data.get(today, {})
        
        high_today_change = sum(today_data[name].change_pct or 0 
                               for name, _ in high_sectors if name in today_data) / len(high_sectors)
        low_today_change = sum(today_data[name].change_pct or 0 
                              for name, _ in low_sectors if name in today_data) / len(low_sectors)
        
        return {
            'high_sectors': [{'name': n, 'change_5d': c} for n, c in high_sectors],
            'low_sectors': [{'name': n, 'change_5d': c} for n, c in low_sectors],
            'high_today_avg': high_today_change,
            'low_today_avg': low_today_change,
            'switch_signal': low_today_change > high_today_change,  # 低位跑赢高位=高低切换
            'switch_magnitude': low_today_change - high_today_change,
        }
    
    def _judge_rotation_pattern(self, analysis: Dict) -> str:
        """判断轮动模式"""
        one_day_count = len(analysis.get('one_day_wonder', []))
        continuous_hot_count = len(analysis.get('continuous_hot', []))
        switch_signal = analysis.get('high_low_switch', {}).get('switch_signal', False)
        
        if one_day_count >= 5:
            return '快速轮动（一日游频繁）'
        elif continuous_hot_count >= 3:
            return '主线明确（持续热点多）'
        elif switch_signal:
            return '高低切换（资金换仓）'
        else:
            return '震荡分化（无明显规律）'
    
    def _judge_rotation_speed(self, analysis: Dict) -> str:
        """判断轮动速度"""
        one_day_count = len(analysis.get('one_day_wonder', []))
        reversal_count = len(analysis.get('reversal_up', [])) + len(analysis.get('reversal_down', []))
        
        if one_day_count >= 5 or reversal_count >= 5:
            return '极快（日内切换频繁，不宜追高）'
        elif one_day_count >= 3 or reversal_count >= 3:
            return '较快（2-3日切换，注意止盈）'
        else:
            return '正常（可适当持有）'
    
    # ========== 新增：资金面数据存取 ==========
    
    def save_capital_flow_daily(self, data: Dict[str, Any], target_date: Optional[date] = None) -> bool:
        """保存资金面日度数据"""
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            try:
                existing = session.execute(select(CapitalFlowDaily).where(CapitalFlowDaily.date == target_date)).scalar_one_or_none()
                if existing:
                    for key, value in data.items():
                        if hasattr(existing, key) and key != 'date':
                            setattr(existing, key, value)
                else:
                    session.add(CapitalFlowDaily(date=target_date, **data))
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"保存资金面数据失败: {e}")
                return False
    
    def get_capital_flow_history(self, days: int = 20) -> List[Dict[str, Any]]:
        """获取资金面历史数据"""
        with self.db.get_session() as session:
            results = session.execute(
                select(CapitalFlowDaily).order_by(desc(CapitalFlowDaily.date)).limit(days)
            ).scalars().all()
            return [r.to_dict() for r in results]
    
    def has_capital_flow_data(self, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            return session.execute(select(CapitalFlowDaily).where(CapitalFlowDaily.date == target_date)).scalar_one_or_none() is not None
    
    def get_capital_flow_daily(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """获取指定日期的资金面数据"""
        if target_date is None:
            target_date = date.today()
        with self.db.get_session() as session:
            result = session.execute(select(CapitalFlowDaily).where(CapitalFlowDaily.date == target_date)).scalar_one_or_none()
            return result.to_dict() if result else None
    
    # ========== 新增：增强版历史对比上下文 ==========
    
    def get_enhanced_historical_context(self, days: int = 10) -> Dict[str, Any]:
        """
        获取增强版历史对比上下文
        
        整合指数技术面、情绪周期、板块轮动、资金面四个维度
        """
        context = {
            'has_history': False,
            'index_technical': {},
            'sentiment_cycle': {},
            'sector_rotation': {},
            'capital_flow': {},
        }
        
        # 1. 指数技术面
        for code in ['000001', '399001', '399006']:  # 上证、深证、创业板
            index_history = self.get_index_history(code, days)
            if index_history:
                context['index_technical'][code] = {
                    'history': index_history,
                    'trend': self._analyze_index_trend(index_history),
                }
                context['has_history'] = True
        
        # 2. 情绪周期
        sentiment_history = self.get_sentiment_history(days)
        if sentiment_history:
            context['sentiment_cycle'] = {
                'history': sentiment_history,
                'trend': self._analyze_sentiment_trend(sentiment_history),
            }
            context['has_history'] = True
        
        # 3. 板块轮动（获取最近几天的数据）
        rotation_data = []
        for i in range(min(days, 5)):
            check_date = date.today() - timedelta(days=i)
            daily_rotation = self.get_sector_rotation_daily(check_date)
            if daily_rotation:
                rotation_data.append({'date': check_date.isoformat(), 'sectors': daily_rotation})
        if rotation_data:
            context['sector_rotation'] = {
                'history': rotation_data,
                'hot_sectors': self._find_hot_sectors(rotation_data),
                'cold_sectors': self._find_cold_sectors(rotation_data),
            }
            context['has_history'] = True
        
        # 4. 资金面
        capital_history = self.get_capital_flow_history(days)
        if capital_history:
            context['capital_flow'] = {
                'history': capital_history,
                'trend': self._analyze_capital_trend(capital_history),
            }
            context['has_history'] = True
        
        return context
    
    def _analyze_index_trend(self, history: List[Dict]) -> Dict[str, Any]:
        """分析指数趋势"""
        if len(history) < 2:
            return {'status': '数据不足'}
        
        today = history[0]
        
        # 均线位置判断
        close = today.get('close', 0)
        ma5 = today.get('ma5', 0)
        ma10 = today.get('ma10', 0)
        ma20 = today.get('ma20', 0)
        ma60 = today.get('ma60', 0)
        
        ma_status = '震荡'
        if close and ma5 and ma10 and ma20:
            if close > ma5 > ma10 > ma20:
                ma_status = '多头排列'
            elif close < ma5 < ma10 < ma20:
                ma_status = '空头排列'
            elif close > ma5 and close > ma10:
                ma_status = '短期偏强'
            elif close < ma5 and close < ma10:
                ma_status = '短期偏弱'
        
        # MACD判断
        macd = today.get('macd', 0)
        macd_hist = today.get('macd_hist', 0)
        macd_status = '中性'
        if macd and macd_hist:
            if macd > 0 and macd_hist > 0:
                macd_status = '多头强势'
            elif macd < 0 and macd_hist < 0:
                macd_status = '空头强势'
            elif macd < 0 and macd_hist > 0:
                macd_status = '底部背离'
            elif macd > 0 and macd_hist < 0:
                macd_status = '顶部背离'
        
        # RSI判断
        rsi = today.get('rsi_6', 50)
        rsi_status = '中性'
        if rsi:
            if rsi > 80:
                rsi_status = '超买'
            elif rsi > 60:
                rsi_status = '偏强'
            elif rsi < 20:
                rsi_status = '超卖'
            elif rsi < 40:
                rsi_status = '偏弱'
        
        # 连续涨跌天数
        up_days = 0
        down_days = 0
        for i in range(len(history) - 1):
            change = history[i].get('change_pct', 0)
            if change and change > 0:
                up_days += 1
            elif change and change < 0:
                down_days += 1
            else:
                break
        
        return {
            'ma_status': ma_status,
            'macd_status': macd_status,
            'rsi_status': rsi_status,
            'rsi_value': rsi,
            'continuous_up': up_days if up_days > 0 else 0,
            'continuous_down': down_days if down_days > 0 else 0,
            'close': close,
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
        }
    
    def _analyze_sentiment_trend(self, history: List[Dict]) -> Dict[str, Any]:
        """分析情绪周期趋势"""
        if len(history) < 2:
            return {'status': '数据不足'}
        
        today = history[0]
        yesterday = history[1] if len(history) > 1 else {}
        
        # 溢价率趋势
        premiums = [h.get('zt_premium', 0) for h in history if h.get('zt_premium') is not None]
        avg_premium = sum(premiums) / len(premiums) if premiums else 0
        
        # 炸板率趋势
        zb_rates = [h.get('zb_rate', 0) for h in history if h.get('zb_rate') is not None]
        avg_zb_rate = sum(zb_rates) / len(zb_rates) if zb_rates else 0
        
        # 连板高度趋势
        max_heights = [h.get('max_continuous', 0) for h in history if h.get('max_continuous') is not None]
        avg_height = sum(max_heights) / len(max_heights) if max_heights else 0
        
        # 情绪得分趋势
        scores = [h.get('sentiment_score', 50) for h in history if h.get('sentiment_score') is not None]
        
        # 判断情绪周期位置
        current_score = today.get('sentiment_score', 50)
        cycle_position = '中性'
        if current_score:
            if current_score >= 80:
                cycle_position = '亢奋期'
            elif current_score >= 60:
                cycle_position = '活跃期'
            elif current_score >= 40:
                cycle_position = '中性期'
            elif current_score >= 20:
                cycle_position = '低迷期'
            else:
                cycle_position = '冰点期'
        
        # 趋势方向
        trend_direction = '震荡'
        if len(scores) >= 3:
            recent_avg = sum(scores[:3]) / 3
            older_avg = sum(scores[3:6]) / 3 if len(scores) >= 6 else recent_avg
            if recent_avg > older_avg * 1.1:
                trend_direction = '回暖'
            elif recent_avg < older_avg * 0.9:
                trend_direction = '降温'
        
        return {
            'cycle_position': cycle_position,
            'trend_direction': trend_direction,
            'current_score': current_score,
            'avg_premium': avg_premium,
            'today_premium': today.get('zt_premium'),
            'avg_zb_rate': avg_zb_rate,
            'today_zb_rate': today.get('zb_rate'),
            'avg_height': avg_height,
            'today_height': today.get('max_continuous'),
        }
    
    def _find_hot_sectors(self, rotation_data: List[Dict]) -> List[Dict]:
        """找出热门板块（连续上涨、资金流入）"""
        if not rotation_data:
            return []
        
        # 统计各板块表现
        sector_stats = {}
        for day_data in rotation_data:
            for sector in day_data.get('sectors', []):
                name = sector.get('sector_name')
                if not name:
                    continue
                if name not in sector_stats:
                    sector_stats[name] = {'up_days': 0, 'total_change': 0, 'total_inflow': 0}
                change = sector.get('change_pct', 0) or 0
                inflow = sector.get('net_inflow', 0) or 0
                if change > 0:
                    sector_stats[name]['up_days'] += 1
                sector_stats[name]['total_change'] += change
                sector_stats[name]['total_inflow'] += inflow
        
        # 筛选热门板块
        hot_sectors = []
        for name, stats in sector_stats.items():
            if stats['up_days'] >= 3 or stats['total_change'] > 5:
                hot_sectors.append({
                    'name': name,
                    'up_days': stats['up_days'],
                    'total_change': round(stats['total_change'], 2),
                    'total_inflow': round(stats['total_inflow'], 2),
                })
        
        hot_sectors.sort(key=lambda x: x['total_change'], reverse=True)
        return hot_sectors[:10]
    
    def _find_cold_sectors(self, rotation_data: List[Dict]) -> List[Dict]:
        """找出冷门板块（连续下跌、资金流出）"""
        if not rotation_data:
            return []
        
        sector_stats = {}
        for day_data in rotation_data:
            for sector in day_data.get('sectors', []):
                name = sector.get('sector_name')
                if not name:
                    continue
                if name not in sector_stats:
                    sector_stats[name] = {'down_days': 0, 'total_change': 0, 'total_outflow': 0}
                change = sector.get('change_pct', 0) or 0
                inflow = sector.get('net_inflow', 0) or 0
                if change < 0:
                    sector_stats[name]['down_days'] += 1
                sector_stats[name]['total_change'] += change
                sector_stats[name]['total_outflow'] += abs(min(0, inflow))
        
        cold_sectors = []
        for name, stats in sector_stats.items():
            if stats['down_days'] >= 3 or stats['total_change'] < -5:
                cold_sectors.append({
                    'name': name,
                    'down_days': stats['down_days'],
                    'total_change': round(stats['total_change'], 2),
                    'total_outflow': round(stats['total_outflow'], 2),
                })
        
        cold_sectors.sort(key=lambda x: x['total_change'])
        return cold_sectors[:10]
    
    def _analyze_capital_trend(self, history: List[Dict]) -> Dict[str, Any]:
        """分析资金面趋势"""
        if len(history) < 2:
            return {'status': '数据不足'}
        
        today = history[0]
        
        # 两融趋势
        margins = [h.get('margin_balance', 0) for h in history if h.get('margin_balance')]
        margin_trend = '稳定'
        if len(margins) >= 3:
            if margins[0] > margins[1] > margins[2]:
                margin_trend = '连续增加'
            elif margins[0] < margins[1] < margins[2]:
                margin_trend = '连续减少'
        
        # 机构动向
        org_buys = [h.get('lhb_org_net_buy', 0) for h in history if h.get('lhb_org_net_buy') is not None]
        org_trend = '中性'
        if org_buys:
            recent_sum = sum(org_buys[:5])
            if recent_sum > 10:
                org_trend = '机构积极买入'
            elif recent_sum < -10:
                org_trend = '机构持续卖出'
        
        # 成交额趋势
        amounts = [h.get('total_amount', 0) for h in history if h.get('total_amount')]
        amount_trend = '正常'
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            if amounts[0] > avg_amount * 1.3:
                amount_trend = '放量'
            elif amounts[0] < avg_amount * 0.7:
                amount_trend = '缩量'
        
        return {
            'margin_trend': margin_trend,
            'margin_balance': today.get('margin_balance'),
            'margin_change': today.get('margin_balance_change'),
            'org_trend': org_trend,
            'org_net_buy_5d': sum(org_buys[:5]) if org_buys else 0,
            'amount_trend': amount_trend,
            'today_amount': today.get('total_amount'),
        }


# === 新增：指数日度技术指标数据 ===

class IndexDaily(Base):
    """
    指数日度技术指标数据
    
    存储主要指数的日线数据和技术指标，用于趋势判断
    """
    __tablename__ = 'index_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    code = Column(String(20), nullable=False, index=True)  # sh000001, sz399001等
    name = Column(String(50))
    
    # OHLC数据
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    change_pct = Column(Float)
    amplitude = Column(Float)
    turnover = Column(Float)
    
    # 均线指标
    ma5 = Column(Float)
    ma10 = Column(Float)
    ma20 = Column(Float)
    ma60 = Column(Float)
    
    # 技术指标
    rsi_6 = Column(Float)
    rsi_12 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    
    # 布林带
    boll_upper = Column(Float)
    boll_mid = Column(Float)
    boll_lower = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('date', 'code', name='uix_index_date_code'),
        Index('ix_index_date_code', 'date', 'code'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'code': self.code,
            'name': self.name,
            'open': self.open,
            'close': self.close,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'change_pct': self.change_pct,
            'amplitude': self.amplitude,
            'turnover': self.turnover,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'rsi_6': self.rsi_6,
            'rsi_12': self.rsi_12,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_hist': self.macd_hist,
            'boll_upper': self.boll_upper,
            'boll_mid': self.boll_mid,
            'boll_lower': self.boll_lower,
        }


# === 新增：情绪周期日度数据 ===

class SentimentDaily(Base):
    """
    情绪周期日度数据
    
    存储市场情绪相关指标，用于情绪周期判断
    """
    __tablename__ = 'sentiment_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # 涨停溢价率（昨日涨停今日平均涨跌幅）
    zt_premium = Column(Float)
    zt_premium_up_count = Column(Integer)    # 昨日涨停今日上涨数
    zt_premium_down_count = Column(Integer)  # 昨日涨停今日下跌数
    
    # 炸板率
    zb_rate = Column(Float)                  # 炸板率 = 炸板数 / (涨停数 + 炸板数)
    zb_count = Column(Integer)               # 炸板数
    
    # 连板高度
    max_continuous = Column(Integer)         # 最高连板数
    continuous_count = Column(Integer)       # 连板股数量
    
    # 首板数据
    first_board_count = Column(Integer)      # 首板数量
    first_board_success_rate = Column(Float) # 首板封板成功率
    
    # 赚钱效应
    up_count = Column(Integer)               # 上涨家数
    down_count = Column(Integer)             # 下跌家数
    up_ratio = Column(Float)                 # 上涨比例
    limit_up_count = Column(Integer)         # 涨停数
    limit_down_count = Column(Integer)       # 跌停数
    
    # 情绪综合得分（0-100）
    sentiment_score = Column(Float)
    sentiment_level = Column(String(20))     # 冰点/低迷/中性/活跃/亢奋
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'zt_premium': self.zt_premium,
            'zt_premium_up_count': self.zt_premium_up_count,
            'zt_premium_down_count': self.zt_premium_down_count,
            'zb_rate': self.zb_rate,
            'zb_count': self.zb_count,
            'max_continuous': self.max_continuous,
            'continuous_count': self.continuous_count,
            'first_board_count': self.first_board_count,
            'first_board_success_rate': self.first_board_success_rate,
            'up_count': self.up_count,
            'down_count': self.down_count,
            'up_ratio': self.up_ratio,
            'limit_up_count': self.limit_up_count,
            'limit_down_count': self.limit_down_count,
            'sentiment_score': self.sentiment_score,
            'sentiment_level': self.sentiment_level,
        }


# === 新增：板块轮动日度数据 ===

class SectorRotationDaily(Base):
    """
    板块轮动日度数据
    
    存储板块排名变化和资金流向，用于板块轮动分析
    """
    __tablename__ = 'sector_rotation_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    sector_name = Column(String(50), nullable=False, index=True)
    
    # 涨跌数据
    change_pct = Column(Float)
    rank_today = Column(Integer)             # 今日排名
    rank_yesterday = Column(Integer)         # 昨日排名
    rank_change = Column(Integer)            # 排名变化（正数=上升）
    
    # 连续涨跌
    continuous_up_days = Column(Integer)     # 连续上涨天数
    continuous_down_days = Column(Integer)   # 连续下跌天数
    change_5d = Column(Float)                # 5日涨跌幅
    change_10d = Column(Float)               # 10日涨跌幅
    
    # 资金流向
    net_inflow = Column(Float)               # 今日主力净流入（亿元）
    net_inflow_5d = Column(Float)            # 5日主力净流入（亿元）
    net_inflow_10d = Column(Float)           # 10日主力净流入（亿元）
    
    # 成交数据
    amount = Column(Float)                   # 成交额（亿元）
    amount_ratio = Column(Float)             # 成交额相对5日均值比
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('date', 'sector_name', name='uix_rotation_date_sector'),
        Index('ix_rotation_date_sector', 'date', 'sector_name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'sector_name': self.sector_name,
            'change_pct': self.change_pct,
            'rank_today': self.rank_today,
            'rank_yesterday': self.rank_yesterday,
            'rank_change': self.rank_change,
            'continuous_up_days': self.continuous_up_days,
            'continuous_down_days': self.continuous_down_days,
            'change_5d': self.change_5d,
            'change_10d': self.change_10d,
            'net_inflow': self.net_inflow,
            'net_inflow_5d': self.net_inflow_5d,
            'net_inflow_10d': self.net_inflow_10d,
            'amount': self.amount,
            'amount_ratio': self.amount_ratio,
        }


# === 新增：资金面日度数据 ===

class CapitalFlowDaily(Base):
    """
    资金面日度数据
    
    存储市场资金流向汇总，用于资金面分析
    """
    __tablename__ = 'capital_flow_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # 两融数据
    margin_balance = Column(Float)           # 融资余额（亿元）
    margin_balance_change = Column(Float)    # 融资余额变化（亿元）
    margin_buy = Column(Float)               # 融资买入额（亿元）
    short_balance = Column(Float)            # 融券余额（亿元）
    
    # 龙虎榜数据
    lhb_net_buy = Column(Float)              # 龙虎榜净买入（亿元）
    lhb_org_net_buy = Column(Float)          # 机构净买入（亿元）
    lhb_org_buy_count = Column(Integer)      # 机构买入次数
    lhb_org_sell_count = Column(Integer)     # 机构卖出次数
    
    # 大宗交易
    block_trade_amount = Column(Float)       # 大宗交易成交额（亿元）
    block_trade_premium_ratio = Column(Float)  # 溢价成交占比
    block_trade_discount_ratio = Column(Float) # 折价成交占比
    
    # 市场资金流向
    market_net_inflow = Column(Float)        # 市场主力净流入（亿元）
    super_large_net = Column(Float)          # 超大单净流入（亿元）
    large_net = Column(Float)                # 大单净流入（亿元）
    medium_net = Column(Float)               # 中单净流入（亿元）
    small_net = Column(Float)                # 小单净流入（亿元）
    
    # 成交额
    total_amount = Column(Float)             # 两市成交额（亿元）
    amount_change_pct = Column(Float)        # 成交额环比变化
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.isoformat() if self.date else None,
            'margin_balance': self.margin_balance,
            'margin_balance_change': self.margin_balance_change,
            'margin_buy': self.margin_buy,
            'short_balance': self.short_balance,
            'lhb_net_buy': self.lhb_net_buy,
            'lhb_org_net_buy': self.lhb_org_net_buy,
            'lhb_org_buy_count': self.lhb_org_buy_count,
            'lhb_org_sell_count': self.lhb_org_sell_count,
            'block_trade_amount': self.block_trade_amount,
            'block_trade_premium_ratio': self.block_trade_premium_ratio,
            'block_trade_discount_ratio': self.block_trade_discount_ratio,
            'market_net_inflow': self.market_net_inflow,
            'super_large_net': self.super_large_net,
            'large_net': self.large_net,
            'medium_net': self.medium_net,
            'small_net': self.small_net,
            'total_amount': self.total_amount,
            'amount_change_pct': self.amount_change_pct,
        }


def get_market_history_manager() -> MarketHistoryManager:
    """获取市场历史数据管理器实例"""
    return MarketHistoryManager()
