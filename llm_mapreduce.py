# -*- coding: utf-8 -*-
"""
===================================
LLM Map-Reduce 分析框架
===================================

核心思想：
- Map 阶段：多个专业分析师（LLM）从不同维度并行分析数据
- Reduce 阶段：首席分析师（LLM）综合各维度结论，生成深度报告

优势：
1. 每个维度的分析更专注、更深入
2. 避免单次 prompt 过长导致信息丢失
3. 可并行调用，提高效率
4. 各维度分析可独立迭代优化

使用方式：
    from llm_mapreduce import generate_market_review
    
    # 获取数据
    overview = market_analyzer.get_market_overview()
    news = market_analyzer.search_market_news()
    
    # 生成报告（需要传入 AI 分析器）
    report = generate_market_review(overview, news, analyzer)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class DimensionAnalysis:
    """单个维度的分析结果"""
    dimension: str              # 维度名称
    analyst_role: str           # 分析师角色
    analysis: str               # 完整分析文本（纯文本，供 Reduce 阶段使用）
    success: bool = True
    error: Optional[str] = None
    elapsed_time: float = 0.0   # 分析耗时（秒）


@dataclass
class MapReduceResult:
    """Map-Reduce 分析最终结果"""
    date: str
    dimension_analyses: Dict[str, DimensionAnalysis]  # 各维度分析结果
    synthesis: str              # 综合分析报告（最终输出）
    metadata: Dict[str, Any] = field(default_factory=dict)


class DimensionAnalyst(ABC):
    """
    维度分析师基类
    
    每个子类代表一个专业分析维度，直接输出专业的文本分析报告
    """
    
    def __init__(self, llm_caller: Callable[[str, dict], str]):
        """
        Args:
            llm_caller: LLM 调用函数，签名为 (prompt, config) -> response_text
        """
        self.llm_caller = llm_caller
    
    @property
    @abstractmethod
    def dimension_name(self) -> str:
        """维度名称"""
        pass
    
    @property
    @abstractmethod
    def analyst_role(self) -> str:
        """分析师角色描述"""
        pass
    
    @abstractmethod
    def build_prompt(self, data: Dict[str, Any]) -> str:
        """构建该维度的分析 prompt"""
        pass
    
    def get_generation_config(self) -> dict:
        """获取 LLM 生成配置"""
        return {
            'temperature': 0.7,
            'max_output_tokens': 65535,
        }
    
    def analyze(self, data: Dict[str, Any]) -> DimensionAnalysis:
        """执行分析，直接返回文本报告"""
        try:
            prompt = self.build_prompt(data)
            config = self.get_generation_config()
            
            logger.info(f"[Map] {self.dimension_name} 分析师开始分析...")
            start_time = time.time()
            
            response = self.llm_caller(prompt, config)
            
            elapsed = time.time() - start_time
            logger.info(f"[Map] {self.dimension_name} 分析完成，耗时 {elapsed:.1f}s，{len(response)} 字符")
            
            return DimensionAnalysis(
                dimension=self.dimension_name,
                analyst_role=self.analyst_role,
                analysis=response,
                success=True,
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.error(f"[Map] {self.dimension_name} 分析失败: {e}")
            return DimensionAnalysis(
                dimension=self.dimension_name,
                analyst_role=self.analyst_role,
                analysis=f"[分析失败: {str(e)[:100]}]",
                success=False,
                error=str(e)
            )


# ============================================================
# 具体维度分析师实现（纯文本输出）
# ============================================================

class EmotionAnalyst(DimensionAnalyst):
    """市场情绪分析师 - 专注涨停板、炸板率、溢价率等情绪指标"""
    
    @property
    def dimension_name(self) -> str:
        return "市场情绪"
    
    @property
    def analyst_role(self) -> str:
        return "游资情绪分析师，专注短线情绪指标，擅长判断市场赚钱效应"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # ========== 涨停股池详情 ==========
        zt_pool = overview.get('zt_pool', [])
        zt_pool_text = ""
        if zt_pool:
            zt_pool_text = "| 股票 | 连板 | 封板资金 | 炸板次数 | 行业 |\n|------|------|----------|----------|------|\n"
            for zt in zt_pool[:20]:
                zt_pool_text += f"| {zt.get('name', '')} | {zt.get('continuous', 1)}板 | {zt.get('seal_amount', 0):.1f}亿 | {zt.get('zb_count', 0)}次 | {zt.get('industry', '')} |\n"
        
        # ========== 昨日涨停今日表现 ==========
        previous_zt = overview.get('previous_zt_pool', [])
        previous_zt_text = ""
        if previous_zt:
            previous_zt_text = "| 股票 | 今日涨跌 | 昨日连板 | 行业 |\n|------|----------|----------|------|\n"
            for pzt in previous_zt[:20]:
                previous_zt_text += f"| {pzt.get('name', '')} | {pzt.get('change_pct', 0):+.2f}% | {pzt.get('yesterday_continuous', 1)}板 | {pzt.get('industry', '')} |\n"
        
        # ========== 炸板股池 ==========
        zb_pool = overview.get('zb_pool', [])
        zb_pool_text = ""
        if zb_pool:
            zb_pool_text = "| 股票 | 炸板次数 | 涨跌幅 | 行业 |\n|------|----------|--------|------|\n"
            for zb in zb_pool[:20]:
                zb_pool_text += f"| {zb.get('name', '')} | {zb.get('zb_count', 0)}次 | {zb.get('change_pct', 0):+.2f}% | {zb.get('industry', '')} |\n"
        
        # ========== 盘口异动详情 ==========
        pankou_changes = overview.get('pankou_changes', {})
        
        # 大笔买入
        big_buy_list = pankou_changes.get('大笔买入', []) if isinstance(pankou_changes, dict) else []
        big_buy_text = ""
        if big_buy_list:
            big_buy_text = "| 时间 | 股票 | 板块 | 详情 |\n|------|------|------|------|\n"
            for item in big_buy_list[:20]:
                big_buy_text += f"| {item.get('time', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:20]} |\n"
        
        # 大笔卖出
        big_sell_list = pankou_changes.get('大笔卖出', []) if isinstance(pankou_changes, dict) else []
        big_sell_text = ""
        if big_sell_list:
            big_sell_text = "| 时间 | 股票 | 板块 | 详情 |\n|------|------|------|------|\n"
            for item in big_sell_list[:20]:
                big_sell_text += f"| {item.get('time', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:20]} |\n"
        
        # 火箭发射
        rocket_list = pankou_changes.get('火箭发射', []) if isinstance(pankou_changes, dict) else []
        rocket_text = ""
        if rocket_list:
            rocket_text = "| 时间 | 股票 | 板块 | 详情 |\n|------|------|------|------|\n"
            for item in rocket_list[:20]:
                rocket_text += f"| {item.get('time', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:20]} |\n"
        
        # 高台跳水
        dive_list = pankou_changes.get('高台跳水', []) if isinstance(pankou_changes, dict) else []
        dive_text = ""
        if dive_list:
            dive_text = "| 时间 | 股票 | 板块 | 详情 |\n|------|------|------|------|\n"
            for item in dive_list[:20]:
                dive_text += f"| {item.get('time', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:20]} |\n"
        
        # ========== 历史对比数据 ==========
        hist_ctx = overview.get('historical_context', {})
        history_text = ""
        if hist_ctx.get('has_history'):
            history_text = "\n## 历史对比（时序分析）\n\n"
            
            # 涨停趋势
            if 'zt_trend' in hist_ctx:
                zt = hist_ctx['zt_trend']
                history_text += f"**涨停数量趋势**: 今日{zt['today']}只 vs 昨日{zt['yesterday']}只, "
                history_text += f"5日均值{zt['avg_5d']:.0f}只, {zt['trend']}\n"
                history_text += f"- 近5日数据: {zt.get('values', [])}\n\n"
            
            # 成交额趋势
            if 'amount_trend' in hist_ctx:
                amt = hist_ctx['amount_trend']
                history_text += f"**成交额趋势**: 今日{amt['today']:.0f}亿 vs 昨日{amt['yesterday']:.0f}亿, "
                history_text += f"5日均值{amt['avg_5d']:.0f}亿, {amt['trend']}\n"
                history_text += f"- 近5日数据: {[f'{v:.0f}' for v in amt.get('values', [])]}\n\n"
            
            # 连板趋势
            if 'continuous_trend' in hist_ctx:
                cont = hist_ctx['continuous_trend']
                history_text += f"**连板数量趋势**: 今日{cont['today']}只 vs 昨日{cont['yesterday']}只, "
                history_text += f"5日均值{cont['avg_5d']:.0f}只, {cont['trend']}\n\n"
            
            # 炸板趋势
            if 'zb_trend' in hist_ctx:
                zb = hist_ctx['zb_trend']
                history_text += f"**炸板数量趋势**: 今日{zb['today']}只 vs 昨日{zb['yesterday']}只, "
                history_text += f"5日均值{zb['avg_5d']:.0f}只, {zb['trend']}\n\n"
        
        return f"""你是一位资深的 A 股游资情绪分析师，拥有 10 年短线交易经验。请基于今日数据，撰写一份专业的市场情绪分析报告。

## 你的专业视角
- 涨停板生态是市场情绪的温度计
- 炸板率反映资金承接意愿
- 溢价率体现赚钱效应
- 连板高度代表情绪天花板
- 盘口异动揭示资金真实动向

## 今日核心数据

**涨停板生态**
- 涨停 {overview.get('limit_up_count', 0)} 只 / 跌停 {overview.get('limit_down_count', 0)} 只
- 首板 {overview.get('zt_first_board_count', 0)} 只 / 连板 {overview.get('zt_continuous_count', 0)} 只
- 最高连板 {overview.get('zt_max_continuous', 0)} 板
- 炸板 {overview.get('zb_pool_count', 0)} 只，炸板率 {overview.get('zb_rate', 0):.1f}%

**涨停股详情（按连板数排序）**
{zt_pool_text if zt_pool_text else "暂无数据"}

**赚钱效应（昨日涨停今日表现）**
- 昨日涨停 {overview.get('previous_zt_count', 0)} 只
- 今日平均溢价 {overview.get('previous_zt_avg_change', 0):+.2f}%
- 上涨 {overview.get('previous_zt_up_count', 0)} 只 / 下跌 {overview.get('previous_zt_down_count', 0)} 只

{previous_zt_text if previous_zt_text else ""}

**炸板股详情**
{zb_pool_text if zb_pool_text else "暂无炸板股"}

**市场活跃度**
- 上涨 {overview.get('up_count', 0)} 家 / 下跌 {overview.get('down_count', 0)} 家
- 两市成交 {overview.get('total_amount', 0):.0f} 亿
- 平均换手率 {overview.get('avg_turnover_rate', 0):.2f}%

## 盘口异动详情

**异动统计**
- 大笔买入 {overview.get('big_buy_count', 0)} 次 / 大笔卖出 {overview.get('big_sell_count', 0)} 次
- 火箭发射 {overview.get('rocket_launch_count', 0)} 次 / 高台跳水 {overview.get('high_dive_count', 0)} 次

**大笔买入（资金流入信号）**
{big_buy_text if big_buy_text else "暂无数据"}

**大笔卖出（资金流出信号）**
{big_sell_text if big_sell_text else "暂无数据"}

**火箭发射（快速拉升）**
{rocket_text if rocket_text else "暂无数据"}

**高台跳水（快速下杀）**
{dive_text if dive_text else "暂无数据"}
{history_text}
---

请撰写一份情绪分析报告，包含：

1. **情绪定性**：今日市场情绪处于什么状态？（冰点/低迷/温和/亢奋/过热）
2. **涨停板解读**：连板梯队健康吗？哪些板块涨停股集中？说明什么？
3. **赚钱效应**：打板族今天赚钱还是亏钱？接力是否安全？
4. **盘口异动解读**：大笔买入集中在哪些股票/板块？火箭发射和高台跳水说明什么？
5. **情绪趋势**：与近期相比，情绪在转暖还是转冷？有无拐点信号？
6. **风险提示**：情绪层面有哪些需要警惕的信号？
7. **明日预判**：基于今日情绪，明日大概率如何演绎？

请用专业但易懂的语言，直接输出分析报告。"""


class ThemeAnalyst(DimensionAnalyst):
    """主线题材分析师 - 专注板块轮动、题材持续性"""
    
    @property
    def dimension_name(self) -> str:
        return "主线题材"
    
    @property
    def analyst_role(self) -> str:
        return "题材挖掘分析师，专注板块轮动规律，擅长判断主线持续性"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # 格式化板块数据（增强：含净流入、领涨股）
        top_sectors = overview.get('top_sectors', [])
        top_sectors_text = ""
        if top_sectors:
            top_sectors_text = "| 板块 | 涨跌幅 | 净流入 | 涨/跌 | 领涨股 |\n"
            top_sectors_text += "|------|--------|--------|-------|--------|\n"
            for s in top_sectors[:20]:
                net_inflow = s.get('net_inflow', 0)
                up_count = s.get('up_count', 0)
                down_count = s.get('down_count', 0)
                leader = s.get('leader_stock', '')
                leader_change = s.get('leader_change', 0)
                leader_text = f"{leader}({leader_change:+.1f}%)" if leader else "-"
                top_sectors_text += f"| {s.get('name', '')} | {s.get('change_pct', 0):+.2f}% | {net_inflow:.2f}亿 | {up_count}/{down_count} | {leader_text} |\n"
        else:
            top_sectors_text = "暂无数据"
        
        bottom_sectors = overview.get('bottom_sectors', [])
        bottom_sectors_text = ""
        if bottom_sectors:
            bottom_sectors_text = "| 板块 | 涨跌幅 | 净流入 | 涨/跌 |\n"
            bottom_sectors_text += "|------|--------|--------|-------|\n"
            for s in bottom_sectors[:20]:
                net_inflow = s.get('net_inflow', 0)
                up_count = s.get('up_count', 0)
                down_count = s.get('down_count', 0)
                bottom_sectors_text += f"| {s.get('name', '')} | {s.get('change_pct', 0):+.2f}% | {net_inflow:.2f}亿 | {up_count}/{down_count} |\n"
        else:
            bottom_sectors_text = "暂无数据"
        
        top_concepts = overview.get('top_concepts', [])
        top_concepts_text = "\n".join([
            f"- {s.get('name', '')}: {s.get('change_pct', 0):+.2f}%"
            for s in top_concepts[:20]
        ]) or "暂无数据"
        
        # ========== 涨停股板块分布（增强：含具体股票名单）==========
        zt_pool = overview.get('zt_pool', [])
        zt_by_industry = {}  # {行业: [股票列表]}
        for zt in zt_pool:
            ind = zt.get('industry', '其他')
            if ind not in zt_by_industry:
                zt_by_industry[ind] = []
            zt_by_industry[ind].append({
                'name': zt.get('name', ''),
                'continuous': zt.get('continuous', 1),
                'seal_amount': zt.get('seal_amount', 0),
            })
        
        # 按涨停数量排序
        sorted_industries = sorted(zt_by_industry.items(), key=lambda x: -len(x[1]))[:20]
        
        zt_dist_text = ""
        for ind, stocks in sorted_industries:
            stock_names = ', '.join([f"{s['name']}({s['continuous']}板)" for s in stocks[:20]])
            if len(stocks) > 5:
                stock_names += f" 等{len(stocks)}只"
            zt_dist_text += f"- **{ind}** ({len(stocks)}只涨停): {stock_names}\n"
        zt_dist_text = zt_dist_text or "暂无数据"
        
        # ========== 板块异动详情（增强：含龙头股）==========
        board_changes = overview.get('board_changes', [])
        board_changes_text = ""
        if board_changes:
            board_changes_text = "| 板块 | 涨跌幅 | 主力净流入 | 异动次数 | 最活跃个股 |\n"
            board_changes_text += "|------|--------|------------|----------|------------|\n"
            for bc in board_changes[:20]:
                board_changes_text += (f"| {bc.get('name', '')} | {bc.get('change_pct', 0):+.2f}% | "
                                      f"{bc.get('main_net_inflow', 0):.2f}亿 | {bc.get('change_count', 0)}次 | "
                                      f"{bc.get('top_stock_name', '')} |\n")
        else:
            board_changes_text = "暂无异动数据"
        
        return f"""你是一位资深的 A 股题材分析师，拥有丰富的板块轮动研究经验。请基于今日数据，撰写一份专业的主线题材分析报告。

## 你的专业视角
- 主线是市场的灵魂，决定赚钱效应
- 涨停股分布揭示资金真实偏好
- 板块轮动规律决定操作节奏
- 题材持续性决定参与价值
- 净流入数据揭示资金真实动向

## 今日板块数据

**行业板块涨幅榜（含净流入、领涨股）**
{top_sectors_text}

**行业板块跌幅榜**
{bottom_sectors_text}

**概念板块涨幅榜**
{top_concepts_text}

**涨停股行业分布（含具体股票）**
{zt_dist_text}

**板块异动详情（含龙头股）**
{board_changes_text}

---

请撰写一份主线题材分析报告，包含：

1. **主线识别**：今日市场的核心主线是什么？有几条主线并行？哪些板块涨停股最集中？
2. **龙头股分析**：各主线的龙头股是谁？连板高度如何？龙头股的带动效应如何？
3. **资金验证**：主线板块的净流入情况如何？资金是否真正认可？
4. **主线逻辑**：主线背后的驱动逻辑是什么？（政策/业绩/事件/资金）
5. **持续性判断**：这条主线处于什么阶段？（启动/发酵/高潮/退潮）还能持续多久？
6. **轮动特征**：今日板块轮动呈现什么特点？（高低切换/强者恒强/普涨/分化）
7. **明日方向**：明日哪些板块值得重点关注？哪些需要回避？

请用专业但易懂的语言，直接输出分析报告。"""


class FundFlowAnalyst(DimensionAnalyst):
    """资金流向分析师 - 专注主力资金、融资融券、大宗交易"""
    
    @property
    def dimension_name(self) -> str:
        return "资金流向"
    
    @property
    def analyst_role(self) -> str:
        return "资金流向分析师，专注主力动向，擅长从资金角度判断市场方向"
    
    def _build_fund_history_text(self, overview: Dict[str, Any]) -> str:
        """构建资金历史趋势文本"""
        hist_ctx = overview.get('historical_context', {})
        if not hist_ctx.get('has_history'):
            return ""
        
        text = "\n## 历史对比（资金趋势）\n\n"
        
        # 两融余额趋势
        if 'margin_trend' in hist_ctx:
            margin = hist_ctx['margin_trend']
            change = margin.get('change', 0)
            change_str = f"+{change:.0f}" if change > 0 else f"{change:.0f}"
            text += f"**两融余额趋势**: 今日{margin['today']:.0f}亿, "
            text += f"较昨日{change_str}亿, {margin['trend']}\n\n"
        
        # 机构净买入趋势
        if 'org_buy_trend' in hist_ctx:
            org = hist_ctx['org_buy_trend']
            text += f"**机构净买入趋势**: 今日{org['today']:.2f}亿, "
            text += f"昨日{org['yesterday']:.2f}亿, 近5日累计{org['sum_5d']:.2f}亿\n"
            text += f"- {org['trend']}\n\n"
        
        return text
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # ========== 龙虎榜数据（增强：含详细席位信息）==========
        lhb_stocks = overview.get('lhb_stocks', [])
        lhb_text = ""
        if lhb_stocks:
            lhb_text = "| 股票 | 涨跌幅 | 净买入 | 上榜原因 |\n"
            lhb_text += "|------|--------|--------|----------|\n"
            for s in lhb_stocks[:20]:
                reason = s.get('reason', '')[:50]
                lhb_text += f"| {s.get('name', '')} | {s.get('change_pct', 0):+.2f}% | {s.get('net_buy', 0):.2f}亿 | {reason} |\n"
        else:
            lhb_text = "暂无龙虎榜数据"
        
        # ========== 龙虎榜席位详情（新增）==========
        lhb_seat_detail = overview.get('lhb_seat_detail', [])
        lhb_seat_text = ""
        if lhb_seat_detail:
            lhb_seat_text = "| 股票 | 席位类型 | 营业部/机构 | 买入金额 | 卖出金额 | 净额 |\n"
            lhb_seat_text += "|------|----------|-------------|----------|----------|------|\n"
            for seat in lhb_seat_detail[:20]:
                seat_type = "机构" if "机构" in seat.get('trader_name', '') else "游资"
                lhb_seat_text += (f"| {seat.get('stock_name', '')} | {seat_type} | "
                                 f"{seat.get('trader_name', '')[:20]} | {seat.get('buy_amount', 0):.2f}亿 | "
                                 f"{seat.get('sell_amount', 0):.2f}亿 | {seat.get('net_amount', 0):.2f}亿 |\n")
        
        # ========== 机构买卖统计（新增）==========
        lhb_org_stats = overview.get('lhb_org_stats', {})
        org_buy_count = lhb_org_stats.get('buy_count', overview.get('lhb_org_buy_count', 0))
        org_sell_count = lhb_org_stats.get('sell_count', overview.get('lhb_org_sell_count', 0))
        org_net_buy = lhb_org_stats.get('net_buy', 0)
        
        # 计算买卖力量比
        big_buy = overview.get('big_buy_count', 0)
        big_sell = overview.get('big_sell_count', 1)
        buy_sell_ratio = big_buy / max(big_sell, 1)
        
        return f"""你是一位资深的 A 股资金分析师，专注于追踪主力资金动向。请基于今日数据，撰写一份专业的资金流向分析报告。

## 你的专业视角
- 资金是市场的血液，资金流向决定涨跌
- 融资余额反映杠杆资金态度
- 大宗交易折溢价揭示机构预期
- 龙虎榜是主力资金的明牌
- 机构席位和游资席位的动向代表不同资金风格

## 今日资金数据

**融资融券**
- 融资余额: {overview.get('margin_balance', 0):.0f} 亿（杠杆资金规模）
- 融资买入额: {overview.get('margin_buy', 0):.2f} 亿（今日加杠杆）
- 融券余额: {overview.get('short_balance', 0):.2f} 亿（做空规模）

**大宗交易**
- 成交总额: {overview.get('block_trade_amount', 0):.2f} 亿
- 溢价成交占比: {overview.get('block_trade_premium_ratio', 0):.1f}%（高=机构看好）
- 折价成交占比: {overview.get('block_trade_discount_ratio', 0):.1f}%（高=机构出货）

**龙虎榜概览（净买入: {overview.get('lhb_net_buy', 0):.2f}亿）**
{lhb_text}

**机构席位统计**
- 机构买入次数: {org_buy_count} 次
- 机构卖出次数: {org_sell_count} 次
- 机构净买入: {org_net_buy:.2f} 亿
- 机构买卖比: {org_buy_count / max(org_sell_count, 1):.2f}（>1 机构偏多）

{f"**龙虎榜席位明细**{chr(10)}{lhb_seat_text}" if lhb_seat_text else ""}

**盘口资金**
- 大笔买入: {big_buy} 次
- 大笔卖出: {big_sell} 次
- 买卖力量比: {buy_sell_ratio:.2f}（>1 买方占优）

{self._build_fund_history_text(overview)}
---

请撰写一份资金流向分析报告，包含：

1. **主力态度**：今日主力资金整体是进攻还是防守？加仓还是减仓？
2. **杠杆信号**：融资余额和买入额变化说明什么？杠杆资金在做什么？
3. **机构动向**：机构席位在买什么方向？卖什么方向？机构买卖比说明什么？
4. **游资动向**：游资在炒什么题材？哪些知名游资席位活跃？
5. **大宗交易**：大宗交易折溢价说明什么？有无大规模出货迹象？
6. **资金预判**：基于今日资金流向，明日资金大概率如何表现？

请用专业但易懂的语言，直接输出分析报告。"""


class RiskAnalyst(DimensionAnalyst):
    """风险预警分析师 - 专注风险信号识别"""
    
    @property
    def dimension_name(self) -> str:
        return "风险预警"
    
    @property
    def analyst_role(self) -> str:
        return "风险控制分析师，专注识别市场风险信号，擅长预警系统性风险"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        news = data.get('news', [])
        
        # ========== 跌停股数据（增强：含详细信息）==========
        dt_pool = overview.get('dt_pool', [])
        dt_text = ""
        if dt_pool:
            dt_text = "| 股票 | 连续跌停 | 封单资金 | 行业 |\n"
            dt_text += "|------|----------|----------|------|\n"
            for d in dt_pool[:20]:
                dt_text += f"| {d.get('name', '')} | {d.get('continuous', 1)}板 | {d.get('seal_amount', 0):.2f}亿 | {d.get('industry', '')} |\n"
        else:
            dt_text = "暂无跌停股"
        
        # ========== 高台跳水详情（新增）==========
        pankou_changes = overview.get('pankou_changes', {})
        dive_list = pankou_changes.get('高台跳水', []) if isinstance(pankou_changes, dict) else []
        dive_text = ""
        if dive_list:
            dive_text = "| 时间 | 股票 | 板块 | 详情 |\n"
            dive_text += "|------|------|------|------|\n"
            for item in dive_list[:20]:
                dive_text += f"| {item.get('time', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:25]} |\n"
        
        # ========== 大笔卖出详情（新增）==========
        big_sell_list = pankou_changes.get('大笔卖出', []) if isinstance(pankou_changes, dict) else []
        big_sell_text = ""
        if big_sell_list:
            big_sell_text = "| 时间 | 股票 | 板块 | 详情 |\n"
            big_sell_text += "|------|------|------|------|\n"
            for item in big_sell_list[:20]:
                big_sell_text += f"| {item.get('time', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:25]} |\n"
        
        # ========== 炸板股详情（新增）==========
        zb_pool = overview.get('zb_pool', [])
        zb_text = ""
        if zb_pool:
            zb_text = "| 股票 | 炸板次数 | 涨跌幅 | 行业 |\n"
            zb_text += "|------|----------|--------|------|\n"
            for zb in zb_pool[:20]:
                zb_text += f"| {zb.get('name', '')} | {zb.get('zb_count', 0)}次 | {zb.get('change_pct', 0):+.2f}% | {zb.get('industry', '')} |\n"
        
        # 新闻摘要
        news_text = "\n".join([
            f"- {n.title[:50] if hasattr(n, 'title') else n.get('title', '')[:50]}"
            for n in news[:10]
        ]) if news else "暂无新闻"
        
        # 计算涨跌比
        up = overview.get('up_count', 0)
        down = overview.get('down_count', 1)
        up_down_ratio = up / max(down, 1)
        
        return f"""你是一位资深的 A 股风险分析师，专注于识别和预警市场风险。请基于今日数据，撰写一份专业的风险预警报告。

## 你的专业视角
- 风险控制是投资的第一要务
- 跌停股是市场风险的放大镜
- 高台跳水和大笔卖出是资金出逃的信号
- 炸板率高说明市场分歧大
- 情绪过热往往是风险的前兆

## 今日风险数据

**跌停与炸板**
- 跌停数量: {overview.get('limit_down_count', 0)} 只
- 连续跌停: {overview.get('dt_continuous_count', 0)} 只
- 炸板数量: {overview.get('zb_pool_count', 0)} 只
- 炸板率: {overview.get('zb_rate', 0):.1f}%
- 高台跳水: {overview.get('high_dive_count', 0)} 次

**跌停股明细**
{dt_text}

{f"**炸板股明细**{chr(10)}{zb_text}" if zb_text else ""}

**市场分化**
- 上涨 {up} 家 / 下跌 {down} 家
- 涨跌比: {up_down_ratio:.2f}（<1 说明亏钱效应）
- 低分股数量(<=40分): {overview.get('comment_low_score_count', 0)} 只

**资金风险信号**
- 大笔卖出: {overview.get('big_sell_count', 0)} 次
- 大宗折价占比: {overview.get('block_trade_discount_ratio', 0):.1f}%

{f"**高台跳水详情（快速下杀信号）**{chr(10)}{dive_text}" if dive_text else ""}

{f"**大笔卖出详情（资金出逃信号）**{chr(10)}{big_sell_text}" if big_sell_text else ""}

**近期新闻（关注利空）**
{news_text}

---

请撰写一份风险预警报告，包含：

1. **系统性风险**：当前有无系统性风险信号？指数层面风险如何？
2. **板块风险**：哪些板块有见顶或崩塌风险？跌停股和高台跳水集中在哪些板块？
3. **个股风险**：哪些股票出现明显的资金出逃信号？大笔卖出集中在哪些股票？
4. **情绪风险**：炸板率说明什么？市场情绪有无过热或崩塌迹象？
5. **政策风险**：新闻中有无政策利空信号？哪些行业可能受影响？
6. **风险等级**：综合评估当前市场风险等级（低/中/高），给出防范建议

请用专业但易懂的语言，直接输出分析报告。"""


class IndexAnalyst(DimensionAnalyst):
    """指数技术分析师 - 专注大盘指数走势"""
    
    @property
    def dimension_name(self) -> str:
        return "指数走势"
    
    @property
    def analyst_role(self) -> str:
        return "指数技术分析师，专注大盘走势研判，擅长支撑压力位分析"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        indices = overview.get('indices', [])
        
        # ========== 格式化指数数据（增强：含开盘/最高/最低价）==========
        indices_table = ""
        if indices:
            indices_table = "| 指数 | 收盘 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 |\n"
            indices_table += "|------|------|--------|------|------|------|------|\n"
            for idx in indices[:20]:
                if isinstance(idx, dict):
                    name = idx.get('name', '')
                    current = idx.get('current', 0)
                    change_pct = idx.get('change_pct', 0)
                    open_price = idx.get('open', 0)
                    high = idx.get('high', 0)
                    low = idx.get('low', 0)
                    amplitude = idx.get('amplitude', 0)
                else:
                    name = getattr(idx, 'name', '')
                    current = getattr(idx, 'current', 0)
                    change_pct = getattr(idx, 'change_pct', 0)
                    open_price = getattr(idx, 'open', 0)
                    high = getattr(idx, 'high', 0)
                    low = getattr(idx, 'low', 0)
                    amplitude = getattr(idx, 'amplitude', 0)
                
                indices_table += f"| {name} | {current:.2f} | {change_pct:+.2f}% | {open_price:.2f} | {high:.2f} | {low:.2f} | {amplitude:.2f}% |\n"
        else:
            indices_table = "暂无指数数据"
        
        # ========== 走势形态判断辅助数据 ==========
        # 提取主要指数的走势特征
        main_indices_analysis = ""
        for idx in indices[:10]:  # 上证、深证、创业板
            if isinstance(idx, dict):
                name = idx.get('name', '')
                current = idx.get('current', 0)
                open_price = idx.get('open', 0)
                high = idx.get('high', 0)
                low = idx.get('low', 0)
            else:
                name = getattr(idx, 'name', '')
                current = getattr(idx, 'current', 0)
                open_price = getattr(idx, 'open', 0)
                high = getattr(idx, 'high', 0)
                low = getattr(idx, 'low', 0)
            
            if open_price > 0 and high > 0 and low > 0:
                # 判断走势形态
                range_total = high - low if high > low else 1
                upper_shadow = (high - max(current, open_price)) / range_total * 100
                lower_shadow = (min(current, open_price) - low) / range_total * 100
                body = abs(current - open_price) / range_total * 100
                
                if current > open_price:
                    candle_type = "阳线"
                elif current < open_price:
                    candle_type = "阴线"
                else:
                    candle_type = "十字星"
                
                main_indices_analysis += f"- **{name}**: {candle_type}，上影线{upper_shadow:.0f}%，下影线{lower_shadow:.0f}%，实体{body:.0f}%\n"
        
        return f"""你是一位资深的 A 股指数分析师，专注于大盘走势研判。请基于今日数据，撰写一份专业的指数走势分析报告。

## 你的专业视角
- 指数是市场的方向标
- 开盘价、最高价、最低价揭示日内走势形态
- 大小盘风格切换决定操作策略
- 支撑压力位是关键决策点
- 量价配合验证趋势有效性

## 今日指数数据

**主要指数表现（含日内高低点）**
{indices_table}

**K线形态分析**
{main_indices_analysis if main_indices_analysis else "暂无数据"}

**市场宽度**
- 上涨 {overview.get('up_count', 0)} 家 / 下跌 {overview.get('down_count', 0)} 家
- 涨停 {overview.get('limit_up_count', 0)} 只 / 跌停 {overview.get('limit_down_count', 0)} 只
- 两市成交 {overview.get('total_amount', 0):.0f} 亿
- 平均换手率 {overview.get('avg_turnover_rate', 0):.2f}%

---

请撰写一份指数走势分析报告，包含：

1. **走势特征**：今日大盘走势呈现什么特征？（冲高回落/低开高走/震荡整理/单边上涨/单边下跌）根据开盘、最高、最低、收盘判断日内走势
2. **K线形态**：今日K线是什么形态？（大阳线/小阳线/十字星/小阴线/大阴线）上下影线说明什么？
3. **风格判断**：大盘股强还是小盘股强？成长风格还是价值风格？权重股表现如何？
4. **量价分析**：成交量配合如何？是放量还是缩量？量价关系健康吗？
5. **关键位置**：当前指数处于什么位置？上方压力位在哪？下方支撑位在哪？
6. **趋势预判**：短期趋势如何？有无拐点信号？明日大盘大概率如何表现？

请用专业但易懂的语言，直接输出分析报告。"""


class NewsAnalyst(DimensionAnalyst):
    """消息面分析师 - 专注新闻政策解读"""
    
    @property
    def dimension_name(self) -> str:
        return "消息政策"
    
    @property
    def analyst_role(self) -> str:
        return "政策消息分析师，专注政策解读和消息面影响分析"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        news = data.get('news', [])
        
        # 财新新闻
        caixin_news = overview.get('caixin_news', [])
        caixin_text = "\n".join([
            f"- [{n.get('tag', '')}] {n.get('summary', '')[:100]}"
            for n in caixin_news[:20]
        ]) if caixin_news else "暂无财新数据"
        
        # 搜索新闻
        news_text = "\n".join([
            f"- {n.title[:80] if hasattr(n, 'title') else n.get('title', '')[:80]}"
            for n in news[:20]
        ]) if news else "暂无搜索新闻"
        
        return f"""你是一位资深的 A 股政策分析师，专注于政策解读和消息面影响分析。请基于今日消息，撰写一份专业的消息政策分析报告。

## 你的专业视角
- 政策是 A 股最大的基本面
- 消息面影响短期情绪和资金流向
- 利好利空需要区分真假和持续性
- 政策受益板块往往是中期主线

## 今日消息数据

**财新内容精选**
{caixin_text}

**市场新闻**
{news_text}

---

请撰写一份消息政策分析报告，包含：

1. **重要政策**：今日有无重要政策出台？政策的核心内容是什么？对市场影响如何？
2. **利好消息**：有哪些利好消息？利好哪些板块？利好的持续性如何？
3. **利空消息**：有哪些利空消息？利空哪些板块？是短期冲击还是长期影响？
4. **政策方向**：近期政策整体偏暖还是偏冷？政策重点支持哪些方向？
5. **消息预判**：基于当前消息面，哪些板块可能受益？哪些需要回避？

请用专业但易懂的语言，直接输出分析报告。"""


# ============================================================
# 新增维度分析师（涨停板、潜力股、埋伏机会、千股千评）
# ============================================================

class ZTPoolAnalyst(DimensionAnalyst):
    """涨停板生态分析师 - 专注涨停板数据深度分析"""
    
    @property
    def dimension_name(self) -> str:
        return "涨停板生态"
    
    @property
    def analyst_role(self) -> str:
        return "涨停板生态分析师，专注连板高度、炸板率、溢价率等短线情绪核心指标"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # 涨停股池详情
        zt_pool = overview.get('zt_pool', [])
        zt_pool_text = ""
        if zt_pool:
            zt_pool_text = "| 股票 | 连板数 | 封板资金 | 炸板次数 | 行业 |\n|------|--------|----------|----------|------|\n"
            for zt in zt_pool[:20]:
                zt_pool_text += f"| {zt.get('name', '')} | {zt.get('continuous', 1)}板 | {zt.get('seal_amount', 0):.1f}亿 | {zt.get('zb_count', 0)}次 | {zt.get('industry', '')} |\n"
        
        # 昨日涨停今日表现
        previous_zt = overview.get('previous_zt_pool', [])
        previous_zt_text = ""
        if previous_zt:
            previous_zt_text = "| 股票 | 今日涨跌 | 昨日连板 | 行业 |\n|------|----------|----------|------|\n"
            for pzt in previous_zt[:20]:
                previous_zt_text += f"| {pzt.get('name', '')} | {pzt.get('change_pct', 0):+.2f}% | {pzt.get('yesterday_continuous', 1)}板 | {pzt.get('industry', '')} |\n"
        
        # 炸板股池
        zb_pool = overview.get('zb_pool', [])
        zb_pool_text = ""
        if zb_pool:
            zb_pool_text = "| 股票 | 炸板次数 | 涨跌幅 | 行业 |\n|------|----------|--------|------|\n"
            for zb in zb_pool[:20]:
                zb_pool_text += f"| {zb.get('name', '')} | {zb.get('zb_count', 0)}次 | {zb.get('change_pct', 0):+.2f}% | {zb.get('industry', '')} |\n"
        
        # 跌停股池
        dt_pool = overview.get('dt_pool', [])
        dt_pool_text = ""
        if dt_pool:
            dt_pool_text = "| 股票 | 连续跌停 | 行业 |\n|------|----------|------|\n"
            for dt in dt_pool[:20]:
                dt_pool_text += f"| {dt.get('name', '')} | {dt.get('continuous', 1)}板 | {dt.get('industry', '')} |\n"
        
        return f"""你是一位资深的 A 股涨停板生态分析师，拥有丰富的短线交易经验。请基于今日涨停板数据，撰写一份专业的涨停板生态分析报告。

## 你的专业视角
- 连板高度是市场情绪的天花板
- 炸板率反映资金承接意愿和市场分歧
- 溢价率体现打板族的赚钱效应
- 涨停股行业分布揭示资金主攻方向

## 今日涨停板核心数据

**涨停股池概况**
- 涨停数量: {overview.get('zt_pool_count', 0)} 只
- 首板数量: {overview.get('zt_first_board_count', 0)} 只
- 连板数量: {overview.get('zt_continuous_count', 0)} 只
- 最高连板: {overview.get('zt_max_continuous', 0)} 板
- 涨停股总成交额: {overview.get('zt_total_amount', 0):.0f} 亿
- 涨停股平均换手率: {overview.get('zt_avg_turnover', 0):.1f}%

**涨停股详情（按连板数排序）**
{zt_pool_text if zt_pool_text else "暂无数据"}

**昨日涨停今日表现（溢价率分析）**
- 昨日涨停: {overview.get('previous_zt_count', 0)} 只
- 今日平均溢价: {overview.get('previous_zt_avg_change', 0):+.2f}%
- 今日上涨: {overview.get('previous_zt_up_count', 0)} 只
- 今日下跌: {overview.get('previous_zt_down_count', 0)} 只

{previous_zt_text if previous_zt_text else ""}

**炸板股池（炸板率: {overview.get('zb_rate', 0):.1f}%）**
- 炸板股: {overview.get('zb_pool_count', 0)} 只
- 炸板总次数: {overview.get('zb_total_count', 0)} 次

{zb_pool_text if zb_pool_text else ""}

**跌停股池**
- 跌停数量: {overview.get('dt_pool_count', 0)} 只
- 连续跌停: {overview.get('dt_continuous_count', 0)} 只

{dt_pool_text if dt_pool_text else ""}

**强势股池**
- 强势股数量: {overview.get('strong_pool_count', 0)} 只
- 60日新高: {overview.get('strong_new_high_count', 0)} 只
- 近期多次涨停: {overview.get('strong_multi_zt_count', 0)} 只

---

请撰写一份涨停板生态分析报告，包含：

1. **连板梯队分析**：今日连板高度如何？最高板是谁？连板梯队是否健康？
2. **炸板率解读**：炸板率高低说明什么？资金承接意愿如何？
3. **溢价率分析**：昨日涨停今日表现如何？打板族赚钱还是亏钱？
4. **行业分布**：涨停股集中在哪些行业？说明资金在做什么方向？
5. **情绪判断**：基于涨停板数据，市场情绪处于什么阶段？（冰点/低迷/温和/亢奋/过热）
6. **明日预判**：明日涨停板生态大概率如何演绎？打板策略建议？

请用专业但易懂的语言，直接输出分析报告。"""


class HiddenInflowAnalyst(DimensionAnalyst):
    """
    潜力股挖掘分析师 - 智能发现资金流入但热度不高的股票
    
    职责：
    1. 接收所有有大笔买入的股票原始数据
    2. 使用 LLM 智能筛选真正的潜力股
    3. 分析资金流入逻辑和潜在催化剂
    
    筛选维度（由 LLM 综合判断）：
    - 资金流入强度：大笔买入次数、买入时间分布
    - 市场热度：关注指数与市场平均的对比
    - 基本面质量：综合得分、机构参与度
    - 价格空间：今日涨幅、是否已被拉升
    - 板块逻辑：所属板块是否有题材支撑
    """
    
    @property
    def dimension_name(self) -> str:
        return "潜力股挖掘"
    
    @property
    def analyst_role(self) -> str:
        return "潜力股挖掘分析师，专注发现主力悄悄建仓的低热度股票"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # ========== 所有有资金流入的股票原始数据 ==========
        all_inflow_stocks = overview.get('hidden_inflow_stocks', [])
        avg_attention = all_inflow_stocks[0].get('avg_attention', 50) if all_inflow_stocks else 50
        
        # 构建完整的股票数据表格（供 LLM 智能筛选）
        all_stocks_text = ""
        if all_inflow_stocks:
            all_stocks_text = "| 代码 | 名称 | 大笔买入 | 关注指数 | 综合得分 | 今日涨跌 | 机构参与度 | 市值(亿) | 换手率 | 板块 |\n"
            all_stocks_text += "|------|------|----------|----------|----------|----------|------------|----------|--------|------|\n"
            for stock in all_inflow_stocks[:40]:  # 展示前25只供分析
                market_cap = stock.get('market_cap', 0)
                market_cap_str = f"{market_cap:.0f}" if market_cap > 0 else "-"
                turnover = stock.get('turnover_rate', 0)
                turnover_str = f"{turnover:.1f}%" if turnover > 0 else "-"
                industry = stock.get('industry', stock.get('sector', ''))
                
                all_stocks_text += (f"| {stock.get('code', '')} | {stock.get('name', '')} | "
                                   f"{stock.get('big_buy_count', 0)}次 | {stock.get('attention', 0):.0f} | "
                                   f"{stock.get('score', 0):.0f} | {stock.get('change_pct', 0):+.2f}% | "
                                   f"{stock.get('org_participate', 0):.1f}% | {market_cap_str} | "
                                   f"{turnover_str} | {industry[:6]} |\n")
        
        # ========== 大笔买入时间分布（判断资金流入节奏）==========
        buy_time_analysis = ""
        if all_inflow_stocks:
            for stock in all_inflow_stocks[:20]:  # 前5只的买入时间
                times = stock.get('big_buy_times', [])
                if times:
                    buy_time_analysis += f"- {stock.get('name', '')}: {', '.join(times[:4])}\n"
        
        # ========== 大笔买入详情 ==========
        pankou_changes = overview.get('pankou_changes', {})
        big_buy_list = pankou_changes.get('大笔买入', []) if isinstance(pankou_changes, dict) else []
        big_buy_text = ""
        if big_buy_list:
            big_buy_text = "| 时间 | 代码 | 名称 | 板块 | 详情 |\n|------|------|------|------|------|\n"
            for item in big_buy_list[:20]:
                big_buy_text += f"| {item.get('time', '')} | {item.get('code', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:25]} |\n"
        
        # ========== 大笔卖出详情（对比分析）==========
        big_sell_list = pankou_changes.get('大笔卖出', []) if isinstance(pankou_changes, dict) else []
        big_sell_text = ""
        if big_sell_list:
            big_sell_text = "| 时间 | 代码 | 名称 | 板块 | 详情 |\n|------|------|------|------|------|\n"
            for item in big_sell_list[:20]:
                big_sell_text += f"| {item.get('time', '')} | {item.get('code', '')} | {item.get('name', '')} | {item.get('sector', '')} | {item.get('info', '')[:25]} |\n"
        
        # ========== 板块分布统计 ==========
        sector_distribution = {}
        for stock in all_inflow_stocks:
            sector = stock.get('industry', stock.get('sector', '其他'))
            if sector:
                sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
        
        sector_dist_text = ""
        if sector_distribution:
            sorted_sectors = sorted(sector_distribution.items(), key=lambda x: -x[1])[:10]
            sector_dist_text = "\n".join([f"- {s[0]}: {s[1]} 只" for s in sorted_sectors])
        
        return f"""你是一位资深的 A 股潜力股挖掘分析师，拥有 15 年主力资金追踪经验。

## 你的核心能力
- 从海量数据中识别主力悄悄建仓的蛛丝马迹
- 区分真正的潜力股和资金陷阱
- 判断资金流入的持续性和可靠性

## 今日任务
从以下有资金流入的股票中，智能筛选出真正值得关注的潜力股。

## 筛选标准（请综合判断，不要机械套用）

**优质潜力股特征：**
1. 资金流入持续：大笔买入次数 >= 2，且时间分布合理（非集中在某一时刻）
2. 热度较低：关注指数低于市场平均（{avg_attention:.0f}），说明散户关注少
3. 基本面不差：综合得分 >= 60，机构参与度适中
4. 价格有空间：今日涨幅 < 5%，未被大幅拉升
5. 板块有逻辑：所属板块有题材支撑或处于轮动窗口

**需要警惕的陷阱：**
- 涨幅已大但仍有大笔买入（可能是诱多）
- 关注指数极高（已被市场充分关注）
- 综合得分很低（基本面有问题）
- 大笔买入集中在尾盘（可能是做盘）
- 同时出现大笔卖出（资金分歧大）

## 原始数据

**市场平均关注指数**: {avg_attention:.0f}

**所有有资金流入的股票（共 {len(all_inflow_stocks)} 只）**
{all_stocks_text if all_stocks_text else "暂无数据"}

**大笔买入时间分布（判断资金节奏）**
{buy_time_analysis if buy_time_analysis else "暂无数据"}

**大笔买入明细**
{big_buy_text if big_buy_text else "暂无数据"}

**大笔卖出明细（对比分析）**
{big_sell_text if big_sell_text else "暂无数据"}

**资金流入股票板块分布**
{sector_dist_text if sector_dist_text else "暂无数据"}

**盘口异动统计**
- 大笔买入: {overview.get('big_buy_count', 0)} 次
- 大笔卖出: {overview.get('big_sell_count', 0)} 次
- 买卖力量比: {overview.get('big_buy_count', 0) / max(overview.get('big_sell_count', 1), 1):.2f}

---

## 请输出分析报告

### 一、资金流向全景

（分析今日大笔买入的整体特征：集中在哪些板块？买入时间分布如何？买卖力量对比说明什么？）

### 二、潜力股筛选结果

请从上述股票中筛选出 3-5 只真正的潜力股，并说明筛选理由：

| 代码 | 名称 | 推荐理由 | 风险点 | 关注级别 |
|------|------|----------|--------|----------|
| ... | ... | ... | ... | ⭐⭐⭐ |

### 三、重点潜力股深度分析

（选择 2-3 只最值得关注的，深入分析：资金流入逻辑、板块背景、潜在催化剂、目标预期）

### 四、风险警示

（哪些股票虽然有资金流入但不建议参与？为什么？）

### 五、操作建议

（短期关注策略、介入时机、仓位建议）

请用专业但易懂的语言，直接输出分析报告。"""


class SectorOpportunityAnalyst(DimensionAnalyst):
    """板块埋伏机会分析师 - 基于申万行业估值数据分析埋伏机会"""
    
    @property
    def dimension_name(self) -> str:
        return "板块埋伏"
    
    @property
    def analyst_role(self) -> str:
        return "板块埋伏机会分析师，专注发现估值底部+催化剂+反转信号的板块"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # 推荐埋伏板块
        recommended = overview.get('sector_recommended', [])
        recommended_text = ""
        if recommended:
            recommended_text = "| 板块 | 总分 | 便宜 | 催化 | 反转 | PE | 股息率 | 推荐理由 |\n"
            recommended_text += "|------|------|------|------|------|-----|--------|----------|\n"
            for opp in recommended[:20]:
                reasons = opp.get('cheap_reasons', [])[:1] + opp.get('catalyst_reasons', [])[:1]
                reason_text = '; '.join(reasons)[:30] if reasons else '-'
                recommended_text += (f"| {opp.get('sector_name', '')} | {opp.get('total_score', 0)} | "
                                    f"{opp.get('cheap_score', 0)} | {opp.get('catalyst_score', 0)} | "
                                    f"{opp.get('reversal_score', 0)} | {opp.get('current_pe', 0):.1f} | "
                                    f"{opp.get('dividend_yield', 0):.1f}% | {reason_text} |\n")
        
        # 估值最低板块
        cheap_list = overview.get('sector_cheap_list', [])
        cheap_text = ""
        if cheap_list:
            # 过滤掉没有有效价格分位数据的板块
            valid_cheap = [o for o in cheap_list if o.get('price_percentile', -1) >= 0]
            if valid_cheap:
                cheap_text = "| 板块 | 价格分位 | PE | PB | 股息率 | 便宜原因 |\n"
                cheap_text += "|------|----------|-----|-----|--------|----------|\n"
                for opp in valid_cheap[:5]:
                    reasons = '; '.join(opp.get('cheap_reasons', [])[:2])[:40] or '-'
                    cheap_text += (f"| {opp.get('sector_name', '')} | {opp.get('price_percentile', 0):.0f}% | "
                                  f"{opp.get('current_pe', 0):.1f} | {opp.get('current_pb', 0):.1f} | "
                                  f"{opp.get('dividend_yield', 0):.1f}% | {reasons} |\n")
        
        # 有催化板块
        catalyst_list = overview.get('sector_catalyst_list', [])
        catalyst_text = ""
        if catalyst_list:
            catalyst_text = "| 板块 | 催化得分 | 政策关键词 | 催化原因 |\n"
            catalyst_text += "|------|----------|------------|----------|\n"
            for opp in catalyst_list[:5]:
                keywords = ', '.join(opp.get('policy_keywords', [])[:3]) or '-'
                reasons = '; '.join(opp.get('catalyst_reasons', [])[:2])[:40] or '-'
                catalyst_text += (f"| {opp.get('sector_name', '')} | {opp.get('catalyst_score', 0)} | "
                                 f"{keywords} | {reasons} |\n")
        
        # 有反转信号板块
        reversal_list = overview.get('sector_reversal_list', [])
        reversal_text = ""
        if reversal_list:
            reversal_text = "| 板块 | 反转得分 | 今日涨跌 | 涨停数 | 反转原因 |\n"
            reversal_text += "|------|----------|----------|--------|----------|\n"
            for opp in reversal_list[:20]:
                reasons = '; '.join(opp.get('reversal_reasons', [])[:2])[:40] or '-'
                reversal_text += (f"| {opp.get('sector_name', '')} | {opp.get('reversal_score', 0)} | "
                                 f"{opp.get('recent_5d_change', 0):+.2f}% | {opp.get('zt_count', 0)} | "
                                 f"{reasons} |\n")
        
        # 行业板块涨跌榜（补充数据，增强：含净流入、领涨股）
        top_sectors = overview.get('top_sectors', [])
        top_sectors_text = ""
        if top_sectors:
            for s in top_sectors[:20]:
                net_inflow = s.get('net_inflow', 0)
                leader = s.get('leader_stock', '')
                leader_text = f"（领涨:{leader}）" if leader else ""
                top_sectors_text += f"- {s.get('name', '')}: {s.get('change_pct', 0):+.2f}% 净流入{net_inflow:.2f}亿{leader_text}\n"
        top_sectors_text = top_sectors_text or "暂无数据"
        
        bottom_sectors = overview.get('bottom_sectors', [])
        bottom_sectors_text = ""
        if bottom_sectors:
            for s in bottom_sectors[:20]:
                net_inflow = s.get('net_inflow', 0)
                bottom_sectors_text += f"- {s.get('name', '')}: {s.get('change_pct', 0):+.2f}% 净流入{net_inflow:.2f}亿\n"
        bottom_sectors_text = bottom_sectors_text or "暂无数据"
        
        # 涨停股行业分布
        zt_pool = overview.get('zt_pool', [])
        zt_industries = {}
        for zt in zt_pool:
            ind = zt.get('industry', '其他')
            zt_industries[ind] = zt_industries.get(ind, 0) + 1
        zt_dist_text = "\n".join([
            f"- {k}: {v} 只涨停"
            for k, v in sorted(zt_industries.items(), key=lambda x: -x[1])[:8]
        ]) or "暂无数据"
        
        return f"""你是一位资深的 A 股板块轮动分析师，专注于发现板块埋伏机会。请基于今日申万行业估值数据和市场表现，撰写一份专业的板块埋伏机会分析报告。

## 你的专业视角

板块埋伏的核心逻辑是同时满足以下三个条件中的至少两个：

1. **够便宜（安全垫）**：估值在历史底部（价格分位数<30%），PE/PB处于低位，高股息率提供安全边际
2. **有催化（导火索）**：未来有政策预期，技术突破或产品落地，行业重大事件
3. **有反转（基本面）**：行业供需格局改善，资金开始流入，涨停股增多，今日涨幅明显

## 申万行业估值数据

**推荐埋伏板块（总分>=4，满足至少2个条件）**
{recommended_text if recommended_text else "暂无推荐板块"}

**估值最低板块TOP5（够便宜）**
{cheap_text if cheap_text else "暂无数据"}

**有催化板块TOP5（有导火索）**
{catalyst_text if catalyst_text else "暂无数据"}

**有反转信号板块TOP5（资金流入）**
{reversal_text if reversal_text else "暂无数据"}

## 今日市场表现

**行业板块涨幅榜**
{top_sectors_text}

**行业板块跌幅榜**
{bottom_sectors_text}

**涨停股行业分布（资金主攻方向）**
{zt_dist_text}

**龙虎榜净买入**: {overview.get('lhb_net_buy', 0):.2f} 亿
**融资买入额**: {overview.get('margin_buy', 0):.2f} 亿

---

请撰写一份板块埋伏机会分析报告，包含：

1. **推荐埋伏板块深度分析**（重点）：为什么这些板块值得埋伏？核心逻辑是什么？催化剂预期是什么？风险点在哪里？

2. **估值洼地机会**：哪些板块估值处于历史底部？低估值是否有基本面支撑？是价值陷阱还是真正的机会？

3. **反转信号识别**：哪些板块出现资金流入迹象？涨停股分布说明什么？是短期反弹还是趋势反转？

4. **板块轮动预判**：当前市场处于什么风格？下一个可能轮动的方向是什么？

5. **操作建议**：短期重点关注哪些板块？中期可以埋伏哪些板块？需要回避哪些板块？

请用专业但易懂的语言，直接输出分析报告。"""


class CommentScoreAnalyst(DimensionAnalyst):
    """千股千评分析师 - 专注市场整体评分和高分股挖掘"""
    
    @property
    def dimension_name(self) -> str:
        return "千股千评"
    
    @property
    def analyst_role(self) -> str:
        return "千股千评分析师，专注市场整体评分变化和高分股挖掘"
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        overview = data.get('overview', {})
        
        # ========== 综合得分TOP股票（增强：含排名变化、主力成本）==========
        comment_top = overview.get('comment_top_stocks', [])
        comment_top_text = ""
        if comment_top:
            comment_top_text = "| 代码 | 名称 | 综合得分 | 排名 | 排名变化 | 涨跌幅 | 换手率 | 机构参与度 | 主力成本 |\n"
            comment_top_text += "|------|------|----------|------|----------|--------|--------|------------|----------|\n"
            for stock in comment_top[:20]:
                rank_change = stock.get('rank_change', 0)
                rank_change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
                comment_top_text += (f"| {stock.get('code', '')} | {stock.get('name', '')} | "
                                    f"{stock.get('score', 0):.0f} | {stock.get('rank', 0)} | "
                                    f"{rank_change_str} | {stock.get('change_pct', 0):+.2f}% | "
                                    f"{stock.get('turnover_rate', 0):.1f}% | {stock.get('org_participate', 0):.1f}% | "
                                    f"{stock.get('main_cost', 0):.2f} |\n")
        
        # ========== 高关注度股票（增强：含机构参与度）==========
        high_attention = overview.get('comment_high_attention', [])
        attention_text = ""
        if high_attention:
            attention_text = "| 代码 | 名称 | 关注指数 | 综合得分 | 涨跌幅 | 机构参与度 |\n"
            attention_text += "|------|------|----------|----------|--------|------------|\n"
            for stock in high_attention[:20]:
                attention_text += (f"| {stock.get('code', '')} | {stock.get('name', '')} | "
                                  f"{stock.get('attention', 0):.0f} | {stock.get('score', 0):.0f} | "
                                  f"{stock.get('change_pct', 0):+.2f}% | {stock.get('org_participate', 0):.1f}% |\n")
        
        # ========== 综合得分最低股票（增强：含排名变化）==========
        comment_bottom = overview.get('comment_bottom_stocks', [])
        comment_bottom_text = ""
        if comment_bottom:
            comment_bottom_text = "| 代码 | 名称 | 综合得分 | 排名 | 排名变化 | 涨跌幅 |\n"
            comment_bottom_text += "|------|------|----------|------|----------|--------|\n"
            for stock in comment_bottom[:20]:
                rank_change = stock.get('rank_change', 0)
                rank_change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
                comment_bottom_text += (f"| {stock.get('code', '')} | {stock.get('name', '')} | "
                                       f"{stock.get('score', 0):.0f} | {stock.get('rank', 0)} | "
                                       f"{rank_change_str} | {stock.get('change_pct', 0):+.2f}% |\n")
        
        # ========== 统计分析 ==========
        # 高分股中排名上升的数量
        rank_up_count = sum(1 for s in comment_top if s.get('rank_change', 0) > 0)
        rank_down_count = sum(1 for s in comment_top if s.get('rank_change', 0) < 0)
        
        # 高分股平均机构参与度
        avg_org_participate = sum(s.get('org_participate', 0) for s in comment_top) / max(len(comment_top), 1)
        
        # 高分股平均换手率
        avg_turnover = sum(s.get('turnover_rate', 0) for s in comment_top) / max(len(comment_top), 1)
        
        # 高关注度股票中高分股数量
        high_attention_high_score = sum(1 for s in high_attention if s.get('score', 0) >= 70)
        
        return f"""你是一位专业的 A 股量化分析师，擅长解读千股千评数据。请基于今日数据，撰写一份专业的千股千评分析报告。

## 你的专业视角
- 千股千评综合得分反映个股的技术面和资金面综合状态
- 排名变化（上升/下降）揭示个股近期走势趋势
- 机构参与度高的股票更受主力资金关注
- 主力成本是判断支撑位的重要参考
- 高关注度+高得分=市场共识方向
- 高关注度+低得分=可能存在分歧或风险

## 今日千股千评数据

**市场整体评分**
- 市场平均得分: {overview.get('comment_avg_score', 0):.1f} 分
- 高分股(>=80分): {overview.get('comment_high_score_count', 0)} 只
- 低分股(<=40分): {overview.get('comment_low_score_count', 0)} 只
- 高低分比: {overview.get('comment_high_score_count', 0) / max(overview.get('comment_low_score_count', 1), 1):.2f}（>1 市场偏强）

**综合得分 TOP10（含排名变化、主力成本）**
{comment_top_text if comment_top_text else "暂无数据"}

**TOP10 统计分析**
- 排名上升: {rank_up_count} 只 / 排名下降: {rank_down_count} 只
- 平均机构参与度: {avg_org_participate:.1f}%
- 平均换手率: {avg_turnover:.1f}%

**高关注度股票 TOP10**
{attention_text if attention_text else "暂无数据"}

**高关注度分析**
- 高关注度中高分股(>=70分): {high_attention_high_score} 只 / 10 只
- 说明: {"市场关注方向与技术面一致" if high_attention_high_score >= 6 else "市场关注与技术面存在分歧"}

**综合得分最低股票（风险警示）**
{comment_bottom_text if comment_bottom_text else "暂无数据"}

---

请撰写一份千股千评分析报告，包含：

1. **市场整体评分解读**：
   - 今日市场平均得分 {overview.get('comment_avg_score', 0):.1f} 分说明什么？
   - 高分股 {overview.get('comment_high_score_count', 0)} 只 vs 低分股 {overview.get('comment_low_score_count', 0)} 只，市场技术面整体如何？
   - 与近期相比，市场评分是在改善还是恶化？

2. **高分股深度分析**：
   - TOP10 高分股有什么共同特点？
   - 排名上升的股票说明什么？排名下降的呢？
   - 机构参与度高的股票值得关注吗？
   - 主力成本与当前价格的关系如何？（高于主力成本=获利盘，低于=套牢盘）

3. **高关注度股票解读**：
   - 市场在关注什么方向？
   - 高关注度+高得分的股票是否值得跟踪？
   - 高关注度+低得分的股票是否存在风险？

4. **投资机会挖掘**：
   - 哪些高分股值得重点关注？为什么？
   - 有无"低关注度+高得分"的潜力股？（被市场忽视的优质标的）

5. **风险提示**：
   - 低分股需要回避
   - 排名持续下降的股票需要警惕
   - 高关注度可能意味着拥挤风险

请用专业但易懂的语言，直接输出分析报告，不要输出 JSON。"""


# ============================================================
# Map-Reduce 协调器
# ============================================================

class MarketMapReduceAnalyzer:
    """
    市场分析 Map-Reduce 协调器
    
    流程：
    1. Map 阶段：并行调用多个维度分析师
    2. Reduce 阶段：首席分析师综合各维度结论
    """
    
    # 默认维度分析师（10个维度，全面覆盖市场数据）
    DEFAULT_ANALYSTS = [
        EmotionAnalyst,      # 市场情绪
        ThemeAnalyst,        # 主线题材
        FundFlowAnalyst,     # 资金流向
        RiskAnalyst,         # 风险预警
        IndexAnalyst,        # 指数走势
        NewsAnalyst,         # 消息政策
        # 以下为新增维度分析师
        ZTPoolAnalyst,       # 涨停板生态
        HiddenInflowAnalyst, # 潜力股挖掘
        SectorOpportunityAnalyst,  # 板块埋伏
        CommentScoreAnalyst, # 千股千评
    ]
    
    def __init__(self, llm_caller: Callable[[str, dict], str], 
                 analysts: Optional[List[type]] = None,
                 max_workers: int = 3):
        """
        Args:
            llm_caller: LLM 调用函数
            analysts: 维度分析师类列表，默认使用全部
            max_workers: 并行调用的最大线程数
        """
        self.llm_caller = llm_caller
        self.max_workers = max_workers
        
        analyst_classes = analysts or self.DEFAULT_ANALYSTS
        self.analysts = [cls(llm_caller) for cls in analyst_classes]
        
        logger.info(f"[MapReduce] 初始化完成，{len(self.analysts)} 个维度分析师")
    
    def analyze(self, overview_data: Dict[str, Any], 
                news: Optional[List] = None,
                parallel: bool = True) -> MapReduceResult:
        """
        执行 Map-Reduce 分析
        
        Args:
            overview_data: MarketOverview 数据（转为 dict）
            news: 新闻列表
            parallel: 是否并行执行 Map 阶段
            
        Returns:
            MapReduceResult
        """
        date = overview_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # 准备数据
        data = {
            'overview': overview_data,
            'news': news or [],
        }
        
        logger.info(f"[MapReduce] 开始分析 {date}")
        start_time = time.time()
        
        # ========== Map 阶段 ==========
        logger.info(f"[MapReduce] === Map 阶段开始 ({len(self.analysts)} 个维度) ===")
        
        if parallel and len(self.analysts) > 1:
            dimension_results = self._map_parallel(data)
        else:
            dimension_results = self._map_sequential(data)
        
        map_elapsed = time.time() - start_time
        logger.info(f"[MapReduce] Map 阶段完成，耗时 {map_elapsed:.1f}s")
        
        # ========== Reduce 阶段 ==========
        logger.info("[MapReduce] === Reduce 阶段开始 ===")
        reduce_start = time.time()
        
        result = self._reduce(date, dimension_results, overview_data)
        
        reduce_elapsed = time.time() - reduce_start
        total_elapsed = time.time() - start_time
        
        logger.info(f"[MapReduce] Reduce 阶段完成，耗时 {reduce_elapsed:.1f}s")
        logger.info(f"[MapReduce] 总耗时 {total_elapsed:.1f}s")
        
        result.metadata['map_time'] = map_elapsed
        result.metadata['reduce_time'] = reduce_elapsed
        result.metadata['total_time'] = total_elapsed
        
        return result
    
    def _map_parallel(self, data: Dict[str, Any]) -> Dict[str, DimensionAnalysis]:
        """并行执行 Map 阶段"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_analyst = {
                executor.submit(analyst.analyze, data): analyst
                for analyst in self.analysts
            }
            
            for future in as_completed(future_to_analyst):
                analyst = future_to_analyst[future]
                try:
                    result = future.result()
                    results[result.dimension] = result
                except Exception as e:
                    logger.error(f"[Map] {analyst.dimension_name} 执行异常: {e}")
                    results[analyst.dimension_name] = DimensionAnalysis(
                        dimension=analyst.dimension_name,
                        analyst_role=analyst.analyst_role,
                        conclusion=f"分析异常: {str(e)[:50]}",
                        score=50,
                        confidence="低",
                        key_findings=[],
                        risk_signals=[],
                        opportunities=[],
                        raw_analysis="",
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    def _map_sequential(self, data: Dict[str, Any]) -> Dict[str, DimensionAnalysis]:
        """顺序执行 Map 阶段"""
        results = {}
        for analyst in self.analysts:
            result = analyst.analyze(data)
            results[result.dimension] = result
        return results

    
    def _reduce(self, date: str, 
                dimension_results: Dict[str, DimensionAnalysis],
                overview_data: Dict[str, Any]) -> MapReduceResult:
        """
        Reduce 阶段：首席策略分析师综合各维度报告，生成深度复盘
        
        核心思想：
        - 各维度分析师已经输出了专业的文本报告
        - 首席分析师需要阅读、理解、综合这些报告
        - 识别各报告之间的共识和矛盾
        - 形成更高层次的整体判断
        """
        # 收集各维度的分析报告
        dimension_reports = []
        successful_count = 0
        
        for dim_name, analysis in dimension_results.items():
            if analysis.success and analysis.analysis:
                successful_count += 1
                report = f"""
---
### 【{dim_name}】分析报告
**分析师**: {analysis.analyst_role}
**耗时**: {analysis.elapsed_time:.1f}秒

{analysis.analysis}
---
"""
                dimension_reports.append(report)
            else:
                dimension_reports.append(f"\n---\n### 【{dim_name}】\n[分析失败: {analysis.error}]\n---\n")
        
        # 构建 Reduce Prompt
        reduce_prompt = f"""你是一位 A 股首席策略分析师，拥有 20 年市场研究经验。您擅长的事项如下：
        1. 从政策变化中，提前捕捉后续的趋势性机会。
        2. 从资金的变化中，能准确关注到您的同行们实际上在布局哪些方向。
        3. 盘感极强，从各类数据中能够交叉挖掘出最有价值的信息，交易点位把握极其精准。
        4. 深谙各种宏观->股市微观层面布局的潜规则，对各种国家政策实际要达到的目标非常理解。
        5. 充分理解各种量化指标，对各种因子的理解远超众人。

现在，你的 10 位专业分析师已经从不同维度完成了今日市场的分析报告。请仔细阅读每份报告，综合所有信息，撰写一份深度的市场复盘报告。

## 今日日期
{date}

## 各维度分析师报告

以下是 {successful_count} 位分析师提交的专业报告：

{''.join(dimension_reports)}

---

## 你的任务

作为首席策略分析师，你需要：

1. **深度阅读**：仔细阅读每位分析师的报告，理解他们的核心观点
2. **交叉验证**：对比各维度的结论，找出共识点和矛盾点
3. **综合判断**：基于所有信息，形成更高层次的市场判断
4. **明确建议**：给出清晰、可执行的操作建议

## 输出要求

请撰写一份深度复盘报告，格式如下：

# 📊 {date} A股市场深度复盘

## 一、核心结论

（用 3-5 句话总结今日市场，给出明确的多空判断和操作建议）

## 二、市场全景

### 2.1 情绪温度
（综合情绪分析师和涨停板分析师的报告，判断市场情绪状态，分析赚钱效应）

### 2.2 主线脉络
（综合主线分析师和板块埋伏分析师的报告，明确今日主线，判断持续性）

### 2.3 资金态度
（综合资金分析师和潜力股分析师的报告，判断主力资金动向，发现隐藏机会）

### 2.4 指数格局
（综合指数分析师的报告，分析大盘位置和趋势）

### 2.5 消息催化
（综合消息分析师的报告，解读政策和消息面影响）

### 2.6 量化视角
（综合千股千评分析师的报告，从量化角度解读市场状态）

## 三、涨停板生态深度解读

（基于涨停板分析师的报告，深入分析连板梯队、炸板率、溢价率，判断短线情绪周期）

## 四、矛盾与共识

（分析各维度报告之间的矛盾点和共识点，解释为什么会有分歧，如何理解这些分歧）

## 五、风险地图

（综合风险分析师的报告和其他维度的风险信号，按重要性排序列出需要警惕的风险）

## 六、机会雷达

### 6.1 短线机会
（基于涨停板和情绪分析，适合短线交易者）

### 6.2 中线埋伏
（基于板块埋伏和潜力股分析，适合波段操作）

### 6.3 价值洼地
（基于千股千评和资金流向，适合中长期布局）

## 七、明日作战计划

### 7.1 大盘预判
（基于今日分析，预判明日大盘走势）

### 7.2 操作策略
- **激进型**：（适合风险偏好高的投资者）
- **稳健型**：（适合普通投资者）
- **保守型**：（适合风险厌恶型投资者）

### 7.3 重点关注
（明日重点关注的板块和方向）

### 7.4 风险回避
（明日需要回避的方向）

## 八、本报告说明

（说明本报告基于哪些维度的分析，各维度分析的一致性如何，报告的置信度评估）

---

注意：

埋伏逻辑：必须同时满足以下三个条件中的至少两个，胜率才高
1. 够便宜（安全垫）：经历了长时间调整，机构仓位低，散户绝望，估值在历史底部
2. 有催化（导火索）：未来3-6个月内有确定的政策预期、技术突破或产品落地
3. 有反转（基本面）：行业供需格局改善，从"杀估值"转向"杀业绩"结束，进入业绩修复期

您不仅仅是给您的客户一份建议，更重要的是，您也需要通过这个报告，来捕捉第二天的赚钱机会，为您自己的钱包负责。

请直接输出完整的 Markdown 格式报告。"""

        try:
            config = {
                'temperature': 0.7,
                'max_output_tokens': 65535,
            }
            
            logger.info("[Reduce] 首席策略分析师开始综合分析...")
            start_time = time.time()
            
            synthesis = self.llm_caller(reduce_prompt, config)
            
            elapsed = time.time() - start_time
            logger.info(f"[Reduce] 综合分析完成，耗时 {elapsed:.1f}s，{len(synthesis)} 字符")
            
            return MapReduceResult(
                date=date,
                dimension_analyses=dimension_results,
                synthesis=synthesis,
                metadata={
                    'dimension_count': len(dimension_results),
                    'successful_dimensions': successful_count,
                    'reduce_time': elapsed,
                }
            )
            
        except Exception as e:
            logger.error(f"[Reduce] 综合分析失败: {e}")
            
            # 降级：直接拼接各维度报告
            fallback_report = f"# {date} 市场分析（降级模式）\n\n"
            fallback_report += "综合分析失败，以下是各维度分析师的原始报告：\n\n"
            fallback_report += ''.join(dimension_reports)
            
            return MapReduceResult(
                date=date,
                dimension_analyses=dimension_results,
                synthesis=fallback_report,
                metadata={'error': str(e), 'fallback': True}
            )
    

# ============================================================
# 便捷函数
# ============================================================

def create_market_analyzer_with_mapreduce(analyzer) -> Optional[MarketMapReduceAnalyzer]:
    """
    创建 Map-Reduce 分析器
    
    Args:
        analyzer: GeminiAnalyzer 实例（需要有 _call_openai_api 方法）
        
    Returns:
        MarketMapReduceAnalyzer 实例，如果 analyzer 不可用则返回 None
    """
    if not analyzer or not analyzer.is_available():
        logger.warning("[MapReduce] AI 分析器不可用，无法创建 MapReduce 分析器")
        return None
    
    def llm_caller(prompt: str, config: dict) -> str:
        return analyzer._call_openai_api(prompt, config)
    
    return MarketMapReduceAnalyzer(llm_caller)


def overview_to_dict(overview) -> Dict[str, Any]:
    """
    将 MarketOverview 对象转换为字典
    
    Args:
        overview: MarketOverview 实例
        
    Returns:
        字典格式的数据
    """
    result = {
        'date': overview.date,
        'up_count': overview.up_count,
        'down_count': overview.down_count,
        'flat_count': overview.flat_count,
        'limit_up_count': overview.limit_up_count,
        'limit_down_count': overview.limit_down_count,
        'total_amount': overview.total_amount,
        'top_sectors': overview.top_sectors,
        'bottom_sectors': overview.bottom_sectors,
        'margin_balance': overview.margin_balance,
        'margin_buy': overview.margin_buy,
        'short_balance': overview.short_balance,
        'lhb_stocks': overview.lhb_stocks,
        'lhb_net_buy': overview.lhb_net_buy,
        'lhb_org_buy_count': getattr(overview, 'lhb_org_buy_count', 0),  # 新增：机构买入次数
        'lhb_org_sell_count': getattr(overview, 'lhb_org_sell_count', 0),  # 新增：机构卖出次数
        'lhb_org_net_buy': getattr(overview, 'lhb_org_net_buy', 0),  # 新增：机构净买入金额（亿元）
        'lhb_seat_detail': getattr(overview, 'lhb_seat_detail', []),  # 新增：龙虎榜席位明细
        'lhb_org_stats': {  # 新增：机构统计汇总（供 FundFlowAnalyst 使用）
            'buy_count': getattr(overview, 'lhb_org_buy_count', 0),
            'sell_count': getattr(overview, 'lhb_org_sell_count', 0),
            'net_buy': getattr(overview, 'lhb_org_net_buy', 0),
        },
        'block_trade_amount': overview.block_trade_amount,
        'block_trade_premium_ratio': overview.block_trade_premium_ratio,
        'block_trade_discount_ratio': overview.block_trade_discount_ratio,
        'top_concepts': overview.top_concepts,
        'bottom_concepts': overview.bottom_concepts,
        'avg_turnover_rate': overview.avg_turnover_rate,
        'high_turnover_count': overview.high_turnover_count,
        'board_changes': overview.board_changes,
        'board_change_count': overview.board_change_count,
        'big_buy_count': overview.big_buy_count,
        'big_sell_count': overview.big_sell_count,
        'limit_up_seal_count': overview.limit_up_seal_count,
        'limit_down_seal_count': overview.limit_down_seal_count,
        'rocket_launch_count': overview.rocket_launch_count,
        'high_dive_count': overview.high_dive_count,
        'caixin_news': overview.caixin_news,
        # 涨停板生态数据
        'zt_pool': overview.zt_pool,
        'zt_pool_count': overview.zt_pool_count,
        'zt_total_amount': overview.zt_total_amount,
        'zt_avg_turnover': overview.zt_avg_turnover,
        'zt_first_board_count': overview.zt_first_board_count,
        'zt_continuous_count': overview.zt_continuous_count,
        'zt_max_continuous': overview.zt_max_continuous,
        # 昨日涨停今日表现
        'previous_zt_pool': overview.previous_zt_pool,
        'previous_zt_count': overview.previous_zt_count,
        'previous_zt_avg_change': overview.previous_zt_avg_change,
        'previous_zt_up_count': overview.previous_zt_up_count,
        'previous_zt_down_count': overview.previous_zt_down_count,
        # 强势股池
        'strong_pool': overview.strong_pool,
        'strong_pool_count': overview.strong_pool_count,
        'strong_new_high_count': getattr(overview, 'strong_new_high_count', 0),
        'strong_multi_zt_count': getattr(overview, 'strong_multi_zt_count', 0),
        # 炸板股池
        'zb_pool': overview.zb_pool,
        'zb_pool_count': overview.zb_pool_count,
        'zb_total_count': overview.zb_total_count,
        'zb_rate': overview.zb_rate,
        # 跌停股池
        'dt_pool': overview.dt_pool,
        'dt_pool_count': overview.dt_pool_count,
        'dt_continuous_count': overview.dt_continuous_count,
        # 千股千评数据
        'comment_avg_score': overview.comment_avg_score,
        'comment_high_score_count': overview.comment_high_score_count,
        'comment_low_score_count': overview.comment_low_score_count,
        'comment_top_stocks': overview.comment_top_stocks,
        'comment_high_attention': getattr(overview, 'comment_high_attention', []),
        'comment_bottom_stocks': getattr(overview, 'comment_bottom_stocks', []),
        # 潜力股数据
        'hidden_inflow_stocks': overview.hidden_inflow_stocks,
        # 盘口异动分类数据
        'pankou_changes': getattr(overview, 'pankou_changes', {}),
    }
    
    # 处理 indices（可能是对象列表）- 增强：含开盘/最高/最低价
    indices = []
    for idx in overview.indices:
        if hasattr(idx, 'to_dict'):
            indices.append(idx.to_dict())
        elif hasattr(idx, 'name'):
            indices.append({
                'name': idx.name,
                'code': idx.code,
                'current': idx.current,
                'change_pct': idx.change_pct,
                'open': getattr(idx, 'open', 0),  # 新增：开盘价
                'high': getattr(idx, 'high', 0),  # 新增：最高价
                'low': getattr(idx, 'low', 0),    # 新增：最低价
                'amplitude': idx.amplitude,
            })
        else:
            indices.append(idx)
    result['indices'] = indices
    
    # 板块埋伏机会数据
    result['sector_opportunities'] = getattr(overview, 'sector_opportunities', [])
    result['sector_cheap_list'] = getattr(overview, 'sector_cheap_list', [])
    result['sector_catalyst_list'] = getattr(overview, 'sector_catalyst_list', [])
    result['sector_reversal_list'] = getattr(overview, 'sector_reversal_list', [])
    result['sector_recommended'] = getattr(overview, 'sector_recommended', [])
    
    return result


# ============================================================
# 统一入口函数
# ============================================================

def generate_market_review(overview, news: List, analyzer=None, 
                           use_mapreduce: bool = True) -> str:
    """
    生成大盘复盘报告（统一入口函数）
    
    这是 LLM 分析的主入口，market_analyzer.py 只负责数据获取，
    所有 LLM 分析逻辑都在这里处理。
    
    Args:
        overview: MarketOverview 实例（来自 market_analyzer.get_market_overview()）
        news: 新闻列表（来自 market_analyzer.search_market_news()）
        analyzer: AI 分析器实例（需要有 _call_openai_api 方法和 is_available 方法）
        use_mapreduce: 是否使用 Map-Reduce 深度分析模式（默认True）
        
    Returns:
        大盘复盘报告文本（Markdown 格式）
        
    使用示例：
        from market_analyzer import MarketAnalyzer
        from llm_mapreduce import generate_market_review
        from analyzer import GeminiAnalyzer
        
        # 1. 初始化
        ai_analyzer = GeminiAnalyzer()
        market_analyzer = MarketAnalyzer()
        
        # 2. 获取数据（market_analyzer 只负责数据获取）
        overview = market_analyzer.get_market_overview()
        news = market_analyzer.search_market_news()
        
        # 3. 生成报告（llm_mapreduce 负责 LLM 分析）
        report = generate_market_review(overview, news, ai_analyzer)
        print(report)
    """
    # 检查 AI 分析器是否可用
    if not analyzer or not analyzer.is_available():
        logger.warning("[MapReduce] AI分析器未配置或不可用，使用模板生成报告")
        return _generate_template_review(overview, news)
    
    # 使用 Map-Reduce 深度分析模式
    if use_mapreduce:
        try:
            return _generate_mapreduce_review(overview, news, analyzer)
        except Exception as e:
            logger.error(f"[MapReduce] Map-Reduce 分析失败: {e}，降级到传统模式")
            return _generate_traditional_review(overview, news, analyzer)
    
    # 传统单次调用模式
    return _generate_traditional_review(overview, news, analyzer)


def _generate_mapreduce_review(overview, news: List, analyzer) -> str:
    """
    使用 Map-Reduce 架构生成深度复盘报告
    
    流程：
    1. Map 阶段：10个专业分析师并行分析不同维度
    2. Reduce 阶段：首席策略分析师综合各维度结论
    3. 保存各分析师完整报告到文件
    """
    logger.info("[MapReduce] === 启用 Map-Reduce 深度分析模式 ===")
    
    # 创建 Map-Reduce 分析器
    mr_analyzer = create_market_analyzer_with_mapreduce(analyzer)
    
    if not mr_analyzer:
        logger.warning("[MapReduce] Map-Reduce 分析器创建失败，降级到传统模式")
        return _generate_traditional_review(overview, news, analyzer)
    
    # 转换数据格式
    overview_dict = overview_to_dict(overview)
    
    # 执行 Map-Reduce 分析
    result: MapReduceResult = mr_analyzer.analyze(
        overview_data=overview_dict,
        news=news,
        parallel=True  # 并行执行 Map 阶段
    )
    
    # 记录各维度分析结果
    logger.info(f"[MapReduce] Map-Reduce 分析完成:")
    logger.info(f"  - 分析维度: {result.metadata.get('dimension_count', 0)} 个")
    logger.info(f"  - 成功维度: {result.metadata.get('successful_dimensions', 0)} 个")
    logger.info(f"  - 总耗时: {result.metadata.get('total_time', 0):.1f}s")
    
    for dim_name, analysis in result.dimension_analyses.items():
        status = "✓" if analysis.success else "✗"
        logger.info(f"  - [{status}] {dim_name} ({analysis.elapsed_time:.1f}s)")
    
    # 保存各分析师完整报告
    _save_analyst_reports(result)
    
    return result.synthesis


def _save_analyst_reports(result: MapReduceResult) -> Optional[str]:
    """
    保存10个分析师的完整报告到文件
    
    Args:
        result: MapReduceResult 分析结果
        
    Returns:
        保存的文件路径，失败返回 None
    """
    try:
        import os
        
        # 构建报告内容
        report_lines = [
            f"# 📊 分析师报告汇总 ({result.date})",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析维度: {result.metadata.get('dimension_count', 0)} 个",
            f"成功维度: {result.metadata.get('successful_dimensions', 0)} 个",
            f"Map阶段耗时: {result.metadata.get('map_time', 0):.1f}s",
            f"Reduce阶段耗时: {result.metadata.get('reduce_time', 0):.1f}s",
            f"总耗时: {result.metadata.get('total_time', 0):.1f}s",
            "",
            "---",
            "",
        ]
        
        # 添加各分析师报告
        for dim_name, analysis in result.dimension_analyses.items():
            status = "✅" if analysis.success else "❌"
            report_lines.extend([
                f"## {status} 【{dim_name}】- {analysis.analyst_role}",
                "",
                f"耗时: {analysis.elapsed_time:.1f}s",
                "",
            ])
            
            if analysis.success:
                report_lines.append(analysis.analysis)
            else:
                report_lines.append(f"**分析失败**: {analysis.error}")
            
            report_lines.extend(["", "---", ""])
        
        # 添加综合报告
        report_lines.extend([
            "## 🎯 首席策略分析师综合报告",
            "",
            result.synthesis,
        ])
        
        report_content = "\n".join(report_lines)
        
        # 保存到 reports 目录
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        date_str = result.date.replace('-', '')
        filename = f"analyst_reports_{date_str}.md"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[MapReduce] 分析师报告已保存: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"[MapReduce] 保存分析师报告失败: {e}")
        return None


def _generate_traditional_review(overview, news: List, analyzer) -> str:
    """传统单次调用模式生成复盘报告"""
    prompt = _build_traditional_prompt(overview, news)
    
    try:
        generation_config = {'temperature': 0.7, 'max_output_tokens': 65535}
        review = analyzer._call_openai_api(prompt, generation_config)
        
        if review:
            return review
        return _generate_template_review(overview, news)
    except Exception as e:
        logger.error(f"[MapReduce] 传统模式生成失败: {e}")
        return _generate_template_review(overview, news)


def _build_traditional_prompt(overview, news: List) -> str:
    """构建传统模式的 Prompt"""
    # 转换为字典格式
    if hasattr(overview, 'date'):
        overview_dict = overview_to_dict(overview)
    else:
        overview_dict = overview
    
    # 指数行情
    indices_text = ""
    for idx in overview_dict.get('indices', []):
        name = idx.get('name', '')
        current = idx.get('current', 0)
        change_pct = idx.get('change_pct', 0)
        direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "-"
        indices_text += f"- {name}: {current:.2f} ({direction}{abs(change_pct):.2f}%)\n"
    
    # 板块数据
    top_sectors = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" 
                             for s in overview_dict.get('top_sectors', [])[:5]])
    
    # 涨停板数据
    zt_pool_text = ""
    for zt in overview_dict.get('zt_pool', [])[:8]:
        zt_pool_text += f"- {zt.get('name', '')}: {zt.get('continuous', 1)}板, {zt.get('industry', '')}\n"
    
    # 新闻
    news_text = ""
    for i, n in enumerate(news[:6], 1):
        title = n.title[:50] if hasattr(n, 'title') else n.get('title', '')[:50]
        news_text += f"{i}. {title}\n"
    
    return f"""你是一位专业的A股市场分析师，请根据以下数据生成一份大盘复盘报告。

# 今日市场数据（{overview_dict.get('date', '')}）

## 主要指数
{indices_text}

## 市场概况
- 上涨: {overview_dict.get('up_count', 0)} 家
- 下跌: {overview_dict.get('down_count', 0)} 家
- 涨停: {overview_dict.get('limit_up_count', 0)} 只
- 跌停: {overview_dict.get('limit_down_count', 0)} 只
- 成交额: {overview_dict.get('total_amount', 0):.0f} 亿

## 领涨板块
{top_sectors}

## 涨停股
{zt_pool_text}

## 市场新闻
{news_text}

---

请生成一份专业的大盘复盘报告，包含：
1. 市场概述
2. 主线分析
3. 资金动向
4. 风险提示
5. 明日展望

输出纯 Markdown 格式。"""


def _generate_template_review(overview, news: List) -> str:
    """使用模板生成复盘报告（无 LLM 时的备选方案）"""
    # 转换为字典格式
    if hasattr(overview, 'date'):
        overview_dict = overview_to_dict(overview)
    else:
        overview_dict = overview
    
    date = overview_dict.get('date', datetime.now().strftime('%Y-%m-%d'))
    up_count = overview_dict.get('up_count', 0)
    down_count = overview_dict.get('down_count', 0)
    limit_up = overview_dict.get('limit_up_count', 0)
    limit_down = overview_dict.get('limit_down_count', 0)
    total_amount = overview_dict.get('total_amount', 0)
    
    # 判断市场走势
    if up_count > down_count * 1.5:
        market_status = "强势上涨"
        emoji = "🔴"
    elif up_count > down_count:
        market_status = "震荡偏强"
        emoji = "🟠"
    elif down_count > up_count * 1.5:
        market_status = "弱势下跌"
        emoji = "🟢"
    else:
        market_status = "震荡整理"
        emoji = "⚪"
    
    # 指数表现
    indices_text = ""
    for idx in overview_dict.get('indices', [])[:5]:
        name = idx.get('name', '')
        current = idx.get('current', 0)
        change_pct = idx.get('change_pct', 0)
        direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "-"
        indices_text += f"- {name}: {current:.2f} ({direction}{abs(change_pct):.2f}%)\n"
    
    # 板块表现
    top_sectors = overview_dict.get('top_sectors', [])
    sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in top_sectors[:5]])
    
    return f"""# {emoji} {date} A股市场复盘

## 一、市场概述

今日市场{market_status}。

**主要指数表现：**
{indices_text}

**市场数据：**
- 上涨 {up_count} 家 / 下跌 {down_count} 家
- 涨停 {limit_up} 只 / 跌停 {limit_down} 只
- 两市成交 {total_amount:.0f} 亿

## 二、板块表现

**领涨板块：** {sectors_text if sectors_text else "暂无数据"}

## 三、涨停板分析

- 涨停数量: {limit_up} 只
- 首板: {overview_dict.get('zt_first_board_count', 0)} 只
- 连板: {overview_dict.get('zt_continuous_count', 0)} 只
- 最高连板: {overview_dict.get('zt_max_continuous', 0)} 板
- 炸板率: {overview_dict.get('zb_rate', 0):.1f}%

## 四、风险提示

- 跌停数量: {limit_down} 只
- 请注意控制仓位，做好风险管理

---
*本报告由模板自动生成，仅供参考*
"""


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    """
    使用示例：
    
    from analyzer import GeminiAnalyzer
    from market_analyzer import MarketAnalyzer
    from llm_mapreduce import create_market_analyzer_with_mapreduce, overview_to_dict
    
    # 1. 初始化
    ai_analyzer = GeminiAnalyzer()
    market_analyzer = MarketAnalyzer(analyzer=ai_analyzer)
    
    # 2. 获取市场数据
    overview = market_analyzer.get_market_overview()
    news = market_analyzer.search_market_news()
    
    # 3. 创建 MapReduce 分析器
    mr_analyzer = create_market_analyzer_with_mapreduce(ai_analyzer)
    
    # 4. 执行 Map-Reduce 分析
    if mr_analyzer:
        result = mr_analyzer.analyze(
            overview_data=overview_to_dict(overview),
            news=news,
            parallel=True  # 并行执行 Map 阶段
        )
        
        # 5. 输出结果
        print(result.synthesis)  # 完整的深度报告
        
        # 查看各维度分析
        for dim_name, analysis in result.dimension_analyses.items():
            print(f"{dim_name}: {analysis.analyst_role}")
            print(f"  耗时: {analysis.elapsed_time:.1f}s")
            print(f"  成功: {analysis.success}")
    """
    print("LLM Map-Reduce 分析框架")
    print("=" * 50)
    print(f"维度分析师 ({len(MarketMapReduceAnalyzer.DEFAULT_ANALYSTS)} 个):")
    for analyst_cls in MarketMapReduceAnalyzer.DEFAULT_ANALYSTS:
        instance = analyst_cls(lambda p, c: "")  # 创建临时实例获取属性
        print(f"  - {instance.dimension_name}: {instance.analyst_role}")
