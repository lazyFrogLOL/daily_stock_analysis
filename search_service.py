# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 搜索服务模块
===================================

职责：
1. 提供统一的新闻搜索接口
2. 支持 Tavily 和 SerpAPI 两种搜索引擎
3. 多 Key 负载均衡和故障转移
4. 搜索结果缓存和格式化
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from itertools import cycle

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    snippet: str  # 摘要
    url: str
    source: str  # 来源网站
    published_date: Optional[str] = None
    
    def to_text(self) -> str:
        """转换为文本格式"""
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"【{self.source}】{self.title}{date_str}\n{self.snippet}"


@dataclass 
class SearchResponse:
    """搜索响应"""
    query: str
    results: List[SearchResult]
    provider: str  # 使用的搜索引擎
    success: bool = True
    error_message: Optional[str] = None
    search_time: float = 0.0  # 搜索耗时（秒）
    
    def to_context(self, max_results: int = 5) -> str:
        """将搜索结果转换为可用于 AI 分析的上下文"""
        if not self.success or not self.results:
            return f"搜索 '{self.query}' 未找到相关结果。"
        
        lines = [f"【{self.query} 搜索结果】（来源：{self.provider}）"]
        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"\n{i}. {result.to_text()}")
        
        return "\n".join(lines)


class BaseSearchProvider(ABC):
    """搜索引擎基类"""
    
    def __init__(self, api_keys: List[str], name: str):
        """
        初始化搜索引擎
        
        Args:
            api_keys: API Key 列表（支持多个 key 负载均衡）
            name: 搜索引擎名称
        """
        self._api_keys = api_keys
        self._name = name
        self._key_cycle = cycle(api_keys) if api_keys else None
        self._key_usage: Dict[str, int] = {key: 0 for key in api_keys}
        self._key_errors: Dict[str, int] = {key: 0 for key in api_keys}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_available(self) -> bool:
        """检查是否有可用的 API Key"""
        return bool(self._api_keys)
    
    def _get_next_key(self) -> Optional[str]:
        """
        获取下一个可用的 API Key（负载均衡）
        
        策略：轮询 + 跳过错误过多的 key
        """
        if not self._key_cycle:
            return None
        
        # 最多尝试所有 key
        for _ in range(len(self._api_keys)):
            key = next(self._key_cycle)
            # 跳过错误次数过多的 key（超过 3 次）
            if self._key_errors.get(key, 0) < 3:
                return key
        
        # 所有 key 都有问题，重置错误计数并返回第一个
        logger.warning(f"[{self._name}] 所有 API Key 都有错误记录，重置错误计数")
        self._key_errors = {key: 0 for key in self._api_keys}
        return self._api_keys[0] if self._api_keys else None
    
    def _record_success(self, key: str) -> None:
        """记录成功使用"""
        self._key_usage[key] = self._key_usage.get(key, 0) + 1
        # 成功后减少错误计数
        if key in self._key_errors and self._key_errors[key] > 0:
            self._key_errors[key] -= 1
    
    def _record_error(self, key: str) -> None:
        """记录错误"""
        self._key_errors[key] = self._key_errors.get(key, 0) + 1
        logger.warning(f"[{self._name}] API Key {key[:8]}... 错误计数: {self._key_errors[key]}")
    
    @abstractmethod
    def _do_search(self, query: str, api_key: str, max_results: int) -> SearchResponse:
        """执行搜索（子类实现）"""
        pass
    
    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            SearchResponse 对象
        """
        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} 未配置 API Key"
            )
        
        start_time = time.time()
        try:
            response = self._do_search(query, api_key, max_results)
            response.search_time = time.time() - start_time
            
            if response.success:
                self._record_success(api_key)
                logger.info(f"[{self._name}] 搜索 '{query}' 成功，返回 {len(response.results)} 条结果，耗时 {response.search_time:.2f}s")
            else:
                self._record_error(api_key)
            
            return response
            
        except Exception as e:
            self._record_error(api_key)
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=str(e),
                search_time=elapsed
            )


class TavilySearchProvider(BaseSearchProvider):
    """
    Tavily 搜索引擎
    
    特点：
    - 专为 AI/LLM 优化的搜索 API
    - 免费版每月 1000 次请求
    - 返回结构化的搜索结果
    
    文档：https://docs.tavily.com/
    """
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Tavily")
    
    def _do_search(self, query: str, api_key: str, max_results: int) -> SearchResponse:
        """执行 Tavily 搜索"""
        try:
            from tavily import TavilyClient
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="tavily-python 未安装，请运行: pip install tavily-python"
            )
        
        try:
            client = TavilyClient(api_key=api_key)
            
            # 执行搜索（优化：使用advanced深度、限制最近7天）
            response = client.search(
                query=query,
                search_depth="advanced",  # advanced 获取更多结果
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                days=7,  # 只搜索最近7天的内容
            )
            
            # 记录原始响应到日志
            logger.info(f"[Tavily] 搜索完成，query='{query}', 返回 {len(response.get('results', []))} 条结果")
            logger.debug(f"[Tavily] 原始响应: {response}")
            
            # 解析结果
            results = []
            for item in response.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('content', '')[:500],  # 截取前500字
                    url=item.get('url', ''),
                    source=self._extract_domain(item.get('url', '')),
                    published_date=item.get('published_date'),
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )
            
        except Exception as e:
            error_msg = str(e)
            # 检查是否是配额问题
            if 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower():
                error_msg = f"API 配额已用尽: {error_msg}"
            
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except:
            return '未知来源'


class SerpAPISearchProvider(BaseSearchProvider):
    """
    SerpAPI 搜索引擎
    
    特点：
    - 支持 Google、Bing、百度等多种搜索引擎
    - 免费版每月 100 次请求
    - 返回真实的搜索结果
    
    文档：https://serpapi.com/
    """
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "SerpAPI")
    
    def _do_search(self, query: str, api_key: str, max_results: int) -> SearchResponse:
        """执行 SerpAPI 搜索"""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="google-search-results 未安装，请运行: pip install google-search-results"
            )
        
        try:
            # 使用百度搜索（对中文股票新闻更友好）
            params = {
                "engine": "baidu",  # 使用百度搜索
                "q": query,
                "api_key": api_key,
            }
            
            search = GoogleSearch(params)
            response = search.get_dict()
            
            # 记录原始响应到日志
            logger.debug(f"[SerpAPI] 原始响应 keys: {response.keys()}")
            
            # 解析结果
            results = []
            organic_results = response.get('organic_results', [])
            
            for item in organic_results[:max_results]:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('snippet', '')[:500],
                    url=item.get('link', ''),
                    source=item.get('source', self._extract_domain(item.get('link', ''))),
                    published_date=item.get('date'),
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )
            
        except Exception as e:
            error_msg = str(e)
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '') or '未知来源'
        except:
            return '未知来源'


class BochaSearchProvider(BaseSearchProvider):
    """
    博查搜索引擎
    
    特点：
    - 专为AI优化的中文搜索API
    - 结果准确、摘要完整
    - 支持时间范围过滤和AI摘要
    - 兼容Bing Search API格式
    
    文档：https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK
    """
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Bocha")
    
    def _do_search(self, query: str, api_key: str, max_results: int) -> SearchResponse:
        """执行博查搜索"""
        try:
            import requests
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="requests 未安装，请运行: pip install requests"
            )
        
        try:
            # API 端点
            url = "https://api.bocha.cn/v1/web-search"
            
            # 请求头
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 请求参数（严格按照API文档）
            payload = {
                "query": query,
                "freshness": "oneMonth",  # 搜索近一个月，适合捕获财报、公告等信息
                "summary": True,  # 启用AI摘要
                "count": min(max_results, 50)  # 最大50条
            }
            
            # 执行搜索
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            # 检查HTTP状态码
            if response.status_code != 200:
                # 尝试解析错误信息
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        error_data = response.json()
                        error_message = error_data.get('message', response.text)
                    else:
                        error_message = response.text
                except:
                    error_message = response.text
                
                # 根据错误码处理
                if response.status_code == 403:
                    error_msg = f"余额不足: {error_message}"
                elif response.status_code == 401:
                    error_msg = f"API KEY无效: {error_message}"
                elif response.status_code == 400:
                    error_msg = f"请求参数错误: {error_message}"
                elif response.status_code == 429:
                    error_msg = f"请求频率达到限制: {error_message}"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_message}"
                
                logger.warning(f"[Bocha] 搜索失败: {error_msg}")
                
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )
            
            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[Bocha] {error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )
            
            # 检查响应code
            if data.get('code') != 200:
                error_msg = data.get('msg') or f"API返回错误码: {data.get('code')}"
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )
            
            # 记录原始响应到日志
            logger.info(f"[Bocha] 搜索完成，query='{query}'")
            logger.debug(f"[Bocha] 原始响应: {data}")
            
            # 解析搜索结果
            results = []
            web_pages = data.get('data', {}).get('webPages', {})
            value_list = web_pages.get('value', [])
            
            for item in value_list[:max_results]:
                # 优先使用summary（AI摘要），fallback到snippet
                snippet = item.get('summary') or item.get('snippet', '')
                
                # 截取摘要长度
                if snippet:
                    snippet = snippet[:500]
                
                results.append(SearchResult(
                    title=item.get('name', ''),
                    snippet=snippet,
                    url=item.get('url', ''),
                    source=item.get('siteName') or self._extract_domain(item.get('url', '')),
                    published_date=item.get('datePublished'),  # UTC+8格式，无需转换
                ))
            
            logger.info(f"[Bocha] 成功解析 {len(results)} 条结果")
            
            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )
            
        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            logger.error(f"[Bocha] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[Bocha] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[Bocha] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except:
            return '未知来源'


class SearchService:
    """
    搜索服务
    
    功能：
    1. 管理多个搜索引擎
    2. 自动故障转移
    3. 结果聚合和格式化
    """
    
    def __init__(
        self,
        bocha_keys: Optional[List[str]] = None,
        tavily_keys: Optional[List[str]] = None,
        serpapi_keys: Optional[List[str]] = None,
    ):
        """
        初始化搜索服务
        
        Args:
            bocha_keys: 博查搜索 API Key 列表
            tavily_keys: Tavily API Key 列表
            serpapi_keys: SerpAPI Key 列表
        """
        self._providers: List[BaseSearchProvider] = []
        
        # 初始化搜索引擎（按优先级排序）
        # 1. Bocha 优先（中文搜索优化，AI摘要）
        if bocha_keys:
            self._providers.append(BochaSearchProvider(bocha_keys))
            logger.info(f"已配置 Bocha 搜索，共 {len(bocha_keys)} 个 API Key")
        
        # 2. Tavily（免费额度更多，每月 1000 次）
        if tavily_keys:
            self._providers.append(TavilySearchProvider(tavily_keys))
            logger.info(f"已配置 Tavily 搜索，共 {len(tavily_keys)} 个 API Key")
        
        # 3. SerpAPI 作为备选（每月 100 次）
        if serpapi_keys:
            self._providers.append(SerpAPISearchProvider(serpapi_keys))
            logger.info(f"已配置 SerpAPI 搜索，共 {len(serpapi_keys)} 个 API Key")
        
        if not self._providers:
            logger.warning("未配置任何搜索引擎 API Key，新闻搜索功能将不可用")
    
    @property
    def is_available(self) -> bool:
        """检查是否有可用的搜索引擎"""
        return any(p.is_available for p in self._providers)
    
    def search_stock_news(
        self,
        stock_code: str,
        stock_name: str,
        max_results: int = 5,
        focus_keywords: Optional[List[str]] = None,
        custom_query: Optional[str] = None
    ) -> SearchResponse:
        """
        搜索股票相关新闻
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_results: 最大返回结果数
            focus_keywords: 重点关注的关键词列表（会拼接到查询中）
            custom_query: 自定义查询词（如果提供，直接使用此查询）
            
        Returns:
            SearchResponse 对象
        """
        # 如果提供了自定义查询，直接使用
        if custom_query:
            query = custom_query
        else:
            # 默认重点关注关键词（基于交易理念）
            if focus_keywords is None:
                focus_keywords = [
                    "年报预告", "业绩预告", "业绩快报",  # 业绩相关
                    "减持", "增持", "回购",              # 股东动向
                    "机构调研", "机构评级",              # 机构动向
                    "利好", "利空",                      # 消息面
                    "合同", "订单", "中标",              # 业务进展
                ]
            
            # 构建搜索查询
            # 如果 focus_keywords 是完整的搜索短语（如 "A股 大盘 今日 走势分析"），直接使用
            if focus_keywords and len(focus_keywords) > 2:
                # 检查是否是完整的搜索短语（包含空格或多个词）
                first_keyword = focus_keywords[0] if focus_keywords else ""
                if ' ' in first_keyword or len(focus_keywords) >= 4:
                    # 这是一个完整的搜索短语列表，使用第一个作为主查询
                    query = ' '.join(focus_keywords[:5])
                else:
                    # 传统模式：股票名 + 关键词
                    query = f"{stock_name} {stock_code} 股票 最新消息"
            else:
                query = f"{stock_name} {stock_code} 股票 最新消息"
        
        logger.info(f"搜索股票新闻: {stock_name}({stock_code})")
        
        # 依次尝试各个搜索引擎
        for provider in self._providers:
            if not provider.is_available:
                continue
            
            response = provider.search(query, max_results)
            
            if response.success and response.results:
                logger.info(f"使用 {provider.name} 搜索成功")
                return response
            else:
                logger.warning(f"{provider.name} 搜索失败: {response.error_message}，尝试下一个引擎")
        
        # 所有引擎都失败
        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="所有搜索引擎都不可用或搜索失败"
        )
    
    def search_stock_events(
        self,
        stock_code: str,
        stock_name: str,
        event_types: Optional[List[str]] = None
    ) -> SearchResponse:
        """
        搜索股票特定事件（年报预告、减持等）
        
        专门针对交易决策相关的重要事件进行搜索
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            event_types: 事件类型列表
            
        Returns:
            SearchResponse 对象
        """
        if event_types is None:
            event_types = ["年报预告", "减持公告", "业绩快报"]
        
        # 构建针对性查询
        event_query = " OR ".join(event_types)
        query = f"{stock_name} ({event_query})"
        
        logger.info(f"搜索股票事件: {stock_name}({stock_code}) - {event_types}")
        
        # 依次尝试各个搜索引擎
        for provider in self._providers:
            if not provider.is_available:
                continue
            
            response = provider.search(query, max_results=5)
            
            if response.success:
                return response
        
        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="事件搜索失败"
        )
    
    def search_comprehensive_intel(
        self,
        stock_code: str,
        stock_name: str,
        max_searches: int = 3
    ) -> Dict[str, SearchResponse]:
        """
        多维度情报搜索（同时使用多个引擎、多个维度）
        
        搜索维度：
        1. 最新消息 - 近期新闻动态
        2. 风险排查 - 减持、处罚、利空
        3. 业绩预期 - 年报预告、业绩快报
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_searches: 最大搜索次数
            
        Returns:
            {维度名称: SearchResponse} 字典
        """
        results = {}
        search_count = 0
        
        # 定义搜索维度
        search_dimensions = [
            {
                'name': 'latest_news',
                'query': f"{stock_name} {stock_code} 最新 新闻 2026年1月",
                'desc': '最新消息'
            },
            {
                'name': 'risk_check', 
                'query': f"{stock_name} 减持 处罚 利空 风险",
                'desc': '风险排查'
            },
            {
                'name': 'earnings',
                'query': f"{stock_name} 年报预告 业绩预告 业绩快报 2025年报",
                'desc': '业绩预期'
            },
        ]
        
        logger.info(f"开始多维度情报搜索: {stock_name}({stock_code})")
        
        # 轮流使用不同的搜索引擎
        provider_index = 0
        
        for dim in search_dimensions:
            if search_count >= max_searches:
                break
            
            # 选择搜索引擎（轮流使用）
            available_providers = [p for p in self._providers if p.is_available]
            if not available_providers:
                break
            
            provider = available_providers[provider_index % len(available_providers)]
            provider_index += 1
            
            logger.info(f"[情报搜索] {dim['desc']}: 使用 {provider.name}")
            
            response = provider.search(dim['query'], max_results=3)
            results[dim['name']] = response
            search_count += 1
            
            if response.success:
                logger.info(f"[情报搜索] {dim['desc']}: 获取 {len(response.results)} 条结果")
            else:
                logger.warning(f"[情报搜索] {dim['desc']}: 搜索失败 - {response.error_message}")
            
            # 短暂延迟避免请求过快
            time.sleep(0.5)
        
        return results
    
    def format_intel_report(self, intel_results: Dict[str, SearchResponse], stock_name: str) -> str:
        """
        格式化情报搜索结果为报告
        
        Args:
            intel_results: 多维度搜索结果
            stock_name: 股票名称
            
        Returns:
            格式化的情报报告文本
        """
        lines = [f"【{stock_name} 情报搜索结果】"]
        
        # 最新消息
        if 'latest_news' in intel_results:
            resp = intel_results['latest_news']
            lines.append(f"\n📰 最新消息 (来源: {resp.provider}):")
            if resp.success and resp.results:
                for i, r in enumerate(resp.results[:3], 1):
                    date_str = f" [{r.published_date}]" if r.published_date else ""
                    lines.append(f"  {i}. {r.title}{date_str}")
                    lines.append(f"     {r.snippet[:100]}...")
            else:
                lines.append("  未找到相关消息")
        
        # 风险排查
        if 'risk_check' in intel_results:
            resp = intel_results['risk_check']
            lines.append(f"\n⚠️ 风险排查 (来源: {resp.provider}):")
            if resp.success and resp.results:
                for i, r in enumerate(resp.results[:3], 1):
                    lines.append(f"  {i}. {r.title}")
                    lines.append(f"     {r.snippet[:100]}...")
            else:
                lines.append("  未发现明显风险信号")
        
        # 业绩预期
        if 'earnings' in intel_results:
            resp = intel_results['earnings']
            lines.append(f"\n📊 业绩预期 (来源: {resp.provider}):")
            if resp.success and resp.results:
                for i, r in enumerate(resp.results[:3], 1):
                    lines.append(f"  {i}. {r.title}")
                    lines.append(f"     {r.snippet[:100]}...")
            else:
                lines.append("  未找到业绩相关信息")
        
        return "\n".join(lines)
    
    def batch_search(
        self,
        stocks: List[Dict[str, str]],
        max_results_per_stock: int = 3,
        delay_between: float = 1.0
    ) -> Dict[str, SearchResponse]:
        """
        批量搜索多只股票新闻
        
        Args:
            stocks: 股票列表 [{"code": "300389", "name": "艾比森"}, ...]
            max_results_per_stock: 每只股票的最大结果数
            delay_between: 每次搜索之间的延迟（秒）
            
        Returns:
            {股票代码: SearchResponse} 字典
        """
        results = {}
        
        for i, stock in enumerate(stocks):
            if i > 0:
                time.sleep(delay_between)
            
            code = stock.get('code', '')
            name = stock.get('name', '')
            
            response = self.search_stock_news(code, name, max_results_per_stock)
            results[code] = response
        
        return results


# === 便捷函数 ===
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """获取搜索服务单例"""
    global _search_service
    
    if _search_service is None:
        from config import get_config
        config = get_config()
        
        _search_service = SearchService(
            bocha_keys=config.bocha_api_keys,
            tavily_keys=config.tavily_api_keys,
            serpapi_keys=config.serpapi_keys,
        )
    
    return _search_service


def reset_search_service() -> None:
    """重置搜索服务（用于测试）"""
    global _search_service
    _search_service = None


class LLMSearchOptimizer:
    """
    LLM 驱动的智能搜索优化器
    
    功能：
    1. 使用 LLM 生成更精准的搜索关键词
    2. 根据搜索目的（板块分析、个股分析等）定制搜索策略
    3. 对搜索结果进行智能筛选和摘要
    """
    
    def __init__(self, analyzer=None):
        """
        初始化搜索优化器
        
        Args:
            analyzer: AI 分析器实例（GeminiAnalyzer）
        """
        self.analyzer = analyzer
    
    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return self.analyzer is not None and self.analyzer.is_available()
    
    def generate_sector_search_queries(
        self,
        sector_name: str,
        policy_keywords: List[str],
        search_purpose: str = "catalyst"
    ) -> List[str]:
        """
        为板块分析生成智能搜索关键词
        
        Args:
            sector_name: 板块名称（如"银行"、"房地产"）
            policy_keywords: 相关政策关键词
            search_purpose: 搜索目的
                - "catalyst": 寻找催化剂（政策、技术突破）
                - "risk": 风险排查
                - "reversal": 反转信号
            
        Returns:
            优化后的搜索关键词列表
        """
        if not self.is_available():
            # LLM 不可用时，使用默认关键词
            return self._get_default_queries(sector_name, policy_keywords, search_purpose)
        
        try:
            prompt = self._build_query_generation_prompt(sector_name, policy_keywords, search_purpose)
            
            generation_config = {
                'temperature': 0.3,  # 低温度，更精确
                'max_output_tokens': 500,
            }
            
            response = self.analyzer._call_openai_api(prompt, generation_config)
            
            # 解析 LLM 返回的搜索词
            queries = self._parse_query_response(response)
            
            if queries:
                logger.info(f"[LLM搜索优化] 为 {sector_name} 生成 {len(queries)} 个搜索词")
                return queries
            
        except Exception as e:
            logger.warning(f"[LLM搜索优化] 生成搜索词失败: {e}")
        
        # 失败时返回默认关键词
        return self._get_default_queries(sector_name, policy_keywords, search_purpose)
    
    def _build_query_generation_prompt(
        self,
        sector_name: str,
        policy_keywords: List[str],
        search_purpose: str
    ) -> str:
        """构建搜索词生成的 Prompt"""
        
        purpose_desc = {
            "catalyst": "寻找该板块未来3-6个月的催化剂，包括：政策预期、技术突破、产品落地、行业事件等",
            "risk": "排查该板块的风险因素，包括：政策利空、行业困境、估值泡沫、资金流出等",
            "reversal": "寻找该板块的反转信号，包括：资金流入、业绩改善、供需改善、估值修复等"
        }
        
        current_month = datetime.now().strftime('%Y年%m月')
        
        prompt = f"""你是一位专业的A股行业分析师，请为以下板块生成精准的搜索关键词。

## 任务
为 **{sector_name}** 板块生成搜索关键词

## 搜索目的
{purpose_desc.get(search_purpose, purpose_desc['catalyst'])}

## 相关政策关键词（参考）
{', '.join(policy_keywords) if policy_keywords else '无'}

## 当前时间
{current_month}

## 要求
1. 生成 3-5 个搜索关键词/短语
2. 关键词要具体、精准，能搜索到有价值的信息
3. 包含时间限定词（如"2024年"、"最新"、"近期"）
4. 针对中文搜索引擎优化

## 输出格式
请直接输出搜索关键词，每行一个，不要编号，不要解释：

示例输出：
银行 2024年 净息差 企稳
银行股 高股息 险资增持
银行业 化债政策 资产质量
"""
        return prompt
    
    def _parse_query_response(self, response: str) -> List[str]:
        """解析 LLM 返回的搜索词"""
        if not response:
            return []
        
        queries = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # 跳过空行和注释
            if line and not line.startswith('#') and not line.startswith('示例'):
                # 移除可能的编号
                if line[0].isdigit() and '.' in line[:3]:
                    line = line.split('.', 1)[1].strip()
                if line:
                    queries.append(line)
        
        return queries[:5]  # 最多返回5个
    
    def _get_default_queries(
        self,
        sector_name: str,
        policy_keywords: List[str],
        search_purpose: str
    ) -> List[str]:
        """获取默认搜索关键词（LLM 不可用时）"""
        current_year = datetime.now().year
        current_month = datetime.now().strftime('%Y年%m月')
        
        if search_purpose == "catalyst":
            queries = [
                f"{sector_name} {current_year}年 政策 利好",
                f"{sector_name} 行业 最新 动态 {current_month}",
            ]
            if policy_keywords:
                queries.append(f"{sector_name} {policy_keywords[0]} 最新")
        
        elif search_purpose == "risk":
            queries = [
                f"{sector_name} 风险 利空 {current_year}年",
                f"{sector_name} 行业 困境 问题",
            ]
        
        elif search_purpose == "reversal":
            queries = [
                f"{sector_name} 资金流入 机构 {current_month}",
                f"{sector_name} 业绩 改善 复苏",
            ]
        
        else:
            queries = [f"{sector_name} 最新 消息 {current_month}"]
        
        return queries
    
    def summarize_search_results(
        self,
        results: List[SearchResult],
        sector_name: str,
        summary_purpose: str = "catalyst"
    ) -> Optional[str]:
        """
        使用 LLM 对搜索结果进行智能摘要
        
        Args:
            results: 搜索结果列表
            sector_name: 板块名称
            summary_purpose: 摘要目的
            
        Returns:
            智能摘要文本
        """
        if not self.is_available() or not results:
            return None
        
        try:
            # 构建搜索结果文本
            results_text = ""
            for i, r in enumerate(results[:10], 1):
                results_text += f"\n{i}. 【{r.source}】{r.title}\n   {r.snippet[:200]}\n"
            
            purpose_desc = {
                "catalyst": "提取对该板块有利的催化剂信息（政策、技术、事件）",
                "risk": "提取该板块面临的风险和利空因素",
                "reversal": "提取该板块可能反转的信号"
            }
            
            prompt = f"""请分析以下关于 **{sector_name}** 板块的搜索结果，{purpose_desc.get(summary_purpose, '')}。

## 搜索结果
{results_text}

## 要求
1. 提取最重要的 2-3 条信息
2. 用简洁的语言总结
3. 标注信息来源
4. 如果没有有价值的信息，直接说"未发现有价值信息"

## 输出格式
直接输出摘要，不超过 200 字。
"""
            
            generation_config = {
                'temperature': 0.3,
                'max_output_tokens': 300,
            }
            
            summary = self.analyzer._call_openai_api(prompt, generation_config)
            return summary.strip() if summary else None
            
        except Exception as e:
            logger.warning(f"[LLM搜索优化] 摘要生成失败: {e}")
            return None


class SmartSearchService(SearchService):
    """
    智能搜索服务（继承自 SearchService，增加 LLM 优化能力）
    
    功能：
    1. 继承基础搜索能力
    2. 使用 LLM 优化搜索关键词
    3. 对搜索结果进行智能筛选和摘要
    4. 专门针对板块埋伏分析优化
    """
    
    def __init__(
        self,
        bocha_keys: Optional[List[str]] = None,
        tavily_keys: Optional[List[str]] = None,
        serpapi_keys: Optional[List[str]] = None,
        analyzer=None
    ):
        """
        初始化智能搜索服务
        
        Args:
            bocha_keys: 博查搜索 API Key 列表
            tavily_keys: Tavily API Key 列表
            serpapi_keys: SerpAPI Key 列表
            analyzer: AI 分析器实例（用于 LLM 优化）
        """
        super().__init__(bocha_keys, tavily_keys, serpapi_keys)
        self.optimizer = LLMSearchOptimizer(analyzer)
        
        if self.optimizer.is_available():
            logger.info("智能搜索服务已启用 LLM 优化")
    
    def search_sector_catalyst(
        self,
        sector_name: str,
        policy_keywords: Optional[List[str]] = None,
        max_results: int = 5,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        搜索板块催化剂信息
        
        专门用于板块埋伏分析，寻找政策、技术、事件等催化剂
        
        Args:
            sector_name: 板块名称
            policy_keywords: 相关政策关键词
            max_results: 最大结果数
            use_llm: 是否使用 LLM 优化
            
        Returns:
            {
                'queries': 使用的搜索词列表,
                'results': 搜索结果列表,
                'summary': LLM 摘要（如果可用）,
                'success': 是否成功
            }
        """
        policy_keywords = policy_keywords or []
        
        # 生成搜索关键词
        if use_llm and self.optimizer.is_available():
            queries = self.optimizer.generate_sector_search_queries(
                sector_name, policy_keywords, "catalyst"
            )
        else:
            queries = self.optimizer._get_default_queries(
                sector_name, policy_keywords, "catalyst"
            )
        
        logger.info(f"[智能搜索] 板块催化剂搜索: {sector_name}, 关键词: {queries}")
        
        # 执行搜索
        all_results = []
        for query in queries[:3]:  # 最多搜索3次
            response = self._search_with_fallback(query, max_results=3)
            if response.success and response.results:
                all_results.extend(response.results)
            time.sleep(0.3)  # 避免请求过快
        
        # 去重
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # LLM 摘要
        summary = None
        if use_llm and unique_results:
            summary = self.optimizer.summarize_search_results(
                unique_results, sector_name, "catalyst"
            )
        
        return {
            'queries': queries,
            'results': unique_results[:max_results],
            'summary': summary,
            'success': len(unique_results) > 0
        }
    
    def search_sector_risks(
        self,
        sector_name: str,
        max_results: int = 5,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        搜索板块风险信息
        
        Args:
            sector_name: 板块名称
            max_results: 最大结果数
            use_llm: 是否使用 LLM 优化
            
        Returns:
            搜索结果字典
        """
        # 生成搜索关键词
        if use_llm and self.optimizer.is_available():
            queries = self.optimizer.generate_sector_search_queries(
                sector_name, [], "risk"
            )
        else:
            queries = self.optimizer._get_default_queries(sector_name, [], "risk")
        
        logger.info(f"[智能搜索] 板块风险搜索: {sector_name}, 关键词: {queries}")
        
        # 执行搜索
        all_results = []
        for query in queries[:2]:
            response = self._search_with_fallback(query, max_results=3)
            if response.success and response.results:
                all_results.extend(response.results)
            time.sleep(0.3)
        
        # 去重
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # LLM 摘要
        summary = None
        if use_llm and unique_results:
            summary = self.optimizer.summarize_search_results(
                unique_results, sector_name, "risk"
            )
        
        return {
            'queries': queries,
            'results': unique_results[:max_results],
            'summary': summary,
            'success': len(unique_results) > 0
        }
    
    def search_market_policy(
        self,
        focus_areas: Optional[List[str]] = None,
        max_results: int = 10,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        搜索市场政策和宏观信息
        
        用于大盘复盘分析，获取最新的政策动向
        
        Args:
            focus_areas: 重点关注领域（如["房地产", "科技", "金融"]）
            max_results: 最大结果数
            use_llm: 是否使用 LLM 优化
            
        Returns:
            搜索结果字典
        """
        current_month = datetime.now().strftime('%Y年%m月')
        
        # 默认关注领域
        if focus_areas is None:
            focus_areas = ["宏观经济", "货币政策", "产业政策"]
        
        # 生成搜索关键词
        if use_llm and self.optimizer.is_available():
            queries = self._generate_policy_queries_with_llm(focus_areas)
        else:
            queries = [
                f"A股 政策 利好 {current_month}",
                f"央行 货币政策 最新 {current_month}",
                f"产业政策 扶持 {current_month}",
            ]
        
        logger.info(f"[智能搜索] 市场政策搜索, 关键词: {queries}")
        
        # 执行搜索
        all_results = []
        for query in queries[:4]:
            response = self._search_with_fallback(query, max_results=3)
            if response.success and response.results:
                all_results.extend(response.results)
            time.sleep(0.3)
        
        # 去重
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        return {
            'queries': queries,
            'results': unique_results[:max_results],
            'success': len(unique_results) > 0
        }
    
    def _generate_policy_queries_with_llm(self, focus_areas: List[str]) -> List[str]:
        """使用 LLM 生成政策搜索关键词"""
        if not self.optimizer.is_available():
            return []
        
        try:
            current_month = datetime.now().strftime('%Y年%m月')
            
            prompt = f"""你是一位专业的A股市场分析师，请生成搜索关键词来获取最新的市场政策信息。

## 重点关注领域
{', '.join(focus_areas)}

## 当前时间
{current_month}

## 要求
1. 生成 4 个搜索关键词
2. 关键词要能搜索到最新的政策动向、监管信息、宏观经济数据
3. 包含时间限定词

## 输出格式
每行一个关键词，不要编号：
"""
            
            generation_config = {
                'temperature': 0.3,
                'max_output_tokens': 200,
            }
            
            response = self.optimizer.analyzer._call_openai_api(prompt, generation_config)
            return self.optimizer._parse_query_response(response)
            
        except Exception as e:
            logger.warning(f"[智能搜索] 生成政策搜索词失败: {e}")
            return []
    
    def _search_with_fallback(self, query: str, max_results: int = 5) -> SearchResponse:
        """带故障转移的搜索"""
        for provider in self._providers:
            if not provider.is_available:
                continue
            
            response = provider.search(query, max_results)
            if response.success and response.results:
                return response
        
        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="所有搜索引擎都不可用"
        )
    
    def search_sector_comprehensive(
        self,
        sector_name: str,
        policy_keywords: Optional[List[str]] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        板块综合搜索（催化剂 + 风险 + 反转信号）
        
        专门为板块埋伏分析设计的综合搜索
        
        Args:
            sector_name: 板块名称
            policy_keywords: 相关政策关键词
            use_llm: 是否使用 LLM 优化
            
        Returns:
            {
                'catalyst': 催化剂搜索结果,
                'risk': 风险搜索结果,
                'combined_summary': 综合摘要
            }
        """
        policy_keywords = policy_keywords or []
        
        logger.info(f"[智能搜索] 开始板块综合搜索: {sector_name}")
        
        # 1. 搜索催化剂
        catalyst_result = self.search_sector_catalyst(
            sector_name, policy_keywords, max_results=5, use_llm=use_llm
        )
        
        # 2. 搜索风险
        risk_result = self.search_sector_risks(
            sector_name, max_results=3, use_llm=use_llm
        )
        
        # 3. 生成综合摘要
        combined_summary = None
        if use_llm and self.optimizer.is_available():
            combined_summary = self._generate_combined_summary(
                sector_name, catalyst_result, risk_result
            )
        
        return {
            'sector_name': sector_name,
            'catalyst': catalyst_result,
            'risk': risk_result,
            'combined_summary': combined_summary
        }
    
    def _generate_combined_summary(
        self,
        sector_name: str,
        catalyst_result: Dict[str, Any],
        risk_result: Dict[str, Any]
    ) -> Optional[str]:
        """生成综合摘要"""
        if not self.optimizer.is_available():
            return None
        
        try:
            # 构建输入
            catalyst_text = ""
            if catalyst_result.get('results'):
                for r in catalyst_result['results'][:5]:
                    catalyst_text += f"- {r.title}: {r.snippet[:100]}\n"
            
            risk_text = ""
            if risk_result.get('results'):
                for r in risk_result['results'][:3]:
                    risk_text += f"- {r.title}: {r.snippet[:100]}\n"
            
            prompt = f"""请为 **{sector_name}** 板块生成埋伏分析摘要。

## 催化剂信息
{catalyst_text if catalyst_text else '未搜索到相关信息'}

## 风险信息
{risk_text if risk_text else '未搜索到明显风险'}

## 要求
1. 用 2-3 句话总结该板块的埋伏价值
2. 明确指出主要催化剂和风险点
3. 给出是否值得埋伏的初步判断

## 输出格式
直接输出摘要，不超过 150 字。
"""
            
            generation_config = {
                'temperature': 0.3,
                'max_output_tokens': 200,
            }
            
            summary = self.optimizer.analyzer._call_openai_api(prompt, generation_config)
            return summary.strip() if summary else None
            
        except Exception as e:
            logger.warning(f"[智能搜索] 生成综合摘要失败: {e}")
            return None


if __name__ == "__main__":
    # 测试搜索服务
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    )
    
    # 手动测试（需要配置 API Key）
    service = get_search_service()
    
    if service.is_available:
        print("=== 测试股票新闻搜索 ===")
        response = service.search_stock_news("300389", "艾比森")
        print(f"搜索状态: {'成功' if response.success else '失败'}")
        print(f"搜索引擎: {response.provider}")
        print(f"结果数量: {len(response.results)}")
        print(f"耗时: {response.search_time:.2f}s")
        print("\n" + response.to_context())
    else:
        print("未配置搜索引擎 API Key，跳过测试")
