"""
Enterprise-CRAG 状态图与条件路由单元测试 (Graph State & Router Unit Tests)
===========================================================================
测试设计哲学与工程原则：
1. 【零网络依赖与确定性测试 (Deterministic Testing)】：
   - 条件路由交警函数 (decide_to_generate 与 decide_after_check) 属于纯工程逻辑，
     完全无需连接外部大语言模型或向量数据库即可进行毫秒级快速测试；
2. 【严格的边界值分析法 (Boundary Value Analysis)】：
   - 重点验证幻觉反思熔断阈值 MAX_RETRIES 的边界情况：
     * retry_count = MAX_RETRIES - 1 (应继续反思重试 'regenerate')
     * retry_count = MAX_RETRIES (应立刻触发安全熔断 'end')
     * retry_count = MAX_RETRIES + 2 (超出阈值应稳健兜底 'end')；
3. 【高频回归测试防死循环】：
   - 确保未来任何代码重构均不会破坏图状态机的熔断保护网。
"""

import pytest
from config.config import config
from agent.graph import decide_to_generate, decide_after_check


class TestDecideToGenerate:
    """测试相关性评分后的路由分支逻辑 (十字路口 1)"""

    def test_routes_to_web_search_when_docs_not_relevant(self):
        """
        测试用例：当文档评分为不相关 (web_search_required=True) 时，
        必须正确路由调度至 'web_search' 节点进行联网知识补充
        """
        state = {"web_search_required": True}
        assert decide_to_generate(state) == "web_search"

    def test_routes_to_generate_when_docs_relevant(self):
        """
        测试用例：当文档评分为相关 (web_search_required=False) 时，
        必须直接路由至 'generate' 节点进入回答生成
        """
        state = {"web_search_required": False}
        assert decide_to_generate(state) == "generate"


class TestDecideAfterCheck:
    """测试事实质检后的自反思回路与安全熔断机制 (十字路口 2)"""

    def test_ends_when_no_hallucination(self):
        """
        测试用例：当事实质检通过 (hallucination_detected=False) 时，
        无论当前重试了几次，工作流必须立即结束 ('end') 并顺利交付答案
        """
        state = {"hallucination_detected": False, "retry_count": 0}
        assert decide_after_check(state) == "end"

    def test_regenerates_when_hallucination_detected_and_under_limit(self):
        """
        测试用例：当发现事实不一致 (hallucination_detected=True) 且重试次数未超标时，
        必须返回 'regenerate'，将状态重新打回生成节点重写
        """
        state = {"hallucination_detected": True, "retry_count": 1}
        assert decide_after_check(state) == "regenerate"

    def test_ends_when_max_retries_reached(self):
        """
        【关键边界测试】：当重试次数恰好等于 MAX_RETRIES 时，
        即使依然存在幻觉，也必须强制触发熔断并返回 'end'，彻底杜绝无限循环
        """
        state = {
            "hallucination_detected": True,
            "retry_count": config.MAX_RETRIES,
        }
        assert decide_after_check(state) == "end"

    def test_ends_when_retry_count_exceeds_max(self):
        """
        【容灾防御测试】：极端情况下若重试计数超出 MAX_RETRIES，
        必须依然稳健触发熔断返回 'end'
        """
        state = {
            "hallucination_detected": True,
            "retry_count": config.MAX_RETRIES + 2,
        }
        assert decide_after_check(state) == "end"

    def test_regenerate_at_boundary_max_minus_one(self):
        """
        【边界临界点测试】：当重试次数为 MAX_RETRIES - 1 时 (最后一次重试机会)，
        检测到幻觉仍允许返回 'regenerate' 尝试最后一次自纠错
        """
        state = {
            "hallucination_detected": True,
            "retry_count": config.MAX_RETRIES - 1,
        }
        assert decide_after_check(state) == "regenerate"
