"""
Enterprise-CRAG 状态图工作流编排模块 (LangGraph StateGraph Workflow)
====================================================================
核心架构职责与拓扑流图：
1. 【基于图结构的有向状态转移网络】：
   - 彻底打破单向不可逆的线性 LCEL 管道局限，引入自反思回路 (Loop) 与动态条件分支 (Branching)；
2. 【两大关键决策十字路口 (Conditional Edges)】：
   - 十字路口 1: decide_to_generate -> 文档充足直接生成；文档匮乏触发公网搜索补偿；
   - 十字路口 2: decide_after_check -> 质检通过直接交付；存在幻觉且在限额内打回重写；达上限安全熔断退出；
3. 【工业级安全熔断机制 (Circuit Breaker)】：
   - 通过 retry_count >= MAX_RETRIES (默认3次) 强制拦截无限递归死循环，保护模型 Token 与响应时延。

拓扑流转示意图：
                             ┌───────────────────────────────────────────────┐
                             │       检测到幻觉 且 retry_count < MAX        │
                             ▼                                               │
   [retrieve] ──> [grade_documents] ──> [判断] ──> [generate] ──> [hallucination_check] ─┤
                        │                 │                                  │
                        │ (不相关)         │ (相关)                           │ (通过 / 达上限)
                        ▼                 │                                  ▼
                   [web_search] ──────────┘                                [END]
"""

import logging
from langgraph.graph import StateGraph, END

from agent.state import GraphState
from agent.nodes import (
    retrieve_node,
    grade_documents_node,
    web_search_node,
    generate_node,
    hallucination_check_node,
)
from config.config import config

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 条件路由交警函数 (Conditional Edge Router Functions)
# =============================================================================
def decide_to_generate(state: GraphState) -> str:
    """
    【十字路口 1：文档评估后的分支决策】
    业务逻辑：
    - 读取 state["web_search_required"] 标志位；
    - 若为 True（代表本地检索内容不足或无关），路由调度至 'web_search' 节点启动外部网络搜索；
    - 若为 False（本地知识库已有充分依据），直接路由至 'generate' 节点进入回答生成。
    """
    web_required = state.get("web_search_required", False)
    if web_required:
        logger.info("🚦 [路由调度] 本地文档不充分 -> 引导至 [web_search] 节点进行联网知识补充")
        return "web_search"

    logger.info("🚦 [路由调度] 本地文档充足 -> 直接引导至 [generate] 生成回答")
    return "generate"


def decide_after_check(state: GraphState) -> str:
    """
    【十字路口 2：事实一致性核查后的回路决策 (含防死循环熔断)】
    业务逻辑：
    1. 【质检通过放行】：hallucination_detected 为 False，说明无幻觉捏造，返回 'end' 终止工作流并交付答案；
    2. 【安全熔断防死循环】：若存在幻觉，但 retry_count 已达到或超过 MAX_RETRIES (默认3次)，
       强制返回 'end' 终止重试！防止在刁钻、自相矛盾的问题下大模型陷入永无休止的打回循环；
    3. 【打回重新思考】：存在幻觉且尚在重试配额内，返回 'regenerate'，将状态重新打回 'generate' 节点重写。
    """
    hallucination_detected = state.get("hallucination_detected", False)
    retry_count = state.get("retry_count", 0)

    # 1. 事实完全一致，直接放行交付
    if not hallucination_detected:
        logger.info("🚦 [路由调度] 幻觉质检通过 ✅ -> 顺利结束工作流 [END]")
        return "end"

    # 2. 存在幻觉，但重试次数已触达安全阈值，触发熔断兜底
    if retry_count >= config.MAX_RETRIES:
        logger.warning(
            "🚦 [路由调度] 触发重试安全熔断 (已重试 %d/%d 次) -> 强制终止重试以防死循环 [END]",
            retry_count,
            config.MAX_RETRIES,
        )
        return "end"

    # 3. 存在幻觉且尚有重试机会，打回重新反思生成
    logger.info("🚦 [路由调度] 检测到事实不一致 -> 打回 [generate] 重新反思生成 (第 %d 次重试)", retry_count)
    return "regenerate"


# =============================================================================
# 2. 组装并编译状态图 (Workflow Assembly & Compilation)
# =============================================================================
def build_crag_graph():
    """
    组装并编译 Enterprise-CRAG 智能体工作流运行时：
    步骤：
    1. 实例化 StateGraph 并传入全局强类型契约 GraphState；
    2. 注册 5 大核心执行节点；
    3. 设置工作流起点 entry_point 为 'retrieve'；
    4. 添加确定性固定连线（Fixed Edges）；
    5. 配置条件分支连线（Conditional Edges）；
    6. 执行 workflow.compile() 编译为高性能图运行时对象。
    """
    workflow = StateGraph(GraphState)

    # 1. 注册核心节点 (节点名 -> 节点可调用函数)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # 2. 配置工作流全局入口 (每次 invoke 从 retrieve 开始)
    workflow.set_entry_point("retrieve")

    # 3. 确定性固定转移边
    # retrieve 执行完毕后，无条件进入 grade_documents 评分
    workflow.add_edge("retrieve", "grade_documents")
    # web_search 联网补充完成后，无条件汇合至 generate 节点
    workflow.add_edge("web_search", "generate")
    # generate 生成回答完毕后，无条件送入 hallucination_check 质检
    workflow.add_edge("generate", "hallucination_check")

    # 4. 条件分支路由边
    # 评分节点出口：根据 decide_to_generate 返回的字符串分流
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    # 幻觉核查出口：根据 decide_after_check 返回的字符串决定交付还是打回重写
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_check,
        {
            "end": END,
            "regenerate": "generate",
        },
    )

    logger.info("✅ Enterprise-CRAG 状态图组装完成，编译运行时...")
    return workflow.compile()
