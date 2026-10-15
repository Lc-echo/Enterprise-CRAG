"""
Enterprise-CRAG 核心节点函数定义模块 (LangGraph Execution Nodes)
==============================================================
核心架构职责与工程设计：
1. 【强类型数据契约 (Pydantic Contract)】：
   - GradeResult          : 严格结构化约束文档评估结果 (is_relevant: bool, reasoning: str)；
   - HallucinationResult  : 严格结构化约束事实反思质检结果 (has_hallucination: bool, reasoning: str)；
2. 【五大核心节点工作流 (5 Core Nodes)】：
   - 节点 1: retrieve_node            -> Dense 向量 + BM25 混合召回子块并逆向回溯提取大父块语境；
   - 节点 2: grade_documents_node     -> 评测模型结构化判定，兼具双层异常降级保护网；
   - 节点 3: web_search_node          -> DuckDuckGo 外部公网实时搜索知识补偿 (Corrective 纠错)；
   - 节点 4: generate_node            -> LCEL 管道综合会话历史与多源上下文生成回答；
   - 节点 5: hallucination_check_node -> 独立事实一致性核查与自纠错重试计数递增 (Self-Reflection)；
3. 【依赖注入与单例模式】：
   - 提供 get_vector_manager() 与 set_vector_manager()，支持单元测试 Mock 注入。
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

from config.config import config
from agent.state import GraphState
from agent.prompts import GRADE_PROMPT, GENERATE_PROMPT, HALLUCINATION_PROMPT
from core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Pydantic 结构化输出契约模型 (Structured Output Contracts)
# =============================================================================
class GradeResult(BaseModel):
    """
    文档相关性评分结构化契约
    利用大模型 Function Calling / Structured Output 强制返回标准 JSON 结构，
    避免传统 Prompt 输出长篇大论导致布尔值提取不准的问题。
    """
    is_relevant: bool = Field(description="文档是否包含回答用户问题所需的依据与事实")
    reasoning: str = Field(description="评定理由(一句话简述)")


class HallucinationResult(BaseModel):
    """
    事实一致性与幻觉核查结构化契约
    用于质检员节点核实生成内容是否无中生有。
    """
    has_hallucination: bool = Field(description="回答中是否包含参考资料中不存在的捏造信息")
    reasoning: str = Field(description="质检依据与核实理由(一句话简述)")


# =============================================================================
# 2. 全局共享推理组件 (LLM Instances & Singletons)
# =============================================================================
# 主力回答模型：Temperature=0.3，兼顾语言流畅度与对参考资料的忠实度
llm_generator = ChatOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    model=config.LLM_MODEL_NAME,
    temperature=0.3,
    timeout=60,
    max_retries=3,
)

# 评测与决策模型：Temperature=0，消除随机性，保证结构化评估与事实质检的高稳定性和确定性
llm_evaluator = ChatOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    model=config.EVAL_MODEL_NAME,
    temperature=0,
    timeout=60,
    max_retries=3,
)

# 全局向量存储与数据库管理器单例持有者
_global_vector_manager: Optional[VectorStoreManager] = None
_global_db_manager: Optional[Any] = None


def get_vector_manager() -> VectorStoreManager:
    """获取或延迟懒加载向量存储管理器单例"""
    global _global_vector_manager
    if _global_vector_manager is None:
        _global_vector_manager = VectorStoreManager(config)
    return _global_vector_manager


def set_vector_manager(manager: VectorStoreManager) -> None:
    """支持外部依赖注入 (在单元测试中注入 Mock 对象或隔离实例)"""
    global _global_vector_manager
    _global_vector_manager = manager


def set_db_manager(manager: Any) -> None:
    """支持外部依赖注入数据库管理器"""
    global _global_db_manager
    _global_db_manager = manager


# =============================================================================
# 节点 1: 混合检索与父块回溯 (retrieve_node)
# =============================================================================
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    【节点 1：检索与父子回溯】
    执行流程：
    1. 读取当前用户提问 question 与可选的限定文档 ID 列表 filter_doc_ids；
    2. 调用 VectorStoreManager.hybrid_search_child_chunks() 执行 Dense 向量 + BM25 混合召回；
    3. 调用 expand_parent_documents() 优先从关系库毫秒级反查父块 (若无则降级向量库)；
    4. 提炼结构化 sources 供前端渲染引用来源抽屉；
    5. 返回更新字典，增量更新 GraphState。
    """
    question = state["question"]
    filter_doc_ids = state.get("filter_doc_ids")
    execution_steps = list(state.get("execution_steps", []))
    execution_steps.append("retrieve")

    logger.info("🔍 [节点 1: 检索与回溯] 正在多路混合召回: %s", question)
    vm = get_vector_manager()

    # 1. 混合检索命中高精细度子块 (默认 Top-4)
    child_docs = vm.hybrid_search_child_chunks(
        query=question,
        top_k=config.TOP_K,
        filter_doc_ids=filter_doc_ids,
    )

    # 2. 逆向拓展取回完整语义父块 (传入 db_manager 优先从关系数据库毫秒级查询，消除假向量化依赖)
    parent_docs = vm.expand_parent_documents(child_docs, db_manager=_global_db_manager)

    # 3. 构建上下文与引用溯源信息
    parent_context_str = "\n\n---\n\n".join([d.page_content for d in parent_docs])

    sources = []
    for c_doc in child_docs:
        sources.append({
            "filename": c_doc.metadata.get("filename", "未知文件"),
            "parent_id": c_doc.metadata.get("parent_id", ""),
            "child_id": c_doc.metadata.get("child_id", ""),
            "preview": c_doc.page_content[:150] + "..." if len(c_doc.page_content) > 150 else c_doc.page_content,
        })

    logger.info(
        "   -> 召回子块 %d 个, 拓展父块语境 %d 个, 总上下文长度 %d 字符",
        len(child_docs),
        len(parent_docs),
        len(parent_context_str),
    )

    return {
        "parent_context": parent_context_str,
        "sources": sources,
        "execution_steps": execution_steps,
    }


# =============================================================================
# 节点 2: 文档相关性评分 (grade_documents_node)
# =============================================================================
def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    【节点 2：相关性评分与 Corrective 信号判定】
    执行流程：
    1. 若检索结果为空，直接置 web_search_required=True；
    2. 优先使用 Pydantic 结构化输出模式 (with_structured_output(GradeResult))；
    3. 【双层防御性降级】：若大模型 API 结构化解析异常，自动回退到普通文本调用 + 关键词模糊匹配；
    4. 若最终判定不相关且启用了网络搜索，标记 web_search_required=True，后续将路由至联网补充。
    """
    question = state["question"]
    parent_context = state.get("parent_context", "")
    execution_steps = list(state.get("execution_steps", []))
    execution_steps.append("grade_documents")

    logger.info("🧐 [节点 2: 文档评分] 评估上下文与问题的相关性...")

    # 若检索库为空，直接判定不相关
    if not parent_context.strip():
        logger.warning("   -> 本地检索上下文为空，直接标记需要外部补偿")
        return {
            "web_search_required": True,
            "grade_reasoning": "本地知识库检索未召回任何内容",
            "execution_steps": execution_steps,
        }

    is_relevant = False
    reasoning = ""

    try:
        # 第一层防护：Pydantic 结构化强制解析
        grader_chain = GRADE_PROMPT | llm_evaluator.with_structured_output(GradeResult)
        result: GradeResult = grader_chain.invoke({
            "question": question,
            "context": parent_context[:3000],  # 截取前 3000 字符控制 token 开销
        })
        is_relevant = result.is_relevant
        reasoning = result.reasoning
        logger.info("   -> 结构化评分完成: is_relevant=%s, 理由: %s", is_relevant, reasoning)
    except Exception as e:
        # 第二层防护：文本调用降级模式
        logger.warning("   -> 结构化输出失败 (%s)，回退至字符串匹配降级模式...", e)
        try:
            resp = llm_evaluator.invoke(
                GRADE_PROMPT.format_messages(question=question, context=parent_context[:2000])
            )
            content_lower = resp.content.strip().lower()
            is_relevant = "true" in content_lower or "相关" in resp.content or "yes" in content_lower
            reasoning = f"文本降级解析: {resp.content[:60]}"
        except Exception as fallback_err:
            logger.error("   -> 评分降级仍失败: %s，默认放行", fallback_err)
            is_relevant = True
            reasoning = "异常放行"

    web_search_required = (not is_relevant) and config.ENABLE_WEB_SEARCH
    return {
        "web_search_required": web_search_required,
        "grade_reasoning": reasoning,
        "execution_steps": execution_steps,
    }


# =============================================================================
# 节点 3: 外部网络知识补偿 (web_search_node)
# =============================================================================
def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    【节点 3：外部公网搜索引擎知识补偿 (Corrective RAG 核心)】
    执行流程：
    1. 当本地知识库无法支撑提问时触发；
    2. 调用 DuckDuckGoSearchRun 工具发起实时网络检索；
    3. 获取外部搜索摘要，更新至 web_context 字段，有效填补知识盲区。
    """
    question = state["question"]
    execution_steps = list(state.get("execution_steps", []))
    execution_steps.append("web_search")

    logger.info("🌐 [节点 3: 联网补偿] 本地知识不足，正在调用外部搜索: %s", question)
    web_text = ""

    try:
        search_engine = DuckDuckGoSearchRun()
        search_results = search_engine.invoke(question)
        web_text = f"【DuckDuckGo 实时搜索补偿】:\n{search_results}"
        logger.info("   -> 联网搜索成功，已捕获外部补充信息 (%d 字符)", len(search_results))
    except Exception as e:
        logger.error("   -> 联网搜索工具调用失败: %s", e)
        web_text = "（外部网络搜索未能获取有效数据）"

    return {
        "web_context": web_text,
        "execution_steps": execution_steps,
    }


# =============================================================================
# 节点 4: 基于增强上下文的回答生成 (generate_node)
# =============================================================================
def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    【节点 4：结构化回答综合生成】
    执行流程：
    1. 从 state 中提取历史对话 chat_history、本地展开父块 parent_context、外部联网 web_context；
    2. 组装输入槽位，通过 LCEL 链 (GENERATE_PROMPT | llm_generator) 调用主力问答模型；
    3. 若为反思重试 (retry_count > 0)，记录重试日志；
    4. 具备异常兜底：若 LLM 调用超时，提供友好提示并返回已召回的相关资料摘要。
    """
    question = state["question"]
    parent_context = state.get("parent_context", "") or "（无本地相关文档）"
    web_context = state.get("web_context", "") or "（未触发联网补偿）"
    chat_history = state.get("chat_history", "") or "（无前序历史）"
    retry_count = state.get("retry_count", 0)

    execution_steps = list(state.get("execution_steps", []))
    execution_steps.append("generate")

    if retry_count > 0:
        logger.info("✍️ [节点 4: 回答生成] 第 %d 次触发自纠错重新生成...", retry_count + 1)
    else:
        logger.info("✍️ [节点 4: 回答生成] 正在综合上下文撰写回答...")

    try:
        generate_chain = GENERATE_PROMPT | llm_generator
        response = generate_chain.invoke({
            "question": question,
            "parent_context": parent_context,
            "web_context": web_context,
            "chat_history": chat_history,
        })
        answer = response.content
    except Exception as e:
        logger.error("LLM 回答生成调用失败 (%s)，启用降级友好提示", e)
        answer = f"根据当前检索到的相关资料：\n{parent_context[:300]}...\n\n（提示：大模型网络调用暂时受阻: {e}）"

    return {
        "generation": answer,
        "retry_count": retry_count,
        "execution_steps": execution_steps,
    }


# =============================================================================
# 节点 5: 事实核验与反思闭环 (hallucination_check_node)
# =============================================================================
def hallucination_check_node(state: GraphState) -> Dict[str, Any]:
    """
    【节点 5：大模型事实核查质检员 (Self-Reflection Loop 核心)】
    执行流程：
    1. 合并全部输入资料 all_context = parent_context + web_context；
    2. 若无上下文输入，判定为无幻觉放行；
    3. 评测模型以 Pydantic 结构化输出审阅生成的回答 generation 是否存在无依据编造；
    4. 若发现幻觉 (has_hallucination=True)，设置 hallucination_detected=True，并将 retry_count + 1；
    5. 返回更新字典，后续由条件边 decide_after_check 决定打回重试还是熔断交付。
    """
    question = state["question"]
    generation = state.get("generation", "")
    parent_context = state.get("parent_context", "")
    web_context = state.get("web_context", "")
    retry_count = state.get("retry_count", 0)

    execution_steps = list(state.get("execution_steps", []))
    execution_steps.append("hallucination_check")

    all_context = f"{parent_context}\n\n{web_context}".strip()
    if not all_context:
        # 当无参考资料时，模型表达“不知道”或“未找到”不属于幻觉，正常放行
        return {
            "hallucination_detected": False,
            "hallucination_reasoning": "无上下文资料，跳过事实核查",
            "execution_steps": execution_steps,
        }

    logger.info("🔬 [节点 5: 幻觉检测] 核查生成内容的事实一致性...")
    has_hallucination = False
    reasoning = ""

    try:
        checker_chain = HALLUCINATION_PROMPT | llm_evaluator.with_structured_output(HallucinationResult)
        result: HallucinationResult = checker_chain.invoke({
            "context": all_context[:4000],
            "generation": generation,
        })
        has_hallucination = result.has_hallucination
        reasoning = result.reasoning
        logger.info("   -> 质检结论: has_hallucination=%s, 理由: %s", has_hallucination, reasoning)
    except Exception as e:
        logger.warning("   -> 幻觉结构化判定异常 (%s)，回退至文本解析...", e)
        try:
            resp = llm_evaluator.invoke(
                HALLUCINATION_PROMPT.format_messages(
                    context=all_context[:2500],
                    generation=generation,
                )
            )
            content_lower = resp.content.strip().lower()
            has_hallucination = "true" in content_lower or "存在幻觉" in resp.content
            reasoning = f"降级解析: {resp.content[:60]}"
        except Exception:
            has_hallucination = False
            reasoning = "质检异常放行"

    # 若检测出幻觉，重试计数自增 1，否则保持原计数值
    next_retry_count = retry_count + 1 if has_hallucination else retry_count

    return {
        "hallucination_detected": has_hallucination,
        "hallucination_reasoning": reasoning,
        "retry_count": next_retry_count,
        "execution_steps": execution_steps,
    }
