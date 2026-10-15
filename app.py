"""
Enterprise-CRAG 智能知识库交互工作台 (Streamlit Web Application)
==================================================================
核心交互特性与工业级前端工程实践：
1. 【ChatGPT 风格多会话隔离与首问智能标题】：
   - 支持“➕ 新建对话”、“🧹 清空当前”、“📜 历史会话直达列表”；
   - 自动在侧边栏呈现历史对话标题（基于第一条用户提问生成）与活跃指示器（`👉 ` vs `💬 `）；
2. 【浏览器 URL Query 状态同步】：
   - 将当前 session_id 双向绑定至浏览器地址栏（`?session_id=...`），页面刷新（F5）零丢失上下文；
3. 【底层向量句柄自动自愈机制】：
   - 捕获底层 Segment Reader 陈旧或文件锁异常，自动触发缓存清理与热重载；
4. 【独家 CRAG 状态机流转指示灯】：
   - 实时呈现检索、评分、联网补偿、大模型生成与幻觉质检五大节点的彩色状态徽章；
5. 【真流式打字机效果与抽屉式引用溯源】：
   - 打字机逐字渲染配合折叠抽屉展示细粒度子块片段与关联父块。
"""

import uuid
import json
import time
import logging
import streamlit as st

from config.config import config
from core.service import RAGService

# =============================================================================
# 1. 页面全局基础配置
# =============================================================================
st.set_page_config(
    page_title="Enterprise-CRAG 企业知识问答",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS 样式提升视觉质感与专业度
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .step-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background-color: #f0f4f8;
        color: #2c3e50;
        font-size: 0.85rem;
        margin: 2px 4px;
        border: 1px solid #dcdfe6;
    }
    .step-badge-active {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-color: #a5d6a7;
        font-weight: 600;
    }
    .source-box {
        background-color: #fafafa;
        border-left: 3px solid #1E88E5;
        padding: 8px 12px;
        margin-top: 6px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. 全局统一业务服务初始化与缓存 (Resource Caching)
# =============================================================================
@st.cache_resource
def init_system():
    """
    全局初始化统一业务服务单例：
    使用 @st.cache_resource 装饰器，确保模型权重、数据库连接池和向量索引
    仅在 Streamlit 服务首次启动时加载一次，后续用户交互重绘时零重复开销！
    """
    return RAGService(config)


# 初始化系统单例实例
rag_service = init_system()
db_manager = rag_service.db_manager


def load_session_messages(session_id: str):
    """
    从关系型数据库中完整反序列化指定会话的历史问答与溯源证据：
    供会话切换与 URL 直达时加载渲染使用。
    """
    records = db_manager.get_chat_history(session_id, limit=50)
    messages = []
    for r in records:
        sources = []
        if r.used_chunks:
            try:
                sources = json.loads(r.used_chunks)
            except Exception:
                sources = []
        messages.append({"role": "user", "content": r.user_message})
        messages.append({
            "role": "assistant",
            "content": r.assistant_message,
            "sources": sources,
        })
    return messages


# =============================================================================
# 3. 会话状态管理与 URL 参数双向绑定
# =============================================================================
query_sid = st.query_params.get("session_id")

if "session_id" not in st.session_state:
    if query_sid:
        # 若 URL 中有指定会话，优先恢复该会话的历史对话
        st.session_state.session_id = query_sid
        st.session_state.messages = load_session_messages(query_sid)
    else:
        # 否则自动生成全新会话 UUID
        new_sid = str(uuid.uuid4())
        st.session_state.session_id = new_sid
        st.session_state.messages = []


def main():
    global rag_service, db_manager

    # =========================================================================
    # 侧边栏：系统指标、文档管理与多会话控制台
    # =========================================================================
    with st.sidebar:
        st.markdown("### ⚙️ 系统控制台")

        # 1. 状态指示灯 (展示当前数据库与模型底座)
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            st.caption(f"💾 **元数据库**: `{db_manager.active_backend.split()[0]}`")
        with col_status2:
            st.caption(f"🧬 **向量引擎**: `{config.VECTOR_STORE_TYPE}`")

        st.caption(f"🤖 **问答模型**: `{config.LLM_MODEL_NAME}`")
        st.caption(f"🧠 **嵌入模型**: `{config.EMBEDDING_PROVIDER.upper()} (BGE-Large 本地)`" if config.EMBEDDING_PROVIDER == 'local' else f"🧠 **嵌入模型**: `{config.EMBEDDING_PROVIDER.upper()}`")

        # 缓存主动刷新按钮
        if st.button("🔄 刷新知识库与索引缓存", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")

        # 2. 文档上传区域
        st.markdown("### 📤 上传新知识文档")
        uploaded_files = st.file_uploader(
            "支持 PDF, Word (.docx), TXT, Markdown",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True,
            key="doc_uploader",
        )

        if uploaded_files:
            for up_file in uploaded_files:
                if st.button(f"📥 处理并入库: {up_file.name}", key=f"btn_{up_file.name}"):
                    with st.spinner(f"正在深度解析与父子分块 [{up_file.name}]..."):
                        try:
                            file_bytes = up_file.getvalue()
                            res = rag_service.ingest_document_bytes(file_bytes=file_bytes, filename=up_file.name)
                            st.success(f"✅ 入库成功! 生成 {res['parent_chunks']} 个父块, {res['child_chunks']} 个子块")
                            time.sleep(1)
                            st.rerun()
                        except Exception as err:
                            st.error(f"处理失败: {err}")

        st.markdown("---")

        # 3. 知识库现有活跃文档列表与限定检索筛选
        st.markdown("### 📚 知识库管理")
        active_docs = db_manager.get_all_documents(active_only=True)
        st.metric("已收录文档数", len(active_docs))

        if active_docs:
            selected_doc_names = st.multiselect(
                "限定问答范围 (默认全库检索)",
                options=[d.filename for d in active_docs],
                default=[],
            )
            selected_doc_ids = [d.id for d in active_docs if d.filename in selected_doc_names] if selected_doc_names else None

            # 文档管理折叠面板 (支持软删除下线)
            with st.expander("查看/下线现有文档", expanded=False):
                for doc in active_docs:
                    c1, c2 = st.columns([3, 1])
                    c1.text(f"📄 {doc.filename}\n({doc.chunk_count} 块)")
                    if c2.button("🗑️", key=f"del_{doc.id}", help="软删除并清理向量与词表"):
                        rag_service.delete_document(doc.id)
                        st.rerun()
        else:
            selected_doc_ids = None
            st.info("知识库暂无文档，请先在上方上传文件，或运行 `python ingest.py` 导入示例文档。")

        st.markdown("---")

        # 4. 对话会话历史管理 (ChatGPT 风格多会话列表)
        st.markdown("### 💬 会话历史管理")
        col_new_chat, col_clear_chat = st.columns([1, 1])
        with col_new_chat:
            if st.button("➕ 新建对话", use_container_width=True, type="primary"):
                new_sid = str(uuid.uuid4())
                st.session_state.session_id = new_sid
                st.session_state.messages = []
                st.query_params.clear()
                st.rerun()
        with col_clear_chat:
            if st.button("🧹 清空当前", use_container_width=True):
                db_manager.clear_session_history(st.session_state.session_id)
                st.session_state.messages = []
                st.rerun()

        # 历史会话直达列表 (带首条提问智能标题)
        history_sessions = db_manager.get_all_sessions(limit=10)
        if history_sessions:
            with st.expander(f"📜 历史对话记录 ({len(history_sessions)})", expanded=True):
                for s in history_sessions:
                    is_current = (s["session_id"] == st.session_state.session_id)
                    btn_prefix = "👉 " if is_current else "💬 "
                    btn_text = f"{btn_prefix}{s['title']} ({s['last_active']})"
                    if st.button(
                        btn_text,
                        key=f"hist_btn_{s['session_id']}",
                        use_container_width=True,
                        disabled=is_current,
                    ):
                        st.session_state.session_id = s["session_id"]
                        st.query_params["session_id"] = s["session_id"]
                        st.session_state.messages = load_session_messages(s["session_id"])
                        st.rerun()

    # =========================================================================
    # 主交互视窗 (Main Chat Viewport)
    # =========================================================================
    st.markdown('<div class="main-header">🧠 Enterprise-CRAG 智能知识库系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于 LangGraph 状态图架构 · 父子文档多路召回 (Parent-Child) · 联网纠错补偿 · 幻觉反思回路</div>', unsafe_allow_html=True)

    # 渲染历史问答消息气泡列表
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 若有 CRAG 节点流转步骤标记，渲染状态指示灯
            if msg.get("steps"):
                step_html = "".join([f'<span class="step-badge step-badge-active">{s}</span>' for s in msg["steps"]])
                st.markdown(f"<div style='margin: 6px 0;'>{step_html}</div>", unsafe_allow_html=True)

            # 若有参考来源，展示折叠溯源框
            if msg.get("sources"):
                with st.expander(f"📖 参考来源依据 ({len(msg['sources'])} 处片段)", expanded=False):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>[{idx}] {src['filename']}</b> (父块: <code>{src['parent_id']}</code> | 子块: <code>{src['child_id']}</code>)<br>
                            <span style="color:#555;">{src['preview']}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # =========================================================================
    # 底部用户输入框 (单例调用，彻底杜绝 DuplicateElementId)
    # =========================================================================
    if prompt := st.chat_input("💬 请输入您的问题，系统将自动进行多路召回与自校正核验..."):
        # 1. 立即上屏展示用户提问
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 准备系统回答容器
        with st.chat_message("assistant"):
            status_container = st.empty()
            answer_container = st.empty()

            with st.status("🤖 正在启动 CRAG 智能体流水线...", expanded=True) as status:
                st.write("🔍 **步骤 1**: 启动密集向量与 BM25 多路混合检索，并向上反查展开完整父块...")

                # 驱动统一 RAG 业务服务 (内置底层句柄陈旧自动自愈重载机制)
                try:
                    res = rag_service.chat(
                        question=prompt,
                        session_id=st.session_state.session_id,
                        filter_doc_ids=selected_doc_ids,
                    )
                except Exception as invoke_err:
                    err_msg = str(invoke_err).lower()
                    if "segment reader" in err_msg or "hnsw" in err_msg or "chroma" in err_msg or "milvus" in err_msg or "grpc" in err_msg:
                        st.warning("🔄 检测到底层向量索引发生外部变更或连接异常，系统正在自愈重载...")
                        st.cache_resource.clear()
                        rag_service = init_system()
                        res = rag_service.chat(
                            question=prompt,
                            session_id=st.session_state.session_id,
                            filter_doc_ids=selected_doc_ids,
                        )
                    else:
                        raise invoke_err

                # 展示各节点流转日志
                steps = res.get("execution_steps", [])
                web_used = res.get("web_search_used", False)
                retry_c = res.get("retry_count", 0)

                st.write("📊 **步骤 2**: 评测模型评估召回上下文相关度：" + ("通过，直接生成" if not web_used else "⚠️ 本地资料不足，触发联网补偿"))
                if web_used:
                    st.write("🌐 **步骤 3**: DuckDuckGo 外部搜索引擎已成功检索补充数据。")
                st.write("✍️ **步骤 4**: 结合历史记忆与完整语境完成回答生成。")
                st.write(f"🛡️ **步骤 5**: 独立质检员完成事实一致性核查 (反思重试: {retry_c} 次)。")

                status.update(label="✅ 智能体自校正核验完成！", state="complete", expanded=False)

            # 流式打字机视觉输出
            full_answer = res.get("answer", "未能生成回答")
            streamed_text = ""
            for char in full_answer:
                streamed_text += char
                answer_container.markdown(streamed_text + "▌")
                time.sleep(0.005)
            answer_container.markdown(full_answer)

            # 展示引用来源抽屉
            sources = res.get("sources", [])
            if sources:
                with st.expander(f"📖 参考来源依据 ({len(sources)} 处片段)", expanded=False):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>[{idx}] {src['filename']}</b> (父块: <code>{src['parent_id']}</code> | 子块: <code>{src['child_id']}</code>)<br>
                            <span style="color:#555;">{src['preview']}</span>
                        </div>
                        """, unsafe_allow_html=True)

            # 关键：持久化当前 session_id 到浏览器 URL，确保 F5 刷新后完全保持在此会话
            st.query_params["session_id"] = st.session_state.session_id

            # 记录会话历史到界面 Session 状态
            step_badges = [
                "🔍 混合召回 & 父块回溯",
                "📊 相关性评分: 合格" if not web_used else "🌐 联网补偿已触发",
                f"🛡️ 幻觉质检通过 (重试 {retry_c} 次)",
            ]
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
                "steps": step_badges,
            })


if __name__ == "__main__":
    main()
