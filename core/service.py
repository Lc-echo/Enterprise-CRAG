"""
Enterprise-CRAG 核心业务服务层 (Unified Business Service Layer)
==============================================================
消除 main.py, api.py, app.py 与 ingest.py 之间的重复代码与胶水逻辑：
1. 统一管理问答流程：加载滑动窗口历史 -> 组装初始状态 -> 驱动 LangGraph -> 结果持久化；
2. 统一管理文档入库：文件读取/字节流接收 -> 计算 MD5 -> 主文档入库 -> 父子拓扑分块 -> 向量化入库 -> 外键写回；
3. 统一管理文档生命周期与会话管理；
4. 保证单例基础设施一次初始化，自动注入至 LangGraph 节点。
"""

import os
import uuid
import json
import logging
from typing import List, Optional, Dict, Any

from config.config import config
from core.database import DatabaseManager
from core.document_processor import DocumentProcessor
from core.vectorstore import VectorStoreManager
from agent.graph import build_crag_graph
from agent.nodes import set_vector_manager, set_db_manager

logger = logging.getLogger(__name__)


class RAGService:
    """企业级 RAG 业务服务中枢 (面向 API / Web / CLI 的统一接口)"""

    def __init__(
        self,
        custom_config=None,
        db_manager: Optional[DatabaseManager] = None,
        doc_processor: Optional[DocumentProcessor] = None,
        vector_manager: Optional[VectorStoreManager] = None,
        crag_graph=None,
    ):
        self.config = custom_config or config
        self.db_manager = db_manager or DatabaseManager(self.config)
        self.doc_processor = doc_processor or DocumentProcessor(self.config)
        self.vector_manager = vector_manager or VectorStoreManager(self.config)

        # 依赖注入至 Agent 节点环境
        set_vector_manager(self.vector_manager)
        set_db_manager(self.db_manager)

        # 编译 LangGraph 状态机运行时
        self.crag_graph = crag_graph or build_crag_graph()
        logger.info("✅ RAG 统一业务服务已初始化就绪")

    def ingest_document_file(self, file_path: str) -> Dict[str, Any]:
        """
        从本地物理文件路径执行五阶段入库流水线
        """
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return self._execute_ingest(file_bytes=file_bytes, filename=filename, file_path=file_path)

    def ingest_document_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        从内存二进制字节流执行五阶段入库流水线 (供 API 上传与 Web 端使用)
        """
        return self._execute_ingest(file_bytes=file_bytes, filename=filename, file_path=f"uploaded://{filename}")

    def _execute_ingest(self, file_bytes: bytes, filename: str, file_path: str) -> Dict[str, Any]:
        """入库流水线原子实现"""
        if not file_bytes:
            raise ValueError(f"文档内容为空: {filename}")

        file_hash = self.doc_processor.calculate_hash(file_bytes)
        file_size = len(file_bytes)

        # 1. 解析原始文档
        raw_docs = self.doc_processor.load_from_bytes(file_bytes, filename)
        full_text = "\n\n".join([d.page_content for d in raw_docs])
        if not full_text.strip():
            raise ValueError(f"文档未能提取出有效文本: {filename}")

        # 2. 关系数据库主文档落盘 (获取真实自增主键)
        doc_id = self.db_manager.save_document(
            filename=filename,
            file_path=file_path,
            content=full_text,
            file_hash=file_hash,
            file_size=file_size,
        )

        # 3. 两级父子拓扑切分
        parent_docs, child_docs = self.doc_processor.create_parent_child_chunks(
            documents=raw_docs,
            document_id=doc_id,
            filename=filename,
        )

        # 4. 向量库持久化 (架构重构：父文档免向量化，仅子文档做向量化与索引)
        c_vids = self.vector_manager.add_documents(
            child_docs=child_docs,
            document_id=doc_id,
        )

        # 5. 回填父子块外键与向量 ID 映射关系 (父文档直接落盘关系库，权威原文由关系库管理)
        self.db_manager.save_chunks(
            document_id=doc_id,
            parent_docs=parent_docs,
            child_docs=child_docs,
            child_vector_ids=c_vids,
        )

        logger.info("✅ 文档入库成功: %s (Doc ID=%d, 父块=%d, 子块=%d)", filename, doc_id, len(parent_docs), len(child_docs))
        return {
            "status": "success",
            "message": "文档解析与父子分块向量化完成",
            "document_id": doc_id,
            "filename": filename,
            "parent_chunks": len(parent_docs),
            "child_chunks": len(child_docs),
        }

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter_doc_ids: Optional[List[int]] = None,
        history_limit: int = 4,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        统一问答流程：
        1. 校验或生成 session_id；
        2. 提取滑动窗口对话历史；
        3. 初始化状态并驱动 LangGraph 状态机；
        4. 持久化聊天记录与引用证据；
        5. 返回统一格式字典。
        """
        sid = session_id or str(uuid.uuid4())

        # 1. 提取多轮历史上下文
        hist_records = self.db_manager.get_chat_history(sid, limit=history_limit)
        chat_history_str = "\n".join([
            f"用户: {r.user_message}\n助手: {r.assistant_message}"
            for r in hist_records
        ])

        # 2. 组装全局初始状态
        initial_state = {
            "question": question,
            "session_id": sid,
            "chat_history": chat_history_str,
            "filter_doc_ids": filter_doc_ids,
            "parent_context": "",
            "sources": [],
            "web_search_required": False,
            "web_context": "",
            "grade_reasoning": "",
            "generation": "",
            "retry_count": 0,
            "hallucination_detected": False,
            "hallucination_reasoning": "",
            "execution_steps": [],
        }

        # 3. 驱动 LangGraph 状态图
        final_state = self.crag_graph.invoke(initial_state)

        answer = final_state.get("generation", "未能生成有效回答")
        sources = final_state.get("sources", [])
        web_search_used = final_state.get("web_search_required", False)
        retry_count = final_state.get("retry_count", 0)
        execution_steps = final_state.get("execution_steps", [])

        # 4. 持久化至关系型数据库
        if persist:
            doc_ids_str = ",".join(map(str, filter_doc_ids)) if filter_doc_ids else None
            self.db_manager.save_chat_history(
                session_id=sid,
                user_message=question,
                assistant_message=answer,
                document_ids=doc_ids_str,
                used_chunks=json.dumps(sources, ensure_ascii=False),
            )

        return {
            "session_id": sid,
            "question": question,
            "answer": answer,
            "web_search_used": web_search_used,
            "retry_count": retry_count,
            "sources": sources,
            "execution_steps": execution_steps,
            "final_state": final_state,
        }

    def delete_document(self, doc_id: int) -> bool:
        """统一删除文档业务：软删除关系库记录 + 物理清理向量库与 BM25 词表"""
        success = self.db_manager.soft_delete_document(doc_id)
        if success:
            self.vector_manager.delete_document_vectors(doc_id)
        return success
