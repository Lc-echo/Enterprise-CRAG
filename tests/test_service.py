"""
Enterprise-CRAG 业务服务层单元测试 (RAGService Unit Tests)
==========================================================
验证点：
1. 服务实例初始化与依赖注入装配；
2. 统一文档字节流解析、切块入库与状态查询；
3. 统一问答执行与会话持久化集成；
4. 文档统一删除（关系库软删除 + 向量与 BM25 词表同步清理）。
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from config.config import Config
from core.database import DatabaseManager
from core.document_processor import DocumentProcessor
from core.service import RAGService


class TestRAGService:
    """测试统一业务服务中枢核心能力"""

    @pytest.fixture
    def test_service(self, tmp_path):
        """测试夹具：注入临时 SQLite 与隔离配置的测试服务"""
        test_cfg = Config()
        test_cfg.DB_TYPE = "sqlite"
        test_cfg.SQLITE_PATH = str(tmp_path / "service_test.db")
        test_cfg.CHUNKS_PATH = str(tmp_path / "service_chunks.pkl")

        db_mgr = DatabaseManager(test_cfg)
        doc_proc = DocumentProcessor(test_cfg)

        # 模拟向量存储管理器，实现完全无网络隔离测试
        mock_vm = MagicMock()
        mock_vm.add_documents.return_value = ["vid_c1"]
        mock_vm.delete_document_vectors.return_value = None

        # 模拟 LangGraph 状态图
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "generation": "根据制度，年假为5天。",
            "sources": [{"filename": "test.txt", "parent_id": "doc_1_p0", "child_id": "doc_1_c0_0", "preview": "年假5天"}],
            "web_search_required": False,
            "retry_count": 0,
            "execution_steps": ["retrieve", "grade_documents", "generate", "hallucination_check"],
        }

        service = RAGService(
            custom_config=test_cfg,
            db_manager=db_mgr,
            doc_processor=doc_proc,
            vector_manager=mock_vm,
            crag_graph=mock_graph,
        )
        return service

    def test_ingest_bytes_and_query(self, test_service):
        """测试内存字节流统一入库"""
        content = "企业年假规章制度：所有正式员工满一年享受5个工作日带薪年假。".encode("utf-8")
        res = test_service.ingest_document_bytes(content, "年假制度.txt")

        assert res["status"] == "success"
        assert res["document_id"] > 0
        assert res["parent_chunks"] >= 1
        assert res["child_chunks"] >= 1

        # 验证数据库中确实落盘
        doc = test_service.db_manager.get_document_by_id(res["document_id"])
        assert doc is not None
        assert doc.filename == "年假制度.txt"

    def test_chat_pipeline_and_persistence(self, test_service):
        """测试统一问答接口与多轮历史持久化"""
        session_id = "test_session_service"
        res = test_service.chat(
            question="年假几天？",
            session_id=session_id,
        )

        assert res["session_id"] == session_id
        assert res["answer"] == "根据制度，年假为5天。"
        assert len(res["sources"]) == 1
        assert "generate" in res["execution_steps"]

        # 验证关系数据库中自动完成问答持久化
        history = test_service.db_manager.get_chat_history(session_id)
        assert len(history) == 1
        assert history[0].user_message == "年假几天？"
        assert history[0].assistant_message == "根据制度，年假为5天。"

    def test_delete_document_integration(self, test_service):
        """测试删除文档的统一联动"""
        content = "即将下线的临时文件内容".encode("utf-8")
        res = test_service.ingest_document_bytes(content, "临时文件.txt")
        doc_id = res["document_id"]

        success = test_service.delete_document(doc_id)
        assert success is True

        # 验证软删除标记
        doc = test_service.db_manager.get_document_by_id(doc_id)
        assert doc.is_active is False
        # 验证调用了向量库删除
        test_service.vector_manager.delete_document_vectors.assert_called_with(doc_id)
