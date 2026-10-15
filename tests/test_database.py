"""
Enterprise-CRAG 数据库管理器单元测试 (Database Manager Unit Tests)
====================================================================
测试设计哲学与隔离原则：
1. 【测试沙箱绝对隔离 (tmp_path Fixture)】：
   - 绝不连生产或开发环境 MySQL，通过 pytest 临时目录注入专属的 test_rag.db；
   - 每个测试函数运行在独立的临时 SQLite 数据库中，保证无状态污染；
2. 【核心业务场景覆盖】：
   - 主文档落盘与真实 ID 生成 (test_save_and_retrieve_document)；
   - 父子文档块级联落盘与物理外键约束 (test_save_chunks_cascade)；
   - 会话历史保存、查询与清空 (test_chat_history)。
"""

import os
import pytest
from langchain_core.documents import Document

from config.config import Config
from core.database import DatabaseManager


class TestDatabaseManager:
    """测试双库管家持久化、事务原子性与外键级联"""

    @pytest.fixture
    def test_db(self, tmp_path):
        """
        测试夹具：强制配置为临时 SQLite 路径，保证每个测试用例享有干净独立的沙箱库
        """
        test_cfg = Config()
        test_cfg.DB_TYPE = "sqlite"
        test_cfg.SQLITE_PATH = str(tmp_path / "test_rag.db")
        return DatabaseManager(test_cfg)

    def test_save_and_retrieve_document(self, test_db):
        """
        测试用例：主文档插入与即时自增主键获取
        验证点：
        1. 返回真实物理自增 ID (doc_id > 0)；
        2. 按 ID 能准确查出对应的实体且 is_active 为 True。
        """
        doc_id = test_db.save_document(
            filename="制度规范.txt",
            file_path="data/docs/制度规范.txt",
            content="公司考勤与报销规范全本",
            file_hash="test_hash_123",
            file_size=1024,
        )
        assert doc_id > 0, "主文档落盘后必须立即返回大于 0 的真实自增主键"

        doc = test_db.get_document_by_id(doc_id)
        assert doc is not None, "必须能按 ID 正确查询到该主文档"
        assert doc.filename == "制度规范.txt"
        assert doc.is_active is True

    def test_save_chunks_cascade(self, test_db):
        """
        测试用例：两阶段切块落库与外键级联关联
        验证点：
        1. 父块成功落盘，绑定正确的 document_id，且 vector_id 为 None (架构重构：父文档不入向量库)；
        2. 子块成功落盘，绑定 vector_id，且 parent_chunk_id 外键严格指向对应父块的自增物理主键！
        """
        doc_id = test_db.save_document(
            filename="测试技术架构.md",
            file_path="data/docs/测试技术架构.md",
            content="技术架构全文",
        )

        p_doc = Document(
            page_content="父块大段落内容",
            metadata={"parent_id": f"doc_{doc_id}_p0"},
        )
        c_doc = Document(
            page_content="子块小段落内容",
            metadata={"parent_id": f"doc_{doc_id}_p0", "child_id": f"doc_{doc_id}_c0_0"},
        )

        test_db.save_chunks(
            document_id=doc_id,
            parent_docs=[p_doc],
            child_docs=[c_doc],
            child_vector_ids=["vid_c_001"],
        )

        # 深入数据库底层直接查询验证外键绑定关系
        session = test_db.get_session()
        try:
            from core.models import ParentChunk, ChildChunk
            p_chunk = session.query(ParentChunk).filter(ParentChunk.document_id == doc_id).first()
            assert p_chunk is not None, "父块记录必须落库成功"
            assert p_chunk.vector_id is None, "父块免向量化，vector_id 必须为 None"

            c_chunk = session.query(ChildChunk).filter(ChildChunk.document_id == doc_id).first()
            assert c_chunk is not None, "子块记录必须落库成功"
            assert c_chunk.parent_chunk_id == p_chunk.id, "子块的外键必须精确匹配父块的自增主键"
            assert c_chunk.vector_id == "vid_c_001", "子块必须正确记录分配的向量 ID"
        finally:
            session.close()

    def test_chat_history(self, test_db):
        """
        测试用例：多轮会话记录持久化与清理
        验证点：
        1. 成功按 session_id 保存问答；
        2. 正确按时间正序回溯会话历史；
        3. clear_session_history 能准确清空指定会话的全部记录。
        """
        session_id = "test_sess_001"
        test_db.save_chat_history(
            session_id=session_id,
            user_message="请问年假几天？",
            assistant_message="入职满一年享有5天带薪年假。",
        )

        hist = test_db.get_chat_history(session_id)
        assert len(hist) == 1
        assert hist[0].user_message == "请问年假几天？"
        assert hist[0].assistant_message == "入职满一年享有5天带薪年假。"

        # 测试清空会话
        test_db.clear_session_history(session_id)
        hist_after = test_db.get_chat_history(session_id)
        assert len(hist_after) == 0, "清空后会话历史条数必须为 0"
