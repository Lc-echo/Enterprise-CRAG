"""
代码精读问题清单回归测试 (Review Fixes Regression Tests)
========================================================
逐条锁定精读发现的缺陷，防止未来重构回退：
1. P1-4 : 空上下文分支必须遵守 ENABLE_WEB_SEARCH 合规开关；
2. P0-3 : 反思重生成时生成提示词必须携带上一轮答案与质检修正意见；
3. P1-6 : 父块反查必须过滤软删除 (is_active=False) 文档；
4. P0-1 : BM25 中文分词后中文查询必须非零命中；
5. P1-5 : 文本降级路径否定优先解析 (『不相关』『不存在幻觉』不得误判)；
6. P0-2 : 重复内容 MD5 查重幂等入库。

全部零真实 LLM / 零真实向量库依赖，可离线运行。
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from config.config import Config
from core.database import DatabaseManager
from core.document_processor import DocumentProcessor
from core.service import RAGService
from core.vectorstore import cjk_aware_tokenizer
import agent.nodes as nodes


# =============================================================================
# P1-4: 空上下文分支遵守联网开关
# =============================================================================
class TestWebSearchGate:
    def test_empty_context_respects_disabled_switch(self, monkeypatch):
        """ENABLE_WEB_SEARCH=False 时，空上下文也不得触发联网搜索标记"""
        monkeypatch.setattr(nodes.config, "ENABLE_WEB_SEARCH", False)
        state = {"question": "测试", "parent_context": "", "execution_steps": []}
        result = nodes.grade_documents_node(state)
        assert result["web_search_required"] is False

    def test_empty_context_respects_enabled_switch(self, monkeypatch):
        """ENABLE_WEB_SEARCH=True 时，空上下文照常触发联网补偿"""
        monkeypatch.setattr(nodes.config, "ENABLE_WEB_SEARCH", True)
        state = {"question": "测试", "parent_context": "", "execution_steps": []}
        result = nodes.grade_documents_node(state)
        assert result["web_search_required"] is True

    def test_web_search_node_hard_guard(self, monkeypatch):
        """web_search_node 入口兜底：开关关闭时绝不发起外部搜索调用"""
        monkeypatch.setattr(nodes.config, "ENABLE_WEB_SEARCH", False)
        search_mock = MagicMock()
        monkeypatch.setattr(nodes, "DuckDuckGoSearchRun", MagicMock(return_value=search_mock))
        result = nodes.web_search_node({"question": "测试", "execution_steps": []})
        search_mock.invoke.assert_not_called()
        assert "禁用外部联网搜索" in result["web_context"]


# =============================================================================
# P0-3: 反思重生成携带质检反馈
# =============================================================================
class TestReflectionFeedback:
    @staticmethod
    def _mock_llm(monkeypatch):
        """用 RunnableLambda 替换生成 LLM，捕获链式 invoke 的最终入参"""
        captured = {}

        def _capture(prompt_value):
            captured["text"] = prompt_value.to_string()
            resp = MagicMock()
            resp.content = "修正后的回答"
            return resp

        monkeypatch.setattr(nodes, "llm_generator", RunnableLambda(_capture))
        return captured

    def test_regenerate_injects_revision_feedback(self, monkeypatch):
        """retry_count>0 时，提示词必须包含上一轮答案与质检理由"""
        captured = self._mock_llm(monkeypatch)
        state = {
            "question": "年假几天？",
            "parent_context": "年假制度：满一年5天。",
            "web_context": "",
            "chat_history": "",
            "generation": "年假为10天。",           # 上一轮错误草稿
            "hallucination_reasoning": "资料写明5天，回答捏造10天",
            "retry_count": 1,
            "execution_steps": [],
        }
        nodes.generate_node(state)
        prompt_text = captured["text"]
        assert "上一轮回答" in prompt_text
        assert "年假为10天" in prompt_text
        assert "捏造10天" in prompt_text

    def test_first_pass_has_no_revision_block(self, monkeypatch):
        """首轮生成 (retry_count=0) 不注入上一轮反馈"""
        captured = self._mock_llm(monkeypatch)
        state = {
            "question": "年假几天？",
            "parent_context": "年假制度：满一年5天。",
            "web_context": "",
            "chat_history": "",
            "generation": "",
            "hallucination_reasoning": "",
            "retry_count": 0,
            "execution_steps": [],
        }
        nodes.generate_node(state)
        assert "质检发现的修正意见" not in captured["text"]


# =============================================================================
# P1-6: 父块反查过滤软删除文档
# =============================================================================
class TestParentLookupActiveFilter:
    def test_soft_deleted_parent_not_returned(self, tmp_path):
        test_cfg = Config()
        test_cfg.DB_TYPE = "sqlite"
        test_cfg.SQLITE_PATH = str(tmp_path / "review_parent.db")
        db = DatabaseManager(test_cfg)

        parent_doc = Document(
            page_content="完整父块语境",
            metadata={"parent_id": "doc_1_p0", "document_id": "1"},
        )
        child_doc = Document(
            page_content="子块片段",
            metadata={"parent_id": "doc_1_p0", "child_id": "doc_1_c0_0"},
        )
        doc_id = db.save_document(filename="t.txt", file_path="t.txt", content="完整父块语境")
        db.save_chunks(document_id=doc_id, parent_docs=[parent_doc], child_docs=[child_doc])

        # 删除前可以反查到父块
        result = db.get_parent_chunks_by_parent_ids(["doc_1_p0"])
        assert "doc_1_p0" in result

        # 软删除后父块不得再进入生成上下文
        assert db.soft_delete_document(doc_id) is True
        result = db.get_parent_chunks_by_parent_ids(["doc_1_p0"])
        assert "doc_1_p0" not in result


# =============================================================================
# P0-1: BM25 中文分词有效
# =============================================================================
class TestBM25ChineseTokenization:
    def test_cjk_bigram_tokenizer(self):
        assert cjk_aware_tokenizer("公司的年假制度") == ["公司", "司的", "的年", "年假", "假制", "制度"]
        assert cjk_aware_tokenizer("BGE embedding 模型") == ["bge", "embedding", "模型"]

    def test_chinese_query_hits_corpus(self):
        """中文查询在合理规模语料上必须命中目标文档 (BM25 idf 需词项出现在少数文档中)"""
        from langchain_community.retrievers import BM25Retriever

        docs = [
            Document(page_content="公司年假制度：入职满一年享有5天带薪年假。"),
            Document(page_content="采购流程：采购申请需部门主管审批后提交财务。"),
            Document(page_content="报销制度：差旅费需在出行后7个工作日内提交票据。"),
            Document(page_content="考勤规范：员工每日上下班需进行指纹打卡。"),
            Document(page_content="会议室管理：会议室使用需提前一天在系统预约。"),
            Document(page_content="绩效考核：绩效评分每年6月与12月两次评定。"),
        ]
        retriever = BM25Retriever.from_documents(docs, preprocess_func=cjk_aware_tokenizer)
        retriever.k = 1
        hits = retriever.invoke("年假几天")
        assert len(hits) >= 1
        assert "年假" in hits[0].page_content


# =============================================================================
# P1-5: 文本降级否定优先解析
# =============================================================================
class TestKeywordBooleanParse:
    GRADE_FALSE = [r"不相关", r"无关", r"false", r"not\s+relevant", r"\bno\b"]
    GRADE_TRUE = [r"相关", r"true", r"\byes\b"]
    HALLU_FALSE = [r"不存在幻觉", r"无幻觉", r"没有幻觉", r"未发现幻觉", r"false", r"\bno\b"]
    HALLU_TRUE = [r"存在幻觉", r"幻觉", r"true", r"\byes\b"]

    def test_negated_relevance(self):
        assert nodes._keyword_boolean_parse("不相关", self.GRADE_FALSE, self.GRADE_TRUE, True) is False
        assert nodes._keyword_boolean_parse("与问题无关", self.GRADE_FALSE, self.GRADE_TRUE, True) is False

    def test_positive_relevance(self):
        assert nodes._keyword_boolean_parse("该文档包含相关内容", self.GRADE_FALSE, self.GRADE_TRUE, True) is True
        assert nodes._keyword_boolean_parse("true", self.GRADE_FALSE, self.GRADE_TRUE, True) is True

    def test_negated_hallucination(self):
        """『不存在幻觉』必须解析为无幻觉 (False)，不得被子串误判"""
        assert nodes._keyword_boolean_parse("不存在幻觉", self.HALLU_FALSE, self.HALLU_TRUE, False) is False

    def test_positive_hallucination(self):
        assert nodes._keyword_boolean_parse("回答存在幻觉", self.HALLU_FALSE, self.HALLU_TRUE, False) is True

    def test_parse_failure_uses_default(self):
        assert nodes._keyword_boolean_parse("无法判断", self.GRADE_FALSE, self.GRADE_TRUE, True) is True
        assert nodes._keyword_boolean_parse("无法判断", self.HALLU_FALSE, self.HALLU_TRUE, False) is False


# =============================================================================
# P0-2: MD5 查重幂等入库
# =============================================================================
class TestIngestDedup:
    @pytest.fixture
    def service(self, tmp_path):
        test_cfg = Config()
        test_cfg.DB_TYPE = "sqlite"
        test_cfg.SQLITE_PATH = str(tmp_path / "review_dedup.db")
        test_cfg.CHUNKS_PATH = str(tmp_path / "review_chunks.pkl")

        db_mgr = DatabaseManager(test_cfg)
        mock_vm = MagicMock()
        mock_vm.add_documents.return_value = ["vid_c1"]

        service = RAGService(
            custom_config=test_cfg,
            db_manager=db_mgr,
            doc_processor=DocumentProcessor(test_cfg),
            vector_manager=mock_vm,
            crag_graph=MagicMock(),
        )
        return service

    def test_duplicate_content_is_skipped(self, service):
        """同一内容第二次入库必须命中查重并返回既有 document_id"""
        content = "企业报销制度：差旅费需在出行后7个工作日内提交。".encode("utf-8")
        first = service.ingest_document_bytes(content, "报销制度.txt")
        assert first["status"] == "success"

        second = service.ingest_document_bytes(content, "报销制度-重传.txt")
        assert second["status"] == "skipped_duplicate"
        assert second["document_id"] == first["document_id"]

        # 向量库只应被调用一次 (重复入库不再写向量)
        assert service.vector_manager.add_documents.call_count == 1

    def test_failed_ingest_compensates(self, service, tmp_path):
        """向量库写入失败时，主文档必须被补偿回滚，不留孤儿记录"""
        service.vector_manager.add_documents.side_effect = RuntimeError("simulated vector store outage")
        content = "补偿回滚测试文档内容。".encode("utf-8")
        with pytest.raises(RuntimeError):
            service.ingest_document_bytes(content, "补偿测试.txt")

        remaining = service.db_manager.get_all_documents(active_only=False)
        assert len(remaining) == 0
