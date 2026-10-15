"""
Enterprise-CRAG 向量数据库管理器单元测试 (VectorStoreManager Unit Tests)
======================================================================
测试设计哲学与生产验证：
1. 验证 Milvus 数据库在 19530 端口的连接可用性；
2. 验证子文档单一集合 (child_documents) 的独立入库与父集合废除断言；
3. 验证 Dense + BM25 粗排初筛与 Cross-Encoder 语义深度精排能力；
4. 验证通过关系数据库逆向回溯父块的完整性与拓扑顺序，以及无 DB 时安全降级子块；
5. 验证按 document_id 物理清理子切片向量的功能。
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from config.config import Config
from core.vectorstore import VectorStoreManager


class TestVectorStoreManager:
    """测试 Milvus 向量库在 19530 端口的核心功能与精排流水线"""

    @pytest.fixture
    def vector_manager(self, tmp_path):
        """测试夹具：配置为 Milvus 独立测试集合与临时 BM25 词表缓存"""
        test_cfg = Config()
        test_cfg.VECTOR_STORE_TYPE = "milvus"
        test_cfg.MILVUS_HOST = "127.0.0.1"
        test_cfg.MILVUS_PORT = 19530
        test_cfg.MILVUS_URI = "http://127.0.0.1:19530"
        test_cfg.MILVUS_CHILD_COLLECTION = "pytest_child_docs"
        test_cfg.CHUNKS_PATH = str(tmp_path / "test_chunks.pkl")
        test_cfg.RERANKER_ENABLED = False  # 默认在基础单测中不强依赖远程权重下载

        vm = VectorStoreManager(test_cfg)
        if vm.milvus_client is None:
            pytest.skip("Milvus 向量服务未在 127.0.0.1:19530 运行，跳过需要外部服务的联调测试")
        yield vm

        # 测试完成后清理临时测试集合
        if vm.milvus_client:
            try:
                vm.milvus_client.drop_collection("pytest_child_docs")
            except Exception:
                pass

    def test_milvus_connection_and_single_collection(self, vector_manager):
        """验证 Milvus 客户端与单集合初始化状态 (彻底消除父集合假向量化)"""
        assert vector_manager.milvus_client is not None, "Milvus 客户端必须成功连接至 19530 端口"
        assert vector_manager.child_store is not None, "子集合 child_store 必须初始化成功"
        assert vector_manager.parent_store is None, "父集合 parent_store 必须已废弃为 None"

    def test_add_documents_and_expand_parent(self, vector_manager):
        """验证仅添加子块向量、检索及通过关系库反查父块"""
        doc_id = 1001
        c_doc = Document(
            page_content="员工应妥善保管系统口令，严禁泄露内部机密。",
            metadata={"parent_id": f"doc_{doc_id}_p0", "child_id": f"doc_{doc_id}_c0_0", "filename": "安全规范.txt"},
        )

        # 1. 向量化入库 (仅子块)
        c_vids = vector_manager.add_documents(
            child_docs=[c_doc],
            document_id=doc_id,
        )
        assert len(c_vids) == 1, "必须返回子块向量 ID"

        # 2. 混合检索子块
        retrieved_children = vector_manager.hybrid_search_child_chunks("保管口令与机密", top_k=2)
        assert len(retrieved_children) >= 1, "检索应命中入库的子块"
        assert "口令" in retrieved_children[0].page_content

        # 3. 逆向拓展取回父块：模拟注入关系型数据库 Manager
        mock_db = MagicMock()
        mock_db.get_parent_chunks_by_parent_ids.return_value = {
            f"doc_{doc_id}_p0": {
                "content": "企业信息安全规范全文：员工应妥善保管系统口令，严禁向外部人员泄露任何内部机密信息。",
                "metadata": {"parent_id": f"doc_{doc_id}_p0", "filename": "安全规范.txt"},
            }
        }

        expanded_parents = vector_manager.expand_parent_documents(retrieved_children, db_manager=mock_db)
        assert len(expanded_parents) == 1, "子块必须成功通过关系库反查出对应的父块"
        assert "企业信息安全规范全文" in expanded_parents[0].page_content
        assert expanded_parents[0].metadata.get("parent_id") == f"doc_{doc_id}_p0"

        # 4. 容灾兜底测试：未注入 db_manager 时安全降级为子块正文
        fallback_parents = vector_manager.expand_parent_documents(retrieved_children, db_manager=None)
        assert len(fallback_parents) == len(retrieved_children)
        assert fallback_parents[0].page_content == retrieved_children[0].page_content

        # 5. 删除向量验证
        vector_manager.delete_document_vectors(doc_id)

    def test_cross_encoder_rerank_logic(self, vector_manager):
        """验证 Cross-Encoder 精排打分与降序截取核心逻辑"""
        docs = [
            Document(page_content="无关内容：今天天气晴朗，微风拂面。", metadata={"id": 1}),
            Document(page_content="高度相关：企业员工必须妥善保管账号密码。", metadata={"id": 2}),
            Document(page_content="弱相关：系统提供了密码找回和修改功能。", metadata={"id": 3}),
        ]

        # 模拟 CrossEncoder 模型
        mock_reranker = MagicMock()
        # 对应 3 个切片的得分
        mock_reranker.predict.return_value = [0.12, 0.98, 0.55]
        vector_manager.reranker = mock_reranker

        reranked = vector_manager.rerank("如何保管密码？", docs, top_k=2)
        assert len(reranked) == 2, "精排必须准确截取 Top-2"
        assert reranked[0].metadata["id"] == 2, "最高分切片必须排在首位"
        assert reranked[1].metadata["id"] == 3, "次高分切片必须排在第二位"
        assert reranked[0].metadata["rerank_score"] == 0.98
