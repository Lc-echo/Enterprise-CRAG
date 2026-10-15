"""
Enterprise-CRAG 文档解析与父子分块单元测试 (Document Processor Unit Tests)
========================================================================
测试目标与断言原则：
1. 【MD5 指纹计算】：验证文件内容哈希算法产出 32 位标准十六进制字符串；
2. 【父子拓扑树完整性断言 (Topology Invariant)】：
   - 验证粗切父块数量与细切子块数量必须大于等于 1；
   - 核心不变量断言：断言每一个子文档块的 metadata["parent_id"] 必须无一遗漏地
     存在于当前父块集合的 parent_id 集合中；
   - 验证 document_id 与 doc_type 元数据正确继承；
3. 【多源真实文档读取验证】：从 data/docs/ 真实加载示例文档，测试解析流程健壮性。
"""

import os
import pytest
from langchain_core.documents import Document

from config.config import config
from core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """测试多模态解析器与两级父子拓扑分块核心算法"""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(config)

    def test_calculate_hash(self, processor):
        """测试 MD5 哈希计算：验证输出为 32 位标准十六进制字符串"""
        data = b"Enterprise-CRAG test data content"
        hash_val = processor.calculate_hash(data)
        assert len(hash_val) == 32
        assert isinstance(hash_val, str)

    def test_create_parent_child_chunks(self, processor):
        """
        【关键测试】：验证父子两级拓扑切分算法的正确性与映射关系不变量
        执行步骤：
        1. 构造一段模拟中文长文档；
        2. 调用 create_parent_child_chunks，传入虚拟 document_id=999；
        3. 验证生成至少 1 个父块和多个子块；
        4. 核心拓扑不变量断言：断言每一个子块的 parent_id 必须精准存在于父块的 ID 集合中！
        """
        # 构造一段长文本（模拟长文档）
        paragraph = "人工智能与大语言模型正在深刻重塑软件工程和企业知识管理模式。\n\n" * 30
        raw_doc = Document(page_content=paragraph, metadata={"source": "test.txt"})

        parent_docs, child_docs = processor.create_parent_child_chunks(
            documents=[raw_doc],
            document_id=999,
            filename="test.txt",
        )

        assert len(parent_docs) >= 1, "必须成功生成至少 1 个大粒度父块"
        assert len(child_docs) >= 1, "必须成功生成细粒度子块"

        # 验证每一个子块都带有正确的 parent_id 映射
        parent_ids = {p.metadata["parent_id"] for p in parent_docs}
        for c in child_docs:
            assert "parent_id" in c.metadata, "子块必须打上 parent_id 归属"
            assert c.metadata["parent_id"] in parent_ids, "子块关联的 parent_id 必须合法有效"
            assert c.metadata["document_id"] == "999", "子块必须继承正确的主文档 ID"
            assert c.metadata["doc_type"] == "child", "子块 doc_type 必须标记为 child"

    def test_load_sample_documents(self, processor):
        """测试从项目自带的 data/docs 目录真实读取并解析文件"""
        docs_dir = config.DOCS_DIR
        if os.path.exists(docs_dir):
            txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
            if txt_files:
                sample_path = os.path.join(docs_dir, txt_files[0])
                docs = processor.load_from_filepath(sample_path)
                assert len(docs) >= 1
                assert len(docs[0].page_content) > 0
