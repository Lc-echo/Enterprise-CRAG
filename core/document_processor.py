"""
Enterprise-CRAG 文档解析与父子两级分块处理器
=============================================
核心架构职责：
1. 【多源异构格式统一加载】：
   - 支持 PDF (保留页码结构)、Word (.docx, 包含 python-docx 容错)、TXT、Markdown 等格式；
   - 支持内存字节流 (load_from_bytes) 与本地物理路径 (load_from_filepath) 双通道输入；
   - 中文字符编码自适应侦测 (UTF-8 -> GBK -> UTF-16 -> Latin1)，彻底消除 Windows 乱码崩溃；
2. 【内容哈希防重去重 (MD5)】：
   - 快速计算文件指纹，避免重复切分与重复入库；
3. 【核心创新算法：父子拓扑分块 (Parent-Child Document Splitting)】：
   - 粗粒度父块 (~1500字): 维护段落逻辑连续性，防止大模型生成时上下文割裂；
   - 细粒度子块 (~400字): 高信息密度聚焦，支撑向量近邻与 BM25 精准召回；
   - 物理语境隔离原则：子块切分严格限定在单个父块内部递归进行，严禁跨越父块边界！
   - 基于 parent_id 构建严格的两级拓扑树。
"""

import os
import hashlib
import tempfile
import logging
from typing import List, Tuple, Union, Optional
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """多模态文档解析与父子拓扑分块器"""

    def __init__(self, custom_config=None):
        """
        初始化文档处理器与两级递归切分器
        :param custom_config: 可选的自定义配置对象
        """
        self.config = custom_config or config

        # =====================================================================
        # 1. 父文档切分器 (Parent Splitter - 大粒度完整语境)
        # =====================================================================
        # 目标：保持大段落、条文规范与上下文逻辑的完整性
        # chunk_size=1500, chunk_overlap=200
        # 优先在双换行、换行或完整句号处切分，绝不随意中断句子
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.PARENT_CHUNK_SIZE,
            chunk_overlap=self.config.PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "."],
            length_function=len,
        )

        # =====================================================================
        # 2. 子文档切分器 (Child Splitter - 小粒度精准聚焦)
        # =====================================================================
        # 目标：抽取高信息密度的语义单元，专用于向量距离计算和 BM25 词频匹配
        # chunk_size=400, chunk_overlap=60
        # 包含更细的分隔符（分号、逗号、空格），确保在长段落中精准提炼核心句群
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHILD_CHUNK_SIZE,
            chunk_overlap=self.config.CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ";", ".", " ", ""],
            length_function=len,
        )

    @staticmethod
    def calculate_hash(content_bytes: bytes) -> str:
        """
        计算文件二进制内容的 MD5 十六进制哈希值
        用于在文件上传时进行指纹核验，防止同一份文档被多次重复导入造成知识库冗余
        """
        hasher = hashlib.md5()
        hasher.update(content_bytes)
        return hasher.hexdigest()

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        从内存二进制字节流中解析文档 (主要供 Web 工作台与 RESTful 上传接口调用)：
        实现机制：
        1. 获取原始文件名后缀 (如 '.pdf')；
        2. 将内存字节流写入系统临时文件 (tempfile)；
        3. 调用 load_from_filepath 复用成熟的解析逻辑；
        4. 在 finally 块中安全删除临时文件，防止磁盘垃圾残留。
        """
        ext = os.path.splitext(filename)[1].lower()
        suffix = ext if ext else ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return self.load_from_filepath(tmp_path, original_filename=filename)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def load_from_filepath(self, file_path: str, original_filename: Optional[str] = None) -> List[Document]:
        """
        从本地物理文件路径解析文档内容：
        支持多格式适配与防御性容错：
        - PDF: 使用 PyPDFLoader，逐页提取文本并保留页码元数据；
        - DOCX: 优先使用 Docx2txtLoader，若底层解析报错自动降级为 python-docx 段落遍历；
        - TXT / MD: 自适应侦测 UTF-8 / GBK / UTF-16 等编码，杜绝 Windows 常见乱码；
        - 统一标准化 metadata["source"] 字段。
        """
        fname = original_filename or os.path.basename(file_path)
        ext = os.path.splitext(fname)[1].lower()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        documents: List[Document] = []

        try:
            # 1. PDF 格式文档解析
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                documents = loader.load()

            # 2. Word 格式文档解析
            elif ext in (".docx", ".doc"):
                try:
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                except Exception:
                    # 容错降级：使用 python-docx 直接读取段落纯文本
                    import docx
                    doc = docx.Document(file_path)
                    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    documents = [Document(page_content=text, metadata={"source": fname})]

            # 3. 纯文本与 Markdown 格式解析 (自适应多编码尝试)
            elif ext in (".txt", ".md", ".json", ".csv"):
                text = ""
                # 依次尝试主流中英文编码
                for enc in ("utf-8", "gbk", "utf-16", "latin1"):
                    try:
                        with open(file_path, "r", encoding=enc) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                if not text:
                    raise ValueError(f"无法使用支持的编码识别文件内容: {fname}")
                documents = [Document(page_content=text, metadata={"source": fname})]

            else:
                raise ValueError(f"不支持的文件类型: {ext} (仅支持 .pdf, .docx, .txt, .md)")

            # 标准化 metadata，统一 source 为展示文件名
            for doc in documents:
                doc.metadata["source"] = fname

            logger.info("成功解析文档 %s，生成 %d 个初始页面/片段", fname, len(documents))
            return documents
        except Exception as e:
            logger.error("解析文件失败 %s: %s", fname, e)
            raise e

    def create_parent_child_chunks(
        self,
        documents: List[Document],
        document_id: int,
        filename: str,
    ) -> Tuple[List[Document], List[Document]]:
        """
        【核心算法：构建父子两级拓扑树 (Parent-Child Topology Creation)】
        ================================================================
        数学与逻辑执行原理：
        1. 【第一级粗粒度切分 (Parent)】：
           - 针对整篇长文档执行 parent_splitter (1500字符)；
           - 为每一个父块分配全局唯一业务 ID：`doc_{document_id}_p{p_idx}`；
           - 注入元数据：parent_id, document_id, doc_type='parent'。

        2. 【第二级细粒度切分 (Child - 物理语境隔离原则)】：
           - 【关键实现细节】：针对每个单独的父块 [p_doc] 调用 child_splitter (400字符)；
           - 为什么不能拿整篇文档重新切子块？因为若拿全文切子块，子块的起止位置会横跨两个父块，
             导致后续“子块找爸爸”时语境不匹配！在父块内部独立切分，确保子块内容严格归属于其父块！
           - 为每个子块分配唯一业务 ID：`doc_{document_id}_c{p_idx}_{c_idx}`；
           - 在子块 metadata 中打上其父块的 `parent_id` 烙印，这是后续混合检索命中后逆向提取大父块的绝对凭据！

        :param documents: 原始解析得到的基础页面/文档列表
        :param document_id: 关系型数据库分配的真实文档自增主键 (如 1)
        :param filename: 原始文件名
        :return: (parent_docs 列表, child_docs 列表)
        """
        # 1. 第一级切分：生成语义完整的父块 (Parent Chunks)
        parent_docs = self.parent_splitter.split_documents(documents)
        child_docs: List[Document] = []

        for p_idx, p_doc in enumerate(parent_docs):
            # 父块唯一业务标识：例如 'doc_1_p0'
            parent_id = f"doc_{document_id}_p{p_idx}"
            p_doc.metadata.update({
                "parent_id": parent_id,
                "document_id": str(document_id),
                "doc_type": "parent",
                "filename": filename,
                "chunk_index": p_idx,
            })

            # 2. 第二级切分：在当前父块物理边界内部独立切分高聚焦子块 (Child Chunks)
            raw_child_chunks = self.child_splitter.split_documents([p_doc])

            for c_idx, c_doc in enumerate(raw_child_chunks):
                # 子块唯一业务标识：例如 'doc_1_c0_0'
                child_id = f"doc_{document_id}_c{p_idx}_{c_idx}"
                c_doc.metadata.update({
                    "parent_id": parent_id,       # 关键线索：子块认爸爸的逆向溯源 ID
                    "child_id": child_id,
                    "document_id": str(document_id),
                    "doc_type": "child",
                    "filename": filename,
                    "parent_index": p_idx,
                    "child_index": c_idx,
                })
                child_docs.append(c_doc)

        logger.info(
            "父子分块完成 [%s]: 文档ID=%d -> 父块=%d, 子块=%d (父块平均长度=%d, 子块平均长度=%d)",
            filename,
            document_id,
            len(parent_docs),
            len(child_docs),
            int(sum(len(d.page_content) for d in parent_docs) / max(len(parent_docs), 1)),
            int(sum(len(d.page_content) for d in child_docs) / max(len(child_docs), 1)),
        )

        return parent_docs, child_docs
