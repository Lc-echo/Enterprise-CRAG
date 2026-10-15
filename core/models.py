"""
Enterprise-CRAG 数据模型模块 (SQLAlchemy ORM Data Models)
=========================================================
架构职责与设计哲学：
1. 关系型数据库底座：维护企业私有知识库的物理拓扑结构与生命周期状态；
2. 彻底解决纯向量数据库缺乏强一致性、外键约束、多表 JOIN、事务审计的致命缺陷；
3. 四表拓扑体系：
   - Document    : 主文档生命周期表（防重复 MD5 哈希、物理路径、软删除标记）；
   - ParentChunk : 父文档块表（1500 字大颗粒度，用于生成时的完整语境还原）；
   - ChildChunk  : 子文档块表（400 字小颗粒度，绑定 vector_id，两库协同纽带）；
   - ChatHistory : 多轮问答历史表（支持滑动窗口记忆与引用证据回溯）；
4. 跨库平滑兼容：完全兼容 MySQL (InnoDB) 与 SQLite，无缝支持高可用降级。
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

# 创建 SQLAlchemy ORM 声明式基类
Base = declarative_base()


class Document(Base):
    """
    主文档元数据表 (Document Master Table)
    -----------------------------------
    记录每一篇原始企业知识文档的生命周期信息。
    支持 MD5 查重去重、大小统计、入库切块总数以及业务软删除机制。
    """
    __tablename__ = "documents"

    # 主键 ID：MySQL / SQLite 自增真实物理 ID (在两阶段入库中首先生成)
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 原始文件名 (如: '2026年企业员工手册.pdf')
    filename = Column(String(255), nullable=False, comment="原始文件名")

    # 文件物理存储路径或原始位置标识
    file_path = Column(String(500), nullable=False, comment="文件物理存储路径或虚拟上传URI")

    # 文件内容的 MD5 哈希值：用于在用户上传时防重，避免重复切分与重复入库
    file_hash = Column(String(64), nullable=True, comment="文件内容MD5哈希值(防重复入库)")

    # 文件字节大小 (单位: 字节)
    file_size = Column(Integer, default=0, comment="文件大小(字节)")

    # 文档全文原始文本备份：方便在不需要重新解析文件的情况下做全文检视或离线重切分
    content = Column(Text, nullable=False, comment="文档全文原始文本")

    # 该文档成功切分生成的有效子块总数
    chunk_count = Column(Integer, default=0, comment="切分生成的有效子块总数")

    # 上传与入库时间戳
    # 【避坑关键点】：必须传递 datetime.now 函数引用本身，而非 datetime.now()！
    # 若带括号，会在 Python 进程启动加载本模块时直接求值并固化，导致后续入库记录时间全相同！
    created_at = Column(DateTime, default=datetime.now, comment="上传与入库时间")

    # 软删除标志位 (Soft Delete)：
    # True = 处于活跃服务状态；False = 业务下线已软删除 (数据仍保留供合规与溯源审计)
    is_active = Column(Boolean, default=True, comment="软删除标志位(True=正常, False=已删除)")

    # ==================== ORM 级联关联关系 ====================
    # 1 对多关联父块：当主文档记录删除时，自动级联删除其名下的所有父文档块 (delete-orphan)
    parent_chunks = relationship(
        "ParentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # 1 对多关联子块：当主文档记录删除时，自动级联删除其名下的所有子文档块
    child_chunks = relationship(
        "ChildChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # 组合索引设计：提升文件名查询、活跃状态筛选与按时间倒序排查的性能
    __table_args__ = (
        Index("idx_doc_filename", "filename"),
        Index("idx_doc_is_active", "is_active"),
        Index("idx_doc_created_at", "created_at"),
    )

    def to_dict(self):
        """将 ORM 实体序列化为字典字典格式，便于 RESTful API 或 Streamlit 前端渲染"""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "is_active": self.is_active,
        }


class ParentChunk(Base):
    """
    父文档块表 (Parent Chunk Table)
    ------------------------------
    颗粒度：粗粒度大块 (默认 ~1500 字符)。
    核心职责：
    1. 保持段落、规章制度条文与论述的完整连续语境；
    2. 检索阶段不直接做高维计算，而是在子块检索命中后，通过 parent_id 反查提取；
    3. 组装为完整大上下文送入 LLM 生成回答，从根本上破解“文本切太小导致语义割裂”的痛点。
    """
    __tablename__ = "parent_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 所属主文档外键：关联 documents.id，级联删除
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # 父块唯一业务标识字符串：命名规则如 'doc_1_p0' (第1篇文档的第0个父块)
    # 作为唯一业务索引，是子块反查父块的关键凭证
    parent_id = Column(String(100), nullable=False, unique=True, comment="父块唯一业务ID, 如 doc_1_p0")

    # 父块完整文本内容
    content = Column(Text, nullable=False, comment="父块大段落完整文本")

    # JSON 格式扩展元数据 (如原始页码、段落类型等)
    json_metadata = Column(Text, nullable=True, comment="JSON格式扩展元数据")

    # 向量数据库中对应的关联 ID (双库同步标记)
    vector_id = Column(String(100), nullable=True, comment="向量数据库中的关联ID")

    created_at = Column(DateTime, default=datetime.now)

    # 双向关联：反向映射到主文档与正向映射下属子块
    document = relationship("Document", back_populates="parent_chunks")
    child_chunks = relationship(
        "ChildChunk",
        back_populates="parent_chunk",
        cascade="all, delete-orphan",
        lazy="select",
    )

    __table_args__ = (
        Index("idx_parent_doc_id", "document_id"),
        Index("idx_parent_chunk_id", "parent_id"),
    )


class ChildChunk(Base):
    """
    子文档块表 (Child Chunk Table)
    -----------------------------
    颗粒度：高信息密度细粒度小块 (默认 ~400 字符)。
    核心职责：
    1. 在单个父块内部独立切分，专用于向量近邻搜索与 BM25 词频匹配；
    2. 解决大块语义稀释导致相似度距离失真的问题，大幅提升 Top-K 检索信噪比；
    3. 字段 vector_id 记录向量库（Chroma）返回的唯一向量 ID，打通两库数据纽带。
    """
    __tablename__ = "child_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 所属主文档外键
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # 所属父文档块外键：通过数据库物理外键与 parent_chunks.id 紧密绑定
    parent_chunk_id = Column(Integer, ForeignKey("parent_chunks.id", ondelete="CASCADE"), nullable=False)

    # 子块唯一业务标识：命名规则如 'doc_1_c0_0' (第1篇文档第0个父块下的第0个子块)
    child_id = Column(String(100), nullable=False, comment="子块唯一业务ID, 如 doc_1_c0_0")

    # 子块切分正文
    content = Column(Text, nullable=False, comment="子块细粒度切分文本")

    # JSON 格式元数据 (附带 parent_id、filename、页码等)
    json_metadata = Column(Text, nullable=True, comment="JSON格式扩展元数据")

    # 向量数据库为该子块分配的向量 ID：
    # 【两库协同核心】：向量检索命中该 ID 后，可在关系库中实现毫秒级溯源
    vector_id = Column(String(100), nullable=True, comment="向量数据库分配的向量ID")

    created_at = Column(DateTime, default=datetime.now)

    # 双向 ORM 关系
    document = relationship("Document", back_populates="child_chunks")
    parent_chunk = relationship("ParentChunk", back_populates="child_chunks")

    __table_args__ = (
        Index("idx_child_doc_id", "document_id"),
        Index("idx_child_parent_chunk_id", "parent_chunk_id"),
        Index("idx_child_unique_id", "child_id"),
        Index("idx_child_vector_id", "vector_id"),
    )


class ChatHistory(Base):
    """
    多轮对话历史记录表 (Chat Conversation History)
    --------------------------------------------
    基于 session_id 维护用户的连续会话上下文。
    记录用户问题、助手最终回答、参考的文档 ID 集合以及完整的引用来源 JSON，
    支持在界面端展示完整的引用证据链和会话回溯。
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 会话唯一标识 UUID：支持多用户、多窗口会话隔离
    session_id = Column(String(100), nullable=False, comment="会话唯一UUID")

    # 用户原始提问内容
    user_message = Column(Text, nullable=False, comment="用户提问")

    # 系统大模型生成的最终回答
    assistant_message = Column(Text, nullable=False, comment="系统最终回答")

    # 该轮问答所限定检索或命中的文档 ID 集合 (逗号分隔，如 '1,3')
    document_ids = Column(String(500), nullable=True, comment="引用命中参考的文档ID集合(逗号分隔)")

    # JSON 格式序列化的实际命中片段溯源信息 (包含文件名、父块ID、子块ID、正文摘要)
    # 用于前端抽屉式展开核查证据来源
    used_chunks = Column(Text, nullable=True, comment="JSON格式存储实际引用的块及溯源信息")

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_chat_session", "session_id"),
        Index("idx_chat_time", "created_at"),
    )
