"""
Enterprise-CRAG 数据库管理器模块 (Database Infrastructure Manager)
==================================================================
核心架构职责：
1. 【MySQL ↔ SQLite 自动无缝降级】：
   - 生产环境优先尝试连接 MySQL 并启用连接池自愈探针 (pool_pre_ping=True)；
   - 本地开发或外部数据库离线时，自动平滑降级为嵌入式本地 SQLite，零外部依赖、开箱即跑！
2. 【自适应数据架构迁移 (_migrate_schema)】：
   - 自动检测历史已有数据表中是否缺失新字段 (如 file_hash, file_size) 并自动 ALTER TABLE 补充；
3. 【彻底根除开源版本 document_id=0 缺陷】：
   - 实现两阶段落库：先保存主文档并通过 `session.flush()` 获取真实自增 ID，再给父子块打标落盘；
4. 【会话历史与智能标题提炼】：
   - 支撑滑动窗口多轮上下文、第一条提问智能标题提取与溯源证据持久化。
"""

import os
import json
import logging
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from config.config import config
from core.models import Base, Document, ParentChunk, ChildChunk, ChatHistory

logger = logging.getLogger(__name__)


class DatabaseManager:
    """双库协同管家 - 统一管理关系型元数据持久化与数据库事务"""

    def __init__(self, custom_config=None):
        """
        初始化数据库管理器
        :param custom_config: 可选的自定义配置对象（在自动化单元测试中支持注入隔离配置）
        """
        self.config = custom_config or config
        self.engine = None
        self.SessionLocal = None
        self.active_backend = "unknown"  # 记录当前实际激活的数据库后端描述字符串
        self.init_database()

    def init_database(self) -> None:
        """
        初始化数据库引擎与连接池：
        1. 优先尝试连接生产级 MySQL 数据库；
        2. 探针验证连通性，若捕获异常则自动触发 _fallback_to_sqlite() 降级；
        3. 确保所有 ORM 数据表结构均已同步就绪。
        """
        # 1. 尝试配置的首选数据库
        if self.config.DB_TYPE.lower() == "mysql":
            mysql_uri = (
                f"mysql+pymysql://{self.config.MYSQL_USER}:{self.config.MYSQL_PASSWORD}"
                f"@{self.config.MYSQL_HOST}:{self.config.MYSQL_PORT}/{self.config.MYSQL_DATABASE}"
                "?charset=utf8mb4"
            )
            try:
                # 创建 MySQL 连接池：
                # pool_pre_ping=True 可以在每次借出连接前发送一次 SELECT 1 探测心跳，避免连接超时断开
                # pool_recycle=3600 每小时自动回收连接，防止 MySQL 默认 8 小时 wait_timeout 强行关闭
                engine = create_engine(
                    mysql_uri,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False,
                )
                # 探针验证连通性：实际发起一次网络连接，验证账号密码与端口是否通畅
                with engine.connect():
                    pass

                # 创建所有不存在的数据表结构
                Base.metadata.create_all(engine)

                # 自适应字段迁移检查（平滑升级历史已有数据表）
                self._migrate_schema(engine)

                self.engine = engine
                self.active_backend = f"MySQL ({self.config.MYSQL_HOST}:{self.config.MYSQL_PORT}/{self.config.MYSQL_DATABASE})"
                logger.info("✅ 成功连接 MySQL 数据库并同步表结构: %s", self.active_backend)
            except Exception as e:
                # 触发容灾降级：当未启动 MySQL 服务或配置错误时，自动切换至 SQLite
                logger.warning(
                    "⚠️ MySQL 连接/建表异常 (%s)，正在自动无缝降级为本地 SQLite 引擎...",
                    e,
                )
                self._fallback_to_sqlite()
        else:
            self._fallback_to_sqlite()

        # 构建 Session 工厂对象：autocommit=False 保证事务显式受控，autoflush=False 避免隐式刷新
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info("✅ 数据库服务就绪，当前活动后端: %s", self.active_backend)

    def _migrate_schema(self, engine) -> None:
        """
        自适应字段迁移辅助函数：
        检查已有的 documents 表，若缺少 file_hash 或 file_size 列，则自动动态执行 ALTER TABLE，
        防止因项目版本升级导致已有数据库报错列缺失。
        """
        from sqlalchemy import inspect, text
        try:
            inspector = inspect(engine)
            if "documents" in inspector.get_table_names():
                existing_cols = {col["name"] for col in inspector.get_columns("documents")}
                with engine.begin() as conn:
                    if "file_hash" not in existing_cols:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN file_hash VARCHAR(64) NULL"))
                        logger.info("🔧 自动迁移: 已为 documents 表补充 file_hash 字段")
                    if "file_size" not in existing_cols:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN file_size INT DEFAULT 0"))
                        logger.info("🔧 自动迁移: 已为 documents 表补充 file_size 字段")
        except Exception as e:
            logger.warning("自适应架构检查提示: %s", e)

    def _fallback_to_sqlite(self) -> None:
        """降级至本地 SQLite 嵌入式文件数据库，确保零外部依赖开箱即跑"""
        sqlite_path = self.config.SQLITE_PATH
        # 自动创建 data/ 目录
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        sqlite_uri = f"sqlite:///{sqlite_path}"
        self.engine = create_engine(sqlite_uri, echo=False)
        Base.metadata.create_all(self.engine)
        self.active_backend = f"SQLite ({sqlite_path})"
        logger.info("📁 已激活本地 SQLite 存储并同步数据表: %s", sqlite_path)

    def get_session(self) -> Session:
        """从连接池中签出一个独立的数据库会话实例 (Session)"""
        return self.SessionLocal()

    def save_document(
        self,
        filename: str,
        file_path: str,
        content: str,
        file_hash: Optional[str] = None,
        file_size: int = 0,
    ) -> int:
        """
        【第一阶段：保存主文档元数据】
        彻底解决原开源项目将 document_id 硬编码为 0 的架构缺陷！
        设计机制：
        1. 实例化 Document 模型并添加到 Session；
        2. 调用 session.flush() 立即执行 INSERT 语句，从数据库序列/自增生成器中获取真实 id；
        3. 立即 commit 事务并返回该真实 document_id，供后续切分与向量库入库时作为唯一外键使用。
        """
        session = self.get_session()
        try:
            doc = Document(
                filename=filename,
                file_path=file_path,
                content=content,
                file_hash=file_hash,
                file_size=file_size,
                chunk_count=0,
                is_active=True,
            )
            session.add(doc)
            # 关键技术点：flush 会将修改发送到数据库缓冲区并立即获取自增的主键 doc.id
            session.flush()
            doc_id = doc.id
            session.commit()
            logger.info("主文档元数据入库成功, 获得真实 document_id=%d", doc_id)
            return doc_id
        except Exception as e:
            session.rollback()
            logger.error("保存主文档失败: %s", e)
            raise e
        finally:
            session.close()

    def save_chunks(
        self,
        document_id: int,
        parent_docs: list,
        child_docs: list,
        child_vector_ids: Optional[list] = None,
        parent_vector_ids: Optional[list] = None,
    ) -> None:
        """
        【第二阶段：携带真实 document_id 保存父子文档块及其与向量库的关联 ID】
        执行步骤：
        1. 批量保存父文档块，并建立 parent_id 业务字符串到数据库真实自增主键的映射字典；
           【架构重构】：父文档纯用于大语境原文回溯，不进行向量化，vector_id 固化为 None；
        2. 遍历子文档块，通过 parent_chunk_map 精准绑定父块的物理外键 parent_chunk_id；
        3. 保存子块对应的向量 ID (vector_id)，打通关系库与向量库的溯源桥梁；
        4. 同步更新主文档记录的 chunk_count 统计数字。
        """
        session = self.get_session()
        try:
            child_vids = child_vector_ids or [None] * len(child_docs)

            # 1. 批量落盘父文档块，并构建 parent_id 到物理主键的映射表 (父块不入向量库，vector_id=None)
            parent_chunk_map: Dict[str, int] = {}
            for i, p_doc in enumerate(parent_docs):
                p_id_str = p_doc.metadata.get("parent_id", f"doc_{document_id}_p{i}")
                parent_chunk = ParentChunk(
                    document_id=document_id,
                    parent_id=p_id_str,
                    content=p_doc.page_content,
                    json_metadata=json.dumps(p_doc.metadata, ensure_ascii=False),
                    vector_id=None,  # 架构重构：父文档不参与向量化，纯由关系库维护权威原文
                )
                session.add(parent_chunk)
                session.flush()
                # 记录业务 parent_id -> 自增物理 ID 的映射
                parent_chunk_map[p_id_str] = parent_chunk.id

            # 2. 保存子文档块，建立精确的外键约束 (parent_chunk_id)
            for j, (c_doc, vid) in enumerate(zip(child_docs, child_vids)):
                target_p_id_str = c_doc.metadata.get("parent_id")
                # 精确匹配该子块对应的父块自增主键
                fk_parent_id = parent_chunk_map.get(target_p_id_str)
                if not fk_parent_id:
                    # 容错兜底：若子块丢失映射，绑定到该文档的首个父块
                    fk_parent_id = next(iter(parent_chunk_map.values()))

                child_id_str = c_doc.metadata.get("child_id", f"doc_{document_id}_c{j}")
                child_chunk = ChildChunk(
                    document_id=document_id,
                    parent_chunk_id=fk_parent_id,
                    child_id=child_id_str,
                    content=c_doc.page_content,
                    json_metadata=json.dumps(c_doc.metadata, ensure_ascii=False),
                    vector_id=str(vid) if vid else None,
                )
                session.add(child_chunk)

            # 3. 回填更新主文档的有效切块计数
            doc = session.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.chunk_count = len(child_docs)

            session.commit()
            logger.info("文档 (id=%d) 父子块成功落库: 父块=%d, 子块=%d", document_id, len(parent_docs), len(child_docs))
        except Exception as e:
            session.rollback()
            logger.error("保存文档块失败: %s", e)
            raise e
        finally:
            session.close()

    def get_all_documents(self, active_only: bool = True) -> List[Document]:
        """
        获取知识库中的文档列表
        :param active_only: True 则只返回未被软删除的正常文档
        """
        session = self.get_session()
        try:
            query = session.query(Document)
            if active_only:
                query = query.filter(Document.is_active == True)
            return query.order_by(Document.created_at.desc()).all()
        finally:
            session.close()

    def get_document_by_id(self, doc_id: int) -> Optional[Document]:
        """按物理主键 ID 查询单个文档实体"""
        session = self.get_session()
        try:
            return session.query(Document).filter(Document.id == doc_id).first()
        finally:
            session.close()

    def get_parent_chunks_by_parent_ids(self, parent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        按 parent_id 列表批量高效查询父块正文与元数据 (毫秒级关系型数据库索引查询)
        :param parent_ids: 父块业务 ID 列表 (如 ['doc_1_p0', 'doc_1_p1'])
        :return: 字典映射 {parent_id: {"content": str, "metadata": dict}}
        """
        if not parent_ids:
            return {}
        session = self.get_session()
        try:
            records = (
                session.query(ParentChunk)
                .filter(ParentChunk.parent_id.in_(parent_ids))
                .all()
            )
            result = {}
            for r in records:
                meta = {}
                if r.json_metadata:
                    try:
                        meta = json.loads(r.json_metadata)
                    except Exception:
                        meta = {}
                meta["parent_id"] = r.parent_id
                meta["document_id"] = str(r.document_id)
                result[r.parent_id] = {
                    "content": r.content,
                    "metadata": meta,
                }
            return result
        finally:
            session.close()

    def soft_delete_document(self, doc_id: int) -> bool:
        """
        软删除指定文档 (is_active = False)：
        保留文档数据以备合规审计，同时在检索和界面列表展示中自动过滤屏蔽
        """
        session = self.get_session()
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.is_active = False
                session.commit()
                logger.info("文档 (id=%d) 已标记软删除", doc_id)
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error("软删除文档失败: %s", e)
            return False
        finally:
            session.close()

    def save_chat_history(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        document_ids: Optional[str] = None,
        used_chunks: Optional[str] = None,
    ) -> None:
        """
        持久化单轮问答历史记录：
        记录提问、回答、限定检索文档 ID 以及完整的引用证据快照 (used_chunks JSON)
        """
        session = self.get_session()
        try:
            record = ChatHistory(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                document_ids=document_ids,
                used_chunks=used_chunks,
            )
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("保存对话记录失败: %s", e)
        finally:
            session.close()

    def get_chat_history(self, session_id: str, limit: int = 10) -> List[ChatHistory]:
        """
        获取指定会话的历史记录（按时间正序排列，用于大模型构建多轮滑动窗口）：
        先倒序取最近 limit 条，再翻转为时间正序，确保语境顺序正确
        """
        session = self.get_session()
        try:
            records = (
                session.query(ChatHistory)
                .filter(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
                .all()
            )
            return list(reversed(records))
        finally:
            session.close()

    def get_all_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        【ChatGPT 风格会话列表提取算法】：
        1. 按 session_id 分组，聚合统计消息条数与最近活跃时间；
        2. 子查询获取每个会话的【第一条用户提问】，提炼为自然语言短标题（如 '请问公司的年假制度...'）；
        3. 格式化活跃时间为 '月-日 时:分'，便于前端侧边栏直观渲染多会话直达按钮。
        """
        session = self.get_session()
        try:
            from sqlalchemy import func
            subq = (
                session.query(
                    ChatHistory.session_id,
                    func.max(ChatHistory.created_at).label("last_active"),
                    func.count(ChatHistory.id).label("msg_count"),
                )
                .group_by(ChatHistory.session_id)
                .order_by(func.max(ChatHistory.created_at).desc())
                .limit(limit)
                .all()
            )
            result = []
            for row in subq:
                # 提取该会话下最早的一条提问作为标题
                first_msg = (
                    session.query(ChatHistory.user_message)
                    .filter(ChatHistory.session_id == row.session_id)
                    .order_by(ChatHistory.created_at.asc())
                    .first()
                )
                raw_text = first_msg[0] if first_msg and first_msg[0] else "新建对话"
                # 超过 22 字符优雅截断并追加省略号
                short_title = raw_text[:22] + "..." if len(raw_text) > 22 else raw_text
                result.append({
                    "session_id": row.session_id,
                    "title": short_title,
                    "last_active": row.last_active.strftime("%m-%d %H:%M") if row.last_active else "",
                    "message_count": row.msg_count,
                })
            return result
        finally:
            session.close()

    def clear_session_history(self, session_id: str) -> None:
        """清空指定会话的全部问答历史记录 (供前端 '清空当前' 按钮调用)"""
        session = self.get_session()
        try:
            session.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
            session.commit()
            logger.info("已清空会话 (%s) 的所有历史对话", session_id)
        except Exception as e:
            session.rollback()
            logger.error("清空会话历史失败: %s", e)
        finally:
            session.close()
