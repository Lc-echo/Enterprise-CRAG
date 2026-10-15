"""
Enterprise-CRAG 向量存储与两阶段混合检索管理器
==============================================
核心架构职责与生产设计：
1. 【子文档单集合存储 (Single Child Collection)】：
   - 彻底废除父文档假向量化，向量库仅维护单一集合 (child_documents)；
   - 专存 400 字高密度语义小块，支撑密集语义近邻检索与 BM25 词频匹配；
   - 父文档由关系型数据库 (DatabaseManager) 统一维护和反查，零显存/零接口浪费；
2. 【两阶段检索：Dense + BM25 初筛 -> Cross-Encoder 语义精排】：
   - 阶段一：Dense 语义向量 + BM25 稀疏检索初筛 Top-16 候选切片 (EnsembleRetriever RRF 融合)；
   - 阶段二：本地轻量级 Cross-Encoder 重排模型 (bge-reranker) 深度计算交叉注意力，截取最优 Top-4；
3. 【高可用容灾与句柄热自愈机制 (Self-Healing)】：
   - 提供 reload() 方法，针对并发场景下向量库底层句柄陈旧自动热重载并重试；
   - 终极降级网：向量库异常自动切换至 BM25 兜底，Reranker 不可用自动回退初筛顺序；
4. 【子块逆向拓扑回溯父块 (Parent Context Expansion)】：
   - 保持完整段落语境，通过 parent_id 由关系数据库毫秒级回溯，缺失时安全降级子块正文。
"""

import os
import json
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    import pymilvus
    from pymilvus import MilvusClient
    from langchain_milvus import Milvus
except ImportError:
    pymilvus = None
    MilvusClient = None
    Milvus = None

try:
    import chromadb
    from langchain_chroma import Chroma
except ImportError:
    chromadb = None
    Chroma = None

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config.config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """企业级向量数据库与两阶段混合检索管理器 (对接 Milvus:19530，兼容 Chroma)"""

    def __init__(self, custom_config=None):
        """
        初始化向量数据库管理器、嵌入模型、单集合句柄与 Reranker
        :param custom_config: 可选的自定义配置对象
        """
        self.config = custom_config or config
        self.embeddings = self._init_embeddings()  # 嵌入式模型
        self.vector_store_type = getattr(self.config, "VECTOR_STORE_TYPE", "milvus").lower()

        self.chroma_client = None
        self.milvus_client = None
        self.parent_store = None  # 架构重构：父文档不进行向量化，该属性已废弃并置 None
        self.child_store = None   # 单一子块向量集合
        self.vector_store = None  # 别名，指向子块向量集合

        # 初始化底层客户端与单一集合句柄 (仅 child_documents)
        self._init_clients_and_stores()

        # BM25 稀疏检索器实例 (基于本地 chunks.pkl 词表构建)
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._load_bm25_cache()

        # Cross-Encoder 语义精排器 (如 bge-reranker)
        self.reranker = self._init_reranker()

    def _init_clients_and_stores(self) -> None:
        """初始化底层向量引擎客户端与单一子块集合 (child_documents)"""
        if self.vector_store_type == "milvus":
            self.milvus_client = self._init_milvus_client()
            self._init_milvus_stores()
        else:
            self.chroma_client = self._init_chroma_client()
            self._init_chroma_stores()

    def _init_milvus_client(self):
        """初始化 MilvusClient 客户端"""
        if MilvusClient is None:
            logger.warning("pymilvus 未安装，Milvus 客户端不可用")
            return None
        uri = getattr(
            self.config,
            "MILVUS_URI",
            f"http://{getattr(self.config, 'MILVUS_HOST', '127.0.0.1')}:{getattr(self.config, 'MILVUS_PORT', 19530)}",
        )
        logger.info("初始化 Milvus 客户端，连接地址: %s", uri)
        try:
            return MilvusClient(uri=uri)
        except Exception as e:
            logger.error("连接 Milvus 客户端异常 (%s): %s", uri, e)
            return None

    def _init_milvus_stores(self) -> None:
        """初始化 Milvus 单一子文档集合 (child_documents)"""
        if Milvus is None:
            logger.error("langchain_milvus 未安装，无法初始化 Milvus 向量集合")
            self.child_store = None
            self.vector_store = None
            self.parent_store = None
            return

        uri = getattr(
            self.config,
            "MILVUS_URI",
            f"http://{getattr(self.config, 'MILVUS_HOST', '127.0.0.1')}:{getattr(self.config, 'MILVUS_PORT', 19530)}",
        )
        c_col = getattr(self.config, "MILVUS_CHILD_COLLECTION", "child_documents")

        logger.info("连接 Milvus 子块向量集合: URI=%s, 集合=%s", uri, c_col)
        try:
            self.child_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=c_col,
                connection_args={"uri": uri},
                auto_id=True,
            )
            self.vector_store = self.child_store
            self.parent_store = None
        except Exception as e:
            logger.error("初始化 Milvus 集合失败: %s", e)
            self.child_store = None
            self.vector_store = None
            self.parent_store = None

    def _init_chroma_client(self):
        """初始化持久化 Chroma 客户端 (PersistentClient)"""
        if chromadb is None:
            logger.warning("chromadb 未安装，Chroma 客户端不可用")
            return None
        persist_dir = getattr(self.config, "CHROMA_PERSIST_DIR", "")
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            logger.info("初始化 Chroma 持久化客户端，存储目录: %s", persist_dir)
            return chromadb.PersistentClient(path=persist_dir)
        return None

    def _init_chroma_stores(self) -> None:
        """初始化 Chroma 单一子文档集合 (child_documents)"""
        if self.chroma_client and Chroma is not None:
            self.child_store = Chroma(
                client=self.chroma_client,
                collection_name="child_documents",
                embedding_function=self.embeddings,
            )
            self.vector_store = self.child_store
            self.parent_store = None
        else:
            self.child_store = None
            self.vector_store = None
            self.parent_store = None

    def _init_reranker(self):
        """
        初始化 Cross-Encoder 重排模型工厂：
        支持加载本地权重或 HuggingFace 模型标识 (如 BAAI/bge-reranker-base)。
        采用工业级高可用防崩设计：若未安装依赖或模型加载失败，自动记录警告并无缝回退至 RRF 初筛结果。
        """
        if not getattr(self.config, "RERANKER_ENABLED", True):
            logger.info("Reranker 精排模块已按配置禁用 (RERANKER_ENABLED=False)")
            return None

        model_name_or_path = getattr(self.config, "RERANKER_MODEL_PATH", "BAAI/bge-reranker-base")
        try:
            from sentence_transformers import CrossEncoder
            logger.info("正在加载 Cross-Encoder 重排模型: %s", model_name_or_path)
            reranker = CrossEncoder(model_name_or_path, max_length=512, device="cpu")
            logger.info("✅ Cross-Encoder 重排模型加载成功")
            return reranker
        except Exception as e:
            logger.warning(
                "Cross-Encoder 重排模型加载失败 (%s: %s)，将自动回退为基于 RRF 融合打分的初筛结果",
                model_name_or_path,
                e,
            )
            return None

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        使用 Cross-Encoder 对初筛候选切片进行细粒度语义交叉重排：
        :param query: 用户提问
        :param documents: 经 Dense + BM25 初筛出的候选切片列表 (通常 Top 15~20)
        :param top_k: 精排后最终截取的切片数量 (通常 Top 3~4)
        :return: 按 Cross-Encoder 得分降序排序并截取的最优切片列表
        """
        if not documents:
            return []
        if not self.reranker:
            # 优雅降级：若未配置或加载失败，直接按初始顺序截取 Top-K
            return documents[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = []
            for doc, score in doc_score_pairs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)

            logger.info(
                "✅ Cross-Encoder 重排完成: 初筛候选 %d 个 -> 精排输出 %d 个 (Top-1 分数: %.4f)",
                len(documents),
                len(reranked_docs),
                float(doc_score_pairs[0][1]),
            )
            return reranked_docs
        except Exception as e:
            logger.warning("Cross-Encoder 重排执行异常 (%s)，回退直接截取初筛结果", e)
            return documents[:top_k]

    def reload(self) -> None:
        """
        【生产级自愈机制：重新初始化向量客户端与集合句柄】
        适用场景：
        当有外部脚本（如 ingest.py 或多线程 API）修改了底层索引或持久化文件，
        导致当前进程持有的连接句柄陈旧或失效时，通过本方法热重载客户端与集合，自愈连接。
        """
        try:
            self._init_clients_and_stores()
            self._load_bm25_cache()
            logger.info("✅ 向量存储管理器已成功重新载入最新的索引状态 (%s)", self.vector_store_type)
        except Exception as e:
            logger.error("重新载入向量存储异常: %s", e)

    def _init_embeddings(self):
        """
        初始化 Embedding 模型工厂：
        支持本地离线模型与在线百炼 API 自适应路由：
        1. 优先检测 EMBEDDING_PROVIDER='local' 或本地模型物理路径是否存在；
           若存在则加载本地 HuggingFace BGE 模型 (BAAI/bge-large-zh-v1.5)；
        2. 若本地加载失败或配置为 'dashscope'，则初始化阿里云百炼 text-embedding-v1 API。
        """
        provider = getattr(self.config, "EMBEDDING_PROVIDER", "local").lower()
        local_path = getattr(self.config, "LOCAL_EMBEDDING_PATH", "")

        # 1. 本地模型优先检测与加载
        if provider == "local" or (local_path and os.path.exists(local_path) and provider != "dashscope"):
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("加载本地 HuggingFace 嵌入模型: %s", local_path)
                return HuggingFaceEmbeddings(
                    model_name=local_path,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},  # 向量归一化，计算内积即为余弦相似度
                )
            except Exception as e:
                logger.warning("加载本地嵌入模型失败 (%s)，回退至 DashScope API...", e)

        # 2. 阿里云百炼 API 嵌入模型兜底
        from langchain_community.embeddings import DashScopeEmbeddings
        logger.info("初始化 DashScope 向量嵌入服务 (text-embedding-v1)")
        api_key = getattr(self.config, "DASHSCOPE_API_KEY", "") or getattr(self.config, "OPENAI_API_KEY", "")
        return DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=api_key,
        )

    def _load_bm25_cache(self):
        """从磁盘 chunks.pkl 反序列化加载 BM25 词表与稀疏检索器"""
        chunks_path = getattr(self.config, "CHUNKS_PATH", "")
        if chunks_path and os.path.exists(chunks_path):
            try:
                # "rb": 以二进制只读模式读取（非纯文本文件，避免编码乱码与字节损坏）
                # pickle.load: 反序列化，将磁盘二进制数据一键原样复原
                with open(chunks_path, "rb") as f:
                    cached_chunks = pickle.load(f)  # 全库所有已入库文档的全部子块列表 (List[Document]，仅含子块无父块)
                if cached_chunks:
                    # from_documents: 给倒排索引器传入全量子块数据源 (cached_chunks)，在内存中建立词表与倒排索引
                    self.bm25_retriever = BM25Retriever.from_documents(cached_chunks)
                    # .k 属性：设置检索结果数量上限 (Top-K)，控制搜索时最多返回前多少篇最相关的文档切片
                    self.bm25_retriever.k = self.config.TOP_K * 2#给倒排索引器传入值
                    logger.info("✅ 成功加载 BM25 词表缓存，共包含 %d 个子块", len(cached_chunks))
            except Exception as e:
                logger.warning("加载 BM25 缓存文件失败: %s", e)

    def save_bm25_cache(self, all_child_chunks: List[Document]):
        """持久化保存 BM25 词表缓存到 chunks.pkl，并重新实例化检索器"""
        chunks_path = getattr(self.config, "CHUNKS_PATH", "")
        if not chunks_path:
            return
        try:
            # exist_ok=True: 若目录已存在则忽略不报错，不存在则自动递归创建
            os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
            # "wb": 二进制覆盖写入模式 (Write Binary，文件不存在则创建，存在则清空重写；专用于写入二进制字节流)
            # pickle.dump: 将内存对象序列化为二进制字节流，直接写入文件句柄 f (持久化保存到磁盘)
            with open(chunks_path, "wb") as f:
                pickle.dump(all_child_chunks, f)
            # 重新实例化：BM25 不支持增量追加且全局词频/IDF已改变，重新构建以实现免重启实时热更新
            self.bm25_retriever = BM25Retriever.from_documents(all_child_chunks)
            self.bm25_retriever.k = self.config.TOP_K * 2
            logger.info("BM25 缓存持久化完成，更新检索器")
        except Exception as e:
            logger.error("保存 BM25 缓存失败: %s", e)

    def add_documents(
        self,
        child_docs: List[Document],
        document_id: int,
        parent_docs: Optional[List[Document]] = None,
        *args,
        **kwargs,
    ) -> List[str]:
        """
        向向量库添加子切片并建立向量索引 (架构重构：父文档免向量化，仅子块入库)：
        1. 注入真实 document_id 作为前置元数据过滤凭据；
        2. 仅存入 child_store (单集合架构)；
        3. 具备维度自动校验与集合重建保护；
        4. 增量更新 BM25 词表缓存。
        :return: 子块向量分配的 ID 列表 (List[str])
        """
        # 兼容老式位置传参 (parent_docs, child_docs, document_id)
        if isinstance(document_id, list) and args and isinstance(args[0], int):
            real_child_docs = document_id
            real_doc_id = args[0]
        else:
            real_child_docs = child_docs
            real_doc_id = document_id

        doc_id_str = str(real_doc_id)
        for doc in real_child_docs:
            doc.metadata["document_id"] = doc_id_str

        # 仅存入子块集合
        try:
            child_vector_ids = self.child_store.add_documents(real_child_docs)
        except Exception as e:
            # 维度自愈机制：切换 Embedding 模型时自动重置集合
            if "dimension" in str(e).lower():
                logger.warning("检测到向量模型维度变更 (%s)，正在自动重置子块向量集合...", e)
                c_col = getattr(self.config, "MILVUS_CHILD_COLLECTION", "child_documents")
                if self.vector_store_type == "milvus" and self.milvus_client:
                    try:
                        self.milvus_client.drop_collection(c_col)
                    except Exception:
                        pass
                    self._init_milvus_stores()
                elif self.vector_store_type == "chroma" and self.chroma_client:
                    try:
                        self.chroma_client.delete_collection("child_documents")
                    except Exception:
                        pass
                    self._init_chroma_stores()
                else:
                    raise e
                child_vector_ids = self.child_store.add_documents(real_child_docs)
            else:
                raise e

        # 统一将 ID 序列化为字符串列表
        c_ids_str = [str(x) for x in child_vector_ids]

        # 增量追加并持久化 BM25 词表
        existing_chunks = []
        chunks_path = getattr(self.config, "CHUNKS_PATH", "")
        if chunks_path and os.path.exists(chunks_path):
            try:
                with open(chunks_path, "rb") as f:
                    existing_chunks = pickle.load(f)
            except Exception:
                existing_chunks = []
        existing_chunks.extend(real_child_docs)
        self.save_bm25_cache(existing_chunks)

        logger.info(
            "文档 (ID=%d) 向量化完成 [%s]: 仅子块入库=%d (父块免向量化)",
            real_doc_id,
            self.vector_store_type,
            len(c_ids_str),
        )
        return c_ids_str

    def hybrid_search_child_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_ids: Optional[List[int]] = None,
        initial_k: Optional[int] = None,
    ) -> List[Document]:
        """
        【两阶段高精度检索：Dense + BM25 混合初筛 -> Cross-Encoder 语义精排】
        ========================================================================
        执行流程：
        1. 阶段一：粗排初筛 (候选量 cand_k，默认 16，取自 RETRIEVAL_INITIAL_TOP_K)；
           - Dense 向量检索与 BM25 稀疏检索分别召回 cand_k 个候选并由 EnsembleRetriever 做 RRF 融合；
           - 解决传统直接截断 Top-4 导致的候选空间受限、漏召回风险；
        2. 阶段二：Cross-Encoder 交叉重排；
           - 调用 bge-reranker 对初筛候选计算 Query 与 Document 的深度语义交互注意力打分；
           - 剔除弱相关与无关噪声切片，截取最优 Top-K (默认 4) 返回；
        3. 生产级容灾自愈：
           - 句柄陈旧自动热重载重试；
           - 若向量库彻底故障，自动无缝降级为 BM25 检索 + Reranker 兜底。
        """
        final_k = top_k or getattr(self.config, "TOP_K", 4)
        cand_k = initial_k or getattr(self.config, "RETRIEVAL_INITIAL_TOP_K", 16)
        cand_k = max(cand_k, final_k)

        def _do_search() -> List[Document]:
            # 粗排多召回候选切片 (cand_k)，为后续 Cross-Encoder 语义精排提供充足候选空间
            search_kwargs: Dict[str, Any] = {"k": cand_k}

            # 1. 前置元数据过滤：限定只在用户选中的特定几篇文档中检索
            if filter_doc_ids:
                str_ids = [str(i) for i in filter_doc_ids]  # 统一转为字符串匹配数据库 VARCHAR 字段
                if self.vector_store_type == "milvus":
                    # Milvus 使用布尔表达式 (expr) 过滤: 如 'document_id in ["1", "2"]'
                    search_kwargs["expr"] = f"document_id in {json.dumps(str_ids)}"
                else:
                    # Chroma 语法规则: 单文档用直接等值 {"k": "v"}，多文档用 {"$in": [...]} 集合包含操作符
                    if len(str_ids) == 1:
                        search_kwargs["filter"] = {"document_id": str_ids[0]}  # 仅查 1 篇: 精确等于 (等于 SQL 的 WHERE doc_id = '1')
                    else:
                        search_kwargs["filter"] = {"document_id": {"$in": str_ids}}  # 查多篇: 包含命中 (等于 SQL 的 WHERE doc_id IN ('1', '2'))

            # dense_retriever (稠密检索器): 这就是【向量检索器】(Dense 即高维稠密向量，内部自动调用 Embedding 计算向量并在 Milvus 中比对余弦相似度)
            dense_retriever = self.child_store.as_retriever(search_kwargs=search_kwargs)

            # 2. 多路加权混合检索：BM25 可用且无文档范围限定时触发 (【向量检索 Dense】+【关键词检索 Sparse】互补)
            if self.bm25_retriever and not filter_doc_ids:
                try:
                    # 计算两路得分融合权重 (默认 50% 语义向量 + 50% BM25 词频)
                    weight_dense = 1.0 - self.config.HYBRID_SEARCH_RATIO
                    weight_sparse = self.config.HYBRID_SEARCH_RATIO
                    self.bm25_retriever.k = cand_k
                    # EnsembleRetriever: LangChain 提供的多路融合检索器，底层使用互惠排名 (RRF) 算法
                    ensemble = EnsembleRetriever(
                        retrievers=[dense_retriever, self.bm25_retriever],
                        weights=[weight_dense, weight_sparse],
                    )
                    # 执行混合检索: 并非简单列表拼接，而是采用 RRF (互惠排名融合) 算法对两路结果加权交叉重排打分
                    raw_docs = ensemble.invoke(query)
                    candidates = self._deduplicate_docs(raw_docs)[:cand_k]
                    # 接入 Cross-Encoder 语义深度精排
                    return self.rerank(query, candidates, top_k=final_k)
                except Exception as e:
                    logger.warning("EnsembleRetriever 融合召回异常，降级为单路向量检索: %s", e)

            # 3. 单路【纯向量检索】初筛
            raw_docs = dense_retriever.invoke(query)
            candidates = self._deduplicate_docs(raw_docs)[:cand_k]
            return self.rerank(query, candidates, top_k=final_k)

        # ---------------------------------------------------------------------
        # 外层调用与三级容灾自愈体系
        # ---------------------------------------------------------------------
        try:
            # 第一级：正常执行混合/向量检索
            return _do_search()
        except Exception as e:
            # 第二级自愈：若遇到连接超时或索引句柄陈旧，自动调用 reload() 刷新客户端连接并重试
            logger.warning("检索遭遇索引句柄陈旧异常 (%s)，自动刷新向量库连接并重试...", e)
            self.reload()
            try:
                return _do_search()
            except Exception as e2:
                # 第三级终极容灾：向量库彻底故障 (如宕机)，无缝降级为纯 BM25 稀疏检索兜底，力保系统不崩
                logger.error("向量检索重试依然失败 (%s)，使用 BM25 稀疏检索兜底保护系统", e2)
                if self.bm25_retriever:
                    try:
                        self.bm25_retriever.k = cand_k
                        raw_bm25 = self.bm25_retriever.invoke(query)[:cand_k]
                        return self.rerank(query, raw_bm25, top_k=final_k)
                    except Exception:
                        pass
                return []

    def expand_parent_documents(
        self,
        child_docs: List[Document],
        db_manager: Optional[Any] = None,
    ) -> List[Document]:
        """
        【核心架构：子块找爸爸 (Parent Context Expansion)】
        ====================================================
        根据检索命中的高精度子块，提取其 metadata 中的 parent_id，并逆向提取完整大父块。
        - 架构重构：彻底废除从向量库反查父块的伪降级逻辑（父文档不入向量库，关系库为唯一权威源）；
        - 性能加速：由 DatabaseManager (MySQL/SQLite) 基于唯一索引毫秒级单次批量反查；
        - 容灾机制：若未注入 db_manager 或反查未命中，直接安全降级采用子块正文作为上下文，保障流程不中断！
        """
        if not child_docs:
            return []

        # 1. 提取去重后的 parent_id 列表 (保序去重：优先保留首选高相关度子块对应的父块)
        parent_id_order = []
        seen_p_ids = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen_p_ids:
                seen_p_ids.add(pid)
                parent_id_order.append(pid)

        # 2. 从关系型元数据库毫秒级直查完整父块 (唯一权威数据源)
        if db_manager and hasattr(db_manager, "get_parent_chunks_by_parent_ids"):
            try:
                db_parent_map = db_manager.get_parent_chunks_by_parent_ids(parent_id_order)
                if db_parent_map:
                    fetched = []
                    for pid in parent_id_order:
                        if pid in db_parent_map:
                            item = db_parent_map[pid]
                            fetched.append(Document(page_content=item["content"], metadata=item["metadata"]))
                    if fetched:
                        logger.info(
                            "从关系型元数据库毫秒级回溯父块成功: 命中 %d/%d 个父块",
                            len(fetched),
                            len(parent_id_order),
                        )
                        return fetched
            except Exception as db_err:
                logger.warning("关系型数据库回溯父块异常 (%s)，将直接降级为子块正文...", db_err)

        # 3. 终极容灾降级兜底：若未注入关系库或未能查出父块，安全采用子块正文作为上下文
        logger.warning("未能通过关系库反查到父块语境，直接降级为子块正文作为上下文")
        return child_docs

    # @staticmethod: 声明为静态方法 (无需传入 self 实例参数，独立无状态的纯工具函数，也可通过 Class._deduplicate_docs 直接调用)
    @staticmethod
    def _deduplicate_docs(docs: List[Document]) -> List[Document]:
        """
        按正文内容精确去重，并严格保持初始的相关度排序 (保序去重)
        - 避免多路召回 (向量 + BM25) 命中相同文本段落导致的大模型上下文冗余
        """
        seen = set()       # 记录已处理过的正文哈希指纹
        deduped = []       # 存放最终去重且保持顺序的文档列表
        for doc in docs:
            content_key = (doc.page_content or "").strip()
            # 只有正文首次出现时才收录进结果集，避免打乱原本从高到低的相关度排序
            if content_key and content_key not in seen:
                seen.add(content_key)
                deduped.append(doc)
        return deduped

    def delete_document_vectors(self, document_id: int) -> None:
        """
        从向量库中物理清理指定文档的所有子切片向量记录 (父块不在向量库中)
        :param document_id: 关系数据库分配的整数文档 ID
        """
        doc_id_str = str(document_id)  # 统一转为字符串匹配数据库 VARCHAR 字段
        try:
            # 分支一：Milvus 物理删除 (基于布尔表达式批量清理)
            if self.vector_store_type == "milvus":
                expr = f'document_id == "{doc_id_str}"'
                if self.child_store:
                    self.child_store.delete(expr=expr)
                logger.info("已从 Milvus 清除文档 (ID=%d) 的子切片向量记录", document_id)
            # 分支二：Chroma 物理删除 (基于 where 字典过滤清理)
            else:
                if self.child_store:
                    self.child_store.delete(where={"document_id": doc_id_str})
                logger.info("已从 Chroma 清除文档 (ID=%d) 的子切片向量记录", document_id)

            # 同步从 BM25 词表缓存与索引中移除该文档的切片，解决数据不一致
            chunks_path = getattr(self.config, "CHUNKS_PATH", "")
            if chunks_path and os.path.exists(chunks_path):
                try:
                    with open(chunks_path, "rb") as f:
                        cached_chunks = pickle.load(f)
                    filtered_chunks = [
                        c for c in cached_chunks
                        if str(c.metadata.get("document_id")) != doc_id_str
                    ]
                    if len(filtered_chunks) != len(cached_chunks):
                        self.save_bm25_cache(filtered_chunks)
                        logger.info("已同步从 BM25 词表中清除文档 (ID=%d) 的 %d 个切片", document_id, len(cached_chunks) - len(filtered_chunks))
                except Exception as bm_err:
                    logger.warning("清理 BM25 缓存切片失败: %s", bm_err)
        except Exception as e:
            logger.error("删除文档向量失败: %s", e)
