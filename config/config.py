"""
Enterprise-CRAG 配置管理模块 (Unified Configuration Center)
============================================================
架构职责：
1. 统一纳管整个系统所有的基础路径、模型名称、数据库凭证与超参数；
2. 基于 `python-dotenv` 自动加载根目录 `.env` 文件，实现无侵入式环境变量注入；
3. 支持安全缺省值（Safe Defaults）与动态路径计算（Pathlib 推导）；
4. 提供开箱即用的全局单例对象 `config`，供全系统各模块直接导入使用。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 自动定位并加载项目根目录下的 .env 环境变量配置文件
# override=False 保证已有系统环境变量优先级高于 .env
load_dotenv()


class Config:
    """企业级统一配置类 (集中参数中枢)"""

    # =========================================================================
    # 1. 基础路径配置 (Filesystem & Storage Paths)
    # =========================================================================
    # resolve(): 解析为绝对路径并处理软链接，确保跨目录运行时路径依然准确
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # 本地待处理知识库文档存储目录（放置 PDF、Word、TXT、MD 等原始文件）
    DOCS_DIR: str = os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "docs"))

    # BM25 稀疏检索离线分词与词表序列化缓存路径 (避免每次重启都重新构建倒排索引)
    CHUNKS_PATH: str = os.getenv("CHUNKS_PATH", str(BASE_DIR / "data" / "chunks.pkl"))

    # =========================================================================
    # 2. 大语言模型 (LLM) 推理配置
    # =========================================================================
    # 兼容 OpenAI 协议的 API Key (默认对接阿里云百炼 DashScope)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # API 接口的基础请求地址 (DashScope 兼容模式 endpoint)
    OPENAI_BASE_URL: str = os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 主力问答生成模型名称 (用于在 generate_node 中综合上下文撰写最终回答)
    # 推荐: qwen3.7-flash, qwen-plus, 或 gpt-4o-mini
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen3.7-flash")

    # 评测与决策模型名称 (用于 grade_documents 文档打分、hallucination_check 幻觉质检)
    # 通常使用轻量、快速、确定性高的模型，temperature 设为 0 以保证判断稳定
    EVAL_MODEL_NAME: str = os.getenv("EVAL_MODEL_NAME", "qwen3.7-flash")

    # =========================================================================
    # 3. 向量化嵌入模型 (Embedding Engine) 配置
    # =========================================================================
    # 嵌入提供商策略：
    # - 'local'     : 加载本地 HuggingFace 开源嵌入模型 (离线免联网、速度快、数据隐私安全)
    # - 'dashscope' : 调用阿里云百炼在线 text-embedding-v1 API (免本地显存与内存开销)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")

    # 阿里云百炼 API Key (若未单独配置则复用 OPENAI_API_KEY)
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    # 本地开源嵌入模型物理存储路径 (需预先下载 BAAI/bge-large-zh-v1.5 或同等模型)
    LOCAL_EMBEDDING_PATH: str = os.getenv(
        "LOCAL_EMBEDDING_PATH",
        r"D:\llm\Local_model\BAAI\bge-large-zh-v1___5"
    )

    # =========================================================================
    # 4. 向量数据库 (Vector Database) 配置
    # =========================================================================
    # 向量库底层引擎：默认使用 Milvus 分布式/独立向量数据库 (默认端口 19530)
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "milvus")

    # Milvus 向量数据库连接参数 (服务端口 19530:19530)
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", 19530))
    MILVUS_URI: str = os.getenv(
        "MILVUS_URI",
        f"http://{os.getenv('MILVUS_HOST', '127.0.0.1')}:{os.getenv('MILVUS_PORT', 19530)}"
    )
    MILVUS_USER: str = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_PARENT_COLLECTION: str = os.getenv("MILVUS_PARENT_COLLECTION", "parent_documents")
    MILVUS_CHILD_COLLECTION: str = os.getenv("MILVUS_CHILD_COLLECTION", "child_documents")

    # Chroma 本地持久化数据目录 (存储 sqlite 元数据与 hnsw 索引二进制文件，保留作为兼容支持)
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(BASE_DIR / "data" / "chroma_db")
    )

    # =========================================================================
    # 5. 关系型元数据数据库配置 (MySQL / SQLite 双模高可用)
    # =========================================================================
    # 首选数据库类型：'mysql' 或 'sqlite'
    # 高可用容灾设计：若配置为 'mysql' 但网络不通或未安装，系统将自动无缝降级为本地 SQLite！
    DB_TYPE: str = os.getenv("DB_TYPE", "mysql")
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "root")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "rag")

    # SQLite 本地兜底持久化文件路径
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", str(BASE_DIR / "data" / "rag_metadata.db"))

    # =========================================================================
    # 6. RAG 父子拓扑分块与检索超参数 (Chunking & Retrieval Parameters)
    # =========================================================================
    # 父文档块 (Parent Chunk) 大小：
    # 语义完整大颗粒度 (建议 1200~2000 字符)，送入大模型生成，保证段落上下文不被割裂
    PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", 1500))
    PARENT_CHUNK_OVERLAP: int = int(os.getenv("PARENT_CHUNK_OVERLAP", 200))

    # 子文档块 (Child Chunk) 大小：
    # 高信息密度小颗粒度 (建议 300~500 字符)，专用于向量匹配与 BM25 检索，保证召回信噪比
    CHILD_CHUNK_SIZE: int = int(os.getenv("CHILD_CHUNK_SIZE", 400))
    CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", 60))

    # 多路混合检索粗排候选数量 (初筛候选池规模，扩大至 15~20 避免漏召回)
    RETRIEVAL_INITIAL_TOP_K: int = int(os.getenv("RETRIEVAL_INITIAL_TOP_K", 16))

    # 最终送入大模型生成阶段的核心父文档块候选数量 (经 Reranker 交叉注意力精排后截取)
    TOP_K: int = int(os.getenv("TOP_K", 4))

    # 多路混合检索融合比重 (Dense Vector vs Sparse BM25)：
    # 取值范围 0.0 ~ 1.0。0.5 表示向量相似度打分与 BM25 词频匹配打分各占 50% 权重
    HYBRID_SEARCH_RATIO: float = float(os.getenv("HYBRID_SEARCH_RATIO", 0.5))

    # =========================================================================
    # 6.1 Reranker 重排模型配置 (Cross-Encoder 精排层)
    # =========================================================================
    # 是否启用 Cross-Encoder 精排模型 (默认开启)
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")

    # 本地或 HuggingFace 开源 Reranker 模型标识或物理路径
    # 推荐: BAAI/bge-reranker-base, BAAI/bge-reranker-large
    RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-base")

    # =========================================================================
    # 7. CRAG 自校正与反思控制参数 (Self-Correction & Guardrails)
    # =========================================================================
    # 幻觉反思循环最大重试上限：
    # 防止大模型针对极端困难问题在反思回路中无限递归，达到该阈值时触发安全熔断
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))

    # 是否开启外部公网搜索引擎兜底补偿 (当本地私有知识库评定不相关时触发 DuckDuckGo)
    ENABLE_WEB_SEARCH: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("true", "1", "yes")

    # =========================================================================
    # 8. 网络服务与日志配置 (Network & Observability)
    # =========================================================================
    # FastAPI/Uvicorn 后端服务监听 IP（"0.0.0.0" 表示监听所有网卡，允许局域网外部访问；"127.0.0.1" 则仅限本机访问）
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    # 后端 API 服务监听端口号（默认 8000，int() 将环境变量字符串转为整数）
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    # 全局日志记录级别（可选 DEBUG / INFO / WARNING / ERROR，默认 INFO 记录关键业务运行日志）
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# 全局单例配置对象，供外部模块一键导入：from config.config import config
config = Config()
