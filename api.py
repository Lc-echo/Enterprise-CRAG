"""
Enterprise-CRAG 微服务后端 (FastAPI RESTful Microservice API)
=============================================================
核心架构职责与生产标准：
1. 【标准化 RESTful 协议接口】：
   - POST   /api/v1/chat              : 执行完整的 CRAG 状态图智能问答，支持滑动窗口上下文与溯源落盘；
   - POST   /api/v1/documents/upload  : 多格式文件动态上传，原子化完成解析、入库、父子分块与双库同步；
   - GET    /api/v1/documents         : 查询知识库活跃文档元数据清单；
   - DELETE /api/v1/documents/{id}    : 软删除文档并同步清理向量库物理索引与 BM25 词表；
   - GET    /api/v1/sessions          : 获取所有历史会话摘要（ChatGPT 风格首问标题与更新时间）；
   - GET    /api/v1/sessions/{id}/history : 查询指定会话的完整历史问答明细；
   - DELETE /api/v1/sessions/{id}     : 清空指定会话的问答历史；
   - GET    /health                   : 生产级健康检查探针 (输出数据库、向量库与活跃模型指标)；
2. 【Pydantic 严格输入输出类型校验】：
   - 自动生成符合 OpenAPI 规范的 Swagger UI 交互式在线调试文档 (/docs)；
3. 【CORS 跨域资源共享中间件】：
   - 原生支持前端 (React / Vue / 原生 JS) 跨源安全调用。
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.config import config
from core.service import RAGService

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

# =============================================================================
# 1. 初始化核心业务服务单例 (面向各表现层的统一服务层)
# =============================================================================
rag_service = RAGService(config)
db_manager = rag_service.db_manager

# 初始化 FastAPI 应用实例
app = FastAPI(
    title="Enterprise-CRAG 企业知识智能问答系统 API",
    description="基于 LangGraph 状态图、父子文档拓扑分块与自反思回路的高可用企业知识库问答微服务",
    version="2.5.0",
)

# 允许跨域请求 (CORS 中间件配置)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2. Pydantic 请求与响应模型契约 (Request / Response Schemas)
# =============================================================================
class ChatRequest(BaseModel):
    """问答请求数据载荷"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户自然语言问题")
    session_id: Optional[str] = Field(None, description="会话唯一标识 UUID；若不提供则由系统自动生成")
    filter_doc_ids: Optional[List[int]] = Field(None, description="限定检索的知识库主文档 ID 列表；若留空则全库检索")


class SourceCitation(BaseModel):
    """溯源引用片段数据契约"""
    filename: str = Field(description="来源文件名")
    parent_id: str = Field(description="命中的大父块业务ID (如 doc_1_p0)")
    child_id: str = Field(description="命中的细粒度子块业务ID (如 doc_1_c0_0)")
    preview: str = Field(description="命中文本片段内容摘录")


class ChatResponse(BaseModel):
    """问答响应数据载荷"""
    session_id: str = Field(description="所属会话 UUID")
    question: str = Field(description="原始用户提问")
    answer: str = Field(description="大模型综合生成的最终结构化回答")
    web_search_used: bool = Field(description="是否触发了外部公网搜索补充知识")
    retry_count: int = Field(description="事实一致性反思重试次数")
    sources: List[SourceCitation] = Field(description="引用的细粒度证据片段列表")
    execution_steps: List[str] = Field(description="CRAG 状态机流转轨迹节点列表")
    created_at: str = Field(description="问答完成时间戳")


class DocumentItem(BaseModel):
    """知识库文档信息实体"""
    id: int
    filename: str
    file_size: int
    chunk_count: int
    created_at: Optional[str]
    is_active: bool


# =============================================================================
# 3. 核心业务接口实现 (Endpoints)
# =============================================================================
@app.get("/", include_in_schema=False)
async def root():
    """访问根路径自动跳转至交互式 Swagger API 文档"""
    return RedirectResponse(url="/docs")


@app.post("/api/v1/chat", response_model=ChatResponse, summary="执行 CRAG 智能问答")
async def chat_endpoint(req: ChatRequest):
    """
    【核心问答接口】
    调用流程通过 RAGService 统一纳管：
    1. 自动维护会话 ID 与滑动窗口上下文；
    2. 驱动 LangGraph 状态图执行：检索 -> 评分 -> 联网补偿(若需) -> 生成 -> 幻觉核查；
    3. 结果原子落盘并返回结构化响应体。
    """
    try:
        res = rag_service.chat(
            question=req.question,
            session_id=req.session_id,
            filter_doc_ids=req.filter_doc_ids,
        )
    except Exception as e:
        logger.error("LangGraph 状态图执行失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"智能体执行异常: {str(e)}")

    return ChatResponse(
        session_id=res["session_id"],
        question=res["question"],
        answer=res["answer"],
        web_search_used=res["web_search_used"],
        retry_count=res["retry_count"],
        sources=[SourceCitation(**s) for s in res["sources"]],
        execution_steps=res["execution_steps"],
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.post("/api/v1/documents/upload", summary="上传新文档并实时解析分块入库")
async def upload_document(file: UploadFile = File(...)):
    """
    【动态文件上传接口】
    支持格式：PDF, Word (.docx), TXT, Markdown
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="不能上传空文件")

    try:
        return rag_service.ingest_document_bytes(file_bytes=file_bytes, filename=file.filename)
    except Exception as e:
        logger.error("上传处理失败 %s: %s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件处理异常: {str(e)}")


@app.get("/api/v1/documents", response_model=List[DocumentItem], summary="获取知识库文档列表")
async def list_documents():
    """获取所有处于活跃状态 (is_active=True) 的知识库文档元数据"""
    docs = db_manager.get_all_documents(active_only=True)
    return [
        DocumentItem(
            id=d.id,
            filename=d.filename,
            file_size=d.file_size or 0,
            chunk_count=d.chunk_count or 0,
            created_at=d.created_at.strftime("%Y-%m-%d %H:%M:%S") if d.created_at else None,
            is_active=d.is_active,
        )
        for d in docs
    ]


@app.delete("/api/v1/documents/{doc_id}", summary="删除知识库文档")
async def delete_document(doc_id: int):
    """
    下线并删除指定知识库文档：
    1. 关系数据库执行软删除 (is_active=False)，保留合规审计追踪；
    2. 向量数据库物理删除该文档下的所有向量索引；
    3. 同步清理 BM25 词表缓存。
    """
    success = rag_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="文档未找到或已删除")
    return {"status": "success", "message": f"文档 (ID={doc_id}) 已成功下线并清理向量与词表"}


@app.get("/api/v1/sessions", summary="获取历史会话列表")
async def list_sessions(limit: int = Query(20, ge=1, le=100)):
    """获取最近活跃的会话列表，附带首条提问智能提炼的标题与时间"""
    return db_manager.get_all_sessions(limit=limit)


@app.get("/api/v1/sessions/{session_id}/history", summary="获取指定会话问答记录")
async def get_session_history(session_id: str):
    """获取指定会话下的全部历史问答明细"""
    records = db_manager.get_chat_history(session_id, limit=50)
    return [
        {
            "id": r.id,
            "user_message": r.user_message,
            "assistant_message": r.assistant_message,
            "document_ids": r.document_ids,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
        }
        for r in records
    ]


@app.delete("/api/v1/sessions/{session_id}", summary="清空指定会话记录")
async def clear_session(session_id: str):
    """物理清空指定会话的历史聊天记录"""
    db_manager.clear_session_history(session_id)
    return {"status": "success", "message": "会话历史已清空"}


@app.get("/health", summary="系统健康探针")
async def health_check():
    """生产级微服务健康探针，输出系统各项核心组件运行状态"""
    all_docs = db_manager.get_all_documents(active_only=True)
    return {
        "status": "healthy",
        "service": "Enterprise-CRAG",
        "version": "2.5.0",
        "db_backend": db_manager.active_backend,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "llm_model": config.LLM_MODEL_NAME,
        "eval_model": config.EVAL_MODEL_NAME,
        "active_documents": len(all_docs),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, reload=False)
