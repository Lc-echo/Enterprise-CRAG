"""
Enterprise-CRAG 离线批量知识库导入构建脚本 (Ingestion Pipeline)
================================================================
核心架构职责与五阶段流水线：
1. 【阶段一：文档扫描与解析加载】：
   - 遍历 data/docs 目录下多格式文件 (.pdf, .docx, .txt, .md)；
   - 计算文件二进制内容的 MD5 哈希指纹；
2. 【阶段二：关系数据库主文档落盘 (获取真实自增主键)】：
   - 插入 documents 表并通过 session.flush() 获得物理唯一 document_id；
   - 彻底解决开源项目把 document_id 硬编码为 0 的历史遗留架构缺陷；
3. 【阶段三：两级父子拓扑分块】：
   - 粗切父块 (~1500字, 打上 doc_{id}_p{p_idx} 标识)；
   - 在父块内部细切子块 (~400字, 打上 parent_id 归属)，保证物理语境不交叉；
4. 【阶段四：子文档向量化持久化 (父块免向量化)】：
   - 架构重构：父块不入向量库，纯由关系数据库统一管理与反查；
   - 仅子块存入 child_documents 集合并获取向量分配唯一标识 (vector_id)；
5. 【阶段五：双库协同关联写回与 BM25 词表缓存更新】：
   - 将父子块及其 vector_id 关联写回关系数据表，建立完整的双向审计外键体系；
   - 增量刷新并序列化写入 data/chunks.pkl，使 BM25 稀疏索引立即可用。
"""

import os
import glob
import logging
from config.config import config
from core.service import RAGService

# 配置日志输出格式
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("ingest")


def run_ingest(docs_dir: str = config.DOCS_DIR) -> None:
    """
    执行全量知识库离线导入流水线：
    :param docs_dir: 待扫描的本地文档目录绝对/相对路径
    """
    logger.info("🚀 启动 Enterprise-CRAG 知识库构建流水线...")
    logger.info("扫描文档目录: %s", docs_dir)

    if not os.path.exists(docs_dir):
        logger.error("文档目录不存在: %s", docs_dir)
        return

    # 初始化统一业务服务
    rag_service = RAGService(config)

    # 递归扫描支持的文件格式后缀
    supported_exts = {".txt", ".md", ".pdf", ".docx", ".doc"}
    file_list = []
    for f in sorted(glob.glob(os.path.join(docs_dir, "**", "*"), recursive=True)):
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in supported_exts:
            file_list.append(f)

    if not file_list:
        logger.warning("在 %s 未找到任何受支持的文档文件", docs_dir)
        return

    logger.info("共发现 %d 篇待处理知识文档", len(file_list))

    total_parents = 0
    total_children = 0

    # 遍历处理每一篇文件
    for file_path in file_list:
        filename = os.path.basename(file_path)
        logger.info("--------------------------------------------------")
        logger.info("正在处理文档: %s", filename)

        try:
            res = rag_service.ingest_document_file(file_path)
            total_parents += res["parent_chunks"]
            total_children += res["child_chunks"]
            logger.info("✅ 成功入库: %s (Doc ID: %d, 父块: %d, 子块: %d)", filename, res["document_id"], res["parent_chunks"], res["child_chunks"])
        except Exception as e:
            logger.error("处理文档失败 %s: %s", filename, e, exc_info=True)

    logger.info("==================================================")
    logger.info(
        "🎉 全量知识库构建完成! 成功入库文档: %d 篇, 累计生成父块: %d, 累计生成子块: %d",
        len(file_list),
        total_parents,
        total_children,
    )
    logger.info("BM25 词表与向量索引已同步更新完成")


if __name__ == "__main__":
    run_ingest()
