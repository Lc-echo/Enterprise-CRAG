"""
Enterprise-CRAG 命令行交互式客户端 (CLI Terminal Interactive Client)
====================================================================
核心架构职责与终端交互特性：
1. 【沉浸式终端会话体验】：
   - 采用纯控制台循环监听，零 Web 依赖，方便研发人员在服务器或轻量终端极速调试；
2. 【实时可视化状态流转轨迹 (Colored Execution Steps)】：
   - 动态映射并打印每个问答经历的图节点流转路径：
     检索 -> 评分(合格/不相关) -> 联网补偿 -> 大模型综合生成 -> 幻觉核查(通过/重试)；
3. 【细粒度引用证据溯源展示】：
   - 打印命中的文件名称、父块 ID 与摘录正文；
4. 【内置运维与调试快捷指令】：
   - /docs    : 查看当前知识库已收录的活跃文档元数据 (ID、文件名、切块数)；
   - /history : 打印当前会话已积累的多轮历史对话；
   - /clear   : 清空当前会话上下文并重置 session_id；
   - /exit    : 安全退出交互式终端。
"""

import sys
import uuid
import logging

from config.config import config
from core.service import RAGService

# 降低第三方库冗余日志，保持控制台界面清爽专注
logging.basicConfig(level=logging.WARNING)


def print_banner():
    """打印控制台字符欢迎横幅与指令指南"""
    banner = r"""
========================================================================
   ____ ____    _    ____   Enterprise Corrective RAG
  / ___|  _ \  / \  / ___|  企业级自校正多路混合检索问答系统
 | |   | |_) |/ _ \| |  _   基于 LangGraph + 父子分块 + 幻觉核查回路
 | |___|  _ </ ___ \ |_| |  
  \____|_| \_\_/   \_\____|  v2.5.0 | 高可用生产架构
========================================================================
指令提示:
  /docs    : 查看知识库文档列表
  /history : 查看当前会话历史问答
  /clear   : 清空当前会话记忆
  /exit    : 退出程序
========================================================================
"""
    print(banner)


def main():
    """CLI 客户端入口主循环"""
    print_banner()

    # 1. 初始化核心业务服务 (统一纳管元数据库、向量库与状态机)
    print("⏳ 正在初始化 Enterprise-CRAG 核心服务...")
    rag_service = RAGService(config)
    db_manager = rag_service.db_manager

    print(f"✅ 数据库已就绪 [{db_manager.active_backend}]")
    print(f"✅ 嵌入模型与向量存储已就绪 [{config.VECTOR_STORE_TYPE}]")
    print("✅ 智能体状态机编译完成! 开始对话体验。\n")

    # 分配初始会话 UUID
    session_id = str(uuid.uuid4())

    while True:
        try:
            # 监听终端用户输入
            user_input = input("\n👤 用户 > ").strip()
            if not user_input:
                continue

            # -------------------------------------------------------------
            # 内置快捷运维命令处理
            # -------------------------------------------------------------
            if user_input.lower() in ("/exit", "exit", "quit"):
                print("👋 感谢使用 Enterprise-CRAG，再见！")
                sys.exit(0)

            if user_input.lower() == "/clear":
                db_manager.clear_session_history(session_id)
                session_id = str(uuid.uuid4())
                print("🧹 当前会话历史已清空，分配新会话 ID。")
                continue

            if user_input.lower() == "/docs":
                docs = db_manager.get_all_documents(active_only=True)
                print(f"\n📚 知识库现有活跃文档 ({len(docs)} 篇):")
                for d in docs:
                    print(f"  - [ID: {d.id}] {d.filename} (切块: {d.chunk_count}, 上传: {d.created_at})")
                continue

            if user_input.lower() == "/history":
                records = db_manager.get_chat_history(session_id)
                print(f"\n📜 当前会话历史 ({len(records)} 轮):")
                for r in records:
                    print(f"  问: {r.user_message}")
                    print(f"  答: {r.assistant_message[:100]}...\n")
                continue

            print("🤖 智能体执行流转ing:")
            # -------------------------------------------------------------
            # 驱动统一服务执行问答与自动持久化
            # -------------------------------------------------------------
            res = rag_service.chat(
                question=user_input,
                session_id=session_id,
            )

            # 格式化打印流转轨迹指示灯
            steps = res.get("execution_steps", [])
            web_used = res.get("web_search_used", False)
            step_icons = {
                "retrieve": "🔍 [多路混合检索与父块回溯]",
                "grade_documents": f"📊 [文档相关度评分: {'合格' if not web_used else '不相关'}]",
                "web_search": "🌐 [触发联网搜索补偿]",
                "generate": "✍️ [大模型综合生成]",
                "hallucination_check": "🛡️ [幻觉一致性质检: 通过]",
            }
            step_display = " -> ".join([step_icons.get(s, s) for s in steps])
            print(f"   轨迹: {step_display}")

            # 打印最终大模型答案
            answer = res.get("answer", "未能生成回答")
            print(f"\n🤖 助手回答:\n{answer}\n")

            # 打印参考来源依据
            sources = res.get("sources", [])
            if sources:
                print("--------------------------------------------------")
                print(f"📖 参考来源依据 (共命中 {len(sources)} 处细粒度文档片段):")
                for idx, src in enumerate(sources, 1):
                    print(f"  [{idx}] 来源文件: {src['filename']} | 关联父块: {src['parent_id']}")
                    print(f"      摘录: {src['preview']}")
                print("--------------------------------------------------")

        except KeyboardInterrupt:
            print("\n👋 用户终止，程序退出。")
            break
        except Exception as e:
            print(f"❌ 运行异常: {e}")


if __name__ == "__main__":
    main()
