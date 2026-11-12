from agent.state import GraphState
from agent.graph import build_crag_graph
# 别名导出编译实例：保持包属性 agent.graph 指向子模块本身，
# 避免实例变量遮蔽同名子模块 (import agent.graph 拿到的应是 module)
from agent.graph import graph as compiled_graph

__all__ = ["GraphState", "build_crag_graph", "compiled_graph"]
